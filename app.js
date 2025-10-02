// ===== Utilities =====
const $ = (sel) => document.querySelector(sel);
const byId = (id) => document.getElementById(id);

const stripPunct = (s) =>
  s.replace(/[\p{P}\p{S}]/gu, ' ')     // 문장부호/기호 제거
   .replace(/\s+/g, ' ')
   .trim()
   .toLowerCase();

const normalizeNumsKo = (s) => {
  // 간단 처리: '2025년' 같은 표기 통일(느슨). 고급 변환은 생략.
  return s.replace(/(\d+)\s*년/g, '$1년').replace(/\s+/g,' ');
};

// 토큰화(공백기반)
const toks = (s) => s.split(/\s+/).filter(Boolean);

// ===== WER(Word Error Rate) =====
// 레벤슈타인 편집거리로 WER 계산 + 정렬 backtrace로 하이라이트용 정렬 생성
function alignTokens(refTokens, hypTokens) {
  const n = refTokens.length, m = hypTokens.length;
  const dp = Array.from({length: n+1}, () => Array(m+1).fill(0));
  const bt = Array.from({length: n+1}, () => Array(m+1).fill(''));

  for (let i=0;i<=n;i++){ dp[i][0]=i; bt[i][0]='D'; }
  for (let j=0;j<=m;j++){ dp[0][j]=j; bt[0][j]='I'; }
  bt[0][0]='';

  for (let i=1;i<=n;i++){
    for (let j=1;j<=m;j++){
      if (refTokens[i-1] === hypTokens[j-1]) { dp[i][j]=dp[i-1][j-1]; bt[i][j]='M'; }
      else {
        const sub = dp[i-1][j-1]+1;
        const del = dp[i-1][j]+1;
        const ins = dp[i][j-1]+1;
        const min = Math.min(sub, del, ins);
        dp[i][j]=min;
        bt[i][j]= (min===sub)?'S':(min===del)?'D':'I';
      }
    }
  }

  // backtrace
  let i=n, j=m;
  const ops = [];
  while (i>0 || j>0){
    const op = bt[i][j];
    if (op==='M' || op==='S'){ ops.push(op); i--; j--; }
    else if (op==='D'){ ops.push('D'); i--; }
    else { ops.push('I'); j--; }
  }
  ops.reverse();
  return {distance: dp[n][m], ops};
}

function calcWER(ref, hyp, options) {
  const { ignorePunct=true, allowNumRead=true } = options;
  let r=ref, h=hyp;
  if (ignorePunct){ r=stripPunct(r); h=stripPunct(h); }
  if (allowNumRead){ r=normalizeNumsKo(r); h=normalizeNumsKo(h); }

  const rt = toks(r), ht = toks(h);
  const {distance, ops} = alignTokens(rt, ht);
  const wer = (rt.length === 0) ? (ht.length === 0 ? 0 : 1) : distance / rt.length;
  return { wer, ops, rt, ht };
}

// 하이라이트 HTML 만들기 (정확 모드)
function renderDiffExact(rt, ht, ops) {
  const refOut = [];
  const hypOut = [];
  let i=0, j=0;
  for (const op of ops){
    if (op==='M'){
      refOut.push(rt[i]); hypOut.push(ht[j]);
      i++; j++;
    } else if (op==='S'){
      refOut.push(`<em class="sub">${rt[i]}</em>`);
      hypOut.push(`<em class="sub">${ht[j]}</em>`);
      i++; j++;
    } else if (op==='D'){
      refOut.push(`<del>${rt[i]}</del>`);
      i++;
    } else if (op==='I'){
      hypOut.push(`<ins>${ht[j]}</ins>`);
      j++;
    }
  }
  // 남은 것(이론상 없음)
  for (; i<rt.length; i++) refOut.push(`<del>${rt[i]}</del>`);
  for (; j<ht.length; j++) hypOut.push(`<ins>${ht[j]}</ins>`);

  return {
    refHTML: refOut.join(' '),
    hypHTML: hypOut.join(' '),
  };
}

// ===== 내용 유사도 (TFJS USE) =====
let useModel = null;
async function loadUSE() {
  if (!useModel) {
    try {
      useModel = await window.universalSentenceEncoder.load();
    } catch (e) {
      console.warn('USE 로드 실패:', e);
    }
  }
  return useModel;
}

async function semanticScore(a, b) {
  // 코사인 유사도 0~1 → 0~100 환산
  const model = await loadUSE();
  if (!model) return null;
  const emb = await model.embed([a, b]);
  const aV = emb.slice([0,0],[1]);   // [1,512]
  const bV = emb.slice([1,0],[1]);
  const sim = await tf.tidy(() => {
    const an = tf.linalg.l2Normalize(aV, 1);
    const bn = tf.linalg.l2Normalize(bV, 1);
    return an.mul(bn).sum(1).array(); // [1]
  });
  const s = sim[0]; // -1~1
  return Math.max(0, Math.min(100, Math.round(((s+1)/2)*100)));
}

// 간단 ROUGE-L 유사도(토큰 공통 LCS 기반, 0~100)
function rougeL(a, b) {
  const A = toks(stripPunct(a)), B = toks(stripPunct(b));
  const n=A.length, m=B.length;
  const dp = Array.from({length:n+1},()=>Array(m+1).fill(0));
  for (let i=1;i<=n;i++) for (let j=1;j<=m;j++)
    dp[i][j] = (A[i-1]===B[j-1]) ? dp[i-1][j-1]+1 : Math.max(dp[i-1][j], dp[i][j-1]);
  const lcs = dp[n][m];
  if (!n && !m) return 100;
  const prec = lcs / (m||1);
  const rec  = lcs / (n||1);
  const beta2 = 1; // F1
  const f = (1+beta2)*prec*rec / ((beta2*prec)+rec || 1);
  return Math.round(f*100);
}

// ===== Speech =====
const live = byId('live');
const recDot = byId('rec-dot');
const btnStart = byId('btn-start');
const btnStop  = byId('btn-stop');

let recog = null;
let finalText = '';

function initSTT() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    live.textContent = '이 브라우저는 음성 인식을 지원하지 않습니다. (Chrome/Edge 권장)';
    btnStart.disabled = true;
    return null;
  }
  const r = new SR();
  r.lang = 'ko-KR';
  r.interimResults = true;
  r.continuous = true;
  r.maxAlternatives = 1;

  r.onstart = () => { recDot.classList.add('live'); live.textContent = '듣는 중…'; finalText=''; };
  r.onresult = (e) => {
    let interim = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const txt = e.results[i][0].transcript;
      if (e.results[i].isFinal) finalText += (finalText ? ' ' : '') + txt;
      else interim += txt + ' ';
    }
    live.textContent = finalText + (interim ? ' ' + interim : '');
  };
  r.onerror = (e) => { console.warn(e); };
  r.onend = () => { recDot.classList.remove('live'); btnStart.disabled=false; btnStop.disabled=true; };
  return r;
}

// ===== Main flow =====
const els = {
  mode: byId('mode'),
  ref: byId('ref'),
  lenRef: byId('len-ref'),
  lenHyp: byId('len-hyp'),
  score: byId('score'),
  refVis: byId('ref-vis'),
  hypVis: byId('hyp-vis'),
  notes: byId('notes'),
  stripPunct: byId('strip-punct'),
  normalizeNum: byId('normalize-num'),
};

function setMetrics({lenRef, lenHyp, score, refHTML, hypHTML, notes}) {
  els.lenRef.textContent = lenRef ?? '-';
  els.lenHyp.textContent = lenHyp ?? '-';
  els.score.textContent  = (score ?? '-').toString();
  els.refVis.innerHTML   = refHTML ?? '';
  els.hypVis.innerHTML   = hypHTML ?? '';
  els.notes.innerHTML    = notes ?? '';
}

btnStart.addEventListener('click', async () => {
  if (!recog) recog = initSTT();
  if (!recog) return;
  setMetrics({ lenRef:'-', lenHyp:'-', score:'-', refHTML:'', hypHTML:'', notes:'' });
  btnStart.disabled = true; btnStop.disabled=false;
  try { recog.start(); } catch(e) { /* already started */ }
});

btnStop.addEventListener('click', async () => {
  if (!recog) return;
  try { recog.stop(); } catch(e) {}
  // 채점
  const refText = els.ref.value.trim();
  const hypText = finalText.trim();
  const ignoreP = els.stripPunct.checked;
  const allowNum = els.normalizeNum.checked;

  const lenRef = toks(ignoreP ? stripPunct(refText) : refText).length;
  const lenHyp = toks(ignoreP ? stripPunct(hypText) : hypText).length;

  if (!refText || !hypText) {
    setMetrics({
      lenRef, lenHyp, score: '-',
      refHTML: refText, hypHTML: hypText,
      notes: `<div class="muted">대본과 발화가 모두 있어야 채점됩니다.</div>`
    });
    return;
  }

  if (els.mode.value === 'exact') {
    const {wer, ops, rt, ht} = calcWER(refText, hypText, {ignorePunct: ignoreP, allowNumRead: allowNum});
    const acc = Math.max(0, Math.round((1 - wer) * 100));
    const {refHTML, hypHTML} = renderDiffExact(rt, ht, ops);
    const notes = `
      <div class="muted">
        • 정확 모드: WER 기반 (치환/삽입/삭제 오차). <em class="sub">노란색</em>=치환, <del>빨강=누락</del>, <ins>초록=불필요</ins>.<br/>
        • 점수 = (1 - WER) × 100
      </div>`;
    setMetrics({ lenRef: rt.length, lenHyp: ht.length, score: acc, refHTML, hypHTML, notes });
  } else {
    // 내용만 모드: USE 코사인 + ROUGE-L 평균
    const use = await semanticScore(refText, hypText); // 0~100 or null
    const rgL = rougeL(refText, hypText);              // 0~100
    const parts = [];
    if (use !== null) parts.push(use);
    parts.push(rgL);
    const acc = Math.round(parts.reduce((a,b)=>a+b,0) / parts.length);

    // 참고로 어긋난 핵심 토큰(정확 모드 프리뷰)도 같이 보여줌
    const {ops, rt, ht} = calcWER(refText, hypText, {ignorePunct:true, allowNumRead:true});
    const {refHTML, hypHTML} = renderDiffExact(rt, ht, ops);

    const notes = `
      <div class="muted">
        • 내용만 모드: 문장 임베딩(USE) 코사인 유사도와 ROUGE-L(F1)의 평균으로 산출.<br/>
        • 임베딩 유사도: ${use === null ? '모델 로드 실패(네트워크 필요)' : use + '/100'} · ROUGE-L: ${rgL}/100
      </div>`;
    setMetrics({ lenRef, lenHyp, score: acc, refHTML, hypHTML, notes });
  }
});
