// ===== Helpers =====
const $ = (s) => document.querySelector(s);
const byId = (id) => document.getElementById(id);
const log = (...a) => console.log('[speech-grader]', ...a);

// ===== Safe query =====
function reqEl(id){ const el = byId(id); if(!el) throw new Error(`#${id} element not found`); return el; }

// ===== Text utils =====
const stripPunct = (s) => s.replace(/[\p{P}\p{S}]/gu,' ').replace(/\s+/g,' ').trim().toLowerCase();
const normalizeNumsKo = (s) => s.replace(/(\d+)\s*년/g,'$1년').replace(/\s+/g,' ');
const toks = (s) => s.split(/\s+/).filter(Boolean);

// ===== WER / Diff =====
function alignTokens(R,H){
  const n=R.length,m=H.length;
  const dp=Array.from({length:n+1},()=>Array(m+1).fill(0));
  const bt=Array.from({length:n+1},()=>Array(m+1).fill(''));
  for(let i=0;i<=n;i++){dp[i][0]=i;bt[i][0]='D'}
  for(let j=0;j<=m;j++){dp[0][j]=j;bt[0][j]='I'}
  bt[0][0]='';
  for(let i=1;i<=n;i++) for(let j=1;j<=m;j++){
    if(R[i-1]===H[j-1]){ dp[i][j]=dp[i-1][j-1]; bt[i][j]='M'; }
    else{
      const sub=dp[i-1][j-1]+1, del=dp[i-1][j]+1, ins=dp[i][j-1]+1;
      const min=Math.min(sub,del,ins);
      dp[i][j]=min; bt[i][j]= (min===sub)?'S':(min===del)?'D':'I';
    }
  }
  let i=n,j=m,ops=[];
  while(i>0||j>0){
    const op=bt[i][j];
    if(op==='M'||op==='S'){ ops.push(op); i--; j--; }
    else if(op==='D'){ ops.push('D'); i--; }
    else { ops.push('I'); j--; }
  }
  ops.reverse(); return {distance:dp[n][m], ops};
}
function calcWER(ref,hyp,{ignorePunct=true,allowNumRead=true}={}){
  let r=ref,h=hyp;
  if(ignorePunct){ r=stripPunct(r); h=stripPunct(h); }
  if(allowNumRead){ r=normalizeNumsKo(r); h=normalizeNumsKo(h); }
  const rt=toks(r), ht=toks(h);
  const {distance,ops}=alignTokens(rt,ht);
  const wer = rt.length ? distance/rt.length : (ht.length?1:0);
  return {wer,ops,rt,ht};
}

function renderDiffExact(rt, ht, ops) {
  const refOut = [];
  const hypOut = [];
  let i=0, j=0;

  for (const op of ops){
    if (op==='M'){ 
      // 일치 → 초록
      refOut.push(`<span class="match">${rt[i]}</span>`);
      hypOut.push(`<span class="match">${ht[j]}</span>`);
      i++; j++;
    } else if (op==='S'){ 
      // 치환 → 노랑
      refOut.push(`<span class="sub">${rt[i]}</span>`);
      hypOut.push(`<span class="sub">${ht[j]}</span>`);
      i++; j++;
    } else if (op==='D'){ 
      // 누락 → 빨강
      refOut.push(`<span class="del">${rt[i]}</span>`);
      i++;
    } else if (op==='I'){ 
      // 삽입 → 회색
      hypOut.push(`<span class="ins">${ht[j]}</span>`);
      j++;
    }
  }

  // 남은 토큰 처리
  for (; i<rt.length; i++) refOut.push(`<span class="del">${rt[i]}</span>`);
  for (; j<ht.length; j++) hypOut.push(`<span class="ins">${ht[j]}</span>`);

  return { refHTML: refOut.join(' '), hypHTML: hypOut.join(' ') };
}

// ===== Semantic (USE + ROUGE-L) =====
let useModel=null;
async function loadUSE(){
  if(!useModel && window.universalSentenceEncoder){
    try{ useModel = await window.universalSentenceEncoder.load(); }
    catch(e){ log('USE load fail', e); }
  }
  return useModel;
}
async function semanticScore(a,b){
  const m=await loadUSE(); if(!m) return null;
  const emb=await m.embed([a,b]);
  const aV=emb.slice([0,0],[1]); const bV=emb.slice([1,0],[1]);
  const sim=await tf.tidy(()=>tf.linalg.l2Normalize(aV,1).mul(tf.linalg.l2Normalize(bV,1)).sum(1).array());
  const s=sim[0]; return Math.max(0,Math.min(100,Math.round(((s+1)/2)*100)));
}
function rougeL(a,b){
  const A=toks(stripPunct(a)), B=toks(stripPunct(b));
  const n=A.length,m=B.length, dp=Array.from({length:n+1},()=>Array(m+1).fill(0));
  for(let i=1;i<=n;i++) for(let j=1;j<=m;j++)
    dp[i][j]= (A[i-1]===B[j-1]) ? dp[i-1][j-1]+1 : Math.max(dp[i-1][j], dp[i][j-1]);
  const l=dp[n][m];
  if(!n&&!m) return 100;
  const p=l/(m||1), r=l/(n||1), f=(2*p*r)/((p+r)||1);
  return Math.round(f*100);
}

// ===== Elements =====
const live   = reqEl('live');
const recDot = reqEl('rec-dot');
const btnStart = reqEl('btn-start');
const btnStop  = reqEl('btn-stop');
const els = {
  lang: byId('lang'),
  mode: reqEl('mode'),
  ref:  reqEl('ref'),
  lenRef: reqEl('len-ref'),
  lenHyp: reqEl('len-hyp'),
  score:  reqEl('score'),
  refVis: reqEl('ref-vis'),
  hypVis: reqEl('hyp-vis'),
  notes:  reqEl('notes'),
  stripPunct: reqEl('strip-punct'),
  normalizeNum: reqEl('normalize-num'),
};

function setMetrics({lenRef, lenHyp, score, refHTML, hypHTML, notes}) {
  els.lenRef.textContent = lenRef ?? '-';
  els.lenHyp.textContent = lenHyp ?? '-';
  els.score.textContent  = (score ?? '-').toString();
  els.refVis.innerHTML   = refHTML ?? '';
  els.hypVis.innerHTML   = hypHTML ?? '';
  els.notes.innerHTML    = notes ?? '';
}

// ===== Mic & STT + Scoring (single source of truth) =====
let recog = null;            // SpeechRecognition
let recording = false;       // 사용자가 Start~Stop 사이에 녹음 의사
let recognizing = false;     // 엔진 onstart~onend
let lastResultAt = 0;        // 결과 수신 시각(워치독용)
let keepTimer = null;        // 워치독
let retryCount = 0;          // 재시도 백오프

// 텍스트 버퍼 (항상 '종료' 누를 때까지 누적해서 보관)
let finalBuf = "";           // 최종(confirmed) 누적 텍스트
let interimBuf = "";         // 진행 중(미확정) 텍스트 스냅샷

function supported(){
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  return !!SR;
}
async function ensureMicPermission(){
  if(!navigator.mediaDevices?.getUserMedia) return true;
  try{
    const s = await navigator.mediaDevices.getUserMedia({audio:true});
    s.getTracks().forEach(t=>t.stop());
    return true;
  }catch(e){
    log('getUserMedia error', e);
    els.notes.innerHTML = `<div class="muted">마이크 권한이 필요합니다. 주소창 옆 🔒에서 허용해 주세요.</div>`;
    return false;
  }
}

// ========== Recognizer ==========
function createRecognizer(lang){
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if(!SR){
    live.textContent = '이 브라우저는 음성 인식을 지원하지 않습니다. (Chrome/Edge 권장)';
    btnStart.disabled = true; return null;
  }
  const r = new SR();
  r.lang = lang || (els.lang?.value || 'ko-KR');
  r.interimResults = true;
  r.continuous = true;
  r.maxAlternatives = 1;

  r.onstart = () => {
    recognizing = true;
    lastResultAt = Date.now();
    recDot.classList.add('live');
    live.textContent = '듣는 중…';
    btnStop.disabled = false;
    startWatchdog();
    log('onstart');
  };

  r.onresult = (e) => {
    // 결과는 "스냅샷"으로 재조립 → 중복/유실 방지
    const res = e.results;
    let final = "", interim = "";
    for (let i=0;i<res.length;i++){
      const txt = res[i][0].transcript;
      if (res[i].isFinal) final += txt + ' ';
      else interim += txt + ' ';
    }
    finalBuf  = final.trim();          // 현재까지 확정본
    interimBuf = interim.trim();       // 현재 진행 중
    lastResultAt = Date.now();
    live.textContent = (finalBuf + (interimBuf?(' '+interimBuf):'')).trim();
  };

  r.onerror = (e) => {
    log('onerror', e);
    if (e.error === 'not-allowed' || e.error === 'service-not-allowed'){
      els.notes.innerHTML = `<div class="muted">마이크 권한이 거부되었습니다. 주소창 옆 🔒에서 허용해 주세요.</div>`;
    } else if (e.error === 'audio-capture') {
      els.notes.innerHTML = `<div class="muted">마이크 장치가 감지되지 않습니다. 입력 장치를 확인해 주세요.</div>`;
    } else if (e.error === 'no-speech') {
      els.notes.innerHTML = `<div class="muted">음성이 감지되지 않았습니다. (자동 재시도 중)</div>`;
    }
    // 사용자가 녹음 중이라면 자동 복구
    if (recording) restartRecognizer();
  };

  r.onend = () => {
    recognizing = false;
    recDot.classList.remove('live');
    btnStart.disabled = false;
    btnStop.disabled = true;
    stopWatchdog();
    log('onend');

    // 사용자가 아직 녹음 원하면(정책/무음으로 끊긴 경우) 재시작
    if (recording) restartRecognizer();
  };

  return r;
}
function ensureRecognizer(){
  if (recog) return recog;
  recog = createRecognizer(els.lang?.value || 'ko-KR');
  return recog;
}

function restartRecognizer(){
  if (!recording) return;
  const delay = Math.min(200 + retryCount*300, 2500);
  try { recog?.stop(); } catch {}
  setTimeout(() => {
    try {
      const r = ensureRecognizer();
      r && r.start();
      retryCount = Math.min(retryCount+1, 5);
      log('restart attempt', retryCount, 'delay', delay);
    } catch (e) { log('restart failed', e); }
  }, delay);
}

function startWatchdog(){
  stopWatchdog();
  keepTimer = setInterval(() => {
    if (!recording) return;
    if (Date.now() - lastResultAt > 4000) {
      log('watchdog idle → restart');
      restartRecognizer();
    }
  }, 1500);
}
function stopWatchdog(){
  if (keepTimer){ clearInterval(keepTimer); keepTimer=null; }
}

// 언어 변경 시 재생성
els.lang?.addEventListener('change', ()=>{
  try { recog?.stop(); } catch {}
  recog = createRecognizer(els.lang.value);
});

// ========== Start / Stop ==========
btnStart.addEventListener('click', async ()=>{
  // 초기화
  setMetrics({lenRef:'-',lenHyp:'-',score:'-',refHTML:'',hypHTML:'',notes:''});
  finalBuf = ""; interimBuf = "";

  if(!supported()){
    live.textContent='이 브라우저는 음성 인식을 지원하지 않습니다. (Chrome/Edge 권장)';
    return;
  }
  const ok = await ensureMicPermission();
  if(!ok) return;

  recording = true;
  retryCount = 0;

  const r = ensureRecognizer();
  if (!r) return;

  btnStart.disabled = true;
  btnStop.disabled  = true;
  try { r.start(); } catch (e) {
    log('start error', e);
    btnStart.disabled = false;
    els.notes.innerHTML = `<div class="muted">음성 인식을 시작할 수 없습니다: <code>${e?.message||e}</code></div>`;
    recording = false;
  }
});

btnStop.addEventListener('click', ()=>{
  // 더 이상 녹음 의사 없음 → 엔진 종료 요청
  recording = false;
  stopWatchdog();
  try { recog?.stop(); } catch {}

  // 마지막 interim까지 포함해서 채점 (엔진이 끝나며 onresult 못 받을 수도 있으니 250ms 대기)
  setTimeout(scoreNow, 250);
});

// ========== Scoring ==========
function scoreNow(){
  // 녹음된 전체 텍스트: 확정 + 미확정(있으면)
  const hypText = (finalBuf + (interimBuf?(' '+interimBuf):'')).trim();

  const refText = els.ref.value.trim();
  const ignoreP = els.stripPunct.checked;
  const allowNum = els.normalizeNum.checked;

  const lenRef = toks(ignoreP ? stripPunct(refText) : refText).length;
  const lenHyp = toks(ignoreP ? stripPunct(hypText) : hypText).length;

  if (!refText || !hypText){
    setMetrics({
      lenRef, lenHyp, score:'-',
      refHTML: refText, hypHTML: hypText,
      notes: `<div class="muted">대본과 발화가 모두 있어야 채점됩니다.</div>`
    });
    return;
  }

  if (els.mode.value === 'exact'){
    const {wer, ops, rt, ht} = calcWER(refText, hypText, {ignorePunct:ignoreP, allowNumRead:allowNum});
    const acc = Math.max(0, Math.round((1-wer)*100));
    const {refHTML, hypHTML} = renderDiffExact(rt, ht, ops);
    const notes = `<div class="muted">• 정확 모드: WER 기반 (치환/삽입/삭제). 점수=(1 - WER)×100</div>`;
    setMetrics({ lenRef: rt.length, lenHyp: ht.length, score: acc, refHTML, hypHTML, notes });
  }else{
    // 내용만: USE + ROUGE-L 평균 (USE 로드 실패 시 ROUGE-L만)
    Promise.resolve().then(async ()=>{
      const use = await semanticScore(refText, hypText);
      const rgL = rougeL(refText, hypText);
      const parts = []; if (use !== null) parts.push(use); parts.push(rgL);
      const acc = Math.round(parts.reduce((a,b)=>a+b,0)/parts.length);

      const {ops, rt, ht} = calcWER(refText, hypText, {ignorePunct:true, allowNumRead:true});
      const {refHTML, hypHTML} = renderDiffExact(rt, ht, ops);
      const notes = `<div class="muted">• 내용만: USE 코사인 + ROUGE-L 평균. 임베딩:${use===null?'모델 로드 실패':use+'/100'} · ROUGE-L:${rgL}/100</div>`;
      setMetrics({ lenRef, lenHyp, score: acc, refHTML, hypHTML, notes });
    });
  }
}
// ===== End STT + Scoring =====
