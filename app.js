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
function renderDiffExact(rt,ht,ops){
  const R=[],H=[]; let i=0,j=0;
  for(const op of ops){
    if(op==='M'){ R.push(rt[i]); H.push(ht[j]); i++; j++; }
    else if(op==='S'){ R.push(`<em class="sub">${rt[i]}</em>`); H.push(`<em class="sub">${ht[j]}</em>`); i++; j++; }
    else if(op==='D'){ R.push(`<del>${rt[i]}</del>`); i++; }
    else { H.push(`<ins>${ht[j]}</ins>`); j++; }
  }
  for(;i<rt.length;i++) R.push(`<del>${rt[i]}</del>`);
  for(;j<ht.length;j++) H.push(`<ins>${ht[j]}</ins>`);
  return {refHTML:R.join(' '), hypHTML:H.join(' ')};
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
function setMetrics({lenRef,lenHyp,score,refHTML,hypHTML,notes}){
  els.lenRef.textContent = lenRef ?? '-';
  els.lenHyp.textContent = lenHyp ?? '-';
  els.score.textContent  = (score ?? '-').toString();
  els.refVis.innerHTML   = refHTML ?? '';
  els.hypVis.innerHTML   = hypHTML ?? '';
  els.notes.innerHTML    = notes ?? '';
}

// ===== Mic & STT =====
let recog=null;           // SpeechRecognition instance
let finalText='';         // 최종 텍스트 스냅샷
let recognizing=false;

function supported(){
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  return !!SR;
}
function createRecognizer(lang){
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if(!SR){
    live.textContent='이 브라우저는 음성 인식을 지원하지 않습니다. (Chrome/Edge 권장)';
    btnStart.disabled=true; return null;
  }
  const r=new SR();
  r.lang = lang || (els.lang?.value || 'ko-KR');
  r.interimResults=true; r.continuous=true; r.maxAlternatives=1;

  r.onstart=()=>{ recognizing=true; recDot.classList.add('live'); live.textContent='듣는 중…'; finalText=''; log('onstart'); btnStop.disabled=false; };
  r.onresult=(e)=>{
    const res=e.results; let final='', interim='';
    for(let i=0;i<res.length;i++){
      const txt=res[i][0].transcript;
      if(res[i].isFinal) final += txt + ' ';
      else interim += txt + ' ';
    }
    finalText=final.trim();
    live.textContent = finalText + (interim ? ' ' + interim.trim() : '');
  };
  r.onerror=(e)=>{
    log('onerror', e);
    if(e.error==='not-allowed' || e.error==='service-not-allowed'){
      els.notes.innerHTML=`<div class="muted">마이크 권한이 거부되었습니다. 주소창 옆 🔒에서 권한을 허용해 주세요.</div>`;
    }
  };
  r.onend=()=>{
    recognizing=false; recDot.classList.remove('live');
    btnStart.disabled=false; btnStop.disabled=true; log('onend');
  };
  return r;
}
els.lang?.addEventListener('change', ()=>{
  try{ recog?.stop(); }catch{}
  recog = createRecognizer(els.lang.value);
});
async function ensureMicPermission(){
  if(!navigator.mediaDevices?.getUserMedia) return true;
  try{
    const stream=await navigator.mediaDevices.getUserMedia({audio:true});
    stream.getTracks().forEach(t=>t.stop());
    return true;
  }catch(e){
    log('getUserMedia error', e);
    els.notes.innerHTML=`<div class="muted">마이크 접근이 필요합니다. 브라우저 권한을 허용해 주세요.</div>`;
    return false;
  }
}

// ===== Main flow =====
btnStart.addEventListener('click', async ()=>{
  setMetrics({lenRef:'-',lenHyp:'-',score:'-',refHTML:'',hypHTML:'',notes:''});
  if(!supported()){ live.textContent='이 브라우저는 음성 인식을 지원하지 않습니다. (Chrome/Edge 권장)'; return; }
  if(recognizing){ log('already recognizing'); return; }
  const ok=await ensureMicPermission(); if(!ok) return;
  if(!recog) recog=createRecognizer(els.lang?.value || 'ko-KR');
  btnStart.disabled=true; btnStop.disabled=true;
  try{
    recog.start();
    setTimeout(()=>{ if(recognizing) btnStop.disabled=false; },150);
  }catch(e){
    log('start error', e);
    btnStart.disabled=false;
    els.notes.innerHTML=`<div class="muted">음성 인식을 시작할 수 없습니다: <code>${e?.message||e}</code></div>`;
  }
});
btnStop.addEventListener('click', ()=>{ try{ recog?.stop(); }catch{} });

// 탭 전환 시 자동 종료
document.addEventListener('visibilitychange', ()=>{
  if(document.visibilityState!=='visible' && recognizing){
    try{ recog?.stop(); }catch{}
  }
});

// ===== Scoring (after stop) =====
btnStop.addEventListener('click', ()=>{
  setTimeout(async ()=>{
    const refText = els.ref.value.trim();
    const hypText = finalText.trim();
    const ignoreP = els.stripPunct.checked;
    const allowNum = els.normalizeNum.checked;

    const lenRef = toks(ignoreP?stripPunct(refText):refText).length;
    const lenHyp = toks(ignoreP?stripPunct(hypText):hypText).length;

    if(!refText || !hypText){
      setMetrics({lenRef,lenHyp,score:'-',refHTML:refText,hypHTML:hypText,notes:`<div class="muted">대본과 발화가 모두 있어야 채점됩니다.</div>`});
      return;
    }

    if(els.mode.value==='exact'){
      const {wer,ops,rt,ht}=calcWER(refText,hypText,{ignorePunct:ignoreP,allowNumRead:allowNum});
      const acc = Math.max(0, Math.round((1-wer)*100));
      const {refHTML,hypHTML}=renderDiffExact(rt,ht,ops);
      const notes = `<div class="muted">• 정확 모드: WER 기반 (치환/삽입/삭제). <em class="sub">노란색</em>=치환, <del>빨강=누락</del>, <ins>초록=불필요</ins>. 점수=(1 - WER)×100</div>`;
      setMetrics({lenRef:rt.length,lenHyp:ht.length,score:acc,refHTML,hypHTML,notes});
    }else{
      const use = await semanticScore(refText,hypText);
      const rgL = rougeL(refText,hypText);
      const parts=[]; if(use!==null) parts.push(use); parts.push(rgL);
      const acc = Math.round(parts.reduce((a,b)=>a+b,0)/parts.length);

      const {ops,rt,ht}=calcWER(refText,hypText,{ignorePunct:true,allowNumRead:true});
      const {refHTML,hypHTML}=renderDiffExact(rt,ht,ops);
      const notes = `<div class="muted">• 내용만: USE 코사인 + ROUGE-L 평균. 임베딩:${use===null?'모델 로드 실패':use+'/100'} · ROUGE-L:${rgL}/100</div>`;
      setMetrics({lenRef,lenHyp,score:acc,refHTML,hypHTML,notes});
    }
  },200);
});
