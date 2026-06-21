/* Eris Cockpit (Tier 7) — single-page controller */
const $ = s => document.querySelector(s);
let convId = null;
let fieldView = 'field';
let voices = [];
let audioCtx=null, analyser=null, curAudio=null, emotion='neutral';

/* ---------------- boot ---------------- */
window.addEventListener('load', () => {
  loadVoices(); loadConvs(); loadDreams(); loadStudy();
  connectVitals(); connectField(); pollSystem(); pollLLM();
  initLive2D();
  $('#in').addEventListener('keydown', e => {
    if (e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); }
  });
  document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
    fieldView=t.dataset.v; document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x===t));
  });
});

/* ---------------- text formatting ---------------- */
// Escape HTML, then render a safe subset of Markdown so bold/italic/code
// show up properly in the chat bubble (instead of raw ** and *).
function esc(s){ return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function mdToHtml(t){
  let s=esc(t);
  s=s.replace(/```([\s\S]*?)```/g,(m,c)=>`<pre>${c.replace(/\n$/,'')}</pre>`);
  s=s.replace(/`([^`]+)`/g,'<code>$1</code>');
  s=s.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');
  s=s.replace(/__([^_]+)__/g,'<strong>$1</strong>');
  s=s.replace(/(^|[^*])\*([^*\n]+)\*/g,'$1<em>$2</em>');
  s=s.replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g,'<a href="$2" target="_blank">$1</a>');
  s=s.replace(/\n/g,'<br>');
  return s;
}
// Strip Markdown so the TTS speaks words, not "asterisk asterisk".
function cleanForTTS(t){
  let s=t||'';
  s=s.replace(/```[\s\S]*?```/g,' ');
  s=s.replace(/`([^`]+)`/g,'$1');
  s=s.replace(/\*\*([^*]+)\*\*/g,'$1');
  s=s.replace(/\*([^*]+)\*/g,'$1');
  s=s.replace(/__([^_]+)__/g,'$1');
  s=s.replace(/_([^_]+)_/g,'$1');
  s=s.replace(/^#{1,6}\s*/gm,'');
  s=s.replace(/\[([^\]]+)\]\([^)]*\)/g,'$1');
  s=s.replace(/[*_~`#>]/g,'');
  return s.replace(/\s+/g,' ').trim();
}

/* ---------------- chat ---------------- */
function addMsg(role, text, meta){
  const d=document.createElement('div'); d.className='msg '+role;
  const body=document.createElement('div'); body.className='body';
  if(role==='eris'){ body.innerHTML=mdToHtml(text); } else { body.textContent=text; }
  d.appendChild(body);
  if(meta){ const m=document.createElement('div'); m.className='mm'; m.textContent=meta; d.appendChild(m);}
  $('#chat').appendChild(d); $('#chat').scrollTop=$('#chat').scrollHeight; return d;
}
async function send(){
  const t=$('#in').value.trim(); if(!t) return;
  $('#in').value=''; addMsg('user',t);
  const thinking=addMsg('eris','…');
  try{
    const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:t, conversation_id:convId||''})});
    const d=await r.json();
    convId=d.conversation_id||convId;
    thinking.firstChild.innerHTML=mdToHtml(d.response||'(no response)');
    const mm=document.createElement('div'); mm.className='mm';
    mm.textContent=`${d.archetype||''} · ${d.regime||''} · dC/dX=${(d.dCdX||0).toFixed(3)} · ${(d.latency_ms||0).toFixed(0)}ms`;
    thinking.appendChild(mm);
    setEmotionFromReply(d);
    if($('#speak').checked) speak(d.response);
    loadConvs();
  }catch(e){ thinking.firstChild.textContent='⚠ '+e; }
}
function newChat(){ convId=null; $('#chat').innerHTML=''; document.querySelectorAll('.conv').forEach(c=>c.classList.remove('on')); }

async function loadConvs(){
  try{
    const d=await (await fetch('/api/conversations')).json();
    const el=$('#convs'); el.innerHTML='';
    (d.conversations||[]).forEach(c=>{
      const x=document.createElement('div'); x.className='conv'+(c.id===convId?' on':'');
      x.onclick=()=>openConv(c.id);
      x.innerHTML=`<div class="t"></div><div class="d"></div>
        <div class="meta">${fmtDate(c.created_at)} · ${c.turn_count} turns · seen ${fmtDate(c.last_accessed)}</div>`;
      x.querySelector('.t').textContent=c.title||'Conversation';
      x.querySelector('.d').textContent=c.description||'';
      el.appendChild(x);
    });
  }catch(e){}
}
async function openConv(id){
  const d=await (await fetch('/api/conversations/'+id)).json();
  convId=id; $('#chat').innerHTML='';
  (d.turns||[]).forEach(t=>{ addMsg('user',t.user); addMsg('eris',t.eris,
    t.meta?`${t.meta.archetype||''} · ${t.meta.regime||''}`:''); });
  loadConvs();
}

/* ---------------- cognitive vitals (WS) ---------------- */
function connectVitals(){
  const ws=new WebSocket(`ws://${location.host}/ws`);
  ws.onopen=()=>{ $('#conn').textContent='● live'; $('#conn').className='status'; };
  ws.onclose=()=>{ $('#conn').textContent='disconnected'; $('#conn').className='status off'; setTimeout(connectVitals,2000); };
  ws.onmessage=e=>{ try{ updVitals(JSON.parse(e.data)); }catch(_){} };
}
function updVitals(v){
  $('#v-regime').textContent=v.regime||'—';
  $('#v-coh').textContent=(v.coherence??0).toFixed(3);
  $('#v-dcdx').textContent=(v.dCdX??0).toFixed(4);
  $('#v-diss').textContent=(v.dissonance??0).toFixed(3);
  $('#v-arch').textContent=v.archetype||'—';
  $('#v-mem').textContent=`${v.stm_size||0}/${v.mtm_size||0}/${v.ltm_size||0}`;
  if(v.llm_backends&&v.llm_backends.length) $('#provider').textContent=v.llm_backends.join(', ');
  setEmotionFromRegime(v.regime);
}

/* ---------------- field canvas (WS) ---------------- */
function connectField(){
  const ws=new WebSocket(`ws://${location.host}/ws/field`);
  ws.onmessage=e=>{ try{ drawField(JSON.parse(e.data)); }catch(_){} };
  ws.onclose=()=>setTimeout(connectField,2500);
}
function hsl(h,s,l){return `hsl(${h},${s}%,${l}%)`;}
function drawField(f){
  const cv=$('#fieldcv'), ctx=cv.getContext('2d'); const N=f.size||64;
  const phi=(f.phi||[]).flat(), th=(f.theta||[]).flat(); if(!phi.length) return;
  const cell=cv.width/N; ctx.fillStyle='#05060a'; ctx.fillRect(0,0,cv.width,cv.height);
  for(let y=0;y<N;y++)for(let x=0;x<N;x++){
    const i=y*N+x, fv=phi[i], t=th[i]; let col;
    if(fieldView==='field'){ if(fv<0.02)continue; col=hsl((t*57.3)%360,92,Math.min(72,14+fv*68)); }
    else if(fieldView==='phase'){ col=hsl((t*57.3)%360,85,52); }
    else if(fieldView==='coher'){ const xp=x<N-1?x+1:x,yp=y<N-1?y+1:y;
      const dth=Math.abs(t-th[y*N+xp])+Math.abs(t-th[yp*N+x]); const c=Math.max(0,1-dth/3.14);
      col=hsl(170-110*c,70,18+44*c); }
    else { col=fv>0.5?'rgba(251,113,133,.9)':'rgba(96,165,250,.5)'; }
    ctx.fillStyle=col; ctx.fillRect(x*cell,y*cell,cell+0.6,cell+0.6);
  }
}

/* ---------------- system vitals (poll) ---------------- */
async function pollSystem(){
  try{
    const s=await (await fetch('/api/system')).json();
    setBar('cpu', s.cpu_pct, s.cpu_pct!=null?s.cpu_pct.toFixed(0)+'%':'n/a');
    setBar('ram', s.ram_pct, s.ram_used_gb!=null?`${s.ram_used_gb}/${s.ram_total_gb} GB`:'n/a');
    const g=(s.gpus&&s.gpus[0])||null;
    if(g){
      setBar('gpu', g.gpu_util_pct, (g.gpu_util_pct!=null?g.gpu_util_pct.toFixed(0)+'%':'n/a'));
      const vp=g.vram_total_mb?100*g.vram_used_mb/g.vram_total_mb:null;
      setBar('vram', vp, g.vram_total_mb?`${(g.vram_used_mb/1024).toFixed(1)}/${(g.vram_total_mb/1024).toFixed(1)} GB`:'n/a');
      $('#s-temp').textContent=`${s.cpu_temp_c!=null?s.cpu_temp_c.toFixed(0)+'°':'—'} / ${g.gpu_temp_c!=null?g.gpu_temp_c.toFixed(0)+'°':'—'}`;
    } else {
      setBar('gpu',null,'no nvidia-smi'); setBar('vram',null,'—');
      $('#s-temp').textContent=`${s.cpu_temp_c!=null?s.cpu_temp_c.toFixed(0)+'°':'—'} / —`;
    }
  }catch(e){}
  setTimeout(pollSystem,2500);
}
function setBar(id,pct,label){
  $('#s-'+id).textContent=label;
  const bar=$('#b-'+id); if(!bar)return; const i=bar.querySelector('i');
  i.style.width=(pct!=null?Math.min(100,pct):0)+'%';
  bar.className='bar'+(pct>=85?' hot':pct>=65?' warn':'');
}
async function pollLLM(){
  try{ const d=await (await fetch('/api/status')).json();
    $('#llm').textContent='llm: '+(d.llm_ready?'ready':'offline');
  }catch(e){ $('#llm').textContent='llm: ?'; }
  setTimeout(pollLLM,5000);
}

/* ---------------- dreams ---------------- */
async function loadDreams(){
  try{
    const d=await (await fetch('/api/dreams')).json(); const el=$('#dreams'); el.innerHTML='';
    if(!(d.dreams||[]).length){ el.innerHTML='<div class="muted">No dreams yet. She reflects on cognitive dissonance during idle/nightly cycles, or on your direction.</div>'; return; }
    d.dreams.forEach(x=>{
      const it=document.createElement('div'); it.className='item'; it.onclick=()=>openDream(x.id);
      it.innerHTML=`<div class="s"></div>${x.question?'<div class="q">❓ needs your input</div>':''}<div class="when">${x.kind} · ${fmtDate(x.timestamp)}</div>`;
      it.querySelector('.s').textContent=x.summary||'(reflection)';
      el.appendChild(it);
    });
  }catch(e){}
}
async function openDream(id){
  const d=await (await fetch('/api/dreams/'+id)).json();
  showModal(d.topic||'Reflection', d.timestamp, d.detail||d.summary||'', d.sources);
}
async function dreamPrompt(){
  const q=prompt('Give Eris something to dream on / ponder:'); if(!q)return;
  const it=$('#dreams'); it.insertAdjacentHTML('afterbegin','<div class="muted" id="pending">💭 pondering…</div>');
  try{ const e=await (await fetch('/api/dream/ponder',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})})).json();
    loadDreams(); showModal(e.topic||q, e.timestamp, e.detail||e.summary, e.sources);
  }catch(err){ alert('ponder failed: '+err);} finally{ const p=$('#pending'); if(p)p.remove(); }
}

/* ---------------- study ---------------- */
async function loadStudy(){
  try{
    const d=await (await fetch('/api/study/reports')).json(); const el=$('#study'); el.innerHTML='';
    if(!(d.reports||[]).length){ el.innerHTML='<div class="muted">Nothing studied yet. Runs nightly on your topics — or hit “study now”.</div>'; return; }
    d.reports.forEach(x=>{
      const it=document.createElement('div'); it.className='item'; it.onclick=()=>openStudy(x.id);
      it.innerHTML=`<div class="s"></div><div class="when">${fmtDate(x.timestamp)} · ${x.total_chunks} passages</div>`;
      it.querySelector('.s').textContent=(x.topics||[]).join(', ')||'study';
      el.appendChild(it);
    });
  }catch(e){}
}
async function openStudy(id){
  const r=await (await fetch('/api/study/reports/'+id)).json();
  const srcs=(r.read||[]).map(x=>x.source);
  showModal('Studied: '+(r.topics||[]).join(', '), r.timestamp, r.summary||'', srcs);
}
async function studyNow(){
  const it=$('#study'); it.insertAdjacentHTML('afterbegin','<div class="muted" id="studying">📚 studying… (reading sources)</div>');
  try{ await fetch('/api/study/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({topics:[]})}); loadStudy(); }
  catch(e){ alert('study failed: '+e);} finally{ const p=$('#studying'); if(p)p.remove(); }
}
async function editTopics(){
  const cur=await (await fetch('/api/study/topics')).json();
  const v=prompt('Topics Eris should study (comma-separated). Reliable nonfiction only.', (cur.topics||[]).join(', '));
  if(v===null)return;
  await fetch('/api/study/topics',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({topics:v.split(',').map(s=>s.trim()).filter(Boolean)})});
  alert('Topics saved. She will study these on the next nightly cycle (or “study now”).');
}

/* ---------------- modal ---------------- */
function showModal(title, ts, body, sources){
  $('#m-title').textContent=title; $('#m-when').textContent=fmtDate(ts);
  $('#m-body').textContent=body||''; const s=$('#m-src'); s.innerHTML='';
  (sources||[]).filter(Boolean).forEach(u=>{ const a=document.createElement('a'); a.href=u; a.target='_blank'; a.textContent=u; s.appendChild(a); });
  $('#modal').classList.add('on');
}
function closeModal(){ $('#modal').classList.remove('on'); }

/* ---------------- voices + TTS lip-sync ---------------- */
async function loadVoices(){
  try{
    const d=await (await fetch('/api/tts/voices')).json(); voices=d.voices||[];
    const sel=$('#voice'); sel.innerHTML='';
    voices.forEach(v=>{ const o=document.createElement('option'); o.value=v.id; o.textContent=v.name; sel.appendChild(o); });
    // prefer an "Emily" voice if present
    const em=voices.find(v=>/emily/i.test(v.name)); if(em) sel.value=em.id;
  }catch(e){}
}
function testVoice(){ speak("Hi, this is Eris. This is how my voice sounds right now."); }
async function speak(text){
  try{
    text=cleanForTTS(text); if(!text) return;
    if(curAudio){ curAudio.pause(); curAudio=null; }
    const r=await fetch('/api/tts/generate',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text, voice_id:$('#voice').value})});
    if(!r.ok) return;
    const blob=await r.blob(); const url=URL.createObjectURL(blob);
    const a=new Audio(url); a.volume=parseFloat($('#vol').value); curAudio=a;
    if(!audioCtx){ audioCtx=new (window.AudioContext||window.webkitAudioContext)(); }
    const src=audioCtx.createMediaElementSource(a); analyser=audioCtx.createAnalyser();
    analyser.fftSize=256; src.connect(analyser); analyser.connect(audioCtx.destination);
    a.onended=()=>{ mouth(0); }; a.play(); lipLoop();
  }catch(e){ console.log('tts',e); }
}
function lipLoop(){
  if(!analyser||!curAudio||curAudio.paused) { mouth(0); return; }
  const buf=new Uint8Array(analyser.frequencyBinCount); analyser.getByteFrequencyData(buf);
  let sum=0; for(let i=0;i<buf.length;i++) sum+=buf[i]; const amp=Math.min(1,(sum/buf.length)/90);
  mouth(amp); requestAnimationFrame(lipLoop);
}
function mouth(open){
  // portrait: subtle scale "pulse"; live2d: ParamMouthOpenY
  const p=$('#portrait'); if(p&&p.style.display!=='none'){ p.style.transform=`scale(${1+open*0.05})`; }
  if(window._l2dModel){ try{ window._l2dModel.internalModel.coreModel.setParameterValueById('ParamMouthOpenY', open); }catch(_){} }
}

/* ---------------- emotion ---------------- */
const EMO={neutral:[125,211,252], plastic:[251,113,133], transfixed:[251,191,36],
  elastic:[96,165,250], warmup:[120,130,150], curious:[167,139,250], calm:[74,222,128]};
function applyEmotion(name){
  emotion=name; $('#emotion').textContent=name;
  const c=EMO[name]||EMO.neutral; const p=$('#portrait');
  if(p) p.style.filter=`drop-shadow(0 0 22px rgba(${c[0]},${c[1]},${c[2]},.45))`;
  if(window._l2dModel){ try{ window._l2dModel.expression && window._l2dModel.expression(name); }catch(_){} }
}
function setEmotionFromRegime(r){
  if(r==='plastic') applyEmotion('plastic');
  else if(r==='transfixed') applyEmotion('transfixed');
  else if(r==='warmup') applyEmotion('warmup');
  else applyEmotion('calm');
}
function setEmotionFromReply(d){
  const t=(d.response||'').toLowerCase();
  if(/[?]\s*$/.test(t)||/curious|wonder|interesting/.test(t)) applyEmotion('curious');
  else setEmotionFromRegime(d.regime);
}

/* ---------------- Live2D (optional; portrait fallback) ---------------- */
async function initLive2D(){
  // Only activates if the Cubism libs + a model are present. Otherwise the
  // portrait (#portrait) stays and is driven by mouth()/applyEmotion().
  const MODEL='/static/models/live2d/eris/eris.model3.json';
  try{
    const head=await fetch(MODEL,{method:'HEAD'}); if(!head.ok) return; // no model -> portrait
    await loadScript('https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js');
    await loadScript('https://cdn.jsdelivr.net/npm/pixi.js@6.5.10/dist/browser/pixi.min.js');
    await loadScript('https://cdn.jsdelivr.net/npm/pixi-live2d-display@0.4.0/dist/cubism4.min.js');
    const app=new PIXI.Application({view:$('#live2d'),transparent:true,resizeTo:$('.avatar-wrap')});
    const model=await PIXI.live2d.Live2DModel.from(MODEL);
    app.stage.addChild(model);
    const fit=()=>{ const s=Math.min(app.renderer.width/model.width, app.renderer.height/model.height)*1.6;
      model.scale.set(s); model.x=app.renderer.width/2; model.y=app.renderer.height*0.1; model.anchor.set(0.5,0); };
    fit(); window.addEventListener('resize',fit);
    window._l2dModel=model; const pt=$('#portrait'); if(pt) pt.style.display='none';
    $('#emotion').textContent='live2d ready';
  }catch(e){ console.log('Live2D not active (portrait fallback):',e.message); }
}
function loadScript(src){ return new Promise((res,rej)=>{ const s=document.createElement('script'); s.src=src; s.onload=res; s.onerror=()=>rej(new Error('load '+src)); document.head.appendChild(s); }); }

/* ---------------- util ---------------- */
function fmtDate(ts){ if(!ts) return '—'; const d=new Date(ts*1000); return d.toLocaleDateString()+' '+d.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}); }
