/* Eris Cockpit (Tier 7) — single-page controller */
const $ = s => document.querySelector(s);
let convId = null;
let fieldView = 'field';
let voices = [];
let audioCtx=null, analyser=null, curAudio=null, curGain=null, emotion='neutral';

/* ---------------- boot ---------------- */
window.addEventListener('load', () => {
  loadVoices(); loadConvs(); loadDreams(); loadStudy(); loadLibrary();
  connectVitals(); connectField(); pollSystem(); pollLLM();
  initLive2D();
  $('#in').addEventListener('keydown', e => {
    if (e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); }
  });
  // live volume: drives auto-speak (gain node) and any per-reply players
  $('#vol').addEventListener('input', e => {
    const v=parseFloat(e.target.value||'1');
    if(curGain) curGain.gain.value=v;
    if(curAudio) curAudio.volume=v;
    document.querySelectorAll('audio.player').forEach(a=>a.volume=v);
  });
  // infinite scroll back through dream history
  const dEl=$('#dreams');
  if(dEl) dEl.addEventListener('scroll', ()=>{
    if(dEl.scrollTop+dEl.clientHeight >= dEl.scrollHeight-24) loadOlderDreams();
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
  if(role==='eris'){
    const bar=document.createElement('div'); bar.className='msgbar';
    const b=document.createElement('button'); b.className='lbtn'; b.textContent='Listen';
    b.onclick=()=>playReply(d,b);
    const c=document.createElement('button'); c.className='lbtn'; c.textContent='Copy';
    c.onclick=()=>copyReply(d,c);
    bar.appendChild(b); bar.appendChild(c); d.appendChild(bar);
  }
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
  }catch(e){ thinking.firstChild.textContent='Error: '+e; }
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
  ws.onopen=()=>{ $('#conn').textContent='live'; $('#conn').className='status'; };
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
let _dreamsOldest=null;
async function loadDreams(){
  try{
    const d=await (await fetch('/api/dreams?limit=8')).json(); const el=$('#dreams'); el.innerHTML='';
    const rows=d.dreams||[];
    if(!rows.length){ el.innerHTML='<div class="muted">No reflections yet. She resolves tensions and crawls a topic on a background loop (a few times an hour), or on your direction.</div>'; return; }
    _dreamsOldest = rows[rows.length-1].timestamp;
    rows.forEach(x=>el.appendChild(dreamRow(x)));
  }catch(e){}
}
function dreamRow(x){
  const it=document.createElement('div'); it.className='item'; it.onclick=()=>openDream(x.id);
  const claude=x.used_claude?' <span class="tag">Claude</span>':'';
  const label=x.guided?'guided':(x.kind||'reflection');
  const counts=(x.source_count!=null)?` &middot; ${x.stored_count||0} kept / ${x.source_count||0} sources`:'';
  it.innerHTML=`<div class="s"></div>${x.question?'<div class="q">(needs your input)</div>':''}<div class="when">${label}${claude} &middot; ${fmtDate(x.timestamp)}${counts}</div>`;
  it.querySelector('.s').textContent=x.summary||x.topic||'(reflection)';
  return it;
}
async function loadOlderDreams(){
  if(_dreamsOldest==null) return;
  const before=_dreamsOldest; _dreamsOldest=null;   // guard against repeat-fire
  try{
    const d=await (await fetch('/api/dreams?limit=8&before='+before)).json();
    const rows=d.dreams||[]; const el=$('#dreams');
    rows.forEach(x=>el.appendChild(dreamRow(x)));
    if(rows.length) _dreamsOldest=rows[rows.length-1].timestamp;
  }catch(e){}
}
async function openDream(id){
  const d=await (await fetch('/api/dreams/'+id)).json();
  if(d.error) return;
  $('#m-title').textContent=`${d.kind||'reflection'}: ${d.topic||''}`+(d.used_claude?'  (consulted Claude)':'');
  $('#m-when').textContent=fmtDate(d.timestamp)+(d.regime?' · feeling '+d.regime:'')+(d.guided?' · you asked':' · self-directed');
  $('#m-body').textContent=d.detail||d.summary||'';   // detail = her reflection + what she read
  const s=$('#m-src'); s.innerHTML='';
  const srcs=d.sources||[];
  if(srcs.length){
    const h=document.createElement('div'); h.className='muted'; h.textContent='Sources'; s.appendChild(h);
    srcs.forEach(src=>{
      const o=(typeof src==='string')?{title:src,url:src}:src;
      if(o.url && o.url.indexOf('anthropic:')===0){ const c=document.createElement('div'); c.textContent='Claude ('+(o.title||'')+')'; s.appendChild(c); }
      else { const a=document.createElement('a'); a.href=o.url||'#'; a.target='_blank'; a.textContent=o.title||o.url; s.appendChild(a); }
    });
  }
  const kept=d.stored||[];
  const btn=document.createElement('button'); btn.className='btn'; btn.style.marginTop='8px';
  btn.textContent=`Show what she kept (${kept.length})`;
  const box=document.createElement('div'); box.style.display='none'; box.style.marginTop='6px';
  if(!kept.length){ box.innerHTML='<em class="muted">Nothing passed the quality filter this cycle.</em>'; }
  else kept.forEach(it=>{
    const bq=document.createElement('blockquote'); bq.className='kept'; bq.textContent=it.snippet||'';
    const m=document.createElement('div'); m.className='muted'; m.style.fontSize='10px';
    if(it.source_url && it.source_url.indexOf('anthropic:')!==0){ const a=document.createElement('a'); a.href=it.source_url; a.target='_blank'; a.textContent='source'; m.appendChild(a); }
    else { m.textContent='Claude'; }
    bq.appendChild(m); box.appendChild(bq);
  });
  btn.onclick=()=>{ box.style.display = (box.style.display==='none')?'block':'none'; };
  s.appendChild(btn); s.appendChild(box);
  $('#modal').classList.add('on');
}
async function steerTopic(){
  const t=prompt('Point Eris at a topic to study next (she keeps choosing on her own too):'); if(!t)return;
  try{
    const d=await (await fetch('/api/study-topic',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:t})})).json();
    toast(d.error?('Error: '+d.error):`Queued "${d.queued}" — she'll study it on her next cycle.`);
  }catch(e){ toast('could not queue topic: '+e); }
}
async function dreamPrompt(){
  const q=prompt('Give Eris something to dream on / ponder:'); if(!q)return;
  const it=$('#dreams'); it.insertAdjacentHTML('afterbegin','<div class="muted" id="pending">pondering...</div>');
  try{ const e=await (await fetch('/api/dream/ponder',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})})).json();
    loadDreams(); showModal(e.topic||q, e.timestamp, e.detail||e.summary, e.sources);
  }catch(err){ alert('ponder failed: '+err);} finally{ const p=$('#pending'); if(p)p.remove(); }
}

/* ---------------- study ---------------- */
async function loadStudy(){
  try{
    const d=await (await fetch('/api/study/reports')).json(); const el=$('#study'); el.innerHTML='';
    if(!(d.reports||[]).length){ el.innerHTML='<div class="muted">Nothing studied yet. Runs nightly on your topics, or hit "study now".</div>'; return; }
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
  const it=$('#study'); it.insertAdjacentHTML('afterbegin','<div class="muted" id="studying">studying... (reading sources)</div>');
  try{ await fetch('/api/study/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({topics:[]})}); loadStudy(); }
  catch(e){ alert('study failed: '+e);} finally{ const p=$('#studying'); if(p)p.remove(); }
}
async function editTopics(){
  const cur=await (await fetch('/api/study/topics')).json();
  const v=prompt('Topics Eris should study (comma-separated). Reliable nonfiction only.', (cur.topics||[]).join(', '));
  if(v===null)return;
  await fetch('/api/study/topics',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({topics:v.split(',').map(s=>s.trim()).filter(Boolean)})});
  alert('Topics saved. She will study these on the next nightly cycle (or "study now").');
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

/* copy a reply's text to the clipboard */
async function copyReply(d, btn){
  const body=d.querySelector('.body'); if(!body) return;
  const text=body.innerText||body.textContent||'';
  try{
    await navigator.clipboard.writeText(text);
  }catch(e){
    const ta=document.createElement('textarea'); ta.value=text; document.body.appendChild(ta);
    ta.select(); try{ document.execCommand('copy'); }catch(_){ } ta.remove();
  }
  const old=btn.textContent; btn.textContent='Copied'; setTimeout(()=>btn.textContent=old, 1200);
}

/* per-reply playback: native <audio controls> = play/pause/seek/volume/speed */
async function playReply(d, btn){
  const body=d.querySelector('.body'); if(!body) return;
  const bar=d.querySelector('.msgbar');
  let au=bar.querySelector('audio.player');
  if(au){ au.paused ? au.play() : au.pause(); return; }   // already loaded → toggle
  const text=cleanForTTS(body.innerText||body.textContent||''); if(!text) return;
  const old=btn.textContent; btn.textContent='generating...'; btn.disabled=true;
  try{
    const r=await fetch('/api/tts/generate',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text, voice_id:$('#voice').value})});
    if(!r.ok){ btn.textContent=old; btn.disabled=false; return; }
    const url=URL.createObjectURL(await r.blob());
    au=document.createElement('audio'); au.className='player'; au.controls=true; au.src=url;
    au.autoplay=true; au.volume=parseFloat($('#vol').value||'1');
    bar.innerHTML=''; bar.appendChild(au);
  }catch(e){ btn.textContent=old; btn.disabled=false; }
}

/* ---------------- document library ---------------- */
function pickFile(){ $('#fileinput').click(); }
async function uploadFile(input){
  const f=input&&input.files&&input.files[0]; if(!f) return;
  toast(`reading ${f.name}...`);
  const fd=new FormData(); fd.append('file', f);
  startProg();
  try{
    const d=await (await fetch('/api/library/upload',{method:'POST',body:fd})).json();
    toast(d.error ? ('Error: '+d.error) : `Ingested ${d.chunks||0} passages from ${f.name}`);
    loadLibrary();
  }catch(e){ toast('upload failed: '+e); }
  finally{ stopProg(); }
  input.value='';
}
async function scanLibrary(force){
  toast(force ? 're-reading your entire ErisLibrary folder...' : 'reading your ErisLibrary folder...');
  startProg();
  try{
    const d=await (await fetch('/api/library/scan'+(force?'?force=true':''),{method:'POST'})).json();
    if(d.error){ toast(`${d.error}: ${d.dir}`); }
    else{ toast(`Library: ${d.ingested} new, ${d.skipped} unchanged, ${d.total_chunks} passages`); }
    loadLibrary();
  }catch(e){ toast('scan failed: '+e); }
  finally{ stopProg(); }
}

/* live ingestion progress bar (polls /api/library/progress) */
let _progTimer=null;
function startProg(){
  const p=$('#libprog'); if(p) p.style.display='block';
  setProg(0,'starting...'); pollProgress();
  if(_progTimer) clearInterval(_progTimer);
  _progTimer=setInterval(pollProgress, 800);
}
function stopProg(){
  if(_progTimer){ clearInterval(_progTimer); _progTimer=null; }
  pollProgress(); setTimeout(()=>{ const p=$('#libprog'); if(p) p.style.display='none'; }, 1800);
}
function setProg(frac, txt){
  const f=$('#libprog-fill'); if(f) f.style.width=Math.max(0,Math.min(100,frac*100)).toFixed(0)+'%';
  const t=$('#libprog-txt'); if(t) t.textContent=txt||'';
}
async function pollProgress(){
  try{
    const p=await (await fetch('/api/library/progress')).json();
    const ft=p.files_total||0, fd=p.files_done||0, bt=p.blocks_total||0, bd=p.blocks_done||0;
    const frac= ft ? (fd + (bt ? bd/bt : 0)) / ft : (p.running?0:1);
    setProg(frac, p.running
      ? `reading ${p.file||''} (file ${Math.min(fd+1,ft)}/${ft}), ${p.chunks||0} passages`
      : `done: ${p.chunks||0} passages`);
  }catch(e){}
}
async function loadLibrary(){
  try{
    const d=await (await fetch('/api/library')).json();
    const el=$('#library'); if(!el) return; el.innerHTML='';
    const dir=$('#libdir'); if(dir) dir.textContent=d.dir||'';
    (d.documents||[]).forEach(x=>{
      const it=document.createElement('div'); it.className='item';
      it.innerHTML=`<div class="s"></div><div class="meta">${(x.chunks||0)} passages · ${fmtDate(x.ingested_at)}</div>`;
      it.querySelector('.s').textContent=x.title||x.file||'document';
      el.appendChild(it);
    });
  }catch(e){}
}
function toast(msg){
  let t=$('#toast'); if(!t){ t=document.createElement('div'); t.id='toast'; document.body.appendChild(t); }
  t.textContent=msg; t.classList.add('on'); clearTimeout(t._h); t._h=setTimeout(()=>t.classList.remove('on'),4500);
}
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
    analyser.fftSize=256;
    curGain=audioCtx.createGain(); curGain.gain.value=parseFloat($('#vol').value||'1');
    src.connect(analyser); analyser.connect(curGain); curGain.connect(audioCtx.destination);
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
