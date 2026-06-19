import os
import re

source_file = "visualizer.html"
with open(source_file, "r", encoding="utf-8") as f:
    html = f.read()

# 1. Master: visualizer_pedestal.html
master_html = html.replace("/* ====================== go ====================== */", """
/* ---- BROADCAST CHANNEL (MASTER) ---- */
const bc = new BroadcastChannel('bcdc_sync');
let lastSyncTime = 0;
function broadcastSync() {
    let now = performance.now();
    if (now - lastSyncTime > 50) {
        bc.postMessage({
            type: 'sync',
            P: P,
            A: A,
            paused: paused,
            view: view,
            showTrails: showTrails
        });
        lastSyncTime = now;
    }
}
const oldLoop = loop;
loop = function() {
    broadcastSync();
    oldLoop();
};
/* ====================== go ====================== */
""")

master_html = master_html.replace("function applyPreset(name){", """
function applyPreset(name){
  bc.postMessage({type: 'action', action: 'preset', name: name});
""")
master_html = master_html.replace("function resetField(){", """
function resetField(){
  bc.postMessage({type: 'action', action: 'resetField'});
""")
master_html = master_html.replace("function addParticles(n){", """
function addParticles(n){
  bc.postMessage({type: 'action', action: 'addParticles', n: n});
""")
master_html = master_html.replace("function addVectors(n){", """
function addVectors(n){
  bc.postMessage({type: 'action', action: 'addVectors', n: n});
""")
master_html = master_html.replace("agents=[];", "agents=[]; bc.postMessage({type:'action', action:'clearAgents'});")
master_html = master_html.replace("function encodeText(text){", """
function encodeText(text){
  bc.postMessage({type: 'action', action: 'encodeText', text: text});
""")
master_html = master_html.replace("function probeReactivity(cx=N/2,cy=N/2){", """
function probeReactivity(cx=N/2,cy=N/2){
  bc.postMessage({type: 'action', action: 'probeReactivity'});
""")
master_html = master_html.replace("agents.push(new Agent(gx,gy));", "agents.push(new Agent(gx,gy)); bc.postMessage({type:'action', action:'addAgent', gx:gx, gy:gy});")
master_html = master_html.replace("agents.push(new Agent(gx,gy,vx,vy));", "agents.push(new Agent(gx,gy,vx,vy)); bc.postMessage({type:'action', action:'addAgentV', gx:gx, gy:gy, vx:vx, vy:vy});")


# 2. Slave: visualizer_display.html
slave_html = html

slave_html = re.sub(r"<header>.*?</header>", "", slave_html, flags=re.DOTALL)
slave_html = re.sub(r"<aside>.*?</aside>", "", slave_html, flags=re.DOTALL)
slave_html = re.sub(r"<aside class=\"met\">.*?</aside>", "", slave_html, flags=re.DOTALL)
slave_html = re.sub(r"<div class=\"hud\".*?</div>", "", slave_html, flags=re.DOTALL)
slave_html = re.sub(r"<div class=\"viewbar.*?</div>", "", slave_html, flags=re.DOTALL)

slave_html = slave_html.replace("body{display:grid;grid-template-columns:288px 1fr 232px;grid-template-rows:48px 1fr;gap:1px;\n       background:var(--line);height:100vh;overflow:hidden}", 
"body{display:flex;align-items:center;justify-content:center;background:#000000;height:100vh;margin:0;overflow:hidden;}")

slave_html = slave_html.replace("<canvas id=\"cv\" width=\"560\" height=\"560\"></canvas>", "<canvas id=\"cv\" width=\"2048\" height=\"2048\" style=\"width:100vw; height:100vh; object-fit:fill;\"></canvas>")
slave_html = slave_html.replace("const cv=document.getElementById('cv'), ctx=cv.getContext('2d'); const cell=cv.width/N;", 
"const cv=document.getElementById('cv'), ctx=cv.getContext('2d'); cv.width=window.innerWidth; cv.height=window.innerHeight; const cell=Math.min(cv.width, cv.height)/N;")

slave_html = slave_html.replace("buildSliders(); buildChips(); syncSliders(); applyPreset('resonant'); loop();", """
const bc = new BroadcastChannel('bcdc_sync');
bc.onmessage = function(e) {
    if(e.data.type === 'sync') {
        Object.assign(P, e.data.P);
        Object.assign(A, e.data.A);
        paused = e.data.paused;
        view = e.data.view;
        showTrails = e.data.showTrails;
    } else if (e.data.type === 'action') {
        if(e.data.action === 'preset') { const pr=PRESETS[e.data.name]; if(pr){ Object.assign(P,pr.p); Object.assign(A,pr.A); showTrails=pr.trails; agents=[]; for(let i=0;i<pr.agents;i++) agents.push(new Agent(Math.random()*N,Math.random()*N)); resetField(); } }
        else if(e.data.action === 'resetField') resetField();
        else if(e.data.action === 'addParticles') addParticles(e.data.n);
        else if(e.data.action === 'addVectors') addVectors(e.data.n);
        else if(e.data.action === 'clearAgents') agents = [];
        else if(e.data.action === 'encodeText') encodeText(e.data.text);
        else if(e.data.action === 'probeReactivity') probeReactivity();
        else if(e.data.action === 'addAgent') agents.push(new Agent(e.data.gx, e.data.gy));
        else if(e.data.action === 'addAgentV') agents.push(new Agent(e.data.gx, e.data.gy, e.data.vx, e.data.vy));
    }
};
applyPreset('resonant'); loop();
""")

slave_html = slave_html.replace("function metrics(){", "function metrics() { return; \n// ")
# also fix the canvas resize on window resize so it fills properly if window resizes
slave_html = slave_html.replace("/* ====================== go ====================== */", """
window.addEventListener('resize', () => {
  cv.width=window.innerWidth; cv.height=window.innerHeight;
});
/* ====================== go ====================== */
""")

with open("visualizer_pedestal.html", "w", encoding="utf-8") as f:
    f.write(master_html)
    
with open("visualizer_display.html", "w", encoding="utf-8") as f:
    f.write(slave_html)

print("HTML split successful.")
