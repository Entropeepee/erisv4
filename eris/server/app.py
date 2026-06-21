"""
Eris Echo v4 — FastAPI Server
================================

Endpoints:
    POST /chat              Main conversation endpoint
    POST /v1/chat/completions OpenAI-compatible endpoint for NVIDIA ACE
    GET  /vitals            Real-time system metrics
    POST /dream             Trigger a dreaming cycle
    GET  /questions          Pending questions from metacognition
    POST /sandbox           Execute code in the sandbox
    POST /ingest            Ingest text into the knowledge base
    GET  /                  Web UI (static HTML)
    WS   /ws                WebSocket for real-time metrics streaming

Usage:
    from eris.server.app import create_app

    app = create_app()
    # Run with: uvicorn eris.server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations
from typing import Optional
import os
import json
import asyncio

# FastAPI is an optional dependency — graceful fallback
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from eris.orchestrator import ErisOrchestrator
from eris.sandbox.executor import SandboxExecutor
from eris.knowledge.extractor import KnowledgeExtractor
from eris.interface.tts import TTSEngine
from eris.server.system_stats import get_system_stats
from eris.memory.conversations import ConversationStore
from eris.knowledge.study import StudyEngine
from eris.knowledge.documents import DocumentLibrary, library_dir
from fastapi import Response, UploadFile, File


class ChatRequest(BaseModel if HAS_FASTAPI else object):
    message: str = ""
    system_context: str = ""
    conversation_id: str = ""

class PonderRequest(BaseModel if HAS_FASTAPI else object):
    question: str = ""

class TopicsRequest(BaseModel if HAS_FASTAPI else object):
    topics: list = []

class StudyRunRequest(BaseModel if HAS_FASTAPI else object):
    topics: list = []

class OpenAIChatRequest(BaseModel if HAS_FASTAPI else object):
    model: str = "eris"
    messages: list = []
    temperature: float = 0.7


class SandboxRequest(BaseModel if HAS_FASTAPI else object):
    code: str = ""
    timeout: int = 60


class IngestRequest(BaseModel if HAS_FASTAPI else object):
    text: str = ""
    title: str = ""

class TTSGenerateRequest(BaseModel if HAS_FASTAPI else object):
    text: str = ""
    voice_id: str = ""


def create_app(
    field_size: int = 64,
    data_dir: str = "eris_data",
    use_frt: bool = False,
) -> "FastAPI":
    """Create the FastAPI application with all endpoints."""
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI not installed. Run: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="Eris Echo v4",
        description="Resonant Cognitive Architecture",
        version="4.0.0",
    )

    # Tier 7: serve the cockpit's static assets (JS, portrait, Live2D model dir).
    _static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(_static_dir, exist_ok=True)
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

    # Core systems
    orchestrator = ErisOrchestrator(
        field_size=field_size,
        data_dir=data_dir,
        use_frt_seeding=use_frt,
    )
    sandbox = SandboxExecutor()
    extractor = KnowledgeExtractor(
        output_dir=os.path.join(data_dir, "knowledge_base"),
        use_frt=use_frt,
    )

    tts_engine = TTSEngine()
    conversations = ConversationStore(
        data_dir=os.path.join(data_dir, "conversations"))
    study = StudyEngine(extractor, orchestrator.memory, data_dir=data_dir,
                        journal=orchestrator.dream_journal, mediator=orchestrator.mediator)
    library = DocumentLibrary(extractor, orchestrator.memory, data_dir=data_dir)

    # WebSocket connections for real-time metrics
    ws_connections: list = []

    # ── Endpoints ─────────────────────────────────────────────

    @app.post("/chat")
    async def chat(req: ChatRequest):
        """Main conversation endpoint."""
        result = await orchestrator.process(
            req.message, system_context=req.system_context
        )
        # Tier 7: persist into a conversation thread for the history sidebar.
        cid = req.conversation_id or conversations.new_thread(req.message)
        try:
            conversations.add_turn(cid, req.message, result.response_text,
                                   meta={"regime": result.regime,
                                         "archetype": result.archetype})
        except Exception:
            pass

        # Broadcast metrics to WebSocket clients
        vitals = orchestrator.get_vitals()
        for ws in ws_connections:
            try:
                await ws.send_json(vitals)
            except Exception:
                pass

        return {
            "response": result.response_text,
            "reasoning": getattr(result, 'reasoning_text', ''),
            "input_bvec": result.input_bvec.as_dict() if result.input_bvec else None,
            "response_bvec": result.response_bvec.as_dict() if result.response_bvec else None,
            "coherence": result.coherence,
            "dCdX": result.dCdX,
            "regime": result.regime,
            "archetype": result.archetype,
            "dissonance": result.dissonance,
            "specialist": result.specialist_source,
            "llm_provider": result.llm_provider,
            "latency_ms": result.latency_ms,
            "contradiction_compiled": result.contradiction_compiled,
            "research_triggered": result.research_triggered,
            "conversation_id": cid,
        }

    # ── Tier 7 cockpit endpoints ─────────────────────────────────────────
    @app.get("/api/system")
    async def api_system():
        """Host telemetry: CPU/RAM/GPU/VRAM/temperatures."""
        return get_system_stats()

    @app.get("/api/conversations")
    async def api_conversations():
        return {"conversations": conversations.list_threads()}

    @app.get("/api/conversations/{cid}")
    async def api_conversation(cid: str):
        d = conversations.get_thread(cid)
        return d or JSONResponse({"error": "not found"}, status_code=404)

    @app.get("/api/dreams")
    async def api_dreams():
        return {"dreams": orchestrator.get_dreams(limit=60)}

    @app.get("/api/dreams/{did}")
    async def api_dream(did: str):
        d = orchestrator.get_dream(did)
        return d or JSONResponse({"error": "not found"}, status_code=404)

    @app.post("/api/dream/ponder")
    async def api_ponder(req: PonderRequest):
        if not req.question.strip():
            return JSONResponse({"error": "question required"}, status_code=400)
        entry = await orchestrator.ponder(req.question.strip())
        return entry

    @app.get("/api/study/topics")
    async def api_get_topics():
        return {"topics": study.get_topics()}

    @app.post("/api/study/topics")
    async def api_set_topics(req: TopicsRequest):
        return {"topics": study.set_topics(list(req.topics))}

    @app.get("/api/study/reports")
    async def api_study_reports():
        return {"reports": study.list_reports(limit=60)}

    @app.get("/api/study/reports/{rid}")
    async def api_study_report(rid: str):
        r = study.get_report(rid)
        return r or JSONResponse({"error": "not found"}, status_code=404)

    @app.post("/api/study/run")
    async def api_study_run(req: StudyRunRequest):
        topics = list(req.topics) or None
        report = await asyncio.to_thread(study.study, topics)
        return report

    # ── Tier 7.4 document library ────────────────────────────────────────
    @app.get("/api/library")
    async def api_library():
        return {"dir": library_dir(), "documents": library.list_documents()}

    @app.post("/api/library/scan")
    async def api_library_scan(force: bool = False):
        """Read & ingest every supported file in the ErisLibrary folder.
        force=true re-ingests everything (use after a physics change)."""
        return await asyncio.to_thread(lambda: library.ingest_dir(force=force))

    # File upload needs python-multipart; register the route only if it's
    # installed so the server still starts (and folder-scan still works)
    # without it. The /api/library/upload route reports how to enable it.
    def _multipart_available() -> bool:
        for _m in ("multipart", "python_multipart"):
            try:
                __import__(_m)
                return True
            except Exception:
                continue
        return False

    if _multipart_available():
        @app.post("/api/library/upload")
        async def api_library_upload(file: UploadFile = File(...)):
            """Ingest a file chosen in the browser (txt/md/pdf/docx/json)."""
            import tempfile
            suffix = os.path.splitext(file.filename or "upload")[1] or ".txt"
            data = await file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            try:
                # Preserve the original name for memory (temp file is random).
                res = await asyncio.to_thread(
                    lambda: library.ingest_file(tmp_path, display_name=file.filename))
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            return res
    else:
        @app.post("/api/library/upload")
        async def api_library_upload_disabled():
            return JSONResponse(
                {"error": "file upload needs python-multipart — run: pip install python-multipart"},
                status_code=501)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: OpenAIChatRequest):
        """OpenAI-compatible chat completions endpoint for NVIDIA ACE."""
        message_text = ""
        system_context = ""
        for msg in req.messages:
            if msg.get("role") == "system":
                system_context += msg.get("content", "") + "\n"
            elif msg.get("role") == "user":
                message_text += msg.get("content", "") + "\n"

        result = await orchestrator.process(
            message_text.strip(), system_context=system_context.strip()
        )

        return {
            "id": "chatcmpl-eris",
            "object": "chat.completion",
            "created": 1677652288,
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.response_text,
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

    @app.get("/vitals")
    async def vitals():
        """Real-time system metrics."""
        return orchestrator.get_vitals()

    @app.post("/dream")
    async def dream():
        """Trigger a dreaming cycle."""
        report = await orchestrator.run_dream_cycle()
        return report

    @app.get("/questions")
    async def questions():
        """Pending questions from the metacognition loop."""
        return {"questions": orchestrator.drain_pending_questions()}

    @app.post("/sandbox")
    async def sandbox_exec(req: SandboxRequest):
        """Execute code in the sandbox."""
        result = sandbox.execute(req.code, timeout=req.timeout)
        return {
            "status": result.status.value,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "duration_ms": result.duration_ms,
            "blocked_reason": result.blocked_reason,
        }

    @app.post("/ingest")
    async def ingest(req: IngestRequest):
        """Ingest text into the knowledge base."""
        descriptors = extractor.extract_text(req.text, title=req.title)
        return {
            "chunks_created": len(descriptors),
            "bvecs": [d.bvec.as_dict() if d.bvec else None for d in descriptors],
        }

    async def _periodic_dream_loop():
        # First reflection ~90s after boot so the cockpit's Dreams panel
        # populates soon after start, then settle into the 15-minute cadence.
        first = True
        while True:
            await asyncio.sleep(90 if first else 60 * 15)
            first = False
            try:
                report = await orchestrator.run_dream_cycle()
                print(f"[Dream Loop] Processed {report['tensions_processed']} tensions. Resolved: {report['tensions_resolved']}")
            except Exception as e:
                print(f"[Dream Loop Error] {e}")

    async def _nightly_learning_loop():
        """Tier 7: once per day (default 03:00 local), study the topic list and
        write a study report for the cockpit. Set ERIS_STUDY_HOUR to change the
        hour; ERIS_STUDY_ENABLED=0 to disable."""
        import datetime
        if os.environ.get("ERIS_STUDY_ENABLED", "1") == "0":
            return
        hour = int(os.environ.get("ERIS_STUDY_HOUR", "3"))
        last_day = None
        while True:
            now = datetime.datetime.now()
            if now.hour == hour and now.date() != last_day:
                last_day = now.date()
                try:
                    rep = await asyncio.to_thread(study.study)
                    print(f"[Study] nightly session: {rep['total_chunks']} passages on {rep['topics']}")
                except Exception as e:
                    print(f"[Study Error] {e}")
            await asyncio.sleep(300)  # check every 5 min

    @app.on_event("startup")
    async def startup_event():
        asyncio.create_task(_periodic_dream_loop())
        asyncio.create_task(_nightly_learning_loop())

    @app.get("/api/tts/voices")
    async def get_tts_voices():
        """Get available TTS voices."""
        return {"voices": tts_engine.get_voices(), "default": {"engine": "pyttsx3", "id": ""}}

    @app.post("/api/tts/generate")
    async def generate_tts(req: TTSGenerateRequest):
        """Generate TTS audio."""
        import re
        # Strip common markdown to prevent reading asterisks/hashes out loud
        clean_text = re.sub(r'[*_`#~]', '', req.text)
        wav_bytes = await tts_engine._generate_audio_async(clean_text, req.voice_id)
        if not wav_bytes:
            return JSONResponse({"error": "TTS generation failed."}, status_code=500)
        return Response(content=wav_bytes, media_type="audio/wav")

    @app.get("/api/status")
    async def get_status():
        """Check if LLM is ready."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get("http://localhost:11434/api/tags")
                if resp.status_code == 200:
                    return {"llm_ready": True}
        except Exception:
            pass
        return {"llm_ready": False}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket for real-time metrics streaming."""
        await websocket.accept()
        ws_connections.append(websocket)
        try:
            while True:
                # Send vitals every 2 seconds
                await websocket.send_json(orchestrator.get_vitals())
                await asyncio.sleep(2)
        except WebSocketDisconnect:
            ws_connections.remove(websocket)
        except Exception:
            if websocket in ws_connections:
                ws_connections.remove(websocket)

    @app.websocket("/ws/field")
    async def websocket_field_endpoint(websocket: WebSocket):
        """WebSocket for real-time field state streaming."""
        await websocket.accept()
        from eris.config import to_numpy
        try:
            while True:
                # Only send if field exists
                if orchestrator.field is not None:
                    # Extract live arrays
                    phi = to_numpy(orchestrator.field.phi).tolist()
                    theta = to_numpy(orchestrator.field.theta).tolist()

                    await websocket.send_json({
                        "size": orchestrator.field.size,
                        "step_count": orchestrator.field.step_count,
                        "phi": phi,
                        "theta": theta,

                        "coherence": orchestrator.field.coherence,
                        "dCdX": orchestrator.field.dCdX,
                        "regime_str": orchestrator.field.detect_regime(),
                    })
                # 10 fps is enough for the visualizer without overloading
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[WS Field Error] {e}")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the web UI."""
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content=_MINIMAL_UI)

    return app

# Create the global application instance for uvicorn
app = create_app() if HAS_FASTAPI else None

# Minimal embedded UI (when static/index.html doesn't exist yet)
_MINIMAL_UI = """<!DOCTYPE html>
<html><head><title>Eris Echo v4</title>
<style>
  body { font-family: system-ui; background: #0a0a0a; color: #e0e0e0; margin: 2em; }
  h1 { color: #7b68ee; }
  #chat { max-width: 700px; }
  #messages { min-height: 300px; border: 1px solid #333; padding: 1em; margin-bottom: 1em;
              border-radius: 8px; overflow-y: auto; max-height: 500px; }
  .user { color: #88ccff; }
  .eris { color: #c8a2c8; }
  .meta { color: #666; font-size: 0.8em; }
  input { width: 80%; padding: 0.5em; background: #1a1a1a; color: #e0e0e0;
          border: 1px solid #444; border-radius: 4px; }
  button { padding: 0.5em 1em; background: #7b68ee; color: white; border: none;
           border-radius: 4px; cursor: pointer; }
  #vitals { position: fixed; right: 2em; top: 2em; background: #111; padding: 1em;
            border-radius: 8px; border: 1px solid #333; font-size: 0.85em; width: 250px; }
</style></head><body>
<h1>Eris Echo v4</h1>
<div id="vitals"><b>Vitals</b><pre id="vitals-data">connecting...</pre></div>
<div id="chat">
  <div id="messages"></div>
  <input id="input" placeholder="Type a message..." onkeydown="if(event.key==='Enter')send()">
  <button onclick="send()">Send</button>
</div>
<script>
const msgs = document.getElementById('messages');
const inp = document.getElementById('input');

async function send() {
  const text = inp.value.trim();
  if (!text) return;
  msgs.innerHTML += `<p class="user"><b>You:</b> ${text}</p>`;
  inp.value = '';
  try {
    const r = await fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({message: text})});
    const d = await r.json();
    msgs.innerHTML += `<p class="eris"><b>Eris:</b> ${d.response}</p>`;
    msgs.innerHTML += `<p class="meta">${d.archetype} | ${d.regime} | dC/dX=${d.dCdX.toFixed(3)} | ${d.latency_ms.toFixed(0)}ms</p>`;
  } catch(e) { msgs.innerHTML += `<p class="meta">Error: ${e}</p>`; }
  msgs.scrollTop = msgs.scrollHeight;
}

// WebSocket vitals
const ws = new WebSocket(`ws://${location.host}/ws`);
ws.onmessage = (e) => {
  document.getElementById('vitals-data').textContent = JSON.stringify(JSON.parse(e.data), null, 2);
};
</script></body></html>"""


# Allow direct run: python -m eris.server.app
if __name__ == "__main__":
    if HAS_FASTAPI:
        import uvicorn
        uvicorn.run("eris.server.app:app", host="0.0.0.0", port=8001, reload=True)
    else:
        print("FastAPI not installed. Run: pip install fastapi uvicorn")
