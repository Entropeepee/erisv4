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
    from fastapi import Response, UploadFile, File, Request
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from eris.orchestrator import ErisOrchestrator
from eris.sandbox.executor import SandboxExecutor, endpoint_guard as _sandbox_endpoint_guard
from eris.knowledge.extractor import KnowledgeExtractor
from eris.interface.tts import TTSEngine
from eris.server.system_stats import get_system_stats
from eris.memory.conversations import ConversationStore
from eris.knowledge.study import StudyEngine
from eris.knowledge.documents import DocumentLibrary, library_dir


class ChatRequest(BaseModel if HAS_FASTAPI else object):
    message: str = ""
    system_context: str = ""
    conversation_id: str = ""
    profile: str = ""   # mode/profile id (e.g. "fast" | "deep"); "" => default

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
    profile: str = ""   # optional Eris mode/profile id (fast|deep|…)


class SandboxRequest(BaseModel if HAS_FASTAPI else object):
    code: str = ""
    timeout: int = 60


class AuthorRequest(BaseModel if HAS_FASTAPI else object):
    brief: str = ""
    formats: list = []        # any of: md, txt, docx, pdf
    audit: bool = True


class ConverseRequest(BaseModel if HAS_FASTAPI else object):
    speakers: list = []       # node names, e.g. ["willow", "npc_c"]
    context: str = ""
    turns: int = 6
    backend: str = ""


class IngestRequest(BaseModel if HAS_FASTAPI else object):
    text: str = ""
    title: str = ""

class DeepReadRequest(BaseModel if HAS_FASTAPI else object):
    source: str = ""   # a file/folder path on the server, 'ltm', or raw text

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

    # Optional access gate (for reaching her from your phone over Tailscale/5G).
    # OFF by default → unchanged local/LAN behavior. When ERIS_AUTH_TOKEN is set,
    # every request must present the token (header X-Eris-Token, ?token=, or the
    # cookie that visiting `/?token=SECRET` once sets). The tunnel/VPN is the
    # transport security; this stops anyone else who reaches the port.
    _auth_token = os.environ.get("ERIS_AUTH_TOKEN", "").strip()
    if _auth_token:
        from eris.server.auth import make_auth_middleware
        app.add_middleware(make_auth_middleware(_auth_token))

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
                        journal=orchestrator.dream_journal, mediator=orchestrator.mediator,
                        thought_stream=getattr(orchestrator, "thought_stream", None))
    library = DocumentLibrary(extractor, orchestrator.memory, data_dir=data_dir)
    from eris.interface.profiles import ProfileStore
    profiles = ProfileStore(data_dir)   # Fast/Deep mode selector (re-read on demand)
    from eris.knowledge.author import DocumentAuthor
    from eris.knowledge.documents import library_dir as _lib_dir
    author = DocumentAuthor(orchestrator.mediator,
                            out_dir=os.path.join(_lib_dir(), "generated"))
    # A2: one workload governor — foreground chat stays responsive while the four
    # background loops + heavy endpoints share the GPU. Background work serializes
    # and defers to active foreground; chat never waits on background.
    from eris.server.governor import Governor
    governor = Governor()

    # WILLOW Part I — the multi-node collective: the pool is orchestrator.memory;
    # 'eris' is the OverSoul, 'willow' a companion node. Unreal addresses nodes
    # by `model` name. Single-Eris behaviour is unchanged (eris is the default).
    from eris.agents.registry import build_default_registry
    from eris.agents.backends import build_backends
    from eris.agents.budget import ConversationBudget, choose_dialogue_plan
    from eris.agents.dialogue import generate_dialogue
    from eris.agents.federation import federate
    agent_backends = build_backends()
    registry = build_default_registry(orchestrator, data_dir=data_dir,
                                      field_size=field_size)
    convo_budget = ConversationBudget(
        per_hour=int(os.environ.get("ERIS_CONVO_PER_HOUR", "60")))

    # WebSocket connections for real-time metrics
    ws_connections: list = []

    # ── Endpoints ─────────────────────────────────────────────

    @app.get("/api/profiles")
    async def api_profiles():
        """The mode/profile dropdown (re-read from disk so edits need no reboot)."""
        profiles.reload()
        return {"profiles": [p.public() for p in profiles.list()],
                "default": profiles.default().id}

    @app.post("/chat")
    async def chat(req: ChatRequest):
        """Main conversation endpoint."""
        prof = profiles.get(req.profile)
        async with governor.foreground():
            result = await orchestrator.process(
                req.message, system_context=req.system_context, profile=prof
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
            "citations": getattr(result, "citations", []),
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
        try:
            d = conversations.get_thread(cid)
        except ValueError:
            return JSONResponse({"error": "invalid conversation id"}, status_code=400)
        return d or JSONResponse({"error": "not found"}, status_code=404)

    @app.get("/api/dreams")
    async def api_dreams(limit: int = 8, before: Optional[float] = None):
        return {"dreams": orchestrator.get_dreams(limit=limit, before=before)}

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

    @app.post("/api/dream/subjective")
    async def api_subjective_dream():
        """Trigger her undirected subjective dream now (decompression on the day; no research,
        no hive). Returns the dream entry, or a note if there was no day to dream about yet."""
        d = await orchestrator.subjective_dream()
        if not d:
            return {"dreamed": False,
                    "message": "Nothing to dream about yet — no recent conversation or tension."}
        return d

    @app.post("/api/dream/reconsider")
    async def api_reconsider(req: PonderRequest):
        """Step 5: have her compare her naive first impression of a topic against her post-hive
        conclusion and write a calibration lesson. Optional question = the topic to reconsider;
        empty = let her pick a topic where both views exist."""
        mc = await orchestrator.metacognitive_review(req.question.strip())
        if not mc:
            return {"reconsidered": False,
                    "message": "No topic yet has both a first impression and an analyzed "
                               "conclusion to compare."}
        return mc

    @app.post("/api/retrospect")
    async def api_retrospect(req: PonderRequest):
        """Look back over her own past thoughts on a topic and synthesize them."""
        topic = req.question.strip()
        if not topic:
            return JSONResponse({"error": "topic required"}, status_code=400)
        return await orchestrator.retrospect(topic)

    @app.post("/api/study-topic")
    async def api_study_topic(req: PonderRequest):
        """Steer Eris's self-directed crawl onto a specific topic (guided)."""
        topic = req.question.strip()
        if not topic:
            return JSONResponse({"error": "topic required"}, status_code=400)
        orchestrator.dreaming_loop.topic_queue.append(topic)
        return {"queued": topic, "queue_len": len(orchestrator.dreaming_loop.topic_queue),
                "message": f"Queued: {topic} — Eris will study this on her next cycle."}

    @app.post("/api/study-now")
    async def api_study_now(req: PonderRequest):
        """Study a topic RIGHT NOW instead of waiting for the dream-loop timer
        (Fix B). Runs the crawl off the event loop so the /ws keepalive isn't
        starved; returns what Eris studied and kept."""
        topic = req.question.strip()
        if not topic:
            return JSONResponse({"error": "topic required"}, status_code=400)
        # Jump the queue so idle_explore picks THIS topic first.
        orchestrator.dreaming_loop.topic_queue.insert(0, topic)
        async with governor.foreground():
            result = await asyncio.to_thread(orchestrator.dreaming_loop.idle_explore)
        if not result:
            return {"studied": topic, "message": f"Nothing new found for '{topic}'."}
        return result

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
        # B5: don't leak the absolute host path — basename only.
        return {"dir": os.path.basename(library_dir().rstrip("/\\")),
                "documents": library.list_documents()}

    @app.get("/api/library/progress")
    async def api_library_progress():
        return library.progress

    @app.get("/api/memory/search")
    async def api_memory_search(q: str = "", k: int = 8):
        """Diagnostic: see exactly what Eris retrieves for a query — confirms a
        document is actually stored and surfaced. Visit /api/memory/search?q=..."""
        from eris.knowledge.embeddings import get_embedding
        emb = get_embedding(q) if q else None
        mem = orchestrator.memory
        named = mem.documents_matching(q, max_chunks=k, query_embedding=emb) if q else []
        results = mem.retrieve(query_embedding=emb, query_text=q, top_k=k) if q else []

        # B5: memory snippets are raw stored content — only expose them with
        # ERIS_DEBUG (info-disclosure if the instance is exposed).
        _debug = os.environ.get("ERIS_DEBUG", "0").lower() not in ("0", "", "off", "false")

        def fmt(r):
            row = {"source": r.source,
                   "title": (r.metadata or {}).get("title"),
                   "chars": len(r.text or "")}
            if _debug:
                row["snippet"] = (r.text or "")[:300]
            return row
        return {
            "query": q,
            "memory_sizes": {"stm": mem.stm.size, "mtm": mem.mtm.size, "ltm": mem.ltm.size},
            "named_documents": [fmt(r) for r in named],
            "retrieved": [fmt(r) for r in results],
        }

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
            # B6: bound the upload — stream in chunks and reject past the cap so a
            # huge file can't exhaust memory/disk. ERIS_MAX_UPLOAD_MB (default 100).
            max_bytes = int(os.environ.get("ERIS_MAX_UPLOAD_MB", "100")) * 1024 * 1024
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = tmp.name
                total = 0
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        tmp.close()
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
                        return JSONResponse(
                            {"error": f"file too large (max {max_bytes // (1024*1024)} MB)"},
                            status_code=413)
                    tmp.write(chunk)
            try:
                # Preserve the original name for memory (temp file is random).
                res = await asyncio.to_thread(
                    lambda: library.ingest_upload(tmp_path, file.filename))
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
        """OpenAI-compatible endpoint. Routes by `model` to a node (WILLOW I.5):
        model:"willow" -> the Willow node, model:"eris" (default) -> the OverSoul.
        Used by NVIDIA ACE and the Unreal NPC plugins."""
        message_text = ""
        system_context = ""
        for msg in req.messages:
            if msg.get("role") == "system":
                system_context += msg.get("content", "") + "\n"
            elif msg.get("role") == "user":
                message_text += msg.get("content", "") + "\n"

        agent = registry.get(req.model)
        if agent is not None and agent.name != "eris":
            reply = await agent.respond(message_text.strip(), agent_backends)
        else:
            async with governor.foreground():
                result = await orchestrator.process(
                    message_text.strip(), system_context=system_context.strip(),
                    profile=profiles.get(req.profile))
            reply = result.response_text

        return {
            "id": "chatcmpl-eris",
            "object": "chat.completion",
            "created": 1677652288,
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    @app.get("/agents")
    async def api_agents():
        """List nodes (debug / Unreal config)."""
        return {"agents": [
            {"name": a.name, "backend": a.backend_id, "has_field": a.has_field,
             "insights": (len(a.insight_log.recent(999)) if a.insight_log else 0)}
            for a in registry.agents.values()]}

    @app.post("/converse")
    async def api_converse(req: ConverseRequest):
        """Puppeteered one-call NPC↔NPC dialogue (WILLOW I.8 Mode B), with the
        degradation ladder. Lines are written into each speaker's private memory."""
        speakers = [s for s in (registry.get(n) for n in req.speakers) if s is not None]
        if len(speakers) < 2:
            return JSONResponse({"error": "need >=2 known speakers"}, status_code=400)
        plan = choose_dialogue_plan(speakers, agent_backends, convo_budget)
        if plan["mode"] == "skip":
            return {"mode": "skip", "script": []}
        backend = (agent_backends.get(plan.get("backend", req.backend or speakers[0].backend_id))
                   or agent_backends.get("ollama"))
        script = await generate_dialogue(backend, speakers, req.context, req.turns)
        convo_budget.charge()
        for line in script:
            for s in speakers:
                framing = "said" if line["speaker"].lower() == s.name.lower() else "heard"
                mem = s.memory
                if hasattr(mem, "store_experience"):
                    try:
                        mem.store_experience(f'{line["speaker"]}: {line["text"]}', kind=framing)
                    except Exception:
                        pass
        return {"mode": plan["mode"], "script": script}

    @app.get("/agents/{name}/insights")
    async def api_agent_insights(name: str):
        a = registry.get(name)
        if a is None or a.insight_log is None:
            return {"agent": name, "insights": []}
        return {"agent": a.name, "insights": [
            {"summary": i.summary, "regime": i.regime, "timestamp": i.timestamp,
             "federated": i.federated} for i in a.insight_log.recent(50)]}

    @app.post("/agents/{name}/reflect")
    async def api_agent_reflect(name: str):
        """Distill ONE insight from the node's recent private experience."""
        a = registry.get(name)
        if a is None or a.insight_log is None:
            return JSONResponse({"error": "no such node / no insight log"}, status_code=404)
        ins = await a.distill(agent_backends)
        return {"insight": (ins.summary if ins else None)}

    @app.post("/agents/{name}/federate")
    async def api_agent_federate(name: str):
        """Push the node's NOVEL insights into the collective pool."""
        a = registry.get(name)
        if a is None or a.insight_log is None:
            return JSONResponse({"error": "no such node / no insight log"}, status_code=404)
        pushed = federate(a.insight_log, a.name, orchestrator.memory)
        return {"federated": pushed}

    @app.get("/health")
    async def health():
        """Lets the Llama-Unreal / NPC plugins connect cleanly."""
        return {"status": "ok"}

    @app.get("/props")
    async def props():
        """Minimal llama.cpp-style props stub for NPC plugins."""
        return {"default_generation_settings": {"model": "eris"},
                "model": "eris", "chat_template": ""}

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

    async def _run_sandbox(req: SandboxRequest):
        # Default-DENY, INDEPENDENT of the auth token. The old gate only blocked when a token was set,
        # so a default no-token box left arbitrary code execution ON to any caller (unauth file
        # read/write/exec — the verified CRITICALs). The endpoint now runs only when explicitly
        # enabled AND under docker isolation (the validator is not a security boundary; host
        # subprocess is not isolation) unless an acknowledged subprocess opt-in is set. See
        # eris.sandbox.executor.endpoint_guard.
        _deny = _sandbox_endpoint_guard(sandbox.mode)
        if _deny:
            return JSONResponse({"error": _deny}, status_code=403)
        # B6: clamp the user-supplied timeout (no unbounded runs).
        timeout = max(1, min(int(req.timeout or 60), 300))
        # Off the event loop — the subprocess/Docker run is blocking.
        result = await asyncio.to_thread(
            lambda: sandbox.execute(req.code, timeout=timeout))
        return {
            "status": result.status.value,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "duration_ms": result.duration_ms,
            "blocked_reason": result.blocked_reason,
        }

    @app.post("/sandbox")
    async def sandbox_exec(req: SandboxRequest):
        """Execute code in the sandbox."""
        return await _run_sandbox(req)

    @app.post("/api/sandbox/run")
    async def api_sandbox_run(req: SandboxRequest):
        """Run code in Eris's isolated sandbox (simulations / code tests)."""
        return await _run_sandbox(req)

    @app.get("/api/sandbox/info")
    async def api_sandbox_info():
        return {"mode": sandbox.mode.value, "timeout": sandbox.timeout,
                "stats": sandbox.stats}

    # ── Tier 7.11 document authoring ─────────────────────────────────────
    @app.post("/api/author/plan")
    async def api_author_plan(req: AuthorRequest):
        if not req.brief.strip():
            return JSONResponse({"error": "brief required"}, status_code=400)
        if not orchestrator.mediator:
            return JSONResponse({"error": "no language model available"}, status_code=503)
        return await asyncio.to_thread(lambda: author.plan(req.brief.strip()))

    @app.post("/api/author/write")
    async def api_author_write(req: AuthorRequest):
        if not req.brief.strip():
            return JSONResponse({"error": "brief required"}, status_code=400)
        if not orchestrator.mediator:
            return JSONResponse({"error": "no language model available"}, status_code=503)
        formats = [f for f in (req.formats or ["md"])
                   if f in ("md", "txt", "docx", "pdf")] or ["md"]
        result = await asyncio.to_thread(
            lambda: author.compose(req.brief.strip(), formats=formats,
                                   do_audit=bool(req.audit)))
        # Surface only download-safe metadata (not absolute paths).
        for f in result.get("files", []):
            if "path" in f:
                f["download"] = f"/api/author/file?name={f['name']}"
        return result

    @app.get("/api/author/progress")
    async def api_author_progress():
        return author.progress

    @app.get("/api/author/docs")
    async def api_author_docs():
        # B5: basename only — no absolute host path.
        return {"dir": os.path.basename(author.out_dir.rstrip("/\\")),
                "documents": author.list_documents()}

    @app.get("/api/author/file")
    async def api_author_file(name: str = ""):
        # Serve a generated file by name only (no path traversal).
        safe = os.path.basename(name or "")
        path = os.path.join(author.out_dir, safe)
        if not safe or not os.path.isfile(path):
            return JSONResponse({"error": "not found"}, status_code=404)
        from fastapi.responses import FileResponse
        return FileResponse(path, filename=safe)

    @app.post("/ingest")
    async def ingest(req: IngestRequest):
        """Ingest text into the knowledge base."""
        descriptors = extractor.extract_text(req.text, title=req.title)
        return {
            "chunks_created": len(descriptors),
            "bvecs": [d.bvec.as_dict() if d.bvec else None for d in descriptors],
        }

    async def _periodic_dream_loop():
        # Constant background learning, day and night: 2-6 cycles/hour. Each
        # cycle resolves tensions if any, else broad-crawls a knowledge topic.
        # ERIS_CRAWL_PERIOD_S sets the base interval (default 900s = 4/hr);
        # jitter avoids lockstep.
        import random
        first = True
        while True:
            if first:
                await asyncio.sleep(90)  # populate the Dreams panel soon after boot
                first = False
            else:
                base = int(os.environ.get("ERIS_CRAWL_PERIOD_S", "900"))
                # GAMING_MODE throttles background thinking so it doesn't fight the
                # renderer (Unreal) for the GPU (WILLOW I.9 / II.11).
                if os.environ.get("ERIS_GAMING_MODE", "0") not in ("0", "", "off", "false"):
                    base *= int(os.environ.get("ERIS_GAMING_THROTTLE", "4"))
                await asyncio.sleep(max(60, base + random.randint(-180, 180)))
            try:
                async with governor.background():
                    report = await orchestrator.run_dream_cycle()
                msg = f"[Dream Loop] Processed {report['tensions_processed']} tensions. Resolved: {report['tensions_resolved']}"
                if report.get("explored_topic"):
                    msg += f" | learned about: {report['explored_topic'][:80]}"
                print(msg)
            except Exception as e:
                print(f"[Dream Loop Error] {e}")
            # Federation pass: each real node distills a fresh insight from its own
            # experience and pushes the NOVEL ones into the collective pool, so a
            # different node can later act on what this one learned (WILLOW I.7).
            try:
                for node in registry.agents.values():
                    if node.insight_log is None:
                        continue
                    await node.distill(agent_backends)
                    n = federate(node.insight_log, node.name, orchestrator.memory)
                    if n:
                        print(f"[Federation] {node.name} -> pool: {n} novel insight(s)")
            except Exception as e:
                print(f"[Federation Error] {e}")

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
                    async with governor.background():
                        rep = await asyncio.to_thread(study.study)
                    print(f"[Study] nightly session: {rep['total_chunks']} passages on {rep['topics']}")
                except Exception as e:
                    print(f"[Study Error] {e}")
                # Sleep also REPLAYS: fold near-duplicate library traces into one reinforced
                # record (re-ingest junk collapses; repeated facts get stronger). Provenance-safe
                # — her reflections/dreams and the thought-stream are never touched.
                try:
                    merged = await asyncio.to_thread(orchestrator.memory.replay_consolidate)
                    if merged.get("mtm_merged") or merged.get("ltm_merged"):
                        print(f"[Replay] consolidated near-duplicates: {merged}")
                except Exception as e:
                    print(f"[Replay Error] {e}")
                # Sleep also DREAMS: a subjective, first-person decompression on the day — no
                # crawl, no hive, no question. Her own voice, separate from the cold logic.
                try:
                    d = await asyncio.to_thread(orchestrator.subjective_dream)
                    if d:
                        print(f"[Dream] subjective ({d.get('regime')}): {d.get('chars', 0)} chars")
                except Exception as e:
                    print(f"[Dream Error] {e}")
                # Sleep also RECONSIDERS: compare a naive first impression against the post-hive
                # conclusion on the same topic and write a calibration lesson (step 5).
                try:
                    mc = await asyncio.to_thread(orchestrator.metacognitive_review)
                    if mc:
                        print(f"[Metacognition] reconsidered '{mc.get('topic')}' — "
                              f"view moved {mc.get('moved')} ({mc.get('revision')})")
                except Exception as e:
                    print(f"[Metacognition Error] {e}")
            await asyncio.sleep(300)  # check every 5 min

    def _study_throttle() -> int:
        return (int(os.environ.get("ERIS_GAMING_THROTTLE", "4"))
                if os.environ.get("ERIS_GAMING_MODE", "0") not in ("0", "", "off", "false")
                else 1)

    async def _study_single_loop():
        """Single-article cadence — she reads + comprehends one self-chosen article
        every ERIS_STUDY_PERIOD_S (default 900s = 15 min). Fully independent of the
        deep-dive loop and of the (subjective) dream loop."""
        import random
        if os.environ.get("ERIS_AUTONOMOUS_STUDY", "1") in ("0", "off", "false"):
            return
        period = int(os.environ.get("ERIS_STUDY_PERIOD_S", "900"))
        await asyncio.sleep(150)                     # settle after boot
        while True:
            try:
                async with governor.background():
                    rep = await asyncio.to_thread(study.study_one)
                print(f"[Study:single] read {rep.get('topics')}: "
                      f"{rep.get('total_chunks', 0)} passages")
            except Exception as e:
                print(f"[Study:single Error] {e}")
            await asyncio.sleep(max(60, period * _study_throttle() + random.randint(-60, 60)))

    async def _study_deepdive_loop():
        """Deep-dive cadence — a multi-reference dive synthesized across sources
        through the calibration critic every ERIS_DEEPDIVE_PERIOD_S (default 1800s
        = twice/hour). Its own task, asynchronous to the single-article loop."""
        import random
        if os.environ.get("ERIS_AUTONOMOUS_STUDY", "1") in ("0", "off", "false"):
            return
        period = int(os.environ.get("ERIS_DEEPDIVE_PERIOD_S", "1800"))
        await asyncio.sleep(450)                     # offset from the single loop's first run
        while True:
            try:
                async with governor.background():
                    rep = await asyncio.to_thread(study.deep_dive)
                print(f"[Study:deep] dive on {rep.get('topics')}: "
                      f"{rep.get('total_chunks', 0)} passages"
                      + (" + synthesis" if rep.get("synthesis") else ""))
            except Exception as e:
                print(f"[Study:deep Error] {e}")
            await asyncio.sleep(max(120, period * _study_throttle() + random.randint(-90, 90)))

    @app.on_event("startup")
    async def startup_event():
        try:
            from eris.knowledge import ask_expert
            connected = ask_expert.is_available()
        except Exception:
            connected = False
        print(f"[Eris] Claude research path: "
              f"{'CONNECTED' if connected else 'dormant (no ANTHROPIC_API_KEY)'}")
        asyncio.create_task(_periodic_dream_loop())
        asyncio.create_task(_nightly_learning_loop())
        asyncio.create_task(_study_single_loop())     # 15-min single article
        asyncio.create_task(_study_deepdive_loop())   # 30-min multi-ref deep dive

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

    @app.post("/api/deep-read")
    async def api_deep_read(req: DeepReadRequest):
        """Comprehend a document/folder/codebase larger than the context window
        (RAPTOR map-reduce): summarize chunks on the Fast profile, recursively
        reduce, synthesize on the Deep profile, and store every node in memory so
        Eris can answer from it. Runs OFF the event loop; resumable + idempotent."""
        from eris.knowledge.deep_read import deep_read
        from eris.interface.mediator import run_blocking
        src = (req.source or "").strip()
        if not src:
            return JSONResponse({"error": "source required (path, 'ltm', or text)"},
                                status_code=400)
        fast, deep = profiles.get("fast"), profiles.get("deep")

        def summarize(text: str, deep_mode: bool) -> str:
            prof = deep if deep_mode else fast
            system = ("You are Eris, synthesizing your understanding of a whole "
                      "document or codebase from section summaries. Write a coherent, "
                      "grounded high-level understanding in prose: what it is, how the "
                      "parts fit, what stands out, what could be improved."
                      if deep_mode else
                      "Summarize this one section faithfully and concisely (2-4 "
                      "sentences). Capture its real content/structure; no preamble.")
            prompt = (("Synthesize these section summaries into one understanding:\n\n"
                       if deep_mode else "Summarize this section:\n\n") + text)
            resp = run_blocking(orchestrator.mediator.generate(
                prompt=prompt, system=system,
                max_tokens=prof.max_tokens, temperature=prof.temperature))
            return (getattr(resp, "text", "") or "").strip() if resp else ""

        async with governor.foreground():
            result = await asyncio.to_thread(
                deep_read, orchestrator.memory, summarize, src, data_dir=data_dir)
        return result

    @app.get("/api/accelerators")
    async def api_accelerators():
        """Which optional accelerator services (embed/rerank/tts/stt) are live vs
        falling back to in-process. Probed off the loop so it never blocks."""
        from eris.interface.accelerators import accelerator_status
        status = await asyncio.to_thread(accelerator_status, True, 2.0)
        return {"accelerators": status}

    @app.post("/api/stt")
    async def api_stt(request: Request):
        """Transcribe audio via the configured STT service (Phase 4). The audio is
        the raw POST body (the cockpit mic sends the recorded blob), avoiding a
        multipart dependency. Returns {text}; 503 if no STT service is configured."""
        from eris.interface import stt
        if not stt.is_configured():
            return JSONResponse({"error": "no STT service configured"}, status_code=503)
        audio = await request.body()
        ctype = request.headers.get("content-type", "audio/wav")
        try:
            text = await asyncio.to_thread(
                stt.transcribe, audio, filename="audio.wav", content_type=ctype)
            return {"text": text}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=502)

    @app.get("/api/status")
    async def get_status():
        """Honest readiness (A3): probe the ACTUALLY-configured LLM backend(s) and
        report whether the embeddings are truly semantic — not a hardcoded Ollama
        URL and not 'a URL string is set'."""
        from eris.knowledge.embeddings import is_semantic
        backends = []
        try:
            # available_backends calls each backend's is_available() (may do a
            # short sync HTTP probe) — run off the event loop.
            avail = await asyncio.to_thread(lambda: orchestrator.mediator.available_backends)
            backends = [getattr(b, "name", b.__class__.__name__) for b in avail]
        except Exception:
            backends = []
        try:
            semantic = await asyncio.to_thread(is_semantic)
        except Exception:
            semantic = False
        return {"llm_ready": bool(backends), "backends": backends,
                "embeddings_semantic": semantic}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket for real-time metrics streaming."""
        await websocket.accept()
        # B6: cap concurrent metric sockets so they can't grow without bound.
        _ws_cap = int(os.environ.get("ERIS_WS_MAX", "32"))
        if len(ws_connections) >= _ws_cap:
            await websocket.close(code=1013)   # try again later
            return
        ws_connections.append(websocket)
        try:
            while True:
                # Compute vitals defensively — a transient error here must never
                # kill the stream (Fix B). The blocking field math is already
                # cheap; if get_vitals ever raises, skip this tick, don't drop.
                try:
                    vitals = orchestrator.get_vitals()
                except Exception as e:
                    vitals = {"error": f"vitals unavailable: {e}"}
                await websocket.send_json(vitals)
                await asyncio.sleep(2)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass  # client vanished mid-send — swallow, don't bubble
        finally:
            if websocket in ws_connections:
                ws_connections.remove(websocket)

    @app.websocket("/ws/field")
    async def websocket_field_endpoint(websocket: WebSocket, agent: str = "eris"):
        """Send-only field stream for a node (WILLOW I.10): /ws/field?agent=willow
        streams that node's mind. Defaults to eris (the existing cockpit viz)."""
        await websocket.accept()
        from eris.config import to_numpy
        node = registry.get(agent)
        field = node.field if node is not None else orchestrator.field
        try:
            while True:
                f = (node.field if node is not None else orchestrator.field)
                if f is not None:
                    phi = to_numpy(f.phi).tolist()
                    theta = to_numpy(f.theta).tolist()
                    await websocket.send_json({
                        "agent": (node.name if node is not None else "eris"),
                        "size": f.size,
                        "step_count": f.step_count,
                        "phi": phi,
                        "theta": theta,
                        "coherence": f.coherence,
                        "dCdX": f.dCdX,
                        "regime_str": f.detect_regime(),
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
            with open(index_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content=_MINIMAL_UI)

    @app.get("/visualizer", response_class=HTMLResponse)
    async def visualizer():
        """Stand-alone Living-Field visualizer (the cockpit 'pop out' target).
        Renders the live PDE field straight off the /ws/field stream."""
        path = os.path.join(os.path.dirname(__file__), "static", "visualizer.html")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content="<h1>visualizer.html missing</h1>", status_code=404)

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
        # Disable uvicorn's protocol-level keepalive ping. That ping path hits a
        # known assertion bug in the legacy `websockets` implementation
        # (websockets/legacy/protocol.py `_drain_helper`) and drops the cockpit
        # `/ws` connection with "keepalive ping failed". We don't need it: the
        # cockpit stream already sends vitals every ~2s, which keeps the socket
        # warm on its own. (ws_ping_interval=None turns the buggy ping off.)
        uvicorn.run("eris.server.app:app", host="0.0.0.0", port=8001, reload=True,
                    ws_ping_interval=None, ws_ping_timeout=None)
    else:
        print("FastAPI not installed. Run: pip install fastapi uvicorn")
