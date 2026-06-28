"""Turnkey Eris arm (Arm B) for the two-arm benchmark.

`make_eris_arm()` returns an answer_fn wired to the REAL pipeline via eris_pipeline_arm:
ingest the provided source into a SCRATCH memory, run the hive over just that document, return
(synthesis, REAL token cost). Use it directly:

    --arm both --eris-factory eris.experiments.benchmarks.eris_arm:make_eris_arm

Safety: a benchmark must never touch or steer the live store, so this arm
  • builds ErisOrchestrator(data_dir="eris_bench_data") — a scratch dir, NEVER eris_data, and
  • force-disables the autonomous side-effects (ERIS_ROUTE_GAPS=0, ERIS_SYNTHESIS_WRITEBACK=0)
    so a run can't write syntheses into, or queue study topics from, the real store.

Cost: the equal-budget comparison needs a REAL token count, so a _TokenMeter wraps the local
backend(s)' generate() and sums LLMResponse.tokens_used (Ollama/vLLM /v1 report usage). If a
backend reports no usage, it falls back to the hive's LLM call count, clearly labeled as a proxy
on stderr."""
import os
import sys
import threading
from typing import Callable, Tuple


class _TokenMeter:
    """Sum real tokens across every LLM call during one ask(): wraps each backend's async
    generate() to accumulate LLMResponse.tokens_used. Thread-safe (the hive reasons in parallel)."""

    def __init__(self):
        self.tokens = 0
        self.calls = 0
        self._lock = threading.Lock()
        self._patched = []

    def _wrap(self, backend) -> None:
        orig = getattr(backend, "generate", None)
        if orig is None or getattr(orig, "_metered", False):
            return
        meter = self

        async def metered(*a, _orig=orig, **kw):
            resp = await _orig(*a, **kw)
            with meter._lock:
                meter.calls += 1
                meter.tokens += int(getattr(resp, "tokens_used", 0) or 0)
            return resp
        metered._metered = True
        backend.generate = metered
        self._patched.append((backend, orig))

    def attach(self, *mediators) -> "_TokenMeter":
        for med in mediators:
            for b in (getattr(med, "_backends", None) or []):
                self._wrap(b)
        return self

    def attach_gateway(self, gateway) -> "_TokenMeter":
        if gateway is None:
            return self
        for tier in ("free", "cheap", "synth"):
            try:
                b = gateway.tier(tier)
            except Exception:
                b = None
            if b is not None:
                self._wrap(b)
        return self

    def reset(self) -> None:
        with self._lock:
            self.tokens = 0
            self.calls = 0


def _wipe_memory(orch) -> None:
    """Clear the scratch memory so item N+1 can never retrieve item N's passage. Wipes MTM/LTM
    (where ingest_text writes) + STM + the thought-stream, in place, and invalidates the LTM
    vector index. The on-disk scratch files are throwaway (eris_bench_data)."""
    m = orch.memory
    for tier in (m.mtm, m.ltm):
        try:
            tier._records = []
            tier._save()
        except Exception:
            pass
    try:                                   # invalidate the LTM FAISS cache so it rebuilds empty
        m.ltm._faiss = None
        m.ltm._faiss_n = -1
    except Exception:
        pass
    try:
        m.stm._buffer.clear()
    except Exception:
        pass
    ts = getattr(orch, "thought_stream", None)
    if ts is not None:
        try:
            ts._thoughts = []
        except Exception:
            pass
    # The cross-attention hub accumulates specialist findings ACROSS calls and feeds synthesis, so
    # without clearing it item N's findings could cross-pollinate item N+1's answer.
    hub = getattr(orch, "hub", None)
    if hub is not None:
        try:
            hub.clear()
        except Exception:
            pass
    # Fail LOUDLY rather than silently contaminate: the wipe reaches into private tier internals,
    # so if a future memory refactor changes them this guard stops the run instead of letting item
    # N+1 retrieve item N's passage and quietly invalidate every downstream number. The hive's _rag
    # reads the thought-stream too, but all_records() doesn't cover it — so count it here.
    ts_left = 0
    if ts is not None:
        try:
            ts_left = len(list(ts.all()))
        except Exception:
            ts_left = 0
    remaining = len(orch.memory.all_records()) + ts_left
    if remaining:
        raise RuntimeError(
            f"benchmark reset did NOT clear the scratch store ({remaining} record(s) remain) — "
            "refusing to continue (cross-item contamination would invalidate the results). A memory "
            "refactor may have changed the tier internals this wipe reaches into.")


def make_eris_arm(data_dir: str = "eris_bench_data",
                  field_size: int = 32) -> Callable[[str], Tuple[str, int]]:
    """Return the Eris answer_fn (prompt -> (text, tokens)), ready for --eris-factory."""
    if os.path.realpath(data_dir) == os.path.realpath("eris_data"):   # realpath: resolve symlinks
        raise ValueError("refusing to benchmark against the live eris_data — use a scratch dir")

    # Force-disable autonomous side-effects BEFORE the orchestrator is built.
    os.environ["ERIS_ROUTE_GAPS"] = "0"
    os.environ["ERIS_SYNTHESIS_WRITEBACK"] = "0"

    import asyncio
    from eris.orchestrator import ErisOrchestrator
    from eris.knowledge import web_reader
    from eris.knowledge.extractor import KnowledgeExtractor
    from eris.experiments.benchmarks.arms import eris_pipeline_arm

    orch = ErisOrchestrator(data_dir=data_dir, field_size=field_size)
    extractor = KnowledgeExtractor(output_dir=os.path.join(data_dir, "knowledge_base"))
    meter = _TokenMeter()
    # mediator (specialist reasoning + local synth), deep_mediator (the synth/expert mediator the
    # hive can route synthesis through), any per-profile mediators, and the gateway tiers.
    meter.attach(orch.mediator, getattr(orch, "deep_mediator", None),
                 *(getattr(orch, "_profile_mediators", {}) or {}).values())
    meter.attach_gateway(getattr(orch, "gateway", None))
    warned = {"proxy": False}

    def reset() -> None:
        _wipe_memory(orch)

    def ingest(ctx: str) -> None:
        if ctx and ctx.strip():
            web_reader.ingest_text(ctx, title="bench", extractor=extractor, memory=orch.memory)

    def ask(q: str):
        meter.reset()
        res = asyncio.run(orch.hive_research(q, scope="doc", document="bench"))
        text = (res.get("synthesis_full") or res.get("synthesis") or "").strip()
        tokens = meter.tokens
        if tokens <= 0:                    # backend reported no usage → labeled call-count proxy
            tc = res.get("tier_calls", {}) or {}
            tokens = sum(v for k, v in tc.items() if not str(k).startswith("_")) or meter.calls
            if not warned["proxy"]:
                warned["proxy"] = True
                print("[eris-arm] WARNING: backend reported no token usage; cost is a CALL-COUNT "
                      "PROXY, not tokens — the equal-budget comparison is approximate. Serve the "
                      "model via an OpenAI-compatible /v1 endpoint (Ollama/vLLM) to get real tokens.",
                      file=sys.stderr)
        return text, int(tokens)

    return eris_pipeline_arm(ingest, ask, reset)
