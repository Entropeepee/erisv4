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
backend(s)' generate() and sums LLMResponse.tokens_used (Ollama/vLLM /v1 report usage; Ollama-native
eval counts read from .raw). If a backend reports no usage, it falls back to the hive's LLM call
count, clearly labeled as a proxy on stderr. NOTE: the meter counts LLM GENERATION tokens (what
both arms spend on the model). The local CPU embedding model (bge-m3) used for retrieval is a
separate, non-LLM cost and is NOT in this number — the comparison is of model-inference tokens."""
import os
import sys
import threading
from typing import Callable, Tuple


def _resp_tokens(resp) -> int:
    """Real token count from an LLMResponse, however the backend reports it: tokens_used (OpenAI/
    /v1 'usage'), else Ollama-native prompt_eval_count + eval_count from .raw, else a nested
    usage.total_tokens. Returns 0 only if nothing is reported."""
    t = int(getattr(resp, "tokens_used", 0) or 0)
    if t:
        return t
    raw = getattr(resp, "raw", None)
    if isinstance(raw, dict):
        n = int(raw.get("prompt_eval_count", 0) or 0) + int(raw.get("eval_count", 0) or 0)
        if n:
            return n
        return int((raw.get("usage") or {}).get("total_tokens", 0) or 0)
    return 0


class _TokenMeter:
    """Sum real tokens across every LLM call during one ask(): wraps each backend's async
    generate() to accumulate its token count (tokens_used OR Ollama-native eval counts).
    Thread-safe (the hive reasons in parallel)."""

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
                meter.tokens += _resp_tokens(resp)
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


def _task_remainder(prompt: str) -> str:
    """Everything after the SOURCE block — the question + any options + the answer-format line."""
    if "=== END SOURCE ===" in prompt:
        return prompt.split("=== END SOURCE ===", 1)[1].strip()
    return prompt.strip()


def _question_line(remainder: str) -> str:
    """The bare question (for the hive topic), without the options/format instructions."""
    for line in remainder.splitlines():
        if line.strip().lower().startswith("question:"):
            return line.split(":", 1)[1].strip()
    return remainder.strip()


def make_eris_arm(data_dir: str = "eris_bench_data",
                  field_size: int = 32) -> Callable[[str], Tuple[str, int]]:
    """Return the Eris answer_fn (prompt -> (text, tokens)), ready for --eris-factory."""
    if os.path.realpath(data_dir) == os.path.realpath("eris_data"):   # realpath: resolve symlinks
        raise ValueError("refusing to benchmark against the live eris_data — use a scratch dir")

    # Force-disable autonomous side-effects BEFORE the orchestrator is built.
    os.environ["ERIS_ROUTE_GAPS"] = "0"
    os.environ["ERIS_SYNTHESIS_WRITEBACK"] = "0"
    # One key covers everything: if the gateway key isn't set explicitly, reuse the bench key (both
    # point at the same OpenRouter account) — so you only paste ONE OpenRouter key into .env.
    if not os.environ.get("ERIS_GATEWAY_API_KEY") and os.environ.get("ERIS_BENCH_API_KEY"):
        os.environ["ERIS_GATEWAY_API_KEY"] = os.environ["ERIS_BENCH_API_KEY"]
    # READ THE WHOLE PASSAGE, don't RAG 6 fragments of it. A grounded benchmark provides the full
    # short document (a ~5k-word QuALITY story is ~17 chunks) and asks a whole-narrative question —
    # so feed the hive (nearly) all of it, not the focused-6 default tuned for live turns. setdefault
    # so an explicit override still wins.
    os.environ.setdefault("ERIS_HIVE_MAX_SOURCES", "50")

    import asyncio
    from eris.orchestrator import ErisOrchestrator
    from eris.knowledge import web_reader
    from eris.knowledge.extractor import KnowledgeExtractor
    from eris.experiments.benchmarks.arms import _split_source
    from eris.interface.mediator import run_blocking

    orch = ErisOrchestrator(data_dir=data_dir, field_size=field_size)
    extractor = KnowledgeExtractor(output_dir=os.path.join(data_dir, "knowledge_base"))
    meter = _TokenMeter()
    # mediator (specialist reasoning + local synth), deep_mediator (the synth/expert mediator the
    # hive can route synthesis through), any per-profile mediators, and the gateway tiers.
    meter.attach(orch.mediator, getattr(orch, "deep_mediator", None),
                 *(getattr(orch, "_profile_mediators", {}) or {}).values())
    meter.attach_gateway(getattr(orch, "gateway", None))
    warned = {"proxy": False}

    # Surface WHERE the hive's ~18-21 calls/question go, so it's never a mystery why a run is fast
    # or slow. OPEN-sensitivity hive calls route to the gateway (e.g. OpenRouter) when
    # ERIS_GATEWAY_BASE_URL is set, else to the local Ollama model. Sovereign work is never affected.
    gw = getattr(orch, "gateway", None)
    if gw is not None and getattr(gw, "enabled", False):
        # VISIBLE PROOF of attributability: the EFFECTIVE model each hive component uses (not the
        # raw env var). With synth_cloud off, synthesis REUSES the specialist/free model
        # (research.py: `synth = synth_model or model`), so we must show the free model for synth —
        # otherwise the header lies. If specialists, synth, and the bare arm aren't the SAME model,
        # the run is NOT attributable and we say so loudly.
        _free = os.environ.get("ERIS_TIER_FREE", "free-pool")
        _synth_cloud = os.environ.get("ERIS_HIVE_SYNTH_CLOUD", "0").strip().lower() in (
            "1", "on", "true", "yes")
        _synth = os.environ.get("ERIS_TIER_SYNTH", "synth") if _synth_cloud else _free
        _bare = os.environ.get("ERIS_BENCH_MODEL", "?")
        _one = (_free == _synth == _bare)
        _verdict = ("✓ ATTRIBUTABLE — one model everywhere" if _one else
                    "✗ MISMATCH — NOT attributable (different models in different slots)")
        print(f"[eris-arm] hive → GATEWAY {os.environ.get('ERIS_GATEWAY_BASE_URL', '?')} | "
              f"specialists+gaps={_free} | synth={_synth} | bare={_bare} | {_verdict}",
              file=sys.stderr)
    else:
        print(f"[eris-arm] hive model → LOCAL Ollama ({os.environ.get('ERIS_LOCAL_MODEL', 'gpt-oss:20b')})"
              " — set ERIS_GATEWAY_BASE_URL/KEY + ERIS_TIER_* to route the hive to OpenRouter for speed.",
              file=sys.stderr)

    def reset() -> None:
        _wipe_memory(orch)

    def ingest(ctx: str) -> None:
        if ctx and ctx.strip():
            n = web_reader.ingest_text(ctx, title="bench", extractor=extractor, memory=orch.memory)
            # Loud guard: a non-empty passage that stores 0 chunks means the source never reached
            # the hive's doc store. FAIL rather than let the hive silently decline and score 0 —
            # an ingest bug must not masquerade as "the architecture lost".
            if n == 0 or len(orch.memory.all_records()) == 0:
                raise RuntimeError(
                    f"benchmark ingest stored 0 chunks for a {len(ctx)}-char context — the passage "
                    "is not reaching the scope='doc' store (title='bench'); aborting instead of "
                    "scoring an empty answer as a loss.")

    def _extract_answer(remainder: str, synthesis: str) -> str:
        """One concise grounded-answer call so Eris emits a SCORABLE answer in the SAME form the
        bare arm uses (a name/number/date, or a single option letter) — fair formatting of the
        hive's synthesis, NOT a second research pass. Metered like every other call. The FULL
        synthesis is passed (no truncation — it's a few thousand chars and the extractor's context
        is large; dropping its tail could drop the answer). Returns '' on failure (the caller flags
        it so a broken extraction never masquerades as a wrong answer)."""
        p = ("Using ONLY the analysis below, give the final answer to the task. Follow the task's "
             "answer format EXACTLY — if it asks for a single letter, output ONLY that letter; "
             "otherwise answer as briefly as possible (a name, number, or date) with no "
             f"explanation.\n\nANALYSIS:\n{synthesis}\n\nTASK:\n{remainder}\n\nFinal answer:")
        try:
            resp = run_blocking(orch.mediator.generate(prompt=p, system=""), timeout=120)
            return (getattr(resp, "text", "") or "").strip()
        except Exception as e:
            print(f"[eris-arm] extraction call FAILED ({type(e).__name__}: {e}) — flagging item",
                  file=sys.stderr)
            return ""

    def _answer(prompt: str):
        context, _ = _split_source(prompt)
        remainder = _task_remainder(prompt)
        question = _question_line(remainder)
        reset()                                  # reset BEFORE ingest (never erase the passage)
        if context and context.strip():
            ingest(context)
        meter.reset()                            # count the hive + the extraction call
        res = asyncio.run(orch.hive_research(question, scope="doc", document="bench"))
        if not isinstance(res, dict):
            res = {}
        hive_error = res.get("error")
        if hive_error:                           # a hive exception, not a reasoning loss — keep it
            print(f"[eris-arm] hive_research returned an error: {hive_error}", file=sys.stderr)
        synthesis = (res.get("synthesis_full") or res.get("synthesis") or "").strip()
        # Cost the hive ALONE before the extraction pass: extraction is an Eris-only extra call the
        # bare arm never makes, so split it out (visibly) instead of folding the overhead invisibly
        # into the equal-budget number. Snapshot the meter between the two stages.
        hive_tokens, hive_calls = meter.tokens, meter.calls
        # Extract a short, scorable answer from the synthesis (the bare arm answers directly; this
        # gives Eris the same answer FORM so exact-match/MC don't penalize the hive for verbosity).
        # If extraction yields nothing we FALL BACK to the synthesis but FLAG it (extraction_ok=
        # False), so a broken/empty extraction can't silently read as "the hive answered wrong".
        extracted = _extract_answer(remainder, synthesis) if synthesis else ""
        extraction_ok = bool(extracted)
        extraction_tokens = meter.tokens - hive_tokens
        extraction_calls = meter.calls - hive_calls
        answer = extracted or synthesis
        tokens = meter.tokens
        token_proxy = False
        if tokens <= 0:
            # Distinguish "no LLM calls happened" (the hive declined with 0 sources — a legitimate
            # 0 cost) from "calls happened but no usage reported" (the real proxy case worth warning).
            tc = res.get("tier_calls", {}) or {}
            calls = sum(v for k, v in tc.items() if not str(k).startswith("_")) or meter.calls
            tokens = calls
            token_proxy = meter.calls > 0        # a call-count substituted for real tokens
            if meter.calls > 0 and not warned["proxy"]:
                warned["proxy"] = True
                print("[eris-arm] WARNING: the hive made calls but the backend reported no token "
                      "usage; cost is a CALL-COUNT PROXY, not tokens — equal-budget is approximate. "
                      "Serve via an OpenAI-compatible /v1 endpoint (Ollama/vLLM) for real tokens.",
                      file=sys.stderr)
        # Return the FULL diagnostics, not just the scored answer. Beyond the synthesis + extraction
        # flag, surface every signal the hive COMPUTED and used to (not) answer — so a right/wrong
        # result is attributable to the architecture, never a black box. These are exactly the
        # fingerprints of the physics + multi-agent machinery the benchmark exists to judge:
        # specialist_divergence (did the lenses actually disagree?), the confidence GEOMETRY
        # (cos-match / sin-unresolved / coherence / torsion), the gaps it could/couldn't close, the
        # adversarial Elos critique, and citation-stripping. All already on `res`; previously all but
        # four were dropped right here. (Per the 39-finding blackbox audit, 2026-06.)
        tier_calls = res.get("tier_calls") or {}
        detail = {
            "synthesis": synthesis,
            "synthesis_pre_ground": (res.get("synthesis_pre_ground_full")
                                     or res.get("synthesis_pre_ground") or ""),
            "extraction_ok": extraction_ok,
            "n_sources": res.get("n_sources", 0),
            "hive_calls": meter.calls,
            "hive_tokens": hive_tokens,
            "extraction_tokens": extraction_tokens,
            "extraction_calls": extraction_calls,
            "tokens_are_proxy": token_proxy,
            # — multi-agent machinery: did the hive engage, and did its second cycle do real work? —
            "cycles": res.get("cycles"),
            "n_active": res.get("n_active"),
            "n_contributors": res.get("n_contributors"),
            "specialist_divergence": res.get("specialist_divergence"),
            "gaps": res.get("gaps"),
            "open_gaps": res.get("open_gaps"),
            "gaps_closed": res.get("gaps_closed"),
            "elos_changed": res.get("elos_changed"),
            "elos_critique": res.get("elos_critique"),
            "stripped_claims": res.get("stripped_claims"),
            "canonized": res.get("canonized"),
            # — the physics readout: cos match, sin/unresolved, coherence, torsion (τ) — the hive's
            #   OWN geometric confidence in the exact answer being scored. None when no sources. —
            "confidence": res.get("confidence"),
            # — cost attribution: WHICH tier/model served the calls + how many were paid —
            "tier_calls": tier_calls,
            "paid_calls": tier_calls.get("_paid_calls", 0),
            # — which source previews the hive actually grounded on (retrieval-miss vs reasoning-miss) —
            "sources": res.get("sources"),
        }
        if hive_error:
            detail["hive_error"] = hive_error
            detail["hive_traceback"] = res.get("traceback")
        return {"text": answer, "tokens": int(tokens), "detail": detail}

    return _answer
