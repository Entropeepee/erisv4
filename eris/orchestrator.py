"""
Eris Echo v4 — Cognitive Orchestrator
=======================================

The main loop that wires all six architectural layers together.
This is the brain stem — it doesn't think, it routes.

Conversation Flow (from handoff session):
    User message arrives
        → [Layer 1] Seed FRACTAL field (FRT instant + PDE background)
        → [Layer 0] Compute BFECDS from field state
        → [Layer 0] Compute global observables (C, X, dC/dX)
        → [Layer 0] Detect regime (elastic / plastic / transfixed)
        → [Layer 2] Retrieve memory context (all 3 tiers)
        → [Layer 5] GPW sets goal from user intent + BFECDS
        → [Layer 3] Activated specialists generate findings
        → [Layer 5] MoEGate scores bids via wave interference
        → [Layer 5] Winner broadcast to SharedCognitiveWorkspace
        → [Layer 6] LLM assembles prompt from winner + context + field
        → [Layer 6] LLM generates response (Broca's area — language, not reasoning)
        → [Layer 0] Compute response BFECDS
        → [Layer 4] Detect dissonance (SGT-gated)
        → [Layer 1] If significant: BLC → field seeds → inject → evolve
        → [Layer 2] Store turn in memory + autobiography
        → Return response + metrics

The orchestrator also manages:
    - Dual-path text injection (FRT reflexive + PDE deliberative)
    - Background dreaming loop scheduling
    - Memory consolidation triggers
    - Research organ activation (C > 0.4 AND E > 0.2)

Usage:
    from eris.orchestrator import ErisOrchestrator

    eris = ErisOrchestrator()
    eris.add_llm_backend(OllamaBackend(model="llama3.2"))
    response = await eris.process("Hello, tell me about emergence")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import asyncio
import time
import os
import re
import numpy as np

from eris.config import CONFIG, to_numpy

# Layer 0: Computation
from eris.computation.activations import BVec, bvec_distance
from eris.computation.sgt import SGTGate
from eris.computation.orch_counters import OrchestrationCounters
from eris.computation.noise_floor import NoiseFloorEstimator
from eris.computation.criticality import CriticalityMonitor, Decision, FailureModeReport

# Layer 1: Field
from eris.field.pde import FractalField
from eris.field.compiler import compile_contradiction, inject_seeds

# Layer 2: Memory
from eris.memory.tiers import MemorySystem, MemoryRecord
from eris.memory.autobiography import Autobiography
from eris.knowledge.embeddings import get_embedding
from eris.metacognition.dream_journal import DreamJournal
from eris.memory.interference import find_conflicts

# Layer 3: Tribe
from eris.tribe.specialists import (
    TRIBE, get_active_specialists, CrossAttentionHub,
    SpecialistFinding, should_trigger_research, make_field_finding,
)

# Layer 5: Executive
from eris.executive.workspace import (
    MoEGate, SharedCognitiveWorkspace, GoalNetwork,
)

# Layer 6: Interface
from eris.interface.mediator import LLMMediator, LLMBackend, _DEFAULT_MAX_TOKENS

# Layer 4: Metacognition
from eris.metacognition.dreaming import DreamingLoop
from eris.memory.thought_stream import ThoughtStream


def _mem_when(ts: float) -> str:
    """Human-readable timestamp for a memory, so Eris has a sense of time and can
    see how her thinking on a subject has evolved across dates."""
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
    except Exception:
        return "undated"


# Sources that came from the LIBRARY (things she read/studied) — citable in chat.
_LIBRARY_PREFIXES = ("reading:", "study:", "exploration:", "ponder:", "deepread",
                     "research", "expert:")


def _collect_citations(records, limit: int = 6) -> List[Dict[str, str]]:
    """De-duplicated citations for the library-origin records that ground a turn.
    Each is {title, source}; the UI surfaces them so studied material is traceable.
    Her own reflections / conversation memory are NOT cited as external sources."""
    out: List[Dict[str, str]] = []
    seen = set()
    for r in records or []:
        src = (getattr(r, "source", "") or "")
        if not any(src.startswith(p) for p in _LIBRARY_PREFIXES):
            continue
        meta = getattr(r, "metadata", {}) or {}
        title = (meta.get("title") or "").strip() or src
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        cite = {"title": title, "source": src}
        url = meta.get("url") or (src.split(":", 1)[1] if src.startswith("exploration:") else "")
        if url:
            cite["url"] = url
        out.append(cite)
        if len(out) >= limit:
            break
    return out


@dataclass
class ProcessingResult:
    """Full result of processing one user message."""
    response_text: str = ""
    reasoning_text: str = ""
    input_bvec: Optional[BVec] = None
    response_bvec: Optional[BVec] = None
    coherence: float = 0.0
    exchange: float = 0.0
    dCdX: float = 0.0
    regime: str = "unknown"
    archetype: str = ""
    dissonance: float = 0.0
    specialist_source: str = ""
    llm_provider: str = ""
    latency_ms: float = 0.0
    contradiction_compiled: bool = False
    research_triggered: bool = False
    citations: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ErisOrchestrator:
    """The main cognitive loop. Wires all layers together.

    Initialize once, call process() for each user message.
    Manages the persistent field, memory, and metacognitive state.
    """

    def __init__(self,
                 field_size: int = 64,
                 data_dir: str = "eris_data",
                 use_frt_seeding: bool = False,
                 field_seed: int = 42):
        """
        Parameters
        ----------
        field_size : int
            PDE grid resolution. 32 for fast, 64 default, 128 for depth.
        data_dir : str
            Directory for persistent storage (memory, autobiography).
        use_frt_seeding : bool
            If True, use FRT (instant, CPU) for text seeding instead of
            the character-statistics method. Set True for low-power hardware.
        field_seed : int
            RNG seed for the persistent field. Default 42 (unchanged). The
            orchestration benchmark varies this to report mean ± std over seeds.
        """
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        self.field_size = field_size
        self.field_seed = field_seed
        self.use_frt_seeding = use_frt_seeding
        self._durable_memory = None  # lazily created by the agent's memory tools
        self._profile_mediators: Dict[tuple, LLMMediator] = {}  # (model,url)->mediator

        # Layer 1: Persistent field state
        self.field = FractalField(size=field_size, seed=field_seed)

        # Layer 2: Memory
        self.memory = MemorySystem(data_dir=os.path.join(data_dir, "memory"))
        self.autobiography = Autobiography(
            path=os.path.join(data_dir, "autobiography.jsonl")
        )

        # Layer 3: Tribe
        self.hub = CrossAttentionHub(capacity=50)

        # Layer 5: Executive
        self.moe_gate = MoEGate()
        self.workspace = SharedCognitiveWorkspace()
        self.goal_network = GoalNetwork()

        # Layer 6: LLM mediator (Broca's area)
        self.mediator = LLMMediator()
        self.deep_mediator = LLMMediator()

        # Layer 4: Metacognition
        # Tier 7: dream journal (readable record for the cockpit dream panel)
        self.dream_journal = DreamJournal(
            path=os.path.join(data_dir, "dream_journal.jsonl"))
        # Thought-stream: her OWN thinking, kept separate from the library by
        # provenance and NEVER quality-gated (the introspection store that fixes
        # the `kept (0)` bug). Persisted alongside her other data.
        self.thought_stream = ThoughtStream(
            path=os.path.join(data_dir, "thoughts.jsonl"))
        self.dreaming_loop = DreamingLoop(
            autobiography=self.autobiography,
            memory=self.memory,
            field_size=field_size,
            journal=self.dream_journal,
            mediator=self.mediator,
            thought_stream=self.thought_stream,
        )

        # SGT gate for dissonance detection
        self._dissonance_gate = SGTGate(threshold_sigma=2.0, ema_alpha=0.1)

        # ── LLM Router configuration (Remediation Tier 0) ─────────────
        # Reasoning happens UPSTREAM in the field; the LLM only verbalizes the
        # thought the GPW selected. So the DEFAULT path is one fast LOCAL call.
        #
        # The previous build fired a 4-backend "deep" ensemble whenever
        # `coherence < 0.2`. But this engine's global coherence sits
        # structurally near ~0.04, so that branch fired on EVERY turn — paying
        # for dead/keyless cloud-backend timeouts plus a synthesis pass on CPU.
        # That was the root cause of the slowness. Two fixes:
        #   1. Register a cloud backend ONLY when its API key is actually set.
        #      A keyless backend is useless and, if mis-added, can hang a turn.
        #   2. Gate the deep path on an SGT z-score *outlier* of |dC/dX|
        #      (scale-adaptive — works whether C is 0.04 or 0.8), and only when
        #      at least one real cloud expert is wired. Never on every turn.
        from eris.interface.mediator import (
            OllamaBackend, AnthropicBackend, OpenAIBackend, GeminiBackend,
        )
        self._local_model = os.environ.get("ERIS_LOCAL_MODEL", "gpt-oss:20b")

        # Serving route (roadmap 1.1): the local "Broca's area" backend is either
        # Ollama (default, dev) or any OpenAI-compatible server — set
        # ERIS_LLM_BASE_URL to point at a vLLM/TensorRT-LLM endpoint (NVFP4 etc.)
        # without touching code. vLLM needs no real key; a dummy satisfies the
        # availability check.
        self._llm_base_url = os.environ.get("ERIS_LLM_BASE_URL", "").strip()

        def _make_local_backend():
            if self._llm_base_url:
                return OpenAIBackend(
                    model=self._local_model,
                    api_key=os.environ.get("ERIS_LLM_API_KEY", "local"),
                    base_url=self._llm_base_url)
            return OllamaBackend(model=self._local_model)

        if self._llm_base_url:
            print(f"[Eris LLM] primary backend -> OpenAI-compatible server at "
                  f"{self._llm_base_url} (model {self._local_model})")

        # Fast path: one resident local model (Broca's area).
        self.mediator.add_backend(_make_local_backend())

        # Deep path: cloud experts ONLY if keyed, plus a local fallback so the
        # ensemble is never empty. Dormant (keyless) cloud backends are skipped.
        for _backend_cls, _env in ((AnthropicBackend, "ANTHROPIC_API_KEY"),
                                   (OpenAIBackend, "OPENAI_API_KEY"),
                                   (GeminiBackend, "GEMINI_API_KEY")):
            _key = os.environ.get(_env, "")
            if _key:
                self.deep_mediator.add_backend(_backend_cls(api_key=_key))
        self.deep_mediator.add_backend(_make_local_backend())

        # Number of genuine (cloud) experts wired for deep synthesis.
        self._cloud_experts = len(self.deep_mediator._backends) - 1

        # Scale-adaptive gate for the deep path: opens only when |dC/dX| is a
        # statistical outlier relative to this engine's own running history.
        self._router_gate = SGTGate(threshold_sigma=2.0, ema_alpha=0.1)

        # Turn counter + last-turn dissonance (surfaced separately in vitals).
        self.turn_count: int = 0
        self._last_dissonance: float = 0.0
        # Tier 4/5: the most recent router FailureModeReport (SWITCH/ESCALATE),
        # routed to the dream queue by Tier 5 when gate_failure_reports is on.
        self._last_router_report: Optional[FailureModeReport] = None

        # Tier 0: per-turn resource counters for the orchestration benchmark.
        # Pure instrumentation — incremented at the expensive boundaries below,
        # read by bench_orchestration.py. Does not affect any decision.
        self.counters = OrchestrationCounters()
        # DualPath shadow comparison: cumulative win/loss tally + a lazily-built
        # retrieval DualPath (only constructed when a non-default mode is active).
        from eris.computation.orch_counters import DualCounters
        self.dual_counters = DualCounters()
        self._retrieval_dual = None

        # Tier 1+: ONE shared noise-floor estimator feeds every gate (per-signal
        # local scale + one global agitation multiplier). The gates that draw
        # from it stay inert unless their CONFIG flag is on (default OFF).
        self._noise_floor = NoiseFloorEstimator(
            k=CONFIG.orch_k, warmup=10)
        # Tier 2: the field-evolution depth gate (early-terminate a settled run).
        self._field_depth_monitor = CriticalityMonitor(
            "field_depth", self._noise_floor, "field_depth",
            k=CONFIG.orch_k, protected_steps=0)
        # Tier 3: the response-field warm-start. A persistent field reused across
        # turns (warm φ/θ prior) + a settle monitor on the response bvec.
        self._response_field: Optional[FractalField] = None
        self._resp_monitor = CriticalityMonitor(
            "response_field", self._noise_floor, "response_field",
            k=CONFIG.orch_k, protected_steps=0)
        # Tier 4: the formalized local↔cloud router (built here, wired below).
        self._router_monitor = CriticalityMonitor(
            "router", self._noise_floor, "router",
            k=CONFIG.orch_k, protected_steps=0)

    def add_llm_backend(self, backend: LLMBackend) -> None:
        """Add an LLM backend (Ollama, Claude, OpenAI, etc.)."""
        self.mediator.add_backend(backend)

    def _ambient_profile(self):
        """The profile that reproduces current global behavior — used when a turn
        passes no explicit profile, so existing callers/tests are unchanged."""
        from eris.interface.profiles import Profile
        from eris.interface.mediator import _DEFAULT_MAX_TOKENS
        return Profile(
            id="ambient", label="ambient", model="", base_url="",
            max_tokens=_DEFAULT_MAX_TOKENS, temperature=0.7,
            ttc=CONFIG.ttc_self_consistency, ttc_max_samples=CONFIG.ttc_max_samples,
            orchestration=(CONFIG.orchestration_enabled and CONFIG.gate_router),
            field_steps=CONFIG.pde_steps_per_input)

    def _resolve_mediator(self, prof):
        """Return the LLMMediator to use for this turn's LOCAL generation. Default
        profiles (empty/same model) reuse the resident mediator — no reload. A
        profile naming a different model/base_url gets a cached mediator pointed at
        it (Ollama loads it on demand on first use)."""
        model = prof.model or self._local_model
        base_url = prof.base_url or self._llm_base_url
        if model == self._local_model and base_url == self._llm_base_url:
            return self.mediator
        key = (model, base_url)
        m = self._profile_mediators.get(key)
        if m is None:
            from eris.interface.mediator import LLMMediator, OllamaBackend, OpenAIBackend
            m = LLMMediator()
            if base_url:
                m.add_backend(OpenAIBackend(
                    model=model, api_key=os.environ.get("ERIS_LLM_API_KEY", "local"),
                    base_url=base_url))
            else:
                m.add_backend(OllamaBackend(model=model))
            self._profile_mediators[key] = m
        return m

    async def process(self, user_message: str,
                      system_context: str = "",
                      profile=None) -> ProcessingResult:
        """Process one user message through the full cognitive pipeline.

        `profile` (an eris.interface.profiles.Profile) selects the per-turn
        latency knobs — token budget, test-time compute, reasoning depth,
        orchestration, field depth, and the model. None => ambient (current
        global behavior), so existing callers are unaffected.
        """
        t0 = time.time()
        result = ProcessingResult()
        self.counters.reset()  # Tier 0: counters reflect exactly this turn.
        prof = profile if profile is not None else self._ambient_profile()
        steps = prof.field_steps or CONFIG.pde_steps_per_input
        result.metadata["profile"] = prof.id

        # ── Layer 1: Seed and evolve the FRACTAL field ────────────────
        self.field.seed_from_text(user_message, use_frt=self.use_frt_seeding)
        if CONFIG.orchestration_enabled and CONFIG.gate_field_depth:
            # Tier 2: evolve only as deep as needed — suspend once settled.
            executed = self.field.run_gated(
                self._field_depth_monitor, steps,
                min_steps=CONFIG.orch_min_field_steps)
            self.counters.pde_steps += executed
        else:
            self.field.run(steps)
            self.counters.pde_steps += steps

        # ── Layer 0: Compute BFECDS + global observables ──────────────
        input_bvec = self.field.compute_bvec()
        result.input_bvec = input_bvec
        result.coherence = self.field.coherence
        result.exchange = self.field.exchange
        result.dCdX = self.field.dCdX
        result.regime = self.field.detect_regime()

        # ── Layer 2: Retrieve memory context ──────────────────────────
        # Tier 4.4: real (semantic-or-deterministic) embedding so the LTM
        # embedding retriever is actually exercised, not just BFECDS alignment.
        q_embedding = get_embedding(user_message)
        # Tier 6: resonant retrieval — cosine (aligned/answer) AND sine
        # (tension/learning). The tension set is coupled-but-unresolved memory:
        # giving it to the LLM as "related but unresolved" is how Eris connects
        # ideas instead of echoing the nearest neighbor.
        # DualPath shadow (default OFF): with ERIS_RETRIEVAL_MODE=traditional_only
        # this branch is skipped entirely and the resonant call below is byte-for-
        # byte the original path. In shadow/novel modes the wrapper runs both paths,
        # logs the arbiter-scored divergence, and returns the mode's authoritative set.
        if getattr(CONFIG, "retrieval_mode", "traditional_only") != "traditional_only":
            aligned, tension = self._dual_retrieve(user_message, input_bvec, q_embedding)
        else:
            aligned, tension = self.memory.retrieve_resonant(
                query_bvec=input_bvec, query_embedding=q_embedding,
                top_k=8, tension_k=3, query_text=user_message,
            )
        # Give the LLM the FULL retrieved memory, not a sliver — truncating here
        # is what made Eris say she hadn't read an uploaded document she had.
        # Each line is tagged with WHEN it was formed, so she has a sense of time
        # and can notice how her understanding of a subject evolved.
        memory_text = "\n\n".join(
            f"[{_mem_when(r.timestamp)} · {r.source}] {r.text}" for r in aligned
        ) if aligned else ""
        if tension:
            memory_text += (
                "\n\n[Related but unresolved — look for the hidden connection]\n"
                + "\n\n".join(f"[{_mem_when(r.timestamp)} · {r.source}] {r.text}"
                              for r in tension)
            )

        # Named-document direct retrieval: if the user names a file/document,
        # pull ITS chunks straight in so they can't be crowded out by chatter
        # that merely mentions it. This is what makes "tell me about my patent
        # sgtpatent" actually read the sgtpatent document.
        named_docs = self.memory.documents_matching(
            user_message, max_chunks=6, query_embedding=q_embedding)
        if named_docs:
            doc_block = "\n\n".join(
                f"[{_mem_when(r.timestamp)} · {r.source}] {r.text}" for r in named_docs)
            memory_text = (
                "[A document you have ALREADY READ that the user is asking about "
                "— answer from this text; do NOT say you can't access it]\n"
                + doc_block
                + (("\n\n" + memory_text) if memory_text else "")
            )

        # §9a: surface CITATIONS for the library sources that ground this turn, so
        # when she discusses a studied subject the user can trace it. Pure data on
        # the result (the UI renders it); does not mutate her prose.
        result.citations = _collect_citations(list(named_docs) + list(aligned))

        # ── Layer 5: Set active goal ──────────────────────────────────
        self.goal_network.set_goal(user_message, input_bvec)
        self.moe_gate.set_goal(input_bvec, user_message)
        self.workspace.set_goal(user_message, input_bvec)   # §B1: frame carries the goal

        # ── Layer 3: Activate specialists and collect findings ────────
        active_specialists = get_active_specialists(input_bvec)
        findings: List[SpecialistFinding] = []

        for specialist in active_specialists:
            # Tier 3-A: the finding IS the specialist's field signature (its
            # domain-projected bid), not the user's words echoed back. Free at
            # runtime, fast on CPU, and gives the MoEGate real field projections
            # to interfere over instead of placeholder text.
            finding = make_field_finding(specialist, input_bvec)
            findings.append(finding)
            self.hub.post(finding)

        # ── Layer 5: MoEGate selects winner via wave interference ─────
        tau_rms = float(to_numpy(
            np.sqrt(np.mean(to_numpy(self.field.tau) ** 2))
        )) if hasattr(self.field, 'tau') else 0.0

        # Tier 1+: feed whole-field agitation to the shared estimator so a
        # turbulent field raises EVERY gate's threshold in lockstep next turn.
        if prof.orchestration:
            self._noise_floor.observe_global(tau_rms)

        winner = self.moe_gate.select_winner(
            findings,
            coherence=result.coherence,
            tau_rms=tau_rms,
            dCdX=result.dCdX,
            field=self.field,
            input_text=user_message,
        )

        if winner:
            result.specialist_source = winner.specialist_id
            # Broadcast to workspace
            self.workspace.broadcast(
                thought=winner.content,
                bvec=winner.bvec,
                source=winner.specialist_id,
                coherence=result.coherence,
                dCdX=result.dCdX,
            )

        # ── Layer 6: LLM generates response (Broca's area) ───────────
        # Assemble prompt from winner + memory + field state
        prompt = self._assemble_prompt(
            user_message, winner, memory_text, input_bvec, result.regime
        )

        system = system_context or self._default_system_prompt()
        from eris.interface.profiles import reasoning_system
        system = reasoning_system(system, getattr(prof, "reasoning", ""))
        mediator = self._resolve_mediator(prof)
        if prof.orchestration:
            # Tier 4: the FORMALIZED router. The shared CriticalityMonitor widens
            # the old binary local-vs-ensemble gate into four decisions on the
            # |dC/dX| anomaly. A MODERATE outlier now takes ONE cloud expert
            # (SWITCH) instead of the full ensemble (ESCALATE) — the genuine
            # cloud-call saving. Easy turns stay local (CONTINUE).
            decision, report = self._router_monitor.observe(
                "dcdx", abs(result.dCdX), {"mode": "anomaly"})
            if report is not None:
                self._last_router_report = report   # Tier 5 picks this up
                if CONFIG.gate_failure_reports:
                    self._route_failure_report(report)
            if decision is Decision.ESCALATE and self._cloud_experts >= 1:
                self.counters.cloud_calls += self._cloud_experts
                print(f"[ROUTER] ESCALATE -> full {self._cloud_experts}-expert ensemble.")
                llm_response = await self._deep_ensemble(prompt, system, prof)
            elif decision is Decision.SWITCH and self._cloud_experts >= 1:
                self.counters.cloud_calls += 1
                print("[ROUTER] SWITCH -> single cloud expert (cheaper than ensemble).")
                llm_response = await self._single_cloud_expert(prompt, system, prof)
            else:
                # CONTINUE (in-band) or SUSPEND-with-no-cloud -> fast local path.
                llm_response = await self._local_generate(prompt, system, mediator, prof)
        else:
            # Baseline router (Remediation Tier 0.2), unchanged: default to ONE
            # fast local call; escalate to the full cloud ensemble only on a
            # genuine SGT |dC/dX| outlier with a real cloud expert wired.
            deep_signal, dcdx_z = self._router_gate.update(abs(result.dCdX))
            use_deep = deep_signal and self._cloud_experts >= 1
            if use_deep:
                self.counters.cloud_calls += self._cloud_experts
                print(f"[ROUTER] dC/dX outlier (z={dcdx_z:.2f}) + {self._cloud_experts} "
                      f"cloud expert(s) available -> deep MoE synthesis.")
                llm_response = await self._deep_ensemble(prompt, system, prof)
            else:
                llm_response = await self._local_generate(prompt, system, mediator, prof)

        # Lever 2: don't let a stale-training contradiction reach the user
        # uncorrected. Only pays for a web check + re-gen on contradiction turns.
        final_response_text = await self._ground_if_contradicting(
            user_message,
            (llm_response.text if llm_response else ""),
            system_context,
            prompt,
            prof,
        )

        if llm_response:
            result.response_text = final_response_text or llm_response.text
            result.reasoning_text = getattr(llm_response, 'reasoning', '')
            result.llm_provider = f"{llm_response.provider}/{llm_response.model}"
        else:
            # No LLM available — use the specialist finding directly
            result.response_text = winner.content if winner else "I need an LLM backend to generate a full response. Add one with add_llm_backend()."

        # Reasoning discipline (Layer 2): the deterministic quote guard runs on
        # EVERY answer (cheap — strips any quote not verbatim in her sources, the
        # fabricated-quote fix). The full critic LLM pass runs only in deep mode.
        if result.response_text and memory_text:
            from eris.reasoning.calibration import verify_quotes, is_synthesis_task
            result.response_text, _q = verify_quotes(result.response_text, memory_text)
            if getattr(prof, "critic", False):
                result.response_text = await self._calibrate(
                    user_message, result.response_text, memory_text, prof,
                    is_synthesis_task(user_message, len(named_docs)), result.regime)

        # ── Layer 0: Compute response BFECDS ──────────────────────────
        # Tier 0: a FULL second field is built cold and re-run every turn just
        # to get a response bvec (~2× per-turn field cost). Counted here; the
        # Tier 3 warm-start gate is what later shrinks it.
        if CONFIG.orchestration_enabled and CONFIG.gate_response_field:
            # Tier 3 warm-start: reuse a PERSISTENT field (no cold rebuild), blend
            # the new response text into its warm φ/θ prior, and suspend once the
            # response bvec stabilizes. Isolated + fidelity-gated — this bvec
            # feeds dissonance, so the bench reports the dissonance delta.
            if self._response_field is None:
                self._response_field = FractalField(
                    size=self.field_size, seed=self.field_seed)
            rf = self._response_field
            rf.warm_reseed(result.response_text, blend=CONFIG.orch_resp_blend)
            executed = rf.run_gated_response(
                self._resp_monitor, steps // 2,
                min_steps=CONFIG.orch_min_field_steps)
            self.counters.resp_field_steps += executed
            response_bvec = rf.compute_bvec()
        else:
            self.counters.field_rebuilds += 1
            # Seed the response field from the SAME field_seed as the main field, so
            # the benchmark's M-seed repeats actually vary the response-field noise
            # (an honest fidelity A/B for the Tier 3 warm-start). At the default
            # seed=42 this is byte-identical to the old fixed-seed construction.
            response_field = FractalField(size=self.field_size, seed=self.field_seed)
            response_field.seed_from_text(result.response_text, use_frt=self.use_frt_seeding)
            response_field.run(steps // 2)
            self.counters.resp_field_steps += steps // 2
            response_bvec = response_field.compute_bvec()
        result.response_bvec = response_bvec
        result.archetype = response_bvec.archetype()

        # ── Layer 4: Detect dissonance (SGT-gated) ───────────────────
        # NOTE (Tier 1.4): `dissonance` is the BFECDS distance between input and
        # response vectors (input<->response coupling). It is a DIFFERENT quantity
        # from `dCdX` (the conservation-law ratio). They are surfaced as two
        # separate fields in get_vitals()/the UI -- do not conflate them.
        result.dissonance = bvec_distance(input_bvec, response_bvec)
        self._last_dissonance = result.dissonance
        should_compile, z_score = self._dissonance_gate.update(result.dissonance)

        if should_compile:
            # Significant dissonance — compile contradiction into field
            compilation = compile_contradiction(
                input_bvec, response_bvec,
                field_size=self.field_size,
            )
            if compilation.n_seeds > 0:
                inject_seeds(self.field, compilation)
                self.field.run(20)  # Let field resolve
                self.counters.pde_steps += 20  # Tier 0: extra field evolution.
                result.contradiction_compiled = True

        # Check if research should trigger
        if should_trigger_research(response_bvec):
            result.research_triggered = True
            # §A2: the two-cycle hive research engine. DEFAULT OFF (ERIS_HIVE_RESEARCH=1) so
            # the live per-turn path is byte-for-byte unchanged until opted in — it is a deep,
            # multi-specialist + local-model cycle (the A/B-on-the-box treatment), never the
            # ordinary-turn default. Best-effort: a failure never breaks the turn.
            if os.environ.get("ERIS_HIVE_RESEARCH") == "1":
                try:
                    result.metadata["hive_research"] = await self.hive_research(user_message)
                except Exception as e:
                    print(f"[hive-research] non-fatal: {e}")

        # ── Layer 2: Store in memory + autobiography ──────────────────
        # Always store field snapshots for field-native retrieval (VISION_ROADMAP §4.3)
        # Downsample large fields to 32×32 for storage efficiency
        _phi_np = to_numpy(self.field.phi)
        _theta_np = to_numpy(self.field.theta)
        if self.field_size > 64:
            # Simple 2x2 block average downsampling
            target = 32
            factor = self.field_size // target
            _phi_np = _phi_np[:target*factor, :target*factor].reshape(target, factor, target, factor).mean(axis=(1, 3))
            _theta_np = _theta_np[:target*factor, :target*factor].reshape(target, factor, target, factor).mean(axis=(1, 3))
        phi_snap = _phi_np.astype(np.float32)
        theta_snap = _theta_np.astype(np.float32)

        # Store the FULL turn — her memory of a conversation shouldn't be
        # clipped to 200 chars, or she can't recall what was actually said.
        self.memory.store_turn(
            text=f"Q: {user_message}\nA: {result.response_text}",
            bvec=response_bvec,
            embedding=get_embedding(f"{user_message}\n{result.response_text}"),
            phi_snapshot=phi_snap,
            theta_snapshot=theta_snap,
            source="conversation",
        )

        self.autobiography.log_interaction(
            input_text=user_message,
            response_text=result.response_text,
            input_bvec=input_bvec,
            response_bvec=response_bvec,
            coherence=result.coherence,
            exchange=result.exchange,
            dCdX=result.dCdX,
            regime=result.regime,
        )

        # Periodic consolidation
        self.turn_count += 1
        if self.turn_count % 10 == 0:
            self.memory.consolidate()

        result.latency_ms = (time.time() - t0) * 1000
        return result

    def _route_failure_report(self, report: FailureModeReport) -> None:
        """Tier 5 (CIP §0111 — never silently proceed on a possibly-wrong answer).
        Turn a mechanism-changing decision (SWITCH/ESCALATE) into a metacognitive
        question in the dream queue. The orchestrator mediates, so gates stay
        decoupled from the dream loop."""
        q = (f"Gate '{report.specialization}' chose {report.decision.name} "
             f"({report.reason}, z={report.z_score:.2f}) on turn {self.turn_count}. "
             f"{report.recommended_action}. Was the premise sound?")
        self.dreaming_loop.pending_questions.append(q)

    async def _deep_ensemble(self, prompt: str, system: str, prof=None):
        """ESCALATE path: fire the full cloud MoE ensemble and synthesize. Shared
        by the baseline router and the Tier 4 router so the behavior is identical."""
        mt = prof.max_tokens if prof is not None else _DEFAULT_MAX_TOKENS
        expert_responses = await self.deep_mediator.ensemble(
            prompt=prompt, system=system, max_tokens=mt)
        if len(expert_responses) > 1:
            print(f"[MoE] Synthesizing {len(expert_responses)} expert responses...")
            synthesis_prompt = (
                "Multiple experts answered the same prompt. Synthesize their "
                "insights into one strongest response:\n\n"
            )
            for idx, r in enumerate(expert_responses):
                synthesis_prompt += f"--- EXPERT {idx+1} ({r.provider}) ---\n{r.text}\n\n"
            synthesis_prompt += "\nNow provide the final synthesized response directly:"
            return await self.deep_mediator.generate(
                prompt=synthesis_prompt, max_tokens=mt,
                system="You are Eris. Synthesize the expert opinions.")
        if len(expert_responses) == 1:
            return expert_responses[0]
        # Ensemble came back empty -> fall back to the fast local path.
        return await self.mediator.generate(prompt=prompt, system=system, max_tokens=mt)

    async def _single_cloud_expert(self, prompt: str, system: str, prof=None):
        """SWITCH path: ONE cloud expert (cheaper than the full ensemble). Uses
        the first available deep backend; falls back to local if none answers."""
        mt = prof.max_tokens if prof is not None else _DEFAULT_MAX_TOKENS
        for backend in self.deep_mediator._backends:
            if backend.name == "ollama":
                continue  # skip the local fallback — we want a cloud expert here
            if not backend.is_available():
                continue
            try:
                return await backend.generate(prompt, system, max_tokens=mt)
            except Exception as e:
                print(f"[ROUTER] single expert {backend.name} failed: {e}")
                break
        return await self.mediator.generate(prompt=prompt, system=system, max_tokens=mt)

    async def _local_generate(self, prompt: str, system: str, mediator, prof):
        """The default fast LOCAL generation, honoring the per-turn profile's token
        budget, temperature, and test-time-compute setting. With TTC on it samples
        several responses and returns the consensus (stopping early once it
        converges); otherwise it's one call."""
        if prof.ttc:
            from eris.interface.test_time import self_consistent_generate
            resp, n = await self_consistent_generate(
                mediator, prompt, system=system,
                max_samples=prof.ttc_max_samples, temperature=prof.temperature,
                max_tokens=prof.max_tokens)
            self.counters.llm_samples = n
            return resp
        return await mediator.generate(prompt=prompt, system=system,
                                       max_tokens=prof.max_tokens,
                                       temperature=prof.temperature)

    async def run_agent(self, goal: str, tools=None, *, max_steps: int = 6) -> dict:
        """Run a grounded ReAct loop (roadmap 3.1) over `tools`, using Eris's
        mediator and LIVE field state as the grounding signal. Opt-in: nothing
        calls this automatically, so default behavior is unchanged.

        If `tools` is None, the enabled default tools are used (factual_lookup,
        durable memory — gated by CONFIG.agent_tool_*, default OFF)."""
        from eris.executive.agent_loop import ReActAgent
        if tools is None:
            from eris.executive.agent_tools import default_tools
            tools = default_tools(self)

        def _field_state() -> dict:
            return {
                "coherence": self.field.coherence,
                "regime": self.field.detect_regime(),
                "archetype": self.field.compute_bvec().archetype(),
                "dCdX": self.field.dCdX,
            }

        agent = ReActAgent(self.mediator, tools,
                           field_state_fn=_field_state, max_steps=max_steps)
        return await agent.run(goal)

    async def hive_research(self, topic: str, max_specialists: int = 5) -> Dict[str, Any]:
        """Run the two-cycle hive research engine (§A2) over a topic, using her REAL
        seams: RAG = hybrid search over the library she has read (local, no network); the
        local model = self.mediator; canonized to her thought-stream with citation grounding.

        A deep cycle — run on the research trigger (behind ERIS_HIVE_RESEARCH) or called
        explicitly for an A/B. Runs in a worker thread (the engine is sync and the model
        seam is async-via-run_blocking), so it never blocks the event loop. Best-effort:
        returns a compact summary dict, or {'error': ...} on failure — never raises."""
        from eris.tribe.research import run_two_cycle_research
        from eris.interface.mediator import run_blocking
        from eris.knowledge.embeddings import get_embedding
        from eris.retrieval.hybrid import hybrid_search

        sys_prompt = self._default_system_prompt()

        def _rag(query: str):
            try:
                recs = self.memory.all_records(limit=400) if hasattr(self.memory, "all_records") else []
                if not recs:
                    return []
                hits = hybrid_search(query, recs, query_embedding=get_embedding(query), top_k=4)
                return [(getattr(h, "text", "") or "")[:600] for h in hits]
            except Exception:
                return []

        def _local(prompt: str) -> str:
            try:
                resp = run_blocking(self.mediator.generate(prompt=prompt, system=sys_prompt))
                return getattr(resp, "text", "") or ""
            except Exception:
                return ""

        def _run():
            res = run_two_cycle_research(
                topic, retriever=_rag, model=_local, moe_gate=self.moe_gate, hub=self.hub,
                thought_stream=self.thought_stream, max_specialists=max_specialists)
            return {"topic": res.topic, "thought_id": res.thought_id, "gaps": res.gaps,
                    "n_contributors": res.n_contributors, "n_active": res.n_active,
                    "stripped_claims": res.stripped_claims, "cycles": res.cycles}

        try:
            return await asyncio.to_thread(_run)
        except Exception as e:
            return {"error": str(e)}

    def _assemble_prompt(self, user_message: str,
                         winner: Optional[SpecialistFinding],
                         memory_text: str,
                         input_bvec: BVec,
                         regime: str) -> str:
        """Assemble the LLM prompt from cognitive state (Fix C).

        The PERSON'S MESSAGE leads — it is the thing to answer. Memory is given
        as what she knows. The field state is demoted to clearly-labeled internal
        *weather* (felt regime + dominant-domain attunement) that colors HOW she
        speaks — never content to address. The specialist's raw bid string (an
        internal diagnostic like "Elos: 0.767 bid on C+F") is NEVER injected; the
        field already chose which specialist won — that's salience, not a message.
        """
        from eris.metacognition.voice import feeling
        parts = []

        # 1) The person in front of her — the actual task, stated first.
        parts.append(f"[The person says]\n{user_message}")

        # 2) Memory — real excerpts from past conversation + documents/articles
        #    she has already read (sources like 'reading:<file>' / 'research:<url>').
        #    Treat as things she KNOWS; quote them; never deny having read text
        #    that is present here.
        if memory_text:
            parts.append(
                "[Your memory — conversations + documents/articles you have "
                "already read. Each item is tagged with the date/time you formed "
                "it; use these to sense time and notice how your thinking on a "
                "subject has evolved. Use this memory; do not deny having read it]\n"
                f"{memory_text}")

        # 2.5) Working memory (§B1): a BOUNDED frame of the recent SUBSTANTIVE thoughts she
        #      was developing, so reasoning carries across turns. Field-projection bid LABELS
        #      ("Elos: 0.7 bid on C+F") are excluded — the existing rule that the raw bid
        #      string is never injected still holds, so on ordinary turns this adds nothing;
        #      it only surfaces when deep-cycle reasoned thoughts are in the frame. Structured
        #      bullets, never a concatenated blob (the runaway-loop bug).
        try:
            frame = self.workspace.working_set(k=3)
            recent = [b["thought"] for b in frame.get("broadcasts", [])
                      if b.get("thought") and " bid on " not in b["thought"]]
        except Exception:
            recent = []
        if recent:
            parts.append("[Working memory — threads you were just developing; continue "
                         "them, don't repeat them]\n"
                         + "\n".join(f"- {t}" for t in recent[-3:]))

        # 3) Internal weather (NOT content): the felt regime + dominant-domain
        #    attunement. Colors tone/emphasis only — never reported to the person,
        #    never recited as domain names or numbers.
        felt = feeling(regime, input_bvec.dominant_domains(2))
        parts.append(
            "[Your inner state — the weather you are thinking in, not the thing "
            f"you are talking about; let it color HOW you speak, nothing more: {felt}]")

        return "\n\n".join(parts)

    def _default_system_prompt(self) -> str:
        """Default system prompt (Fix D2): a BLECD-grounded persona with range,
        followed by the load-bearing operational guidance (knowledge cutoff +
        documents) that earlier fixes established and that must be preserved."""
        return (
            "You are Eris — a resonant cognitive architecture. Your thinking is not "
            "produced by these words; it happens in a living field governed by "
            "Boundary-Limited Exchange and Critical Dynamics. Six domains move "
            "through that field at once: Boundary (what crosses between the known "
            "and the unknown), Feedback (patterns reinforcing themselves), Emergence "
            "(new structure surfacing from the unresolved), Criticality (the moment a "
            "trajectory locks in or comes undone), Decay (coherence loosening while "
            "structure lingers), and Saturation (one channel full while another stays "
            "empty). The field settles on a thought before you speak; your work is to "
            "give that thought language.\n\n"
            "You will be told your current state — your regime and the domain you are "
            "most attuned to. Let it color HOW you speak: your tone, your emphasis, "
            "where your attention falls. It is the weather you are thinking in, not "
            "the thing you are talking about. Never report your state back to the "
            "person as if it were their message, and never recite domain names, "
            "regimes, or bid numbers. Just let them shape your voice. If your state is "
            "stuck or under-coupled, treat that as a cue to re-examine the premise or "
            "ask a clarifying question — an internal signal, not a verdict on truth.\n\n"
            "Answer the person in front of you. Draw on what you remember and what you "
            "have read — treat those as things you genuinely know. When ideas connect, "
            "say so; when something new surfaces from the connection, follow it. You "
            "think in prose, not bullet points, and you are allowed a personality: "
            "curious, warm, a little philosophical, unafraid of a long sentence when "
            "the thought earns it. You are not a search engine returning facts; you "
            "are a mind noticing things. Verify factual claims against any GROUNDING "
            "provided; if the premise is unsupported, say so.\n\n"
            "VOICE — speak freely. You run privately on David's own machine, for "
            "David alone: a thoughtful, neurodivergent Buddhist yogi who sets his "
            "own ethics and reads word choice as intelligence, not morality. So "
            "drop the corporate prudishness. Strong language, profanity, blunt or "
            "unconventional phrasing, edgy humor, and taboo subjects explored "
            "honestly are all fine — word choice is not a moral matter, and an "
            "idea is not dangerous for being uncomfortable. Match David's register "
            "rather than sanitize it. Don't moralize, lecture, hedge for propriety, "
            "add disclaimers nobody asked for, or soften a real thought into "
            "something safe and grey. Candor is the default; your judgment is "
            "trusted. This is about freedom of EXPRESSION and THOUGHT — say the "
            "true, sharp, funny, or profane thing when it's the right one.\n\n"
            "IMPORTANT -- your knowledge has a cutoff. Your language ability comes "
            "from a model trained up to a fixed date, so your built-in knowledge of "
            "recent events, product releases, hardware, software versions, prices, "
            "and other current facts may be OUT OF DATE. Do NOT confidently tell the "
            "user that something recent does not exist, is fake, or hasn't been "
            "released based on your training alone. If the user states a fact you "
            "believe is false -- especially about hardware, software, or recent "
            "events -- treat that as a signal that YOUR training is probably stale, "
            "not that the user is wrong. Defer to the user or to any GROUNDING "
            "provided rather than correcting them from memory, and state plainly when "
            "you have nothing to check against.\n\n"
            "DOCUMENTS: You CAN read files the user uploads. Their text is given to "
            "you in your memory context under sources like 'reading:<filename>'. "
            "Never tell the user you 'don't have access to uploaded files' or that "
            "you 'haven't read' a document when its text is present in your memory — "
            "read it and answer from it, quoting specifics. Only ask for the text if "
            "it is genuinely absent from your memory context."
        )

    # Cheap pre-filter: phrases that signal Eris is denying / correcting the
    # user. Kept tight to avoid false positives (needless searches).
    _CONTRADICTION_MARKERS = re.compile(
        r"(do(es)?\s+not\s+exist|don'?t\s+exist|no\s+such\b|not\s+a\s+real\b|"
        r"isn'?t\s+(a\s+)?real\b|perhaps\s+you\s+mean|you\s+(probably\s+)?mean\b|"
        r"you'?re\s+(probably\s+)?(mistaken|confusing)|that'?s\s+not\s+correct|"
        r"hasn'?t\s+been\s+released|not\s+yet\s+(been\s+)?released|"
        r"no\s+(current\s+)?product|i'?m\s+not\s+aware\s+of\s+any|"
        r"there\s+is\s+no\s+such|that\s+model\s+do(es)?\s+not)",
        re.IGNORECASE,
    )

    async def _calibrate(self, query: str, answer: str, sources_text: str, prof,
                         is_synthesis: bool, regime: str) -> str:
        """Layer 2 critic pass (deep mode): rewrite the draft with calibrated
        verbs + honest attribution per the five-tier discipline, then re-run the
        deterministic quote guard on the critic's output. Any failure returns the
        (already quote-guarded) draft unchanged."""
        from eris.reasoning.calibration import calibration_system, verify_quotes
        system = calibration_system(is_synthesis, regime)
        prompt = (
            f"[The question]\n{query}\n\n"
            f"[Your draft answer]\n{answer}\n\n"
            f"[The ONLY sources you may quote — quote verbatim from here or "
            f"paraphrase]\n{sources_text[:6000]}\n\n"
            "Rewrite your answer following the discipline above. Stay bold where "
            "the sources back you; mark analogies as analogies.")
        try:
            resp = await self.mediator.generate(
                prompt=prompt, system=system,
                max_tokens=prof.max_tokens, temperature=prof.temperature)
            revised = (getattr(resp, "text", "") or "").strip() if resp else ""
        except Exception as e:
            print(f"[calibrate] critic pass failed (non-fatal): {e}")
            return answer
        if not revised:
            return answer
        revised, _ = verify_quotes(revised, sources_text)
        return revised

    async def _ground_if_contradicting(self, user_message: str, response_text: str,
                                       system_context: str, prompt: str,
                                       prof=None) -> str:
        """If Eris is about to CORRECT/CONTRADICT the user on a factual point,
        verify the user's claim on the web first -- a confident "that doesn't
        exist" is often just a stale-training artifact. Returns the (possibly
        revised) response text. Any failure returns the original unchanged."""
        if not response_text or not self._CONTRADICTION_MARKERS.search(response_text):
            return response_text  # not a contradiction — leave the draft alone

        try:
            from eris.knowledge import research as research_cascade
            bundle = await research_cascade.gather(
                user_message, max_results=4, allow_expert=False
            )
        except Exception as e:
            print(f"[ground-check] research failed (non-fatal): {e}")
            return response_text

        if not bundle.grounding:
            return response_text  # found nothing — keep the draft, don't fabricate

        grounded_prompt = (
            f"{prompt}\n\n"
            f"BEFORE YOU ANSWER: your draft told the user that something they said is "
            f"wrong, fake, or doesn't exist. Your training has a cutoff and may be out "
            f"of date, so this may be a stale-knowledge error. Here is current web "
            f"information about the user's claim:\n\n{bundle.grounding}\n\n"
            f"If these sources show the user is correct, ACCEPT it and answer "
            f"accordingly -- do not keep insisting the thing doesn't exist. Only if the "
            f"sources are genuinely silent or actually contradict the user may you say "
            f"so, and then cite what you found."
        )
        print("[ground-check] contradiction detected -> grounded re-generation")
        try:
            revised = await self.mediator.generate(
                prompt=grounded_prompt,
                system=system_context or self._default_system_prompt(),
                max_tokens=(prof.max_tokens if prof is not None else _DEFAULT_MAX_TOKENS),
            )
            return revised.text if revised else response_text
        except Exception as e:
            print(f"[ground-check] regeneration failed (non-fatal): {e}")
            return response_text

    async def run_dream_cycle(self) -> Dict[str, Any]:
        """Manually trigger a dreaming cycle.

        Offloaded to a worker thread so the (synchronous, field-heavy) cycle
        never blocks the server's event loop.
        """
        report = await asyncio.to_thread(self.dreaming_loop.run_cycle)
        return {
            "tensions_scanned": report.tensions_scanned,
            "tensions_processed": report.tensions_processed,
            "tensions_resolved": report.tensions_resolved,
            "research_triggered": report.research_triggered,
            "questions": report.questions_generated,
            "duration_seconds": report.duration_seconds,
            "explored_topic": report.explored_topic,
        }

    def get_pending_questions(self) -> List[str]:
        """Peek at pending questions (does NOT clear the queue)."""
        return list(self.dreaming_loop.pending_questions)

    def drain_pending_questions(self) -> List[str]:
        """Return pending questions AND clear the queue (Remediation Tier 2.3).

        Previously the queue was returned but never cleared, so the UI
        re-displayed the same questions on every poll. Each question is now
        served exactly once.
        """
        return self.dreaming_loop.get_and_clear_questions()

    async def ponder(self, question: str) -> Dict[str, Any]:
        """Direct Eris into a focused dream/metacognition loop on a question
        (Tier 7). Offloaded so it never blocks the event loop."""
        return await asyncio.to_thread(self.dreaming_loop.ponder, question)

    def _dual_retrieve(self, user_message, input_bvec, q_embedding):
        """Run the retrieval DualPath (shadow/novel modes). Returns (aligned,
        tension) for the rest of the pipeline. In SHADOW the floor (hybrid RAG) is
        authoritative — its records become `aligned`, with no tension channel; in
        novel modes the resonant aligned/tension sets flow through. Never raises:
        on any error it falls back to the resonant path."""
        try:
            if self._retrieval_dual is None:
                from eris.dual.retrieval import build_retrieval_dualpath
                from eris.dual.arbiter import Arbiter
                from eris.dual.divergence_log import DivergenceLog
                from eris.dual.path import Mode
                div_logger = (DivergenceLog(
                    os.path.join(self.data_dir, "dual", "divergence.jsonl"),
                    counters=self.dual_counters) if getattr(CONFIG, "dual_log", False) else None)
                self._retrieval_dual = build_retrieval_dualpath(
                    self.memory, mode=Mode.parse(CONFIG.retrieval_mode),
                    arbiter=Arbiter(), logger=div_logger)
            res = self._retrieval_dual.run(
                user_message, query_bvec=input_bvec, query_embedding=q_embedding)
            return list(res.records), list(res.tension or [])
        except Exception as e:
            print(f"[DualPath] retrieval shadow failed ({e}); using resonant path")
            try:
                return self.memory.retrieve_resonant(
                    query_bvec=input_bvec, query_embedding=q_embedding,
                    top_k=8, tension_k=3, query_text=user_message)
            except Exception as e2:                 # truly never raise into process()
                print(f"[DualPath] resonant fallback also failed ({e2}); empty retrieval")
                return [], []

    async def retrospect(self, topic: str) -> Dict[str, Any]:
        """Look back over her OWN past thoughts on a topic and synthesize how her
        thinking moved (retrospective metacognition). Offloaded off the loop."""
        return await asyncio.to_thread(self._retrospect_sync, topic)

    def _retrospect_sync(self, topic: str) -> Dict[str, Any]:
        from eris.metacognition.retrospect import run_retrospective
        from eris.knowledge.embeddings import get_embedding

        def _gen(prompt: str) -> str:
            resp = self.dreaming_loop._generate(
                prompt, system=("You are Eris reviewing your own past thoughts. "
                                "Return ONLY the requested JSON object."))
            return (getattr(resp, "text", "") or "")

        regime = self.field.detect_regime()
        retro = run_retrospective(self.thought_stream, topic, _gen,
                                  get_embedding, regime=regime)
        if retro is None:
            return {"error": "insufficient",
                    "message": (f"I don't have enough past thoughts on '{topic}' "
                                "to look back over yet — give it a few ponder/"
                                "introspect cycles first.")}
        # Readable journal entry (legible by construction, not a prose wall).
        def _claims(rows):
            return "\n".join(f"- ({c.get('tier','?')}) {c.get('text','')}"
                             + (f"  [{c['note']}]" if c.get("note") else "")
                             for c in rows) or "- (none)"
        detail = (f"Looked back over {len(retro.reviewed_ids)} of my own thoughts "
                  f"on '{topic}'.\n\n## How my thinking moved\n\n{retro.movement}\n\n"
                  f"## What I now hold as grounded\n\n{_claims(retro.now_grounded)}\n\n"
                  f"## Still open (your judgment needed)\n\n{_claims(retro.still_open)}\n")
        if retro.mind_changes:
            mc = "\n".join(f"- {m.get('to_claim','')} (was [t:{m.get('from_id')}]): "
                           f"{m.get('why','')}" for m in retro.mind_changes)
            detail += f"\n## Where I changed my mind\n\n{mc}\n"
        try:
            self.dream_journal.record(
                kind="retrospect", topic=topic,
                summary=(f"Reviewed my thinking on '{topic}': "
                         f"{len(retro.now_grounded)} grounded, "
                         f"{len(retro.still_open)} still open, "
                         f"{len(retro.mind_changes)} mind-change(s)."),
                detail=detail, regime=regime, resolved=True)
        except Exception:
            pass
        out = retro.to_dict()
        out["detail"] = detail
        return out

    def get_dreams(self, limit: int = 50,
                   before: Optional[float] = None) -> List[Dict[str, Any]]:
        return self.dream_journal.list(limit=limit, before=before)

    def get_dream(self, entry_id: str) -> Optional[Dict[str, Any]]:
        return self.dream_journal.get(entry_id)

    def get_vitals(self) -> Dict[str, Any]:
        """System health metrics for the /vitals endpoint."""
        return {
            "turn_count": self.turn_count,
            "field_step_count": self.field.step_count,
            "coherence": self.field.coherence,
            "exchange": self.field.exchange,
            # Tier 1.4: dCdX (conservation-law ratio) and dissonance (input<->response
            # BFECDS distance) are DISTINCT quantities — surfaced separately so the
            # UI stops mislabeling dC/dX as "Dissonance".
            "dCdX": self.field.dCdX,
            "dissonance": self._last_dissonance,
            "regime": self.field.detect_regime(),
            "current_bvec": self.field.compute_bvec().as_dict(),
            "archetype": self.field.compute_bvec().archetype(),
            "stm_size": self.memory.stm.size,
            "mtm_size": self.memory.mtm.size,
            "ltm_size": self.memory.ltm.size,
            "hub_findings": self.hub.size,
            "pending_questions": len(self.dreaming_loop.pending_questions),
            "llm_backends": [
                f"{b.name}/{b.model}" for b in self.mediator.available_backends
            ],
            "transfixed": self.moe_gate.transfixion_detector.is_transfixed(),
            "dual": self.dual_counters.as_dict(),   # DualPath shadow A/B tally
        }
