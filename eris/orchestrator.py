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
from eris.interface.mediator import LLMMediator, LLMBackend

# Layer 4: Metacognition
from eris.metacognition.dreaming import DreamingLoop


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
    metadata: Dict[str, Any] = field(default_factory=dict)


class ErisOrchestrator:
    """The main cognitive loop. Wires all layers together.

    Initialize once, call process() for each user message.
    Manages the persistent field, memory, and metacognitive state.
    """

    def __init__(self,
                 field_size: int = 64,
                 data_dir: str = "eris_data",
                 use_frt_seeding: bool = False):
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
        """
        os.makedirs(data_dir, exist_ok=True)
        self.field_size = field_size
        self.use_frt_seeding = use_frt_seeding

        # Layer 1: Persistent field state
        self.field = FractalField(size=field_size)

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
        self.dreaming_loop = DreamingLoop(
            autobiography=self.autobiography,
            memory=self.memory,
            field_size=field_size,
            journal=self.dream_journal,
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

        # Fast path: one resident local model (Broca's area).
        self.mediator.add_backend(OllamaBackend(model=self._local_model))

        # Deep path: cloud experts ONLY if keyed, plus a local fallback so the
        # ensemble is never empty. Dormant (keyless) cloud backends are skipped.
        for _backend_cls, _env in ((AnthropicBackend, "ANTHROPIC_API_KEY"),
                                   (OpenAIBackend, "OPENAI_API_KEY"),
                                   (GeminiBackend, "GEMINI_API_KEY")):
            _key = os.environ.get(_env, "")
            if _key:
                self.deep_mediator.add_backend(_backend_cls(api_key=_key))
        self.deep_mediator.add_backend(OllamaBackend(model=self._local_model))

        # Number of genuine (cloud) experts wired for deep synthesis.
        self._cloud_experts = len(self.deep_mediator._backends) - 1

        # Scale-adaptive gate for the deep path: opens only when |dC/dX| is a
        # statistical outlier relative to this engine's own running history.
        self._router_gate = SGTGate(threshold_sigma=2.0, ema_alpha=0.1)

        # Turn counter + last-turn dissonance (surfaced separately in vitals).
        self.turn_count: int = 0
        self._last_dissonance: float = 0.0

    def add_llm_backend(self, backend: LLMBackend) -> None:
        """Add an LLM backend (Ollama, Claude, OpenAI, etc.)."""
        self.mediator.add_backend(backend)

    async def process(self, user_message: str,
                      system_context: str = "") -> ProcessingResult:
        """Process one user message through the full cognitive pipeline.

        This is the main loop — everything happens here.
        """
        t0 = time.time()
        result = ProcessingResult()

        # ── Layer 1: Seed and evolve the FRACTAL field ────────────────
        self.field.seed_from_text(user_message, use_frt=self.use_frt_seeding)
        self.field.run(CONFIG.pde_steps_per_input)

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
        aligned, tension = self.memory.retrieve_resonant(
            query_bvec=input_bvec, query_embedding=q_embedding,
            top_k=8, tension_k=3, query_text=user_message,
        )
        # Give the LLM the FULL retrieved memory, not a sliver — truncating here
        # is what made Eris say she hadn't read an uploaded document she had.
        memory_text = "\n\n".join(
            f"[{r.source}] {r.text}" for r in aligned
        ) if aligned else ""
        if tension:
            memory_text += (
                "\n\n[Related but unresolved — look for the hidden connection]\n"
                + "\n\n".join(f"[{r.source}] {r.text}" for r in tension)
            )

        # Named-document direct retrieval: if the user names a file/document,
        # pull ITS chunks straight in so they can't be crowded out by chatter
        # that merely mentions it. This is what makes "tell me about my patent
        # sgtpatent" actually read the sgtpatent document.
        named_docs = self.memory.documents_matching(
            user_message, max_chunks=6, query_embedding=q_embedding)
        if named_docs:
            doc_block = "\n\n".join(f"[{r.source}] {r.text}" for r in named_docs)
            memory_text = (
                "[A document you have ALREADY READ that the user is asking about "
                "— answer from this text; do NOT say you can't access it]\n"
                + doc_block
                + (("\n\n" + memory_text) if memory_text else "")
            )

        # ── Layer 5: Set active goal ──────────────────────────────────
        self.goal_network.set_goal(user_message, input_bvec)
        self.moe_gate.set_goal(input_bvec, user_message)

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

        # LLM ROUTER (Remediation Tier 0.2): default to ONE fast local call.
        # Escalate to the cloud MoE ensemble ONLY when |dC/dX| is a genuine SGT
        # outlier for THIS engine AND at least one real cloud expert is wired.
        deep_signal, dcdx_z = self._router_gate.update(abs(result.dCdX))
        use_deep = deep_signal and self._cloud_experts >= 1

        if use_deep:
            print(f"[ROUTER] dC/dX outlier (z={dcdx_z:.2f}) + {self._cloud_experts} "
                  f"cloud expert(s) available -> deep MoE synthesis.")
            expert_responses = await self.deep_mediator.ensemble(
                prompt=prompt,
                system=system_context or self._default_system_prompt(),
            )
            if len(expert_responses) > 1:
                print(f"[MoE] Synthesizing {len(expert_responses)} expert responses...")
                synthesis_prompt = (
                    "Multiple experts answered the same prompt. Synthesize their "
                    "insights into one strongest response:\n\n"
                )
                for idx, r in enumerate(expert_responses):
                    synthesis_prompt += f"--- EXPERT {idx+1} ({r.provider}) ---\n{r.text}\n\n"
                synthesis_prompt += "\nNow provide the final synthesized response directly:"
                llm_response = await self.deep_mediator.generate(
                    prompt=synthesis_prompt,
                    system="You are Eris. Synthesize the expert opinions.",
                )
            elif len(expert_responses) == 1:
                llm_response = expert_responses[0]
            else:
                # Ensemble came back empty -> fall back to the fast local path.
                llm_response = await self.mediator.generate(
                    prompt=prompt,
                    system=system_context or self._default_system_prompt(),
                )
        else:
            llm_response = await self.mediator.generate(
                prompt=prompt,
                system=system_context or self._default_system_prompt(),
            )

        # Lever 2: don't let a stale-training contradiction reach the user
        # uncorrected. Only pays for a web check + re-gen on contradiction turns.
        final_response_text = await self._ground_if_contradicting(
            user_message,
            (llm_response.text if llm_response else ""),
            system_context,
            prompt,
        )

        if llm_response:
            result.response_text = final_response_text or llm_response.text
            result.reasoning_text = getattr(llm_response, 'reasoning', '')
            result.llm_provider = f"{llm_response.provider}/{llm_response.model}"
        else:
            # No LLM available — use the specialist finding directly
            result.response_text = winner.content if winner else "I need an LLM backend to generate a full response. Add one with add_llm_backend()."

        # ── Layer 0: Compute response BFECDS ──────────────────────────
        response_field = FractalField(size=self.field_size)
        response_field.seed_from_text(result.response_text, use_frt=self.use_frt_seeding)
        response_field.run(CONFIG.pde_steps_per_input // 2)
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
                result.contradiction_compiled = True

        # Check if research should trigger
        if should_trigger_research(response_bvec):
            result.research_triggered = True

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

    def _assemble_prompt(self, user_message: str,
                         winner: Optional[SpecialistFinding],
                         memory_text: str,
                         input_bvec: BVec,
                         regime: str) -> str:
        """Assemble the LLM prompt from cognitive state.

        The LLM sees: the user message, the specialist's analysis,
        relevant memory context, and the field state summary.
        The LLM does NOT see raw BFECDS numbers — it sees their
        interpretation in natural language.
        """
        parts = []

        # Specialist analysis (the "thought" the GPW selected)
        if winner:
            parts.append(f"[Specialist {winner.specialist_id} analysis]\n{winner.content}")

        # Memory context. These are real excerpts from YOUR memory — past
        # conversation, plus documents and articles you have already read and
        # ingested (sources like 'reading:<file>' or 'research:<url>' are
        # documents the user gave you or pages you studied). Treat them as
        # things you KNOW. If relevant content appears here, use it and quote it
        # — do NOT claim you haven't seen or read a document when its text is
        # present below.
        if memory_text:
            parts.append(
                "[Your memory — conversations + documents/articles you have "
                f"already read; use this, do not deny having read it]\n{memory_text}")

        # Field state in natural language
        archetype = input_bvec.archetype()
        regime_desc = {
            "elastic": "processing smoothly",
            "plastic": "actively restructuring understanding",
            "transfixed": "the field is stuck / under-coupled on some channel — re-examine or ask for clarification",
            "warmup": "still calibrating",
        }.get(regime, "unknown state")

        parts.append(
            f"[Cognitive state: {archetype}, {regime_desc}]"
        )

        # The actual user message
        parts.append(f"[User message]\n{user_message}")

        return "\n\n".join(parts)

    def _default_system_prompt(self) -> str:
        """Default system prompt for the LLM."""
        return (
            "You are Eris Echo, a resonant cognitive architecture. "
            "You have been given a specialist analysis, memory context, "
            "and cognitive state assessment. Use these to formulate your "
            "response. Be direct, thoughtful, and honest. If the cognitive "
            "state indicates the field is stuck or under-coupled, treat that as "
            "a cue to re-examine the premise or ask a clarifying question -- it "
            "is an internal-processing signal, not a verdict on whether the "
            "content is true. Verify factual claims against any GROUNDING "
            "provided; if the premise is unsupported, say so.\n\n"
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

    async def _ground_if_contradicting(self, user_message: str, response_text: str,
                                       system_context: str, prompt: str) -> str:
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
        }
