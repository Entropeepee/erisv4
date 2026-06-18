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
from eris.memory.interference import find_conflicts

# Layer 3: Tribe
from eris.tribe.specialists import (
    TRIBE, get_active_specialists, CrossAttentionHub,
    SpecialistFinding, should_trigger_research,
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

        # Layer 4: Metacognition
        self.dreaming_loop = DreamingLoop(
            autobiography=self.autobiography,
            memory=self.memory,
            field_size=field_size,
        )

        # SGT gate for dissonance detection
        self._dissonance_gate = SGTGate(threshold_sigma=2.0, ema_alpha=0.1)
        # We can add a default LLM backend here if we want, 
        # but it's typically injected by the caller.
        from eris.interface.mediator import OpenAIBackend
        self.llm_backends: List[LLMBackend] = [
            OpenAIBackend(model="qwen", api_key="sk-no-key", base_url="http://localhost:8080/v1")
        ]

        # Turn counter
        self.turn_count: int = 0

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
        memory_context = self.memory.retrieve(
            query_bvec=input_bvec, top_k=5
        )
        memory_text = "\n".join(
            f"[{r.source}] {r.text[:200]}" for r in memory_context
        ) if memory_context else ""

        # ── Layer 5: Set active goal ──────────────────────────────────
        self.goal_network.set_goal(user_message, input_bvec)
        self.moe_gate.set_goal(input_bvec, user_message)

        # ── Layer 3: Activate specialists and collect findings ────────
        active_specialists = get_active_specialists(input_bvec)
        findings: List[SpecialistFinding] = []

        for specialist in active_specialists:
            # Each specialist generates a finding based on its domain
            # In production: these call the LLM with specialist-specific prompts
            # For now: generate a finding with the specialist's sensitivity profile
            finding = SpecialistFinding(
                specialist_id=specialist.id,
                content=f"[{specialist.name}] Analysis of: {user_message[:100]}",
                bvec=BVec(
                    B=input_bvec.B * specialist.sensitivity_bvec.B,
                    F=input_bvec.F * specialist.sensitivity_bvec.F,
                    E=input_bvec.E * specialist.sensitivity_bvec.E,
                    C=input_bvec.C * specialist.sensitivity_bvec.C,
                    D=input_bvec.D * specialist.sensitivity_bvec.D,
                    S=input_bvec.S * specialist.sensitivity_bvec.S,
                ),
                confidence=bvec_distance(input_bvec, specialist.sensitivity_bvec),
            )
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

        llm_response = await self.mediator.generate(
            prompt=prompt,
            system=system_context or self._default_system_prompt(),
        )

        if llm_response:
            result.response_text = llm_response.text
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
        result.dissonance = bvec_distance(input_bvec, response_bvec)
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

        self.memory.store_turn(
            text=f"Q: {user_message[:200]}\nA: {result.response_text[:200]}",
            bvec=response_bvec,
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

        # Memory context
        if memory_text:
            parts.append(f"[Relevant memory]\n{memory_text}")

        # Field state in natural language
        archetype = input_bvec.archetype()
        regime_desc = {
            "elastic": "processing smoothly",
            "plastic": "actively restructuring understanding",
            "transfixed": "WARNING: may be generating without genuine processing",
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
            "state indicates transfixion, acknowledge uncertainty rather "
            "than generating a confident-sounding response."
        )

    async def run_dream_cycle(self) -> Dict[str, Any]:
        """Manually trigger a dreaming cycle."""
        report = self.dreaming_loop.run_cycle()
        return {
            "tensions_scanned": report.tensions_scanned,
            "tensions_processed": report.tensions_processed,
            "tensions_resolved": report.tensions_resolved,
            "research_triggered": report.research_triggered,
            "questions": report.questions_generated,
            "duration_seconds": report.duration_seconds,
        }

    def get_pending_questions(self) -> List[str]:
        """Get questions the dreaming loop generated for the user."""
        return self.dreaming_loop.pending_questions

    def get_vitals(self) -> Dict[str, Any]:
        """System health metrics for the /vitals endpoint."""
        return {
            "turn_count": self.turn_count,
            "field_step_count": self.field.step_count,
            "coherence": self.field.coherence,
            "exchange": self.field.exchange,
            "dCdX": self.field.dCdX,
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
