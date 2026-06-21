"""
Dreaming Loop — Metacognitive Processing Engine
==================================================

Runs on 15-30 minute cycles (or triggered manually). This is the
system's sleep consolidation — where unresolved tensions get processed,
contradictions get compiled into field dynamics, and genuine learning
happens between conversations.

The cycle:
    1. SCAN autobiography for high-torsion entries
    2. SGT GATE: is this tension above the noise floor?
       NO → skip (it's noise, not real dissonance)
       YES → proceed
    3. COMPILE: BLC generates φ-θ seed geometry from the contradiction
    4. EVOLVE: inject into PDE field, let dynamics resolve
    5. LEARN: if resolved, store pattern in analogy engine / LTM
    6. RESEARCH: if C > 0.4 AND E > 0.2, trigger two-cycle research
    7. QUESTION: if tension persists, formulate question for David
       framed using the dominant BLECD domain

The dreaming loop is what makes Eris Echo more than a chatbot with
memory — it actively processes its own cognitive state, detecting
where its understanding is broken and either fixing it or asking.

From the handoff conversation:
    "The piece that was missing was the gating on the dreaming process.
     In the earlier version, everything flagged as conflicting got sent
     for research. But SGT tells you that some conflicts are below the
     noise floor — they're apparent dissonance that's actually just
     stochastic variation."
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time

from eris.computation.activations import BVec, bvec_distance
from eris.computation.sgt import SGTGate
from eris.field.pde import FractalField
from eris.field.compiler import compile_contradiction, inject_seeds
from eris.memory.autobiography import Autobiography, AutobiographyEntry
from eris.memory.tiers import MemorySystem, MemoryRecord
from eris.tribe.specialists import should_trigger_research
from eris.knowledge import research as research_cascade


@dataclass
class DreamResult:
    """Result of processing one tension in the dreaming loop."""
    entry: AutobiographyEntry
    resolved: bool = False
    resolution_bvec: Optional[BVec] = None
    resolution_regime: str = "unknown"
    triggered_research: bool = False
    generated_question: Optional[str] = None


@dataclass
class DreamCycleReport:
    """Summary of one dreaming cycle."""
    tensions_scanned: int = 0
    tensions_gated_out: int = 0   # Below noise floor
    tensions_processed: int = 0
    tensions_resolved: int = 0
    research_triggered: int = 0
    questions_generated: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class DreamingLoop:
    """Metacognitive processing engine.

    Scans the autobiography for unresolved tensions, SGT-gates them,
    compiles contradictions into field dynamics, and either resolves
    them or generates questions.
    """

    def __init__(self,
                 autobiography: Autobiography,
                 memory: MemorySystem,
                 field_size: int = 32,
                 torsion_threshold: float = 0.2,
                 sgt_threshold: float = 2.0):
        self.autobiography = autobiography
        self.memory = memory
        self.field_size = field_size
        self.torsion_threshold = torsion_threshold

        # SGT gate for tension significance
        self._tension_gate = SGTGate(threshold_sigma=sgt_threshold, ema_alpha=0.1)

        # Questions generated for the user
        self.pending_questions: List[str] = []

    def run_cycle(self, max_tensions: int = 10) -> DreamCycleReport:
        """Run one dreaming cycle.

        Scans today's high-torsion entries, processes up to max_tensions.
        """
        t0 = time.time()
        report = DreamCycleReport()

        # 1. SCAN for high-torsion entries (including prior-session tensions
        #    loaded from disk — Tier 2.1, so restarts don't forget).
        candidates = self.autobiography.get_high_torsion(
            self.torsion_threshold, include_persisted=True
        )
        report.tensions_scanned = len(candidates)

        for entry in candidates[:max_tensions]:
            result = self._process_tension(entry)

            if result is None:
                report.tensions_gated_out += 1
                continue

            report.tensions_processed += 1

            if result.resolved:
                report.tensions_resolved += 1

            if result.triggered_research:
                report.research_triggered += 1

            if result.generated_question:
                report.questions_generated.append(result.generated_question)
                self.pending_questions.append(result.generated_question)

        report.duration_seconds = time.time() - t0
        return report

    def _process_tension(self, entry: AutobiographyEntry) -> Optional[DreamResult]:
        """Process one high-torsion entry.

        Returns None if gated out (below noise floor).
        """
        # 2. SGT GATE: is this dissonance significant?
        should_process, z_score = self._tension_gate.update(entry.dissonance)
        if not should_process:
            return None  # Noise — skip

        result = DreamResult(entry=entry)

        # 3. COMPILE: generate field seed from the contradiction
        if entry.input_bvec and entry.response_bvec:
            compilation = compile_contradiction(
                entry.input_bvec, entry.response_bvec,
                field_size=self.field_size,
            )

            if compilation.n_seeds > 0:
                # 4. EVOLVE: create a fresh field, inject seeds, evolve
                field = FractalField(size=self.field_size)
                # Seed from the original input text
                field.seed_from_text(entry.input_text)
                field.run(20)

                # Inject the contradiction geometry
                inject_seeds(field, compilation)

                # Let the field resolve
                field.run(50)

                resolved_bvec = field.compute_bvec()
                result.resolution_bvec = resolved_bvec
                result.resolution_regime = field.detect_regime()

                # 5. LEARN: check if resolved (low residual torsion)
                residual_torsion = resolved_bvec.C
                if residual_torsion < 0.3:
                    result.resolved = True
                    # Store resolved pattern in LTM
                    self.memory.ltm.store(MemoryRecord(
                        text=f"[Dream resolution] {entry.input_text[:100]}",
                        bvec=resolved_bvec,
                        source="dream",
                    ))

                # 6. RESEARCH (Tier 4.3): if high criticality + emergence, run
                #    the cascade (web search -> expert if keyed) and ingest the
                #    findings into LTM as grounding for future turns.
                if should_trigger_research(resolved_bvec):
                    result.triggered_research = True
                    bundle = self._run_research(entry.input_text)
                    if bundle is not None:
                        for i, txt in enumerate(bundle.full_texts):
                            src = bundle.sources[i] if i < len(bundle.sources) else "expert"
                            self.memory.ltm.store(MemoryRecord(
                                text=f"[Research: {entry.input_text[:60]}] {txt[:2000]}",
                                bvec=resolved_bvec,
                                source=f"research:{src}",
                            ))

                # 7. QUESTION: if tension persists, generate a question
                if not result.resolved:
                    question = self._formulate_question(entry, resolved_bvec)
                    result.generated_question = question

        return result

    def _formulate_question(self, entry: AutobiographyEntry,
                            resolved_bvec: BVec) -> str:
        """Generate a question for the user based on unresolved tension.

        The question is framed using the dominant BLECD domain —
        the system asks about what it can't resolve on its own.
        """
        domain_questions = {
            "B": f"What are the boundaries or constraints around: {entry.input_text[:80]}?",
            "F": f"What feedback loops or recurring patterns do you see in: {entry.input_text[:80]}?",
            "E": f"What new structure might be emerging from: {entry.input_text[:80]}?",
            "C": f"What's the phase transition or critical threshold in: {entry.input_text[:80]}?",
            "D": f"What's being lost or forgotten about: {entry.input_text[:80]}?",
            "S": f"What's at capacity or saturated regarding: {entry.input_text[:80]}?",
        }
        return domain_questions.get(entry.dominant_domain,
                                     f"Can you help me understand: {entry.input_text[:80]}?")

    def _run_research(self, input_text: str):
        """Run the research cascade for an unresolved tension (Tier 4.3).

        Bridges to the async cascade safely whether or not an event loop is
        already running, and never raises (returns None on any failure).
        """
        query = input_text[:120]
        for prefix in ("tell me about", "what is", "explain", "how does",
                       "why is", "can you", "i think", "i wonder"):
            if query.lower().startswith(prefix):
                query = query[len(prefix):].strip()
                break
        try:
            try:
                asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(
                        asyncio.run, research_cascade.gather(query)
                    ).result(timeout=40)
            except RuntimeError:
                return asyncio.run(research_cascade.gather(query))
        except Exception as e:
            logger.warning(f"[Dream Research] failed (non-fatal): {e}")
            return None

    def get_and_clear_questions(self) -> List[str]:
        """Return pending questions AND clear the queue (Remediation Tier 2.3).

        Each question is served exactly once so the UI stops re-displaying
        questions that were already surfaced.
        """
        questions = list(self.pending_questions)
        self.pending_questions.clear()
        return questions
