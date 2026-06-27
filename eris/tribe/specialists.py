"""
The Tribe of Eleven Specialists
=================================

Parallel domain experts that compete for attention via the MoEGate.
Each specialist generates findings with computed BFECDS vectors.
Findings are posted to the Cross-Attention Hub for cross-pollination.

The eleven:
    Logos      — Logic, formal reasoning, mathematical structure
    Mythos     — Philosophy, meaning, symbolic interpretation
    Praxis     — Physics, engineering, material processes
    Elos       — Adversarial skeptic, red team, devil's advocate
    Chronos    — History, temporal patterns, precedent
    Anthropos  — Sociology, human dynamics, culture
    Ploutos    — Finance, economics, resource allocation
    Eros       — Neuroscience, embodiment, somatic intelligence
    Aesthetes  — Art, beauty, pattern recognition, aesthetic judgment
    Techne     — Technology, implementation, systems engineering
    Kairos     — Synthesis, timing, integration across domains

SGT-gated activation: specialists only fire when their domain
relevance exceeds the noise floor. Not every specialist speaks
on every input — only those whose BFECDS alignment with the
current goal is statistically significant.

Research triggering: when C > 0.4 AND E > 0.2, the two-cycle
research engine activates (broad → synthesis → refined → canonize).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
import re
import time

from eris.computation.activations import BVec, bvec_cosine, bvec_resonance
from eris.computation.sgt import SGTGate

Model = Callable[[str], str]            # prompt -> text (injected; stub in tests)
_WORD = re.compile(r"[a-z0-9]{3,}")


@dataclass
class SpecialistFinding:
    """A single finding from a specialist, with computed BFECDS."""
    specialist_id: str
    content: str
    bvec: BVec
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Specialist:
    """A domain expert in the Tribe."""
    id: str
    name: str
    domain: str
    description: str
    # Which BLECD domains this specialist is most sensitive to
    sensitivity_bvec: BVec = field(default_factory=BVec)
    activation_gate: SGTGate = field(default_factory=lambda: SGTGate(threshold_sigma=1.5))

    def should_activate(self, goal_bvec: BVec) -> bool:
        """SGT-gated activation: only fire if relevance exceeds the noise floor. Relevance
        is the full field metric (κ/λ cos+sin coupling, §B3) — the same one the MoEGate and
        hub use — not a plain cosine (§0.3)."""
        relevance = bvec_resonance(self.sensitivity_bvec, goal_bvec)
        should_fire, _ = self.activation_gate.update(relevance)
        # The SGT gate (adaptive z-score) is the primary, scale-free authority. The absolute
        # fallback is calibrated to the RESONANCE scale (≈[-0.8,+0.5], the relevant domains
        # cluster ~0.3-0.5), NOT cosine's compressed ~[0.8,0.99] where >0.6 fired for nearly
        # everything. 0.3 keeps a sparse cold-start set (the top few domains), not all eleven.
        return should_fire or relevance > 0.3


# The Eleven
TRIBE = [
    Specialist("logos", "Logos", "logic",
               "Formal reasoning, mathematical structure, proof",
               sensitivity_bvec=BVec(B=0.6, F=0.7, E=0.3, C=0.2, D=0.1, S=0.3)),
    Specialist("mythos", "Mythos", "philosophy",
               "Meaning, symbolic interpretation, metaphysics",
               sensitivity_bvec=BVec(B=0.3, F=0.4, E=0.7, C=0.5, D=0.3, S=0.2)),
    Specialist("praxis", "Praxis", "physics",
               "Physical processes, engineering, material science",
               sensitivity_bvec=BVec(B=0.5, F=0.5, E=0.4, C=0.6, D=0.3, S=0.4)),
    Specialist("elos", "Elos", "adversarial",
               "Skeptic, red team, devil's advocate, falsification",
               sensitivity_bvec=BVec(B=0.2, F=0.2, E=0.3, C=0.8, D=0.6, S=0.1)),
    Specialist("chronos", "Chronos", "history",
               "Temporal patterns, precedent, historical context",
               sensitivity_bvec=BVec(B=0.4, F=0.6, E=0.2, C=0.3, D=0.5, S=0.5)),
    Specialist("anthropos", "Anthropos", "sociology",
               "Human dynamics, culture, social systems",
               sensitivity_bvec=BVec(B=0.5, F=0.7, E=0.5, C=0.3, D=0.3, S=0.4)),
    Specialist("ploutos", "Ploutos", "finance",
               "Economics, resource allocation, value",
               sensitivity_bvec=BVec(B=0.7, F=0.4, E=0.3, C=0.4, D=0.2, S=0.7)),
    Specialist("eros", "Eros", "neuroscience",
               "Embodiment, somatic intelligence, sensation",
               sensitivity_bvec=BVec(B=0.3, F=0.6, E=0.6, C=0.4, D=0.3, S=0.3)),
    Specialist("aesthetes", "Aesthetes", "art",
               "Beauty, pattern recognition, aesthetic judgment",
               sensitivity_bvec=BVec(B=0.2, F=0.3, E=0.8, C=0.3, D=0.2, S=0.4)),
    Specialist("techne", "Techne", "technology",
               "Implementation, systems engineering, tooling",
               sensitivity_bvec=BVec(B=0.6, F=0.5, E=0.5, C=0.3, D=0.2, S=0.6)),
    Specialist("kairos", "Kairos", "synthesis",
               "Integration, timing, cross-domain connection",
               sensitivity_bvec=BVec(B=0.3, F=0.5, E=0.7, C=0.5, D=0.2, S=0.3)),
]


class CrossAttentionHub:
    """Shared workspace where specialists post and query findings.

    Findings are stored with their BFECDS vectors, enabling
    cross-pollination: a Praxis finding about phase transitions
    might resonate with a Mythos finding about transformation.
    """

    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self._findings: List[SpecialistFinding] = []

    def post(self, finding: SpecialistFinding) -> None:
        self._findings.append(finding)
        if len(self._findings) > self.capacity:
            self._findings = self._findings[-self.capacity:]

    def query(self, bvec: BVec, top_k: int = 5) -> List[SpecialistFinding]:
        """Find findings most resonant with the query BFECDS — the full κ/λ cos+sin field
        metric (§B3), so cross-pollination uses interference, not plain cosine (§0.2)."""
        scored = [(f, bvec_resonance(bvec, f.bvec)) for f in self._findings]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in scored[:top_k]]

    def clear(self) -> None:
        self._findings.clear()

    @property
    def size(self) -> int:
        return len(self._findings)


def get_active_specialists(goal_bvec: BVec, max_k: int = 5) -> List[Specialist]:
    """Active specialists for this goal — TRIBE-RELATIVE (top-K by field resonance),
    which is both scale-free (the resonance metric's absolute level depends on how
    saturated the goal bvec is, so a fixed threshold can't work) and the §2 cost guard:
    cap the active set at `max_k` so a deep cycle never runs all eleven.

    A specialist activates if it is a temporal outlier (its own SGT gate fires) OR it is
    above the tribe's average resonance to this goal; the set is capped at max_k and is
    never empty (the single best always speaks)."""
    scored = []
    for s in TRIBE:
        rel = bvec_resonance(s.sensitivity_bvec, goal_bvec)
        gate_fire = s.should_activate(goal_bvec)        # updates the per-specialist SGT gate
        scored.append((rel, gate_fire, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    mean_rel = sum(r for r, _, _ in scored) / max(1, len(scored))
    active = [s for rel, gate_fire, s in scored
              if gate_fire or rel >= mean_rel][:max(1, max_k)]
    return active or [scored[0][2]]


def should_trigger_research(bvec: BVec) -> bool:
    """Research fires when C > 0.4 AND E > 0.2 — genuine criticality
    combined with emergence signal. Without both, log and move on."""
    return bvec.C > 0.4 and bvec.E > 0.2


def make_field_finding(specialist: Specialist, field_bvec: BVec) -> SpecialistFinding:
    """Build a specialist finding from the FIELD signature (Remediation Tier 3-A).

    The previous build set every finding's content to
    ``f"[{name}] Analysis of: {user_message[:100]}"`` — it echoed the user's
    words with a name attached and added nothing, so the MoEGate scored
    placeholder text.

    Here the finding IS the specialist's field signature: the live field BFECDS
    projected onto the specialist's domain sensitivity (element-wise). The
    projected vector is the bid, its magnitude is the bid strength, and its
    dominant domains name what the specialist 'sees'. Free at runtime (no LLM
    call), fast on CPU, and it makes the MoEGate's wave-interference selection
    operate on real field projections instead of placeholder echoes.
    """
    bid = field_bvec.elementwise(specialist.sensitivity_bvec)
    strength = bid.magnitude()
    top = bid.dominant_domains(k=2)
    text = f"{specialist.name}: {strength:.3f} bid on {'+'.join(top)}"
    return SpecialistFinding(
        specialist_id=specialist.id,
        content=text,
        bvec=bid,
        confidence=strength,
    )


# ── §A1: substantive reasoned findings (deep cycles only, local-model) ───────
def _trigrams(text: str) -> set:
    toks = _WORD.findall((text or "").lower())
    return {tuple(toks[i:i + 3]) for i in range(len(toks) - 2)} if len(toks) >= 3 else set()


def _echo_overlap(reasoning: str, source_text: str) -> float:
    """Fraction of the reasoning's trigrams that merely echo the source text — the
    'placeholder echo of the user's words' failure mode the field-finding fix was
    fighting. High overlap ⇒ the specialist restated instead of reasoning."""
    r = _trigrams(reasoning)
    if not r:
        return 0.0
    return len(r & _trigrams(source_text)) / len(r)


def _text_to_bvec(text: str, field_size: int = 32):
    """Compute a BFECDS vector from the reasoning via the real computed-activations
    bridge (TextFrontend seeds a field, the PDE evolves it, the field is measured) — so
    the MoEGate scores the REASONING's field signature, not a label. Lazy import keeps
    the tribe module light and avoids an import cycle."""
    from eris.knowledge.frontends import TextFrontend
    return TextFrontend().to_bvec(text, size=field_size)


def make_reasoned_finding(specialist: Specialist, goal: str, retrieved_context: str,
                          model: Model, *, bvec_fn: Optional[Callable[[str], BVec]] = None,
                          field_size: int = 32, max_regens: int = 1) -> SpecialistFinding:
    """A genuine domain-lens finding (§A1) — the substantive mode for DEEP cycles only.

    The active specialist reasons through its domain over the RETRIEVED CONTEXT via the
    injected local model, under the truth-contract (grounded reflection, no fabricated
    human autobiography). Its content is the reasoning; its bvec is computed FROM that
    reasoning, so the MoEGate scores real analysis instead of a `"<Name>: <strength> bid"`
    label. Guards: regenerate once if the truth-contract backstop trips or the output just
    echoes the goal; if it still echoes, flag it and down-weight (low confidence) so the
    gate never lets an echo win. Local-model only (the per-turn cost that broke it before
    is avoided by gating this to deep cycles + the top-K active cap)."""
    from eris.metacognition.truth_contract import PONDER_CONTRACT, fabricated_self
    context = (retrieved_context or "").strip() or "(no sources retrieved)"
    base = (
        f"{PONDER_CONTRACT}\n\n"
        f"You are {specialist.name}, the {specialist.domain} specialist "
        f"({specialist.description}). Through your domain lens ONLY, give ONE concrete, "
        f"specific finding about the topic below, grounded STRICTLY in the provided "
        f"sources. Do NOT restate the question or the sources verbatim — add {specialist.domain} "
        f"analysis the sources support. 2-4 sentences.\n\n"
        f"TOPIC: {goal}\n\nSOURCES:\n{context}\n\n{specialist.name}'s finding:"
    )
    reasoning, echoed = "", False
    for attempt in range(max_regens + 1):
        prompt = base if attempt == 0 else (
            base + "\n\n(Your last attempt echoed the prompt. Give ORIGINAL domain analysis.)")
        try:
            reasoning = (model(prompt) or "").strip()
        except Exception:
            reasoning = ""
        if not reasoning:
            continue
        if fabricated_self(reasoning):                 # truth-contract backstop
            base = base + "\n\n(Reflect as Eris — do not invent a human past.)"
            continue
        echoed = _echo_overlap(reasoning, f"{goal}\n{context}") > 0.5
        if not echoed:
            break
    bvec = (bvec_fn or (lambda t: _text_to_bvec(t, field_size)))(reasoning or specialist.name)
    return SpecialistFinding(
        specialist_id=specialist.id,
        content=reasoning or f"({specialist.name}: no grounded finding)",
        bvec=bvec,
        confidence=0.1 if (echoed or not reasoning) else float(bvec.magnitude()),
        metadata={"mode": "reasoned", "grounded": True, "echo": echoed,
                  "empty": not bool(reasoning)},
    )
