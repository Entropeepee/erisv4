"""
Global Predictive Workspace (GPW) — Wave Interference Architecture
====================================================================

The original question: "What if we wanted to transmit or encode knowledge
in a non-binary way?" A resonant architecture, like the dC/dX law.

In a decentralized network (or locally with MoE), waves of information
interact and a democratic system answers to maximize Coherence:
    - Two YES bids = constructive interference → confident affirmation
    - One YES, one NO = destructive interference → undetermined / maybe
    - The DEGREE of interference maps to dC/dX (conservation law)

Components:
    MoEGate:              Wave interference between specialist bids (NOT cosine)
    TransfixionDetector:  Reads dC/dX directly — the conservation law diagnostic
    SharedCognitiveWorkspace: Single-slot broadcast buffer (GWT)
    GoalNetwork:          Top-down objectives from user intent + BFECDS

Hallucination = broken coupling geometry (information theory framework):
    dC/dX ≈ 0 with nonzero C and nonzero input = system reports
    coherence without exchanging anything with reality.

Copyright 2026 Terminus IP Group LLC.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from collections import deque
import numpy as np
from eris.config import to_numpy, xp
import math

from eris.computation.activations import BVec, bvec_cosine, bvec_distance, bvec_resonance
from eris.computation.sgt import SGTGate
from eris.computation.shrinkage import davidian_weight
from eris.tribe.specialists import SpecialistFinding


DOMAIN_NAMES = ["B", "F", "E", "C", "D", "S"]


@dataclass
class Broadcast:
    """A single broadcast in the workspace."""
    thought: str
    bvec: BVec
    source: str
    timestamp: float = 0.0
    coherence: float = 0.0
    dCdX: float = 0.0


class TransfixionDetector:
    """Detects when the system is stuck — reads dC/dX directly.

    The conservation law diagnostic (from dCdX derivation):
        dC/dX = f(BLECD) + T_eff

    Three regimes (from information theory session):
        Elastic:          |dC/dX| moderate → smooth processing
        Plastic:          |dC/dX| large → genuine restructuring / learning
        Transfixed:       dC/dX ≈ 0 WITH nonzero C → fake confidence

    The hallucination signature is NOT just "low diversity in broadcasts."
    It's the conservation law telling us: coherence is being reported
    without any exchange happening. The system thinks it knows but
    hasn't actually coupled to the input.

    Also monitors BFECDS diversity as a secondary signal.
    """

    def __init__(self, history_length: int = 10,
                 diversity_threshold: float = 0.15,
                 dCdX_stagnation_threshold: float = 0.005):
        self.history_length = history_length
        self.diversity_threshold = diversity_threshold
        self.dCdX_stagnation_threshold = dCdX_stagnation_threshold

        self._bvec_history: deque[BVec] = deque(maxlen=history_length)
        self._dCdX_history: deque[float] = deque(maxlen=history_length)
        self._coherence_history: deque[float] = deque(maxlen=history_length)

        # SGT gate for dC/dX stagnation detection
        self._stagnation_gate = SGTGate(threshold_sigma=1.5, ema_alpha=0.1)

    def record(self, bvec: BVec, dCdX: float = 0.0,
               coherence: float = 0.0) -> None:
        """Record a broadcast's metrics."""
        self._bvec_history.append(bvec)
        self._dCdX_history.append(dCdX)
        self._coherence_history.append(coherence)

    def is_transfixed(self, field: Optional[Any] = None, input_text: str = "") -> bool:
        """Are we stuck? Uses physical reactivity probe if field provided, else falls back to history.

        Primary (v4): The BCDC Reactivity Probe directly measures input-reactivity.
        Secondary: Conservation law and BFECDS diversity checks.
        """
        if field is not None and hasattr(field, 'probe_reactivity') and input_text:
            # Physical transfixion test: measure actual input-reactivity
            reactivity = field.probe_reactivity(input_text=input_text, steps=12)
            div = reactivity["field_divergence"]
            dC = reactivity["coherence_response"]
            # If the field absorbs actual new input but doesn't change state
            # or coherence significantly, it's transfixed (stuck in an attractor).
            return div < 0.05 and dC < 0.01

        if len(self._dCdX_history) < self.history_length // 2:
            return False

        # PRIMARY (Tier 1.3): the SGT stagnation gate is the decision authority,
        # not a hardcoded `dCdX_stagnation_threshold`. SGT z-scoring is
        # scale-adaptive (works whether C is 0.04 or 0.8), so the detector no
        # longer depends on magic constants tuned for a since-replaced engine.
        # Transfixion = the latest |dC/dX| is a significant LOW outlier
        # (stagnation) while coherence sits at/above its own median.
        cur = abs(self._dCdX_history[-1])
        gate_open, _z = self._stagnation_gate.update(cur)
        c_hist = list(self._coherence_history)
        c_now = c_hist[-1] if c_hist else 0.0
        c_med = float(np.median(c_hist)) if c_hist else 0.0
        conservation_transfixed = (
            gate_open
            and cur < self._stagnation_gate.running_mean  # LOW outlier, not high
            and c_now >= c_med                              # coherence not collapsing
        )

        # SECONDARY: BFECDS diversity check
        if len(self._bvec_history) >= self.history_length // 2:
            vecs = [b.as_array() for b in self._bvec_history]
            arr = np.array(vecs)
            n = len(arr)
            total_dist = sum(
                float(np.linalg.norm(arr[i] - arr[j]))
                for i in range(n) for j in range(i + 1, n)
            )
            count = max(n * (n - 1) // 2, 1)
            diversity_transfixed = (total_dist / count) < self.diversity_threshold
        else:
            diversity_transfixed = False

        return conservation_transfixed or diversity_transfixed

    def check_empty_confidence_signature(self, bvec: BVec,
                                          coherence: float,
                                          tau_rms: float,
                                          dCdX: float = 0.0) -> bool:
        """Internal-metacognition signal: high coherence with no exchange.

        Remediation Tier 1.1 — this was named `check_hallucination_signature`,
        but it does NOT detect factual hallucination. A fluent fabrication is
        internally coherent by construction, so field coherence cannot tell you
        whether a claim matches the world (that is a GROUNDING failure, caught in
        Tier 4). What this DOES detect is "empty confidence": the field reports
        coherence while exchanging nothing with the input. Route its output to
        internal metacognition (nudge the dream loop / re-query) — never to a
        user-facing "this is a hallucination" claim.

        Conservation-law form:  |dC/dX| ~ 0 + high C            -> empty confidence
        BFECDS backup form:     high C + low tau + low Emergence -> no processing
        """
        if abs(dCdX) < 0.003 and coherence > 0.7:
            return True
        return coherence > 0.7 and tau_rms < 0.05 and bvec.E < 0.1

    # Backwards-compatible alias (deprecated name; do not re-bolt as a fact-checker).
    def check_hallucination_signature(self, bvec: BVec, coherence: float,
                                      tau_rms: float, dCdX: float = 0.0) -> bool:
        return self.check_empty_confidence_signature(bvec, coherence, tau_rms, dCdX)

    def reset(self) -> None:
        self._bvec_history.clear()
        self._dCdX_history.clear()
        self._coherence_history.clear()


class MoEGate:
    """Mixture-of-Experts Gate — wave interference between specialist bids.

    NOT cosine similarity. Uses the CSBA coupling geometry from the
    interference module: per-domain elastic/plastic decomposition
    with Davidian Hill-Power shrinkage on the coupling spectrum.

    The wave interference model:
        Multiple bids → compute pairwise interference → constructive
        regions = agreement → destructive regions = conflict →
        the bid with highest constructive interference with the goal
        AND lowest destructive interference with other bids wins.

    This is the local version of the decentralized network protocol:
        Each specialist is a "node" broadcasting a field state.
        The MoEGate is the interference-based consensus mechanism.
    """

    def __init__(self):
        self.goal_bvec: Optional[BVec] = None
        self.goal_text: str = ""
        self.transfixion_detector = TransfixionDetector()

    def set_goal(self, bvec: BVec, text: str = "") -> None:
        self.goal_bvec = bvec
        self.goal_text = text

    def score_bid(self, finding: SpecialistFinding) -> float:
        """Score a bid using per-domain coupling geometry (not single cosine).

        The κ/λ = cos/sin elastic−plastic decomposition with Davidian shrinkage now lives
        in the shared `bvec_resonance` helper (§B3), so the gate, the cross-attention hub,
        and specialist activation all use ONE field metric instead of three. The math is
        unchanged (verified identical to the prior inline form to float precision)."""
        if self.goal_bvec is None:
            return 0.0
        return bvec_resonance(finding.bvec, self.goal_bvec)

    def select_winner(self, findings: List[SpecialistFinding],
                      coherence: float = 0.0,
                      tau_rms: float = 0.0,
                      dCdX: float = 0.0,
                      field: Optional[Any] = None,
                      input_text: str = "") -> Optional[SpecialistFinding]:
        """Select winning bid via wave interference, with transfixion override.

        Normal: highest interference score wins.
        Transfixed: force highest-Emergence bid to break the loop.
        Hallucinating: force highest-Emergence bid AND flag for review.

        In CIP terms the transfixion override IS a `SWITCH` decision — when the
        field is stuck, switch the selection mechanism (interference score ->
        highest-Emergence) to break the loop. It deliberately keeps its own
        physical reactivity probe (richer than a dC/dX z-score) rather than being
        rewired through CriticalityMonitor, so winner selection stays behavior-
        preserving; the two share the SWITCH vocabulary, not duplicated logic.
        """
        if not findings:
            return None

        # Record for transfixion tracking
        is_stuck = self.transfixion_detector.is_transfixed(field=field, input_text=input_text)
        empty_confidence = False
        if self.goal_bvec:
            empty_confidence = self.transfixion_detector.check_empty_confidence_signature(
                self.goal_bvec, coherence, tau_rms, dCdX
            )

        if is_stuck or empty_confidence:
            # Override: inject novelty to break the attentional loop
            winner = max(findings, key=lambda f: f.bvec.E)
        else:
            # Normal: wave interference scoring
            scored = [(f, self.score_bid(f)) for f in findings]
            scored.sort(key=lambda x: x[1], reverse=True)
            winner = scored[0][0]

        # Record broadcast metrics
        self.transfixion_detector.record(winner.bvec, dCdX, coherence)
        return winner


class SharedCognitiveWorkspace:
    """Single-slot broadcast buffer. Only one thought at a time.

    Implements Global Workspace Theory (Baars/Dehaene).
    Winner gets broadcast to all specialists, updating their context.
    """

    def __init__(self):
        self.current: Optional[Broadcast] = None
        self._history: deque[Broadcast] = deque(maxlen=100)
        self._listeners: List[Any] = []

    def broadcast(self, thought: str, bvec: BVec, source: str,
                  coherence: float = 0.0, dCdX: float = 0.0) -> Broadcast:
        """Push a new thought. Notifies all listeners."""
        import time
        b = Broadcast(thought=thought, bvec=bvec, source=source,
                      timestamp=time.time(), coherence=coherence, dCdX=dCdX)
        self.current = b
        self._history.append(b)

        for listener in self._listeners:
            if hasattr(listener, "on_broadcast"):
                listener.on_broadcast(b)

        return b

    def add_listener(self, listener) -> None:
        self._listeners.append(listener)

    @property
    def history(self) -> List[Broadcast]:
        return list(self._history)


class GoalNetwork:
    """Top-down objective tracking. Keeps specialists on task."""

    def __init__(self):
        self._stack: List[Tuple[str, BVec]] = []

    def set_goal(self, text: str, bvec: BVec) -> None:
        self._stack = [(text, bvec)]

    def push_subgoal(self, text: str, bvec: BVec) -> None:
        self._stack.append((text, bvec))

    def pop_subgoal(self) -> Optional[Tuple[str, BVec]]:
        if len(self._stack) > 1:
            return self._stack.pop()
        return None

    @property
    def active_goal(self) -> Optional[Tuple[str, BVec]]:
        return self._stack[-1] if self._stack else None

    @property
    def active_bvec(self) -> Optional[BVec]:
        return self._stack[-1][1] if self._stack else None
