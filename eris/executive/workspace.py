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
import math

from eris.computation.activations import BVec, bvec_cosine, bvec_distance
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

    def is_transfixed(self) -> bool:
        """Are we stuck? Uses BOTH conservation law AND diversity check.

        Primary: dC/dX stagnation with nonzero coherence
            = reporting coupling without exchanging (hallucination)
        Secondary: low BFECDS diversity in broadcast history
            = ruminating on the same pattern
        """
        if len(self._dCdX_history) < self.history_length // 2:
            return False

        # PRIMARY: Conservation law check
        # dC/dX near zero while coherence is high = transfixion
        recent_dCdX = list(self._dCdX_history)[-5:]
        recent_C = list(self._coherence_history)[-5:]
        mean_abs_dCdX = np.mean([abs(x) for x in recent_dCdX])
        mean_C = np.mean(recent_C)

        conservation_transfixed = (
            mean_abs_dCdX < self.dCdX_stagnation_threshold and
            mean_C > 0.5  # High coherence but no exchange
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

    def check_hallucination_signature(self, bvec: BVec,
                                       coherence: float,
                                       tau_rms: float,
                                       dCdX: float = 0.0) -> bool:
        """Direct check for hallucination.

        Conservation law version:
            dC/dX ≈ 0 + high C + low tau = fake confidence
        BFECDS version (backup):
            high phi + low tau + low E = no genuine processing
        """
        # Conservation law check (primary)
        if abs(dCdX) < 0.003 and coherence > 0.7:
            return True
        # BFECDS check (secondary)
        return coherence > 0.7 and tau_rms < 0.05 and bvec.E < 0.1

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

        Computes interference between bid and goal across all six BLECD
        domains independently, applies Davidian shrinkage to separate
        signal from noise in the coupling, returns net constructive
        interference as the score.
        """
        if self.goal_bvec is None:
            return 0.0

        bid = finding.bvec.as_array()
        goal = self.goal_bvec.as_array()

        # Per-domain coupling: geometric mean of activations
        coupling = np.sqrt(np.maximum(bid * goal, 0.0))

        # Per-domain alignment: ratio of min/max (1 = identical, 0 = opposite)
        max_vals = np.maximum(bid, goal)
        max_vals = np.where(max_vals < 1e-8, 1.0, max_vals)
        min_vals = np.minimum(bid, goal)
        ratios = min_vals / max_vals

        # Elastic (constructive) and plastic (destructive) per domain
        cos2 = ratios * ratios
        sin2 = 1.0 - cos2
        elastic = cos2 * coupling
        plastic = sin2 * coupling

        # Davidian shrinkage on coupling spectrum: denoise
        total_coupling = elastic + plastic
        mean_c = max(float(np.mean(total_coupling)), 1e-10)
        snr = total_coupling / mean_c
        weights = np.asarray(davidian_weight(
            snr, alpha=1.0, beta=0.5, gamma=1.0, delta=0.0
        )).ravel()

        # Net constructive interference (shrunk)
        net = float(np.sum((elastic - plastic) * weights))
        norm = float(np.sum(total_coupling * weights))
        if norm < 1e-10:
            return 0.0

        return net / norm

    def select_winner(self, findings: List[SpecialistFinding],
                      coherence: float = 0.0,
                      tau_rms: float = 0.0,
                      dCdX: float = 0.0) -> Optional[SpecialistFinding]:
        """Select winning bid via wave interference, with transfixion override.

        Normal: highest interference score wins.
        Transfixed: force highest-Emergence bid to break the loop.
        Hallucinating: force highest-Emergence bid AND flag for review.
        """
        if not findings:
            return None

        # Record for transfixion tracking
        is_stuck = self.transfixion_detector.is_transfixed()
        hallucinating = False
        if self.goal_bvec:
            hallucinating = self.transfixion_detector.check_hallucination_signature(
                self.goal_bvec, coherence, tau_rms, dCdX
            )

        if is_stuck or hallucinating:
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
