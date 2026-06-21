"""Criticality monitor — the four-decision boundary interface (Tier 1).

ERIS_ORCHESTRATION_REMEDIATION.md §4.3. A monitor instruments ONE boundary
between computational stages. It draws its noise floor from the SHARED
`NoiseFloorEstimator` (per-signal local scale + one global agitation multiplier),
and decides whether continuing the next stage is worth it:

    CONTINUE  — next stage runs as planned (signal is in-band / still moving)
    SUSPEND   — stop; the current best partial is the answer (settled / deadline)
    SWITCH    — replace the next stage with a cheaper/different mechanism
    ESCALATE  — run the next stage with elevated fidelity / more resources

Not every boundary uses all four (CIP §11.7): the field/response gates are mostly
CONTINUE/SUSPEND (mode="settle"); the router/grounding gates use SWITCH/ESCALATE
(mode="anomaly"). A monitor only emits the decisions its `context` asks for.

Per CIP §0111 ("never silently return a wrong answer"), the monitor emits a
structured `FailureModeReport` — but ONLY on SWITCH/ESCALATE, the decisions that
change the mechanism. CONTINUE and SUSPEND are normal flow and carry no report.

Tier 1 builds and tests this interface; NO gate is wired into the live pipeline
yet. Behavior is unchanged until Tier 2+.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional, Tuple

from eris.computation.noise_floor import NoiseFloorEstimator


class Decision(Enum):
    CONTINUE = auto()    # next stage runs as planned
    SUSPEND = auto()     # stop; current best partial is the answer
    SWITCH = auto()      # replace next stage with a different mechanism
    ESCALATE = auto()    # run next stage with elevated fidelity/resources


@dataclass
class FailureModeReport:
    """Structured account of a mechanism-changing decision (SWITCH/ESCALATE).

    Eris's natural sink for these is metacognition: Tier 5 routes them into the
    dream queue ("gate X switched under high agitation on turn N — revisit?")."""
    monitor_id: str
    specialization: str           # which gate fired (e.g. "field_depth", "router")
    decision: Decision
    reason: str                   # "settled", "stalled", "outlier", "deadline"
    z_score: float
    partial_result: Any = None    # current best (e.g. current bvec)
    recommended_action: str = ""


class CriticalityMonitor:
    """Watches one signal at one boundary, drawing scale from a shared estimator.

    Parameters
    ----------
    monitor_id : str
        Unique id for this monitor instance (for reports/logging).
    estimator : NoiseFloorEstimator
        The SHARED estimator. Per-signal scale + the global agitation multiplier
        both come from here, so monitors stay mutually consistent.
    specialization : str
        Which gate this is ("field_depth", "response_field", "router", ...).
    k : float
        Base threshold in σ. Effective threshold is `k * estimator.g`, so a
        turbulent whole-field state makes every monitor conservative together.
    protected_steps : int
        Protected initialization (CIP "protected init"): the monitor will not
        fire for the first `protected_steps` observations — it only learns scale.
        This is what stops a gate suspending before real work is done.
    """

    def __init__(self, monitor_id: str, estimator: NoiseFloorEstimator,
                 specialization: str, k: float = 2.5, protected_steps: int = 0):
        self.monitor_id = monitor_id
        self.estimator = estimator
        self.specialization = specialization
        self.k = k
        self.protected_steps = protected_steps
        self._step = 0

    def observe(self, signal_name: str, value: float,
                context: Optional[dict] = None
                ) -> Tuple[Decision, Optional[FailureModeReport]]:
        """Feed one observation; return a decision (+ a report iff SWITCH/ESCALATE).

        context keys (all optional):
          mode        : "anomaly" (default) | "settle"
          deadline    : bool — bounded-latency dispatch; force SUSPEND ("deadline")
          escalate_k  : σ multiple above which an anomaly ESCALATEs (default 1.5×k)
          partial     : current best partial result, attached to any report
        """
        context = context or {}
        self._step += 1

        # Protected initialization: learn the signal's scale, never fire early.
        if self._step <= self.protected_steps:
            self.estimator.observe(signal_name, value)
            return Decision.CONTINUE, None

        # Bounded-latency dispatch: a deadline forces us to take the partial now.
        if context.get("deadline"):
            self.estimator.observe(signal_name, value)
            return Decision.SUSPEND, None

        z_signed = self.estimator.observe(signal_name, value)
        # Don't act on a signal whose own scale isn't warmed up yet.
        if self.estimator.in_warmup(signal_name):
            return Decision.CONTINUE, None

        eff_k = self.k * self.estimator.global_multiplier()
        z = abs(z_signed)

        if z < eff_k:
            return Decision.CONTINUE, None     # in-band: keep going

        mode = context.get("mode", "anomaly")

        if mode == "settle":
            # A LOW-side outlier means the trajectory's change has fallen below
            # its own noise floor — it has settled. SUSPEND (no mechanism change,
            # so no report). A high-side outlier means it's still moving: CONTINUE.
            if z_signed <= 0:
                return Decision.SUSPEND, None
            return Decision.CONTINUE, None

        # anomaly mode: a HIGH-side outlier is the actionable one.
        if z_signed > 0:
            escalate_k = context.get("escalate_k", eff_k * 1.5)
            if z >= escalate_k:
                report = FailureModeReport(
                    self.monitor_id, self.specialization, Decision.ESCALATE,
                    "outlier", z_signed, context.get("partial"),
                    "run next stage at elevated fidelity")
                return Decision.ESCALATE, report
            report = FailureModeReport(
                self.monitor_id, self.specialization, Decision.SWITCH,
                "outlier", z_signed, context.get("partial"),
                "switch next stage to a cheaper alternate mechanism")
            return Decision.SWITCH, report

        # Low-side outlier in anomaly mode is not actionable here.
        return Decision.CONTINUE, None

    def reset(self) -> None:
        self._step = 0
