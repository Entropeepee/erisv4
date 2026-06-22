"""Shared noise-floor estimator (Tier 1 — the foundation every gate sits on).

ERIS_ORCHESTRATION_REMEDIATION.md §4. The CIP calls for "a single sigma used by
a plurality of monitors." Implemented naively as ONE float σ for every gate this
would break Eris, because the gated signals are different physical quantities at
wildly different scales (a bvec-distance dissonance ~0.1–1.0; a dC/dX ~1e-3; a
per-step coherence delta ~1e-4). One literal σ cannot serve them.

The correct design, implemented here, is:

  • PER-SIGNAL LOCAL SCALE — a registry keyed by signal name; each signal is
    z-scored against ITS OWN robust running scale (EMA mean/var, reused from
    sgt.update_ema). This preserves correctness across scales.

  • SHARED POLICY — one `k` (threshold in σ), one warmup policy, one robust-scale
    method, applied uniformly. This is where the CIP's "amortized + mutually
    consistent" benefit lives.

  • ONE SHARED GLOBAL-AGITATION MULTIPLIER `g` — a single scalar derived from a
    whole-field quantity (normalized tau_rms, or the z of mean |dC/dX|) that
    multiplies EVERY gate's effective threshold together. When the whole field is
    turbulent, all gates become conservative IN LOCKSTEP — the genuine "shared σ"
    coupling that kills the race-condition class the CIP names (one gate thinking
    it's quiet while another thinks it's noisy).

So: shared instance + shared policy + shared global multiplier, per-signal local
scale. Do not "simplify" this to one global σ — that is the documented trap.

Tier 1 wires NOTHING into the live pipeline; this is the substrate the gates in
Tier 2+ draw from. With no estimator injected, SGTGate behaves byte-identically
to before (see sgt.SGTGate).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

from eris.computation.sgt import update_ema


@dataclass
class _SignalScale:
    """Per-signal robust running scale (EMA mean/var).

    `mean`/`var` are seeded from the FIRST observation (mean=value, var=0) so a
    tiny-scale signal (e.g. dC/dX ~ 1e-3) isn't swamped by a generic var=1.0
    prior for dozens of steps — its scale tracks its own magnitude immediately.
    """
    mean: float = 0.0
    var: float = 0.0
    n: int = 0


class NoiseFloorEstimator:
    """One shared instance, injected into many monitors/gates.

    Each gate calls `observe(name, value)` for its own signal and gets a SIGNED
    z-score against that signal's own scale. Separately, whole-field agitation is
    fed via `observe_global(agitation)`; the resulting multiplier `g` (read with
    `global_multiplier()`) scales every gate's threshold together.
    """

    def __init__(self, k: float = 2.5, ema_alpha: float = 0.1,
                 warmup: int = 10, g_min: float = 1.0, g_max: float = 3.0):
        self.k = k                      # shared threshold (σ) — the policy
        self.ema_alpha = ema_alpha      # shared robust-scale smoothing
        self.warmup = warmup            # shared warmup policy
        self.g_min = g_min
        self.g_max = g_max
        self._signals: Dict[str, _SignalScale] = {}
        # Global agitation state -> the shared multiplier g.
        self._g: float = 1.0
        self._agit = _SignalScale()

    # ── Per-signal local scale ────────────────────────────────────────────
    def observe(self, name: str, value: float) -> float:
        """Update `name`'s local scale with `value`; return its SIGNED z-score.

        Sign carries direction: a settling gate reads a low-side (negative)
        outlier as "settled"; an anomaly gate reads a high-side (positive)
        outlier as "escalate". During this signal's warmup the z is reported
        but callers should treat warmup as non-actionable (the monitor does)."""
        s = self._signals.get(name)
        if s is None:
            # Seed scale from the first value (mean=value, var=0); z is 0 here.
            self._signals[name] = _SignalScale(mean=value, var=0.0, n=1)
            return 0.0
        # Judge the value against the ESTABLISHED scale (history) FIRST — so a
        # genuine outlier reads as one — then fold it into the running scale.
        # (Updating first would let a big value inflate its own variance and
        # mask itself; that is the standard EMA-z pitfall.)
        std = max(s.var ** 0.5, 1e-10)
        z = (value - s.mean) / std
        s.n += 1
        s.mean, s.var = update_ema(value, s.mean, s.var, self.ema_alpha)
        return z

    def z(self, name: str, value: float) -> float:
        """SIGNED z-score of `value` against `name`'s current scale (no update)."""
        s = self._signals.get(name)
        if s is None:
            return 0.0
        std = max(s.var ** 0.5, 1e-10)
        return (value - s.mean) / std

    def observations(self, name: str) -> int:
        s = self._signals.get(name)
        return s.n if s else 0

    def in_warmup(self, name: str) -> bool:
        return self.observations(name) < self.warmup

    # ── Shared global agitation multiplier ────────────────────────────────
    def observe_global(self, agitation: float) -> float:
        """Feed a whole-field agitation scalar (e.g. normalized tau_rms or mean
        |dC/dX|). Returns the updated multiplier g in [g_min, g_max].

        Turbulent field (agitation a high outlier vs its own history) -> g > 1 ->
        every gate's threshold rises -> all gates conservative in lockstep.
        During global warmup g stays at g_min (=1.0): no coupling until there is
        a history to judge "turbulent" against."""
        if self._agit.n == 0:
            self._agit = _SignalScale(mean=agitation, var=0.0, n=1)
            self._g = self.g_min
            return self._g
        # z against established agitation history (before folding this one in).
        std = max(self._agit.var ** 0.5, 1e-10)
        z = (agitation - self._agit.mean) / std       # signed; >0 means turbulent
        self._agit.n += 1
        self._agit.mean, self._agit.var = update_ema(
            agitation, self._agit.mean, self._agit.var, self.ema_alpha)
        if self._agit.n <= self.warmup:
            self._g = self.g_min
            return self._g
        self._g = float(min(self.g_max, max(self.g_min, self.g_min + max(0.0, z))))
        return self._g

    def global_multiplier(self) -> float:
        """Current shared multiplier g. 1.0 until the field has proven turbulent."""
        return self._g

    def reset(self) -> None:
        self._signals.clear()
        self._g = 1.0
        self._agit = _SignalScale()
