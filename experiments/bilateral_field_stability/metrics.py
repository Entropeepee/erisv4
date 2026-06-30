"""
metrics.py -- operational collapse detection (no vibes, raw numbers logged)
===========================================================================

Three terminal states for an evolving coherence field, per the brief:

  LOCK  (standing-wave / transfixion): the field stops CHANGING in time while
        spatial structure persists. Detector: rolling per-cell temporal variance
        of the readout over a window -> below eps_lock for N consecutive steps.
        (Equivalently: lag-1 autocorrelation of consecutive states -> 1.)

  DEATH (decoherence): spatial structure vanishes. Two sub-modes, either is death:
        FLAT  -> spatial variance of phi -> 0 (and tau-RMS -> 0)
        NOISE -> mean local Kuramoto coherence -> 0 (incoherent hash)

  ALIVE: sustained temporal variation AND persistent spatial structure AND
        bounded (no NaN/Inf, amplitude within ceiling). The healthy interior.

COLLAPSE = LOCK or DEATH (or DIVERGE) reached within T steps.

The monitor logs the full raw time series so every threshold is auditable.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import List, Optional
import numpy as np

from field_core import vorticity, local_coherence


@dataclass
class CollapseThresholds:
    window: int = 30          # temporal-variance window (steps)
    n_consec: int = 30        # consecutive steps a condition must hold to "flag"
    eps_lock: float = 3e-5    # temporal var below this == not changing
    eps_flat: float = 2e-4    # spatial var below this == flat
    eps_noise: float = 0.15   # mean Kuramoto coherence below this == hash/noise
    diverge: float = 1e3      # |phi| above this (or NaN) == diverged


@dataclass
class RunResult:
    outcome: str                       # ALIVE | LOCK | DEATH_FLAT | DEATH_NOISE | DIVERGE
    collapse_step: Optional[int]       # first step collapse detected (None if alive)
    collapsed: bool
    # tail (final-state) summary statistics, raw
    temporal_var_final: float
    spatial_var_final: float
    kuramoto_final: float
    tau_rms_final: float
    phi_max_final: float
    # full series (decimated) for plotting/audit
    t: List[int] = field(default_factory=list)
    temporal_var: List[float] = field(default_factory=list)
    spatial_var: List[float] = field(default_factory=list)
    kuramoto: List[float] = field(default_factory=list)
    tau_rms: List[float] = field(default_factory=list)


class CollapseMonitor:
    """Feed it (phi, theta) each step; it classifies the run when finished."""

    def __init__(self, thr: CollapseThresholds, log_every: int = 5):
        self.thr = thr
        self.log_every = log_every
        self._buf: deque = deque(maxlen=thr.window)
        self._s1 = None   # rolling sum of frames over the window
        self._s2 = None   # rolling sum of squares (for O(N^2)/step temporal var)
        self._lock_run = 0
        self._flat_run = 0
        self._noise_run = 0
        self.collapse_step: Optional[int] = None
        self.outcome: Optional[str] = None
        self.res = RunResult("ALIVE", None, False, 0, 0, 0, 0, 0)

    def observe(self, step: int, phi: np.ndarray, theta: np.ndarray):
        if self.outcome is not None:
            return  # already terminal; keep evolving but stop re-flagging

        # divergence guard
        if not np.all(np.isfinite(phi)) or float(np.max(np.abs(phi))) > self.thr.diverge:
            self.outcome = "DIVERGE"
            self.collapse_step = step
            return

        # rolling-window temporal variance via running sum / sum-of-squares
        # (O(N^2) per step instead of restacking the whole window)
        f = phi.copy()
        if self._s1 is None:
            self._s1 = np.zeros_like(f)
            self._s2 = np.zeros_like(f)
        if len(self._buf) == self.thr.window:
            old = self._buf[0]            # about to be evicted by the deque
            self._s1 -= old
            self._s2 -= old * old
        self._buf.append(f)
        self._s1 += f
        self._s2 += f * f

        spatial_var = float(np.var(phi))
        kur = float(np.mean(local_coherence(theta)))

        if len(self._buf) == self.thr.window:
            w = self.thr.window
            cell_var = self._s2 / w - (self._s1 / w) ** 2
            temporal_var = float(np.mean(np.maximum(cell_var, 0.0)))
        else:
            temporal_var = np.nan

        # tau-RMS only needed for the (decimated) log, not the live checks
        tau_rms = (float(np.sqrt(np.mean(vorticity(phi, theta) ** 2)))
                   if step % self.log_every == 0 else self.res.tau_rms_final)

        # raw logging (decimated)
        if step % self.log_every == 0:
            self.res.t.append(step)
            self.res.temporal_var.append(temporal_var)
            self.res.spatial_var.append(spatial_var)
            self.res.kuramoto.append(kur)
            self.res.tau_rms.append(tau_rms)

        # cache finals
        self.res.temporal_var_final = temporal_var
        self.res.spatial_var_final = spatial_var
        self.res.kuramoto_final = kur
        self.res.tau_rms_final = tau_rms
        self.res.phi_max_final = float(np.max(phi))

        if len(self._buf) < self.thr.window:
            return  # not enough history to judge LOCK yet

        # --- LOCK: temporal var below floor AND structure still present ----- #
        if temporal_var < self.thr.eps_lock and spatial_var > self.thr.eps_flat:
            self._lock_run += 1
        else:
            self._lock_run = 0

        # --- DEATH-FLAT: spatial structure gone ----------------------------- #
        if spatial_var < self.thr.eps_flat:
            self._flat_run += 1
        else:
            self._flat_run = 0

        # --- DEATH-NOISE: phase coherence gone ------------------------------ #
        if kur < self.thr.eps_noise:
            self._noise_run += 1
        else:
            self._noise_run = 0

        nc = self.thr.n_consec
        if self._lock_run >= nc:
            self.outcome, self.collapse_step = "LOCK", step
        elif self._flat_run >= nc:
            self.outcome, self.collapse_step = "DEATH_FLAT", step
        elif self._noise_run >= nc:
            self.outcome, self.collapse_step = "DEATH_NOISE", step

    def finalize(self) -> RunResult:
        self.res.outcome = self.outcome or "ALIVE"
        self.res.collapse_step = self.collapse_step
        self.res.collapsed = self.outcome is not None
        return self.res
