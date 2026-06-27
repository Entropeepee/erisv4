"""Two-channel field coupling (RULE 2) + GLNCS debias + SGT unknown-gate.

The glossary defines the two channels explicitly (and the glossary wins over any
paper): elastic = cos²(Δθ)·coupling (in-phase resonance), plastic = sin²(Δθ)·coupling
(out-of-phase tension). Using cos-only (`field_resonance`) silently drops the sine
half and collapses the field to cross-modal cosine — so we always compute BOTH, plus
the phase statistic mean|sin(Δθ)| (RULE 2: "include sine" in three places — the
plastic channel, the τ torsion cross-product, and the phase stats).
"""
from __future__ import annotations
from typing import List, Tuple

import numpy as np

from eris.computation.shrinkage import davidian_weight

_EPS = 1e-20


def field_coupling(mag_a, theta_a, mag_b, theta_b) -> Tuple[float, float, float]:
    """(elastic, plastic, sine_stat) for two (mag=√ρ, θ) fields.

    elastic = mean(magₐ·mag_b · cos²Δθ) / norm
    plastic = mean(magₐ·mag_b · sin²Δθ) / norm
    sine_stat = mean|sinΔθ|   (the phase witness, RULE 2)
    """
    ma = np.asarray(mag_a, dtype=np.float64); ta = np.asarray(theta_a, dtype=np.float64)
    mb = np.asarray(mag_b, dtype=np.float64); tb = np.asarray(theta_b, dtype=np.float64)
    dth = ta - tb
    coupling = ma * mb
    norm = np.sqrt(max(np.mean(ma ** 2) * np.mean(mb ** 2), _EPS))
    cos2 = np.cos(dth) ** 2
    elastic = float(np.mean(coupling * cos2) / norm)
    plastic = float(np.mean(coupling * (1.0 - cos2)) / norm)
    sine_stat = float(np.mean(np.abs(np.sin(dth))))
    return elastic, plastic, sine_stat


def coupling_score(mag_a, theta_a, mag_b, theta_b) -> float:
    """Net field coupling: elastic − plastic (RULE 2 — both channels, never cos-only)."""
    e, p, _ = field_coupling(mag_a, theta_a, mag_b, theta_b)
    return e - p


class FieldDebias:
    """GLNCS debias of fields — strip the dataset-wide (cross-class) nuisance modes
    (overall brightness/DC/shared texture), keeping class-discriminative structure.
    Optional: the spec warns 'if accuracy craters, suspect over-annihilation'."""

    def __init__(self, size: int, bias_fraction: float = 0.05):
        from eris.retrieval.glncs_filter import GLNCSFilter
        self.size = size
        self.bias_fraction = bias_fraction
        self._filter = GLNCSFilter(input_dim=size * size)
        self.fitted = False

    def fit(self, mags: List[np.ndarray]) -> "FieldDebias":
        X = np.asarray([np.asarray(m, dtype=np.float32).ravel() for m in mags])
        if X.shape[0] < 2:
            return self                     # nothing to debias against
        self._filter.calibrate(X, bias_fraction=self.bias_fraction)
        self.fitted = True
        return self

    def apply(self, mag: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return mag
        v = self._filter.apply(np.asarray(mag, dtype=np.float32).ravel())
        return np.asarray(v, dtype=np.float64).reshape(self.size, self.size)


class UnknownGate:
    """'Unknown' gate over per-query top coupling scores: track a running mean/var
    of the best score and call a query `unknown` when its top score falls BELOW the
    noise floor (mean − k·σ). Directional (low = unknown), unlike SGT's |z|; built
    on the same EMA statistics. Returns known during warmup (stats not yet stable)."""

    def __init__(self, threshold_sigma: float = 1.0, warmup: int = 8, alpha: float = 0.2):
        from eris.computation.sgt import update_ema
        self._ema = update_ema
        self.k = threshold_sigma
        self.warmup = warmup
        self.alpha = alpha
        self.mean = 0.0
        self.var = 1.0
        self.n = 0

    def is_known(self, top_score: float) -> bool:
        top_score = float(top_score)
        self.n += 1
        if self.n <= self.warmup:
            self.mean, self.var = self._ema(top_score, self.mean, self.var, self.alpha)
            return True
        std = max(self.var ** 0.5, 1e-9)
        known = top_score >= (self.mean - self.k * std)
        self.mean, self.var = self._ema(top_score, self.mean, self.var, self.alpha)
        return known
