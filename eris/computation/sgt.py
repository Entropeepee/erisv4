"""
Statistical Gating Technology (SGT)
====================================

Every decision in Eris Echo passes through a dual-path gate:
- Path A (drift tracker): running mean of the signal
- Path B (noise estimator): noise floor from signal variance
- Gate opens ONLY when signal exceeds noise floor by threshold_sigma σ

Below threshold: do nothing (it's noise).
Above threshold: act (it's signal).

This module provides both:
- Stateless core functions (for vectorized field operations with thousands of cells)
- Stateful SGTGate class (for scalar signals like memory consolidation decisions)

Usage:
    # Stateless (field operations — you manage the running stats)
    from eris.computation.sgt import gate_decision, batch_gate

    should_act, z_score = gate_decision(value, running_mean, running_var, threshold=2.0)
    mask = batch_gate(values, means, variances, threshold=2.0)

    # Stateful (scalar signals — gate tracks its own EMA stats)
    from eris.computation.sgt import SGTGate

    gate = SGTGate(threshold_sigma=2.0, ema_alpha=0.1)
    should_act, z_score = gate.update(new_value)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any

from eris.config import xp, to_numpy, CONFIG
import numpy as np


# ─── Stateless Core ──────────────────────────────────────────────────────
# Use these for vectorized field operations where you have arrays of signals,
# each with its own running mean and variance.

def gate_decision(
    value: float,
    running_mean: float,
    running_var: float,
    threshold: float = 2.0,
) -> Tuple[bool, float]:
    """Single-value gate decision.

    Parameters
    ----------
    value : float
        The current signal value to test.
    running_mean : float
        Path A output: running mean of this signal.
    running_var : float
        Path B output: running variance of this signal.
    threshold : float
        Number of standard deviations above noise floor to trigger.

    Returns
    -------
    should_act : bool
        True if signal exceeds noise floor by `threshold` sigma.
    z_score : float
        How many sigma above the mean. Useful for logging.
    """
    std = max(running_var ** 0.5, 1e-10)  # Avoid division by zero
    z_score = abs(value - running_mean) / std
    return z_score >= threshold, z_score


def batch_gate(
    values,
    means,
    variances,
    threshold: float = 2.0,
):
    """Vectorized gate decision for arrays of signals.

    All inputs should be same-shaped arrays (CuPy or NumPy).
    Returns a boolean mask: True where the gate opens.

    This is what you use inside the PDE field layer — every cell
    has its own running_mean and running_var, and you gate them all
    in one GPU operation.
    """
    values = xp.asarray(values, dtype=xp.float32)
    means = xp.asarray(means, dtype=xp.float32)
    variances = xp.asarray(variances, dtype=xp.float32)

    stds = xp.maximum(xp.sqrt(variances), 1e-10)
    z_scores = xp.abs(values - means) / stds
    mask = z_scores >= threshold
    return mask, z_scores


def update_ema(
    current_value: float,
    running_mean: float,
    running_var: float,
    alpha: float = 0.1,
) -> Tuple[float, float]:
    """Update exponential moving average statistics.

    Parameters
    ----------
    current_value : float
        New observation.
    running_mean : float
        Previous EMA of the mean.
    running_var : float
        Previous EMA of the variance.
    alpha : float
        Smoothing factor. 0.1 = slow adaptation, 0.5 = fast.

    Returns
    -------
    new_mean : float
        Updated running mean.
    new_var : float
        Updated running variance.
    """
    new_mean = (1 - alpha) * running_mean + alpha * current_value
    diff = current_value - new_mean
    new_var = (1 - alpha) * running_var + alpha * (diff * diff)
    return new_mean, new_var


def batch_update_ema(
    values,
    means,
    variances,
    alpha: float = 0.1,
):
    """Vectorized EMA update for arrays. Returns (new_means, new_variances)."""
    values = xp.asarray(values, dtype=xp.float32)
    means = xp.asarray(means, dtype=xp.float32)
    variances = xp.asarray(variances, dtype=xp.float32)

    new_means = (1 - alpha) * means + alpha * values
    diffs = values - new_means
    new_vars = (1 - alpha) * variances + alpha * (diffs * diffs)
    return new_means, new_vars


# ─── Stateful Wrapper ────────────────────────────────────────────────────
# Use this for scalar signals: memory consolidation, dissonance detection,
# research triggering, dreaming loop decisions.

@dataclass
class SGTGate:
    """Stateful SGT gate for scalar signals.

    Maintains its own running mean and variance via EMA.
    Call `gate.update(value)` each time you have a new observation.

    Attributes
    ----------
    threshold_sigma : float
        Gate opens when |z-score| >= this value.
    ema_alpha : float
        EMA smoothing factor.
    running_mean : float
        Current estimate of signal mean (Path A).
    running_var : float
        Current estimate of signal variance (Path B).
    n_observations : int
        How many values we've seen (for warmup logic).
    """
    threshold_sigma: float = 2.0
    ema_alpha: float = 0.1
    running_mean: float = 0.0
    running_var: float = 1.0
    n_observations: int = 0

    # Warmup: don't gate until we've seen enough data to have stable stats
    warmup_observations: int = 10

    def update(self, value: float) -> Tuple[bool, float]:
        """Observe a new value, update stats, return gate decision.

        During warmup (first `warmup_observations` values), the gate
        always returns False (don't act) to let stats stabilize.

        Returns
        -------
        should_act : bool
            True if signal is significant.
        z_score : float
            How many sigma from the mean.
        """
        self.n_observations += 1

        # Update running statistics
        self.running_mean, self.running_var = update_ema(
            value, self.running_mean, self.running_var, self.ema_alpha
        )

        # During warmup, never trigger
        if self.n_observations < self.warmup_observations:
            std = max(self.running_var ** 0.5, 1e-10)
            z = abs(value - self.running_mean) / std
            return False, z

        # Normal gating
        return gate_decision(
            value, self.running_mean, self.running_var, self.threshold_sigma
        )

    def reset(self) -> None:
        """Reset all state. Use when context changes fundamentally."""
        self.running_mean = 0.0
        self.running_var = 1.0
        self.n_observations = 0

    def snapshot(self) -> Dict[str, Any]:
        """Serialize state for checkpointing."""
        return {
            "threshold_sigma": self.threshold_sigma,
            "ema_alpha": self.ema_alpha,
            "running_mean": self.running_mean,
            "running_var": self.running_var,
            "n_observations": self.n_observations,
            "warmup_observations": self.warmup_observations,
        }

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "SGTGate":
        """Restore from checkpoint."""
        return cls(
            threshold_sigma=data["threshold_sigma"],
            ema_alpha=data["ema_alpha"],
            running_mean=data["running_mean"],
            running_var=data["running_var"],
            n_observations=data["n_observations"],
            warmup_observations=data.get("warmup_observations", 10),
        )
