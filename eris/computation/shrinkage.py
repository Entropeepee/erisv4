"""
Davidian Hill-Power Adaptive Shrinkage
=======================================

The unified shrinkage function from David Pope's CIP that replaces all
discrete classical filters with a single continuous 4-parameter equation:

    w(s; α, β, γ, δ) = [(s-δ)₊^α / ((s-δ)₊^α + β)]^γ

Verified exact recovery: Wiener (α=1,β=1,γ=1,δ=0), Garrote (δ=1),
Horseshoe (α≈0.99,β≈10.8,γ≈1.75,δ=0), and 19+ others as special cases.

The Pope Filter is a TRADE SECRET meta-selector. The Hill-Power function
is the unified REPLACEMENT. For all Eris Echo code, use Hill-Power.

In v4, the four parameters are driven by the computed BFECDS activation
vector — shrinkage adapts continuously to the system's dynamical state.

Copyright 2026 Terminus IP Group LLC. Patent Pending.

Usage:
    from eris.computation.shrinkage import davidian_weight, shrink_eigenvalues
    weights = davidian_weight(snr, alpha=1.0, beta=1.0, gamma=1.0, delta=0.0)
    shrunk = shrink_eigenvalues(eigenvalues, n_samples, n_features, bvec=bvec)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from eris.config import xp, to_numpy
import numpy as np

if TYPE_CHECKING:
    from eris.computation.activations import BVec


def _softplus(x, epsilon: float = 0.05):
    """Smooth C¹ approximation of max(0, x).
    The Hill function's infinite-derivative singularity at the origin
    enables L1/Lq sparsity — softplus preserves this property.
    """
    # For large x/epsilon, softplus ≈ x. Avoid overflow by clamping.
    ratio = x / epsilon
    # Where ratio is large, just return x directly (softplus ≈ x)
    return xp.where(
        ratio > 20.0,
        xp.maximum(x, 0.0),  # For large values, softplus ≈ max(0, x)
        epsilon * xp.log(1.0 + xp.exp(xp.clip(ratio, -50, 20)))
    )


def davidian_weight(s, alpha: float = 1.0, beta: float = 1.0,
                    gamma: float = 1.0, delta: float = 0.0,
                    smooth: bool = True):
    """Compute the Davidian Hill-Power shrinkage weight.

    w(s; α, β, γ, δ) = [(s-δ)₊^α / ((s-δ)₊^α + β)]^γ

    Parameters
    ----------
    s : array — signal strength (SNR, eigenvalue ratio, etc.)
    alpha : float — transition rate (>0). Criticality sharpens this.
    beta : float — threshold scale (>0). Boundary raises this.
    gamma : float — compression exponent (>0). Emergence amplifies.
    delta : float — kill zone (≥0). Decay increases this.
    smooth : bool — use softplus for C¹ differentiability.

    Returns
    -------
    w : array in [0, 1].
    """
    s = xp.asarray(s, dtype=xp.float32)
    u = _softplus(s - delta) if smooth else xp.maximum(s - delta, 0.0)
    u_alpha = xp.where(u > 1e-30, xp.power(u, alpha), xp.zeros_like(u))
    # Hill function: u^α / (u^α + β). For very large u^α, this → 1.0
    h = xp.where(u_alpha > 1e30, xp.ones_like(u_alpha), u_alpha / (u_alpha + beta))
    return xp.power(h, gamma)


@dataclass
class DavidianParams:
    """Four Hill-Power parameters. Defaults recover the Wiener filter."""
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 0.0


def params_from_bvec(bvec: "BVec", psi: float) -> DavidianParams:
    """Map BFECDS activation vector to Davidian parameters.

    This is the bridge: the field state drives the estimation strategy.
      α: Criticality sharpens transitions, Feedback smooths them
      β: Boundary raises threshold (more conservative)
      γ: Emergence amplifies novel signals, Saturation compresses
      δ: Decay creates kill zone (prune decaying components)
    """
    alpha_0 = 1.0 + 0.5 / max(psi, 0.5)
    beta_0 = 0.5 / max(psi, 1.0)
    gamma_0 = 1.0 + 0.3 / max(psi, 0.5)

    alpha = max(alpha_0 + 1.5 * bvec.C - 0.5 * bvec.F, 0.1)
    beta = max(beta_0 + 0.3 * bvec.B, 0.01)
    gamma = max(gamma_0 + 0.5 * bvec.E - 0.3 * bvec.S, 0.1)
    delta = max(0.8 * bvec.D, 0.0)

    return DavidianParams(alpha=alpha, beta=beta, gamma=gamma, delta=delta)


def shrink_eigenvalues(eigenvalues, n_samples: int, n_features: int,
                       bvec: Optional["BVec"] = None,
                       params: Optional[DavidianParams] = None):
    """Shrink eigenvalues toward their mean using Davidian Hill-Power.

    Per-eigenvalue SNR: s_i = eigenvalue_i / mean(eigenvalues).
    Large eigenvalues (s >> 1) preserved; small ones shrunk toward mean.
    """
    eigenvalues = xp.asarray(eigenvalues, dtype=xp.float32)
    psi = n_samples / max(n_features, 1)
    mean_eig = xp.mean(eigenvalues)

    if params is not None:
        p = params
    elif bvec is not None:
        p = params_from_bvec(bvec, psi)
    else:
        p = DavidianParams(alpha=1.0, beta=0.5 / max(psi, 1.0), gamma=1.0, delta=0.0)

    s = eigenvalues / xp.maximum(mean_eig, xp.float32(1e-10))
    w = davidian_weight(s, p.alpha, p.beta, p.gamma, p.delta)
    return w * eigenvalues + (1.0 - w) * mean_eig


def shrink_covariance(cov_matrix, n_samples: int,
                      bvec: Optional["BVec"] = None):
    """Shrink covariance via Davidian Hill-Power eigenvalue shrinkage."""
    cov_matrix = xp.asarray(cov_matrix, dtype=xp.float32)
    n_features = cov_matrix.shape[0]
    eigenvalues, eigenvectors = xp.linalg.eigh(cov_matrix)
    shrunk_eig = shrink_eigenvalues(eigenvalues, n_samples, n_features, bvec=bvec)
    shrunk_eig = xp.maximum(shrunk_eig, 0.0)
    return (eigenvectors * shrunk_eig[None, :]) @ eigenvectors.T
