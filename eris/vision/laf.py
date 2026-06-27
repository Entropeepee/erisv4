"""LAF multiresolution-SVD signature — the (κ, λ) of a coherence field.

Zero-weight, no learned parameters (RULE 1): a fixed linear-algebra transform of
the complex spinor field √ρ·e^{iθ}. Build patch columns of the spinor, average
adjacent columns recursively to form a multi-scale "tower", realify (stack real &
imag), and take the SVD:

  κ (kappa)  = leading left-singular vectors  — the principal geometric "shapes"
               of the field across scales (the structural identity basis).
  λ (lambda) = normalized singular values     — how much each κ-mode matters.
  τ_modes    = mean projection of the tower into κ-space.

Pure NumPy / CuPy (np.linalg.svd) — torch-free by design (sm_120/Blackwell). The
Chimera prototype's tower/factorize is the reference for structure only; the
corrected torsion (∇ρ×∇θ) lives in frontends.py, never the Laplacian.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np

_EPS = 1e-9


@dataclass
class LAFConfig:
    patch: int = 8        # square patch side → each patch is one spinor column
    n_scales: int = 4     # tower depth (adjacent-column averaging)
    n_modes: int = 12     # κ rank kept


def _patch_columns(spinor: np.ndarray, patch: int) -> np.ndarray:
    """(d, n) complex matrix: each non-overlapping patch flattened to a column."""
    H, W = spinor.shape
    patch = max(1, min(patch, H, W))
    cols = []
    for i in range(0, H - patch + 1, patch):
        for j in range(0, W - patch + 1, patch):
            cols.append(spinor[i:i + patch, j:j + patch].ravel())
    if not cols:
        cols = [spinor.ravel()]
    return np.asarray(cols, dtype=np.complex128).T


def tower(X: np.ndarray, n_scales: int) -> np.ndarray:
    """Recursive adjacent-column averaging across scales; concatenate all scales.
    Coarser scales summarize larger neighbourhoods of the field."""
    scales = [X]
    cur = X
    for _ in range(max(0, n_scales - 1)):
        n = cur.shape[1]
        if n < 2:
            break
        m = n // 2
        cur = 0.5 * (cur[:, 0:2 * m:2] + cur[:, 1:2 * m:2])
        scales.append(cur)
    return np.concatenate(scales, axis=1)


def laf_signature(mag: np.ndarray, theta: np.ndarray,
                  cfg: LAFConfig = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(κ, λ, τ_modes) for a (mag=√ρ, θ) field. Deterministic, zero-weight."""
    cfg = cfg or LAFConfig()
    mag = np.asarray(mag, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    spinor = mag * np.exp(1j * theta)
    X = _patch_columns(spinor, cfg.patch)
    T = tower(X, cfg.n_scales)
    # Realify the complex tower so the SVD is a standard real factorization.
    R = np.vstack([T.real, T.imag])
    # Guard degenerate fields.
    if R.size == 0 or not np.isfinite(R).all():
        r = cfg.n_modes
        return np.zeros((1, r)), np.zeros(r), np.zeros(r)
    U, S, _ = np.linalg.svd(R, full_matrices=False)
    r = int(min(cfg.n_modes, S.shape[0]))
    kappa = U[:, :r]                                  # dominant geometric modes
    # Normalized energy spectrum ACROSS THE KEPT κ-modes (a distribution summing
    # to 1) — how much each structural shape matters relative to the others.
    lam = (S[:r] / (S[:r].sum() + _EPS)).astype(np.float64)
    proj = kappa.T @ R                                # (r, cols)
    tau_modes = proj.mean(axis=1).astype(np.float64)
    return kappa, lam, tau_modes


def kappa_overlap(kappa_q: np.ndarray, kappa_p: np.ndarray) -> Tuple[float, float]:
    """Subspace alignment of two κ bases via principal angles. Returns
    (aligned, emergent): aligned = mean cos(principal angle) — modes shared; emergent
    = mean sin(principal angle) — modes in the query NOT in the prototype (RULE 2:
    keep the sine half, don't drop it)."""
    if kappa_q is None or kappa_p is None or kappa_q.size == 0 or kappa_p.size == 0:
        return 0.0, 0.0
    # Orthonormalize columns (SVD already gives orthonormal U, but be safe).
    def _on(M):
        q, _ = np.linalg.qr(np.asarray(M, dtype=np.float64))
        return q
    Qq, Qp = _on(kappa_q), _on(kappa_p)
    m = min(Qq.shape[1], Qp.shape[1])
    if m == 0:
        return 0.0, 0.0
    s = np.linalg.svd(Qq[:, :m].T @ Qp[:, :m], compute_uv=False)
    cos = np.clip(s, 0.0, 1.0)
    aligned = float(np.mean(cos))
    emergent = float(np.mean(np.sqrt(np.clip(1.0 - cos ** 2, 0.0, 1.0))))
    return aligned, emergent


def lambda_distance(lam_q: np.ndarray, lam_p: np.ndarray) -> float:
    """L2 distance between two normalized energy spectra (padded to equal length)."""
    a = np.asarray(lam_q, dtype=np.float64).ravel()
    b = np.asarray(lam_p, dtype=np.float64).ravel()
    n = max(a.shape[0], b.shape[0])
    a = np.pad(a, (0, n - a.shape[0]))
    b = np.pad(b, (0, n - b.shape[0]))
    return float(np.linalg.norm(a - b))
