"""
Multi-Memory Interference Detection (CSBA-Enhanced)
=====================================================

From the DCR White Paper (Pope, 2025):
    R_ij(t) = ∫ φᵢ(x,t) · φⱼ(x,t) · cos(θᵢ - θⱼ) dx

When field snapshots are available, we compute this integral directly.

When field snapshots are NOT available, we DO NOT fall back to a single
cosine similarity. A single cosine between two 6-vectors is one viewing
angle on the coupling sphere — it loses the multi-dimensional structure
that the conservation law requires.

Instead, we use the full BFECDS coupling geometry:
    1. Per-domain coupling angles θ_k between two memories
    2. Conservation law decomposition: cos²(θ_k) + sin²(θ_k) = 1 per domain
    3. Elastic (resonant) vs plastic (conflicting) channels per domain
    4. Davidian Hill-Power shrinkage to separate signal from noise
       in the coupling spectrum (CSBA principle: project out noise,
       keep invariant structure)

This gives a multi-dimensional interference measure that respects:
    - All six BLECD domains independently
    - The conservation law (information = coupling geometry)
    - Signal/noise separation via Davidian shrinkage (not arbitrary clipping)
    - Both resonance AND conflict detection per domain

The result is NOT a single number — it's a 6-vector of per-domain
interferences plus a scalar summary. The per-domain breakdown tells
you WHERE memories agree and WHERE they conflict.

Copyright 2026 Terminus IP Group LLC. All IP used under internal license.

Usage:
    from eris.memory.interference import compute_interference, find_conflicts
    R = compute_interference(memory_a, memory_b)
    # R.total = scalar summary
    # R.per_domain = {"B": 0.3, "F": -0.1, ...}
    # R.regime = "resonant" | "conflicting" | "orthogonal"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np
from eris.config import to_numpy, xp

from eris.computation.activations import BVec, bvec_distance
from eris.computation.shrinkage import davidian_weight, DavidianParams


DOMAIN_NAMES = ["B", "F", "E", "C", "D", "S"]


@dataclass
class InterferenceResult:
    """Full interference measurement between two memories.

    Not just a scalar — carries per-domain breakdown and regime.
    """
    total: float = 0.0                           # Scalar summary in [-1, 1]
    per_domain: Dict[str, float] = field(default_factory=dict)  # Per-domain R_k
    elastic_energy: float = 0.0                   # Total resonant coupling
    plastic_energy: float = 0.0                   # Total conflicting coupling
    regime: str = "orthogonal"                    # resonant | conflicting | orthogonal
    used_field_integral: bool = False             # True if computed from φ-θ snapshots

    def __repr__(self) -> str:
        return (f"Interference(R={self.total:.3f}, regime={self.regime}, "
                f"elastic={self.elastic_energy:.3f}, plastic={self.plastic_energy:.3f})")


def _field_integral(phi_a, theta_a, phi_b, theta_b) -> InterferenceResult:
    """Compute interference from field snapshots (the DCR integral).

    R_ij = ∫ φᵢ · φⱼ · cos(θᵢ - θⱼ) dx

    This is the gold standard — direct measurement of coupling geometry.
    """
    # Handle size mismatches
    min_h = min(phi_a.shape[0], phi_b.shape[0])
    min_w = min(phi_a.shape[1], phi_b.shape[1])
    pa = phi_a[:min_h, :min_w]
    pb = phi_b[:min_h, :min_w]
    ta = theta_a[:min_h, :min_w]
    tb = theta_b[:min_h, :min_w]

    integrand = pa * pb * np.cos(ta - tb)
    raw = float(xp.mean(integrand))

    # Normalize by geometric mean of field energies
    energy_a = float(np.mean(pa ** 2))
    energy_b = float(np.mean(pb ** 2))
    norm = np.sqrt(max(energy_a * energy_b, 1e-20))
    total = raw / norm

    # Decompose into elastic/plastic via the sign structure
    elastic = float(np.mean(np.maximum(integrand, 0.0))) / max(norm, 1e-20)
    plastic = float(np.mean(np.maximum(-integrand, 0.0))) / max(norm, 1e-20)

    if total > 0.3:
        regime = "resonant"
    elif total < -0.3:
        regime = "conflicting"
    else:
        regime = "orthogonal"

    return InterferenceResult(
        total=total,
        elastic_energy=elastic,
        plastic_energy=plastic,
        regime=regime,
        used_field_integral=True,
    )


def _csba_coupling_geometry(bvec_a: BVec, bvec_b: BVec) -> InterferenceResult:
    """Compute interference from BFECDS vectors using full coupling geometry.

    NOT a single cosine. Uses the conservation law per domain:
        cos²(θ_k) + sin²(θ_k) = 1

    For each domain k:
        θ_k = angle between the two memories' activation in that domain
        elastic_k = cos²(θ_k) × coupling_strength_k  (resonance)
        plastic_k = sin²(θ_k) × coupling_strength_k  (conflict)

    Then applies Davidian Hill-Power shrinkage to the coupling spectrum
    to separate real structure from noise (CSBA principle).

    This respects:
        - All six domains independently (not collapsed to one angle)
        - Conservation law (cos²+sin²=1 per domain)
        - Signal/noise via Davidian (not arbitrary threshold)
        - Both agreement AND disagreement per domain
    """
    a = bvec_a.as_array()  # [B, F, E, C, D, S]
    b = bvec_b.as_array()

    per_domain = {}
    elastic_raw = np.zeros(6, dtype=np.float32)
    plastic_raw = np.zeros(6, dtype=np.float32)

    for k in range(6):
        # Coupling strength = geometric mean of activations
        # (if either memory has zero activation in a domain, no coupling)
        coupling_k = np.sqrt(max(a[k] * b[k], 0.0))

        if coupling_k < 1e-8:
            per_domain[DOMAIN_NAMES[k]] = 0.0
            continue

        # Per-domain angle: how aligned are the two memories in this domain?
        # Use the normalized difference as a proxy for angular separation
        max_val = max(a[k], b[k], 1e-10)
        min_val = min(a[k], b[k])
        # ratio ∈ [0, 1]: 1 = identical activation, 0 = maximally different
        ratio = min_val / max_val

        # cos²(θ_k) = ratio² (aligned = high ratio = high cos²)
        # sin²(θ_k) = 1 - ratio² (misaligned = low ratio = high sin²)
        cos2 = ratio * ratio
        sin2 = 1.0 - cos2

        # Elastic (resonant) and plastic (conflicting) contributions
        elastic_raw[k] = cos2 * coupling_k
        plastic_raw[k] = sin2 * coupling_k

        # Per-domain interference: positive = resonant, negative = conflict
        per_domain[DOMAIN_NAMES[k]] = float(elastic_raw[k] - plastic_raw[k])

    # Apply Davidian Hill-Power shrinkage to the coupling spectrum
    # This is the CSBA step: separate signal from noise in the 6D coupling
    # The "signal" is coupling that exceeds the noise floor of BFECDS estimation
    coupling_spectrum = elastic_raw + plastic_raw  # Total coupling per domain
    if coupling_spectrum.max() > 1e-8:
        # SNR for each domain's coupling relative to the mean
        mean_coupling = max(float(np.mean(coupling_spectrum)), 1e-10)
        snr = coupling_spectrum / mean_coupling

        # Davidian weights: which domain couplings are signal vs noise?
        # Use moderate shrinkage (β=0.5) to be conservative
        weights = to_numpy(davidian_weight(
            snr, alpha=1.0, beta=0.5, gamma=1.0, delta=0.0
        )).ravel()

        # Shrunk elastic/plastic: only domains with significant coupling survive
        elastic_shrunk = elastic_raw * weights
        plastic_shrunk = plastic_raw * weights
    else:
        elastic_shrunk = elastic_raw
        plastic_shrunk = plastic_raw

    # Total interference: sum of shrunk elastic minus shrunk plastic
    total_elastic = float(np.sum(elastic_shrunk))
    total_plastic = float(np.sum(plastic_shrunk))

    # Normalize to [-1, 1] range
    total_coupling = max(total_elastic + total_plastic, 1e-10)
    total = (total_elastic - total_plastic) / total_coupling

    if total > 0.3:
        regime = "resonant"
    elif total < -0.3:
        regime = "conflicting"
    else:
        regime = "orthogonal"

    return InterferenceResult(
        total=total,
        per_domain=per_domain,
        elastic_energy=total_elastic,
        plastic_energy=total_plastic,
        regime=regime,
        used_field_integral=False,
    )


def compute_interference(mem_a, mem_b) -> InterferenceResult:
    """Compute interference between two memories.

    Uses field integral when snapshots available, CSBA coupling
    geometry when they're not. Never falls back to a single cosine.

    Parameters
    ----------
    mem_a, mem_b : MemoryRecord
        Two memories to compare.

    Returns
    -------
    InterferenceResult with total, per-domain breakdown, and regime.
    """
    # Prefer field-level integral if snapshots exist
    if (mem_a.phi_snapshot is not None and mem_b.phi_snapshot is not None and
        mem_a.theta_snapshot is not None and mem_b.theta_snapshot is not None):
        return _field_integral(
            mem_a.phi_snapshot, mem_a.theta_snapshot,
            mem_b.phi_snapshot, mem_b.theta_snapshot,
        )

    # Full CSBA coupling geometry — NOT a single cosine
    return _csba_coupling_geometry(mem_a.bvec, mem_b.bvec)


def compute_interference_matrix(memories: list) -> np.ndarray:
    """NxN interference matrix. R[i,j] = total interference between i and j."""
    n = len(memories)
    R = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        R[i, i] = 1.0
        for j in range(i + 1, n):
            result = compute_interference(memories[i], memories[j])
            R[i, j] = result.total
            R[j, i] = result.total
    return R


def find_conflicts(memories: list, threshold: float = -0.1) -> List[Tuple[int, int, InterferenceResult]]:
    """Find pairs in conflict. Returns (i, j, InterferenceResult) sorted by severity."""
    conflicts = []
    for i in range(len(memories)):
        for j in range(i + 1, len(memories)):
            result = compute_interference(memories[i], memories[j])
            if result.total < threshold:
                conflicts.append((i, j, result))
    conflicts.sort(key=lambda x: x[2].total)
    return conflicts


def find_resonances(memories: list, threshold: float = 0.5) -> List[Tuple[int, int, InterferenceResult]]:
    """Find resonant pairs. Returns (i, j, InterferenceResult) sorted by strength."""
    resonances = []
    for i in range(len(memories)):
        for j in range(i + 1, len(memories)):
            result = compute_interference(memories[i], memories[j])
            if result.total > threshold:
                resonances.append((i, j, result))
    resonances.sort(key=lambda x: x[2].total, reverse=True)
    return resonances
