"""
Computed BFECDS Activations
============================

THE CRITICAL BRIDGE: Domain activations computed from field dynamics,
not assigned by an LLM.

Previous versions (v1-v3) asked GPT/Gemini to *guess* what the BFECDS
values should be for a given input. v4 COMPUTES them from the actual
FRACTAL PDE field state:

    B (Boundary)    = fraction of field at boundary clamp
    F (Feedback)    = magnitude of phi-theta gradient coupling (advection)
    E (Emergence)   = integrated positive novelty (phi - prev_phi)
    C (Criticality) = RMS torsion intensity
    D (Decay)       = rate of coherence loss between steps
    S (Saturation)  = fraction of field near saturation

Six validated archetypes (from unsupervised k=6 clustering on chemistry data):
    Feedback Stabilizer, Contained Saturator, Breakdown Hub,
    Structured Emergence, Organic Flow Field, Emergence Catalyst

Usage:
    from eris.computation.activations import BVec, compute_bvec_from_field

    bvec = compute_bvec_from_field(phi, theta, tau, phi_prev)
    print(bvec)              # BVec(B=0.12, F=0.45, E=0.31, C=0.67, D=0.08, S=0.15)
    print(bvec.archetype())  # 'Structured Emergence'
    print(bvec.as_array())   # array([0.12, 0.45, 0.31, 0.67, 0.08, 0.15])
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from eris.config import xp, to_numpy
import numpy as np


# ─── Archetype Reference Vectors ─────────────────────────────────────────
# From unsupervised k=6 clustering on BLECD_rx_v2_4 chemistry data.
# These are the cluster centroids normalized to [0,1] activation space.
# Order: [B, F, E, C, D, S]
#
# Chemistry cluster → Archetype mapping:
#   Cluster 3 (Boundary-Limited Feedback)  → Feedback Stabilizer  (F+B dominant)
#   Cluster 4 (Entropy Sink / Flatline)    → Contained Saturator  (S+B dominant)
#   Cluster 2 (Chaotic Transition Node)    → Breakdown Hub        (C+D dominant)
#   Cluster 0 (Resonant Emergence Loop)    → Structured Emergence (E+B dominant)
#   Cluster 1 (Resonant Expansion Field)   → Organic Flow Field   (E+F dominant)
#   Cluster 5 (Criticality-Fueled Rebound) → Emergence Catalyst   (E+C dominant)

ARCHETYPE_NAMES = [
    "Feedback Stabilizer",
    "Contained Saturator",
    "Breakdown Hub",
    "Structured Emergence",
    "Organic Flow Field",
    "Emergence Catalyst",
]

# Normalized centroid vectors from chemistry clustering.
# These are the "ideal" activation profiles for each archetype.
# Raw cluster centroids were ΔB/ΔF/ΔE/ΔC/ΔD/ΔS (signed changes);
# here they're mapped to [0,1] activation space where each component
# represents "how active is this domain."
_ARCHETYPE_CENTROIDS_NP = np.array([
    # B     F     E     C     D     S
    [0.70, 0.80, 0.10, 0.15, 0.10, 0.10],  # Feedback Stabilizer:  high F+B
    [0.75, 0.15, 0.10, 0.10, 0.15, 0.85],  # Contained Saturator:  high S+B
    [0.15, 0.10, 0.10, 0.80, 0.75, 0.20],  # Breakdown Hub:        high C+D
    [0.65, 0.30, 0.70, 0.35, 0.15, 0.15],  # Structured Emergence: high E+B
    [0.20, 0.70, 0.75, 0.30, 0.20, 0.10],  # Organic Flow Field:   high E+F
    [0.20, 0.25, 0.70, 0.75, 0.30, 0.15],  # Emergence Catalyst:   high E+C
], dtype=np.float32)

# L2 norms for cosine similarity (precomputed)
_CENTROID_NORMS_NP = np.linalg.norm(_ARCHETYPE_CENTROIDS_NP, axis=1)


@dataclass
class BVec:
    """BFECDS domain activation vector.

    Six continuous values in [0, 1] representing how active each
    universal domain of change is in the current system state.

    This is the lingua franca of the entire architecture:
    - Layer 0 computes it from field dynamics
    - Layer 2 uses it to gate memory consolidation
    - Layer 3 uses it to select specialist bids
    - Layer 4 uses it for dissonance detection
    - Layer 5 uses it for transfixion detection
    """
    B: float = 0.0  # Boundary:    constraints, limits, containment
    F: float = 0.0  # Feedback:    coupling loops, recursive influence
    E: float = 0.0  # Emergence:   novel structure from interactions
    C: float = 0.0  # Criticality: phase transitions, torsion, edge-of-chaos
    D: float = 0.0  # Decay:       entropy, forgetting, coherence loss
    S: float = 0.0  # Saturation:  capacity limits, density

    def as_array(self) -> np.ndarray:
        """Return as a 6-element NumPy array [B, F, E, C, D, S]."""
        return np.array([self.B, self.F, self.E, self.C, self.D, self.S],
                        dtype=np.float32)

    def as_dict(self) -> Dict[str, float]:
        """Return as a dictionary for JSON serialization."""
        return {"B": self.B, "F": self.F, "E": self.E,
                "C": self.C, "D": self.D, "S": self.S}

    def elementwise(self, other: "BVec") -> "BVec":
        """Per-domain product with another BVec.

        Used for domain-projected specialist bids (Remediation Tier 3-A):
        projecting the live field vector onto a specialist's sensitivity vector
        yields that specialist's real field signature.
        """
        return BVec(
            B=self.B * other.B, F=self.F * other.F, E=self.E * other.E,
            C=self.C * other.C, D=self.D * other.D, S=self.S * other.S,
        )

    def magnitude(self) -> float:
        """L2 magnitude of the activation vector (bid strength)."""
        return float(np.linalg.norm(self.as_array()))

    def dominant_domains(self, k: int = 2) -> list:
        """Names of the k most-active domains, strongest first."""
        names = ["B", "F", "E", "C", "D", "S"]
        arr = self.as_array()
        order = list(np.argsort(arr)[::-1][:max(1, k)])
        return [names[int(i)] for i in order]

    @classmethod
    def from_array(cls, arr) -> "BVec":
        """Create from a 6-element array."""
        a = to_numpy(arr).ravel()
        return cls(B=float(a[0]), F=float(a[1]), E=float(a[2]),
                   C=float(a[3]), D=float(a[4]), S=float(a[5]))

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "BVec":
        """Create from a dictionary."""
        return cls(B=d.get("B", 0), F=d.get("F", 0), E=d.get("E", 0),
                   C=d.get("C", 0), D=d.get("D", 0), S=d.get("S", 0))

    def archetype(self) -> str:
        """Return the name of the closest validated archetype.

        Uses cosine similarity against the six chemistry-validated
        cluster centroids.
        """
        vec = self.as_array()
        vec_norm = np.linalg.norm(vec)
        if vec_norm < 1e-10:
            return "Feedback Stabilizer"  # Default for zero vector

        cosines = (_ARCHETYPE_CENTROIDS_NP @ vec) / (_CENTROID_NORMS_NP * vec_norm)
        return ARCHETYPE_NAMES[int(np.argmax(cosines))]

    def archetype_scores(self) -> Dict[str, float]:
        """Return cosine similarity to all six archetypes.

        Useful for understanding mixed states — a system might be
        60% Emergence Catalyst and 30% Breakdown Hub simultaneously.
        """
        vec = self.as_array()
        vec_norm = np.linalg.norm(vec)
        if vec_norm < 1e-10:
            return {name: 0.0 for name in ARCHETYPE_NAMES}

        cosines = (_ARCHETYPE_CENTROIDS_NP @ vec) / (_CENTROID_NORMS_NP * vec_norm)
        return {name: float(c) for name, c in zip(ARCHETYPE_NAMES, cosines)}

    def __repr__(self) -> str:
        return (f"BVec(B={self.B:.3f}, F={self.F:.3f}, E={self.E:.3f}, "
                f"C={self.C:.3f}, D={self.D:.3f}, S={self.S:.3f})")


# ─── Distance / Similarity Functions ─────────────────────────────────────

def bvec_distance(a: BVec, b: BVec) -> float:
    """L2 distance between two BVecs in BFECDS space.

    Used for:
    - Dissonance detection (Layer 4): large distance = dissonance
    - Memory novelty scoring (Layer 2): distance from stored attractors
    - Transfixion detection (Layer 5): low diversity = stuck
    """
    va = a.as_array()
    vb = b.as_array()
    return float(np.linalg.norm(va - vb))


def bvec_cosine(a: BVec, b: BVec) -> float:
    """Cosine similarity between two BVecs.

    Used for:
    - MoEGate bid scoring (Layer 5): specialist bid vs. active goal
    - Memory retrieval weighting: query BFECDS vs. stored BFECDS
    """
    va = a.as_array()
    vb = b.as_array()
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def bvec_resonance(a: BVec, b: BVec) -> float:
    """Per-domain wave-interference coupling between two BVecs — the κ/λ = cos/sin
    treatment (NOT a single cosine). Factored verbatim from MoEGate.score_bid so the gate,
    the cross-attention hub, and specialist activation all use ONE field metric (§B3):

      coupling = √(a·b) per domain;  ratio = min/max per domain;
      cos2 = ratio² (elastic/constructive),  sin2 = 1−cos2 (plastic/destructive);
      net = Σ (elastic − plastic)·w  /  Σ (elastic+plastic)·w,
    with Davidian Hill-Power shrinkage w on the coupling spectrum (denoise). Range ≈[−1,1];
    higher = stronger constructive interference. 6-vectors → plain NumPy (no GPU transfer)."""
    from eris.computation.shrinkage import davidian_weight
    va = a.as_array().astype(np.float64)
    vb = b.as_array().astype(np.float64)
    coupling = np.sqrt(np.maximum(va * vb, 0.0))
    max_vals = np.maximum(va, vb)
    max_vals = np.where(max_vals < 1e-8, 1.0, max_vals)
    ratios = np.minimum(va, vb) / max_vals
    cos2 = ratios * ratios
    sin2 = 1.0 - cos2
    elastic = cos2 * coupling
    plastic = sin2 * coupling
    total = elastic + plastic
    mean_c = max(float(np.mean(total)), 1e-10)
    weights = np.asarray(davidian_weight(total / mean_c, alpha=1.0, beta=0.5,
                                         gamma=1.0, delta=0.0)).ravel()
    norm = float(np.sum(total * weights))
    if norm < 1e-10:
        return 0.0
    return float(np.sum((elastic - plastic) * weights) / norm)


def cosine(a, b) -> float:
    """Cosine similarity between two raw vectors (e.g. semantic embeddings).
    Shared helper so the same safe, normalized formula isn't re-inlined across
    memory retrieval, the quality gate, and the dream loop."""
    if a is None or b is None:
        return 0.0
    va = np.asarray(a, dtype=np.float32).ravel()
    vb = np.asarray(b, dtype=np.float32).ravel()
    if va.shape != vb.shape or va.size == 0:
        return 0.0
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


# ─── Field → BFECDS Computation ──────────────────────────────────────────

# Scaling constants for mapping raw field statistics to [0, 1].
# These were calibrated so that the k=6 chemistry reactions cluster
# correctly when run through the PDE.  Tune empirically.
_SCALE_B: float = 1.0    # B is already a fraction
_SCALE_F: float = 10.0   # Advection magnitude is typically small
_SCALE_E: float = 20.0   # Novelty is typically small
_SCALE_C: float = 5.0    # Torsion RMS
_SCALE_D: float = 15.0   # Decay rate
_SCALE_S: float = 1.0    # S is already a fraction


def compute_bvec_from_field(
    phi,
    theta,
    tau,
    phi_prev,
    boundary_threshold: float = 0.95,
    saturation_threshold: float = 0.80,
) -> BVec:
    """Compute BFECDS domain activations from FRACTAL field state.

    This is the function that makes v4 different from all previous versions.
    No LLM calls. No keyword matching. Pure measurement.

    Parameters
    ----------
    phi : 2D array (CuPy or NumPy)
        Coherence field — how "sure" the system is at each point.
    theta : 2D array
        Phase field — what the system "thinks" at each point.
    tau : 2D array
        Torsion field — where certainty is curving (∇²φ).
    phi_prev : 2D array
        Previous timestep's phi (for computing novelty and decay).
    boundary_threshold : float
        phi values above this are counted as "at boundary."
    saturation_threshold : float
        phi values above this are counted as "near saturation."

    Returns
    -------
    BVec with all six domain activations in [0, 1].
    """
    # Ensure GPU arrays if available
    phi = xp.asarray(phi, dtype=xp.float32)
    theta = xp.asarray(theta, dtype=xp.float32)
    tau = xp.asarray(tau, dtype=xp.float32)
    phi_prev = xp.asarray(phi_prev, dtype=xp.float32)

    # B: Boundary — fraction of field at boundary clamp
    b_raw = float(to_numpy(xp.mean(phi > boundary_threshold)))
    B = min(b_raw * _SCALE_B, 1.0)

    # F: Feedback — gradient alignment between phi and theta (advection)
    # Uses cp.roll stencils (NOT cupyx.scipy.ndimage — broken on CUDA 13.2)
    gphi_x = phi - xp.roll(phi, 1, axis=1)  # d(phi)/dx
    gphi_y = phi - xp.roll(phi, 1, axis=0)  # d(phi)/dy
    gtheta_x = theta - xp.roll(theta, 1, axis=1)
    gtheta_y = theta - xp.roll(theta, 1, axis=0)
    advection = xp.abs(gphi_x * gtheta_x + gphi_y * gtheta_y)
    f_raw = float(to_numpy(xp.mean(advection)))
    F = min(f_raw * _SCALE_F, 1.0)

    # E: Emergence — integrated positive novelty (new structure appearing)
    novelty = xp.maximum(0.0, phi - phi_prev)
    e_raw = float(to_numpy(xp.mean(novelty)))
    E = min(e_raw * _SCALE_E, 1.0)

    # C: Criticality — RMS torsion intensity (phase transitions)
    tau_rms = float(to_numpy(xp.sqrt(xp.mean(tau ** 2))))
    C = min(tau_rms * _SCALE_C, 1.0)

    # D: Decay — rate of coherence loss
    decay = xp.maximum(0.0, phi_prev - phi)
    d_raw = float(to_numpy(xp.mean(decay)))
    D = min(d_raw * _SCALE_D, 1.0)

    # S: Saturation — fraction of field near full capacity
    s_raw = float(to_numpy(xp.mean(phi > saturation_threshold)))
    S = min(s_raw * _SCALE_S, 1.0)

    return BVec(B=B, F=F, E=E, C=C, D=D, S=S)
