"""
BLECD Logic Compiler (BLC) — Field Geometry Compiler
=====================================================

Translates contradictions between two BFECDS vectors into φ-θ seed
geometries that are injected into the FRACTAL PDE field. The field then
RESOLVES the contradiction through dynamics — logic is not programmed,
it is grown and resolved geometrically.

From the BLC paper (Pope, 2025):
    "The compiler does not 'run code' — it grows symbolic geometries
     and watches them evolve. It is ontological, not syntactic."

Three compilation stages:
    1. Contradiction Analysis → identify dominant BLECD domain pairs (SGT-gated)
    2. Domain Pairs → Gate Geometry selection (from BLC gate taxonomy)
    3. Gate Geometry → φ-θ seed arrays to inject into the PDE field

Gate taxonomy (from BLC paper Appendix C):
    YES   (B+S): Single φ well, coherent θ gradient — affirms input
    XOR   (C+D): Dual overlapping φ circles, π-phase shift — resolves conflict
    AND   (F+E): Narrow corridor, aligned phase convergence — requires consensus
    DELAY (S+F): Toroidal φ basin, rotational θ trap — temporal buffer
    DIODE (D+B): Asymmetric φ gradient, locked phase barrier — irreversible flow

The hex lattice (lattice.py) is a SECONDARY tracking layer that monitors
how pulses propagate through the resolved field. The PDE is primary.

Usage:
    from eris.field.compiler import BLECDCompiler, compile_contradiction

    compiler = BLECDCompiler(field_size=64)
    seed = compile_contradiction(bvec_input, bvec_response)
    # seed.phi_patch and seed.theta_patch are 2D arrays to inject into the PDE
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
import numpy as np

from eris.computation.activations import BVec, bvec_distance
from eris.computation.sgt import gate_decision


# ─── Gate Types (from BLC paper) ──────────────────────────────────────────

class FieldGateType(Enum):
    """Symbolic gate types, each defined by a φ-θ geometry."""
    YES = "yes"       # Affirmation: single φ well + coherent θ
    XOR = "xor"       # Conflict resolution: dual φ circles, π-shift
    AND = "and"       # Consensus: narrow corridor, aligned convergence
    DELAY = "delay"   # Temporal buffer: toroidal φ basin
    DIODE = "diode"   # Irreversible flow: asymmetric φ gradient


@dataclass
class GateInstruction:
    """A discrete instruction for the hex lattice SLGP worker."""
    q: int
    r: int
    direction: int
    gate_type: FieldGateType
    weight: float

@dataclass
class GateProgram:
    """A collection of instructions for the SLGP worker."""
    instructions: List[GateInstruction] = field(default_factory=list)


# Domain pair → gate type mapping (from BLC paper Appendix C)
# Each gate is defined by its TWO dominant BLECD domains
DOMAIN_PAIR_GATE = {
    ("B", "S"): FieldGateType.YES,    # Boundary + Saturation = affirmation
    ("S", "B"): FieldGateType.YES,
    ("C", "D"): FieldGateType.XOR,    # Criticality + Decay = conflict
    ("D", "C"): FieldGateType.XOR,
    ("F", "E"): FieldGateType.AND,    # Feedback + Emergence = consensus
    ("E", "F"): FieldGateType.AND,
    ("S", "F"): FieldGateType.DELAY,  # Saturation + Feedback = buffer
    ("F", "S"): FieldGateType.DELAY,
    ("D", "B"): FieldGateType.DIODE,  # Decay + Boundary = irreversible
    ("B", "D"): FieldGateType.DIODE,
}

DOMAIN_NAMES = ["B", "F", "E", "C", "D", "S"]


# ─── Seed Geometry (compiler output) ─────────────────────────────────────

@dataclass
class FieldSeed:
    """A φ-θ geometry to inject into the PDE field.

    This is the output of the BLC compiler. It describes WHERE and HOW
    to perturb the field to trigger a specific gate resolution.
    """
    phi_patch: np.ndarray       # 2D φ perturbation to add to field
    theta_patch: np.ndarray     # 2D θ perturbation to add to field
    gate_type: FieldGateType    # What kind of gate this geometry encodes
    center_x: int               # Center of the geometry in field coordinates
    center_y: int               # Center of the geometry in field coordinates
    radius: int                 # Spatial extent of the geometry
    strength: float             # Amplitude scale [0, 1]
    source_domains: Tuple[str, str]  # Which BLECD domains generated this
    contradiction: float        # L2 distance that triggered this

    def __repr__(self) -> str:
        return (f"FieldSeed({self.gate_type.value}, center=({self.center_x},{self.center_y}), "
                f"r={self.radius}, strength={self.strength:.2f}, "
                f"domains={self.source_domains})")


@dataclass
class CompiledContradiction:
    """Full compilation result: one or more field seeds plus metadata."""
    seeds: List[FieldSeed] = field(default_factory=list)
    total_contradiction: float = 0.0
    source_bvec: BVec = field(default_factory=BVec)
    target_bvec: BVec = field(default_factory=BVec)
    significant_domains: List[str] = field(default_factory=list)

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)


# ─── Geometry Generators ──────────────────────────────────────────────────
# Each function creates the φ-θ seed arrays for a specific gate type.
# These are the "geometric programs" from the BLC paper.

def _make_circle(size: int, cx: int, cy: int, radius: float) -> np.ndarray:
    """Create a smooth circular mask centered at (cx, cy)."""
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2).astype(np.float32)
    return np.clip(1.0 - dist / max(radius, 1), 0.0, 1.0)


def _make_corridor(size: int, cx: int, cy: int, length: int,
                   width: int, angle: float = 0.0) -> np.ndarray:
    """Create a narrow corridor (elongated Gaussian) at given angle."""
    y, x = np.ogrid[:size, :size]
    dx = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
    dy = -(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
    return np.exp(-(dx**2 / max(length**2, 1) + dy**2 / max(width**2, 1))).astype(np.float32)


def generate_yes_gate(size: int, cx: int, cy: int, radius: int,
                      strength: float) -> Tuple[np.ndarray, np.ndarray]:
    """YES gate: single φ well + coherent radial θ gradient.

    BLECD domains: Boundary + Saturation
    Behavior: Confirms symbolic input, reinforces coherence.
    """
    phi = _make_circle(size, cx, cy, radius) * strength
    # Coherent radial θ: phase increases outward from center (affirmation)
    y, x = np.ogrid[:size, :size]
    theta = np.arctan2((y - cy).astype(np.float32), (x - cx).astype(np.float32) + 1e-10)
    theta = (theta + np.pi).astype(np.float32)  # [0, 2π]
    theta *= phi  # Only apply where φ is nonzero
    return phi, theta


def generate_xor_gate(size: int, cx: int, cy: int, radius: int,
                      strength: float) -> Tuple[np.ndarray, np.ndarray]:
    """XOR gate: dual overlapping φ circles with π-phase shift.

    BLECD domains: Criticality + Decay
    Behavior: Symbolic bifurcation — outputs coherence only on asymmetric input.
    The π-shift creates destructive interference at the midpoint.
    """
    offset = max(radius // 2, 2)
    circle_a = _make_circle(size, cx - offset, cy, radius) * strength
    circle_b = _make_circle(size, cx + offset, cy, radius) * strength
    phi = np.maximum(circle_a, circle_b)  # Union of circles

    # Opposing phase: left circle at 0, right circle at π
    theta = np.zeros((size, size), dtype=np.float32)
    theta += circle_a * 0.0           # Left lobe: phase 0
    theta += circle_b * np.pi         # Right lobe: phase π
    # Normalize where overlap exists
    overlap = np.minimum(circle_a, circle_b)
    theta = np.where(overlap > 0.1 * strength,
                     np.pi / 2 * strength,  # Midpoint: critical phase
                     theta)
    return phi, theta


def generate_and_gate(size: int, cx: int, cy: int, radius: int,
                      strength: float) -> Tuple[np.ndarray, np.ndarray]:
    """AND gate: narrow corridor with dual φ inputs, aligned phase convergence.

    BLECD domains: Feedback + Emergence
    Behavior: Requires synchronized coherence from multiple inputs.
    The corridor only sustains coherence if both ends are active.
    """
    # Two input circles connected by a narrow corridor
    offset = max(radius, 3)
    circle_a = _make_circle(size, cx - offset, cy, radius // 2 + 1) * strength
    circle_b = _make_circle(size, cx + offset, cy, radius // 2 + 1) * strength
    corridor = _make_corridor(size, cx, cy, length=offset, width=max(radius // 3, 2))
    corridor *= strength * 0.5

    phi = np.maximum(np.maximum(circle_a, circle_b), corridor)

    # Aligned phase: both inputs and corridor at same phase (constructive)
    theta = np.ones((size, size), dtype=np.float32) * np.pi / 4
    theta *= phi  # Phase only where phi is nonzero
    return phi, theta


def generate_delay_gate(size: int, cx: int, cy: int, radius: int,
                        strength: float) -> Tuple[np.ndarray, np.ndarray]:
    """DELAY gate: toroidal φ basin with rotational θ trap.

    BLECD domains: Saturation + Feedback
    Behavior: Holds symbolic potential temporarily; releases later.
    The rotational phase creates a temporal buffer.
    """
    # Toroidal (ring) φ: high at radius, low at center
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2).astype(np.float32)
    ring = np.exp(-((dist - radius)**2) / max(radius, 1)).astype(np.float32)
    phi = ring * strength

    # Rotational θ: phase swirls around the ring (temporal trap)
    theta = np.arctan2((y - cy).astype(np.float32),
                       (x - cx).astype(np.float32) + 1e-10).astype(np.float32)
    theta = ((theta + np.pi) * phi).astype(np.float32)
    return phi, theta


def generate_diode_gate(size: int, cx: int, cy: int, radius: int,
                        strength: float) -> Tuple[np.ndarray, np.ndarray]:
    """DIODE gate: asymmetric φ gradient, locked phase barrier.

    BLECD domains: Decay + Boundary
    Behavior: Allows symbolic coherence in one direction only.
    Forward (right) side has high φ; backward (left) is suppressed.
    """
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2).astype(np.float32)
    base = np.clip(1.0 - dist / max(radius, 1), 0.0, 1.0)

    # Asymmetric: multiply by sigmoid in x-direction
    sigmoid_x = 1.0 / (1.0 + np.exp(-(x - cx).astype(np.float32) * 3.0 / max(radius, 1)))
    phi = (base * sigmoid_x * strength).astype(np.float32)

    # Locked phase barrier: steep θ gradient at the boundary
    theta = np.where(
        (x - cx).astype(np.float32) < 0,
        np.pi * base * strength,     # Backward: high phase (barrier)
        0.1 * base * strength,        # Forward: low phase (passthrough)
    ).astype(np.float32)
    return phi, theta


# Gate type → generator function
_GATE_GENERATORS = {
    FieldGateType.YES: generate_yes_gate,
    FieldGateType.XOR: generate_xor_gate,
    FieldGateType.AND: generate_and_gate,
    FieldGateType.DELAY: generate_delay_gate,
    FieldGateType.DIODE: generate_diode_gate,
}


# ─── The Compiler ─────────────────────────────────────────────────────────

def compile_contradiction(
    bvec_a: BVec,
    bvec_b: BVec,
    field_size: int = 64,
    sgt_threshold: float = 2.0,
    sgt_mean: float = 0.0,
    sgt_var: float = 0.01,
) -> CompiledContradiction:
    """Compile a BFECDS contradiction into field seed geometries.

    Three stages (from BLC paper):
    1. Contradiction Analysis: element-wise |a - b|, SGT-gated
    2. Domain Pair → Gate Type selection
    3. Gate Type → φ-θ seed geometry generation

    Parameters
    ----------
    bvec_a, bvec_b : BVec
        The two activation vectors whose contradiction to resolve.
    field_size : int
        Size of the PDE field (NxN).
    sgt_threshold : float
        SGT gate threshold for domain differences.
    sgt_mean, sgt_var : float
        Running stats for the SGT gate.

    Returns
    -------
    CompiledContradiction with field seeds ready to inject into the PDE.
    """
    arr_a = bvec_a.as_array()
    arr_b = bvec_b.as_array()
    diff = np.abs(arr_a - arr_b)

    result = CompiledContradiction(
        total_contradiction=float(np.linalg.norm(diff)),
        source_bvec=bvec_a,
        target_bvec=bvec_b,
    )

    # Stage 1: Find significant domain contradictions (SGT-gated)
    significant_indices = []
    for i, domain in enumerate(DOMAIN_NAMES):
        significant, z_score = gate_decision(
            float(diff[i]), sgt_mean, sgt_var, sgt_threshold
        )
        if significant:
            significant_indices.append((i, domain, float(diff[i])))
            result.significant_domains.append(domain)

    if not significant_indices:
        return result  # No significant contradictions — nothing to compile

    # Stage 2: Sort by strength, pair top two domains → gate type
    significant_indices.sort(key=lambda x: x[2], reverse=True)

    # Take the top two contradicting domains
    top_domain = significant_indices[0][1]
    second_domain = significant_indices[1][1] if len(significant_indices) > 1 else top_domain
    pair = (top_domain, second_domain)

    # Look up the gate type for this domain pair
    gate_type = DOMAIN_PAIR_GATE.get(pair)
    if gate_type is None:
        # No exact pair match — default based on top domain
        gate_defaults = {"B": FieldGateType.YES, "F": FieldGateType.AND,
                         "E": FieldGateType.AND, "C": FieldGateType.XOR,
                         "D": FieldGateType.DIODE, "S": FieldGateType.DELAY}
        gate_type = gate_defaults.get(top_domain, FieldGateType.XOR)

    # Stage 3: Generate the φ-θ seed geometry
    # Place the gate near the center of the field, with radius proportional
    # to contradiction strength
    strength = min(result.total_contradiction / 2.0, 0.5)  # Cap at 0.5
    radius = max(3, int(field_size * strength * 0.3))
    cx = field_size // 2
    cy = field_size // 2

    # Use deterministic seed from the contradiction itself
    rng = np.random.default_rng(int(np.sum(diff * 10000)))
    # Jitter center slightly so repeated contradictions don't stack exactly
    cx += rng.integers(-field_size // 8, field_size // 8)
    cy += rng.integers(-field_size // 8, field_size // 8)
    cx = np.clip(cx, radius + 2, field_size - radius - 2)
    cy = np.clip(cy, radius + 2, field_size - radius - 2)

    generator = _GATE_GENERATORS[gate_type]
    phi_patch, theta_patch = generator(field_size, cx, cy, radius, strength)

    result.seeds.append(FieldSeed(
        phi_patch=phi_patch,
        theta_patch=theta_patch,
        gate_type=gate_type,
        center_x=cx,
        center_y=cy,
        radius=radius,
        strength=strength,
        source_domains=(top_domain, second_domain),
        contradiction=result.total_contradiction,
    ))

    # If there are additional significant domains beyond the top pair,
    # generate secondary seeds at offset positions
    for idx in range(2, min(len(significant_indices), 4)):
        extra_domain = significant_indices[idx][1]
        extra_strength = float(significant_indices[idx][2]) * 0.3
        extra_pair = (top_domain, extra_domain)
        extra_gate = DOMAIN_PAIR_GATE.get(extra_pair, gate_type)
        extra_radius = max(2, int(radius * 0.6))
        ex = cx + rng.integers(-radius, radius)
        ey = cy + rng.integers(-radius, radius)
        ex = np.clip(ex, extra_radius + 2, field_size - extra_radius - 2)
        ey = np.clip(ey, extra_radius + 2, field_size - extra_radius - 2)

        gen = _GATE_GENERATORS[extra_gate]
        ep, et = gen(field_size, ex, ey, extra_radius, extra_strength)
        result.seeds.append(FieldSeed(
            phi_patch=ep, theta_patch=et, gate_type=extra_gate,
            center_x=ex, center_y=ey, radius=extra_radius,
            strength=extra_strength,
            source_domains=(top_domain, extra_domain),
            contradiction=float(significant_indices[idx][2]),
        ))

    return result


def inject_seeds(field, compilation: CompiledContradiction) -> None:
    """Inject compiled seed geometries into a FractalField.

    This is the final step: the compiled contradiction becomes a
    physical perturbation in the PDE field. The field then evolves
    and resolves the contradiction through its own dynamics.

    Parameters
    ----------
    field : FractalField
        The active PDE field to perturb.
    compilation : CompiledContradiction
        Output of compile_contradiction().
    """
    from eris.config import xp, to_gpu

    for seed in compilation.seeds:
        phi_add = to_gpu(seed.phi_patch)
        theta_add = to_gpu(seed.theta_patch)

        # Add φ perturbation (clamped to [0, 1])
        field.phi = xp.clip(field.phi + phi_add, 0.0, 1.0)

        # Add θ perturbation (wrapped to [0, 2π])
        field.theta = (field.theta + theta_add) % (2.0 * float(np.pi))
