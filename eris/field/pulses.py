"""
Symbolic Pulses
================

A SymbolicPulse is the quantum of information in the hex lattice.
It carries phi (coherence), theta (phase), and metadata through
the gate network. When pulses meet at a cell, they blend via a
9-gate logic that determines the combined output.

The "9-gate blend" is:
    For each of the 6 incoming edges: gate logic from source
    Plus 3 internal operations: blend, threshold, emit

This mirrors how information packets in the brain carry both
content (theta) and confidence (phi), and are combined at
synaptic junctions through nonlinear operations.

Usage:
    from eris.field.pulses import SymbolicPulse, blend_pulses

    p1 = SymbolicPulse(phi=0.8, theta=1.2, origin=(0, 0))
    p2 = SymbolicPulse(phi=0.3, theta=2.5, origin=(1, 0))
    blended = blend_pulses([p1, p2])
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np
from eris.config import to_numpy, xp


@dataclass
class SymbolicPulse:
    """A quantum of information propagating through the hex lattice.

    Attributes
    ----------
    phi : float
        Coherence strength [0, 1]. Decays over propagation distance.
    theta : float
        Phase angle [0, 2π]. Carries the "content" of the pulse.
    origin : tuple
        (q, r) coordinates where this pulse was injected.
    metadata : dict
        Arbitrary payload (text snippets, domain labels, etc.)
    timestamp : float
        When this pulse was created (Unix time).
    hops : int
        How many lattice cells this pulse has traversed.
    """
    phi: float = 0.5
    theta: float = 0.0
    origin: Tuple[int, int] = (0, 0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    hops: int = 0

    # Decay constants
    phi_decay_per_hop: float = 0.05  # Lose 5% coherence per hop

    def decay(self) -> None:
        """Apply one hop of propagation decay."""
        self.phi *= (1.0 - self.phi_decay_per_hop)
        self.hops += 1

    @property
    def is_alive(self) -> bool:
        """Pulse is considered dead below this coherence threshold."""
        return self.phi > 0.01

    def energy(self) -> float:
        """Total energy carried by this pulse."""
        return self.phi

    def as_dict(self) -> Dict[str, Any]:
        """Serialize for storage/logging."""
        return {
            "phi": self.phi,
            "theta": self.theta,
            "origin": list(self.origin),
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "hops": self.hops,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SymbolicPulse":
        return cls(
            phi=d["phi"],
            theta=d["theta"],
            origin=tuple(d["origin"]),
            metadata=d.get("metadata", {}),
            timestamp=d.get("timestamp", time.time()),
            hops=d.get("hops", 0),
        )


def blend_pulses(pulses: List[SymbolicPulse]) -> Optional[SymbolicPulse]:
    """Blend multiple pulses arriving at the same cell.

    The 9-gate blend:
    1-6: Each incoming pulse passes through its edge gate (handled
         by the lattice propagation, not here)
    7: Coherence blend — phi-weighted average of surviving pulses
    8: Phase blend — circular mean of theta values, weighted by phi
    9: Threshold — if blended phi < min_threshold, pulse dies

    Parameters
    ----------
    pulses : list of SymbolicPulse
        All pulses arriving at a single cell this step.

    Returns
    -------
    SymbolicPulse or None
        Blended pulse, or None if below threshold.
    """
    if not pulses:
        return None

    alive = [p for p in pulses if p.is_alive]
    if not alive:
        return None

    # Gate 7: Phi-weighted coherence blend
    total_phi = sum(p.phi for p in alive)
    if total_phi < 1e-10:
        return None

    blended_phi = total_phi / len(alive)  # Average, not sum (prevents explosion)

    # Gate 8: Circular mean of theta, weighted by phi
    sin_sum = sum(p.phi * np.sin(p.theta) for p in alive)
    cos_sum = sum(p.phi * np.cos(p.theta) for p in alive)
    blended_theta = float(np.arctan2(sin_sum, cos_sum)) % (2.0 * np.pi)

    # Gate 9: Threshold check
    min_threshold = 0.02
    if blended_phi < min_threshold:
        return None

    # Merge metadata (combine all keys, latest value wins for conflicts)
    merged_meta: Dict[str, Any] = {}
    for p in alive:
        merged_meta.update(p.metadata)

    # Origin is the highest-phi contributor
    strongest = max(alive, key=lambda p: p.phi)

    return SymbolicPulse(
        phi=blended_phi,
        theta=blended_theta,
        origin=strongest.origin,
        metadata=merged_meta,
        timestamp=time.time(),
        hops=max(p.hops for p in alive),
    )
