"""
Hexagonal Logic Grid
=====================

The hex lattice is the discrete substrate underlying the continuous PDE.
Think of the PDE as the "fluid" and the lattice as the "crystalline scaffold."

Each cell has 6 edges (hexagonal geometry), each assigned a logic gate:
    AND, OR, XOR, NAND, NOR, DIODE, DELAY

Pulses propagate through the lattice by passing through edge gates.
The BLC compiler translates contradictions (BFECDS differences) into
gate programs that rewire the lattice topology.

Axial coordinate system (q, r) for hex grids — the standard choice
for computational hex geometry. Neighbor offsets depend on parity.

Usage:
    from eris.field.lattice import HexLogicGrid, GateType

    grid = HexLogicGrid(radius=16)
    grid.inject_pulse((0, 0), pulse)
    grid.propagate(steps=10)
    hotspots = grid.find_hotspots()
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Tuple, Optional, Set
import numpy as np


class GateType(IntEnum):
    """Logic gates available on hex cell edges."""
    AND = 0
    OR = 1
    XOR = 2
    NAND = 3
    NOR = 4
    DIODE = 5   # One-way: passes signal only in one direction
    DELAY = 6   # Passes signal with a one-step delay


# Axial hex neighbors: 6 directions for axial coordinates (q, r)
# These are the standard cube-coordinate directions projected to axial.
HEX_DIRECTIONS = [
    (1, 0), (1, -1), (0, -1),
    (-1, 0), (-1, 1), (0, 1),
]


def hex_neighbor(q: int, r: int, direction: int) -> Tuple[int, int]:
    """Get axial coordinates of neighbor in given direction (0-5)."""
    dq, dr = HEX_DIRECTIONS[direction]
    return (q + dq, r + dr)


def hex_distance(q1: int, r1: int, q2: int, r2: int) -> int:
    """Hex Manhattan distance between two axial coordinates."""
    dq = q2 - q1
    dr = r2 - r1
    return (abs(dq) + abs(dq + dr) + abs(dr)) // 2


@dataclass
class HexCell:
    """A single cell in the hex logic grid.

    Attributes
    ----------
    q, r : int
        Axial coordinates.
    gates : list of GateType
        Logic gate on each of the 6 edges. Index = direction (0-5).
    energy : float
        Current activation energy in this cell. Pulses deposit energy;
        it decays over time.
    pulse_count : int
        How many pulses have passed through (for hotspot detection).
    """
    q: int = 0
    r: int = 0
    gates: list = field(default_factory=lambda: [GateType.OR] * 6)
    energy: float = 0.0
    pulse_count: int = 0
    delay_buffer: list = field(default_factory=lambda: [0.0] * 6)


class HexLogicGrid:
    """Hexagonal logic grid with pulse propagation.

    Creates a hex grid of given radius (number of rings from center).
    A radius of 16 gives ~817 cells.

    Parameters
    ----------
    radius : int
        Number of rings around center. Total cells ≈ 3r² + 3r + 1.
    default_gate : GateType
        Default gate type for all edges.
    energy_decay : float
        Per-step decay factor for cell energy (0.9 = 10% loss per step).
    """

    def __init__(
        self,
        radius: int = 16,
        default_gate: GateType = GateType.OR,
        energy_decay: float = 0.95,
    ):
        self.radius = radius
        self.energy_decay = energy_decay
        self.cells: Dict[Tuple[int, int], HexCell] = {}

        # Generate hex grid using axial coordinates
        for q in range(-radius, radius + 1):
            r_min = max(-radius, -q - radius)
            r_max = min(radius, -q + radius)
            for r in range(r_min, r_max + 1):
                self.cells[(q, r)] = HexCell(
                    q=q, r=r,
                    gates=[default_gate] * 6,
                )

        self.step_count: int = 0

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def get_cell(self, q: int, r: int) -> Optional[HexCell]:
        """Get cell at (q, r), or None if out of bounds."""
        return self.cells.get((q, r))

    def set_gate(self, q: int, r: int, direction: int, gate: GateType) -> None:
        """Set the gate type on a specific edge of a cell."""
        cell = self.cells.get((q, r))
        if cell is not None and 0 <= direction < 6:
            cell.gates[direction] = gate

    def inject_energy(self, q: int, r: int, amount: float) -> None:
        """Inject energy into a cell (equivalent to injecting a pulse)."""
        cell = self.cells.get((q, r))
        if cell is not None:
            cell.energy += amount
            cell.pulse_count += 1

    def _gate_output(self, gate: GateType, input_a: float, input_b: float) -> float:
        """Compute gate output for two input signals.

        For propagation, input_a is the source cell energy and
        input_b is the destination cell energy. Output determines
        how much energy transfers.
        """
        # Threshold inputs to binary for logic operations
        a = input_a > 0.1
        b = input_b > 0.1

        if gate == GateType.AND:
            return input_a if (a and b) else 0.0
        elif gate == GateType.OR:
            return input_a if (a or b) else 0.0
        elif gate == GateType.XOR:
            return input_a if (a != b) else 0.0
        elif gate == GateType.NAND:
            return input_a if not (a and b) else 0.0
        elif gate == GateType.NOR:
            return 0.0 if (a or b) else input_a
        elif gate == GateType.DIODE:
            return input_a  # Always passes (one-way enforced by caller)
        elif gate == GateType.DELAY:
            return 0.0  # Handled separately via delay buffer
        return 0.0

    def propagate(self, steps: int = 1) -> None:
        """Run pulse propagation for N steps.

        Each step:
        1. For each cell with energy, push through gates to neighbors
        2. Deduct transferred energy from source cells (conservation)
        3. Apply delay buffers
        4. Decay all cell energies
        """
        for _ in range(steps):
            # Collect energy transfers and outflows
            transfers: Dict[Tuple[int, int], float] = {}
            outflows: Dict[Tuple[int, int], float] = {}
            # Transfer fraction per edge. Total max outflow = 6 * 0.08 = 0.48.
            transfer_frac = 0.08

            for (q, r), cell in self.cells.items():
                if cell.energy < 0.01:  # Skip quiet cells
                    continue

                for d in range(6):
                    nq, nr = hex_neighbor(q, r, d)
                    neighbor = self.cells.get((nq, nr))
                    if neighbor is None:
                        continue

                    gate = cell.gates[d]

                    if gate == GateType.DELAY:
                        old_delayed = cell.delay_buffer[d]
                        sent = cell.energy * transfer_frac
                        cell.delay_buffer[d] = sent
                        # Track outflow from source
                        src = (q, r)
                        outflows[src] = outflows.get(src, 0.0) + sent
                        # Release previously delayed energy
                        if old_delayed > 0.01:
                            key = (nq, nr)
                            transfers[key] = transfers.get(key, 0.0) + old_delayed
                    else:
                        output = self._gate_output(gate, cell.energy, neighbor.energy)
                        if output > 0.01:
                            sent = output * transfer_frac
                            key = (nq, nr)
                            transfers[key] = transfers.get(key, 0.0) + sent
                            # Track outflow from source
                            src = (q, r)
                            outflows[src] = outflows.get(src, 0.0) + sent

            # Deduct outflows from source cells (energy conservation)
            for (oq, or_), amount in outflows.items():
                cell = self.cells.get((oq, or_))
                if cell is not None:
                    cell.energy = max(0.0, cell.energy - amount)

            # Apply incoming transfers
            for (tq, tr), amount in transfers.items():
                cell = self.cells.get((tq, tr))
                if cell is not None:
                    cell.energy += amount
                    cell.pulse_count += 1

            # Decay all energies
            for cell in self.cells.values():
                cell.energy *= self.energy_decay
                for d in range(6):
                    cell.delay_buffer[d] *= self.energy_decay

            self.step_count += 1

    def find_hotspots(self, top_n: int = 10) -> List[Tuple[Tuple[int, int], float, int]]:
        """Find cells with highest pulse counts (convergence points).

        Returns list of ((q, r), energy, pulse_count) sorted by pulse_count descending.
        """
        scored = [
            ((c.q, c.r), c.energy, c.pulse_count)
            for c in self.cells.values()
            if c.pulse_count > 0
        ]
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_n]

    def get_energy_map(self) -> np.ndarray:
        """Return all cell energies as a flat array for analysis.

        Returns structured array with columns: q, r, energy, pulse_count.
        """
        data = [(c.q, c.r, c.energy, c.pulse_count)
                for c in self.cells.values()]
        return np.array(data, dtype=[
            ("q", np.int32), ("r", np.int32),
            ("energy", np.float32), ("pulse_count", np.int32),
        ])

    def reset_energy(self) -> None:
        """Clear all energy and pulse counts (keep gate topology)."""
        for cell in self.cells.values():
            cell.energy = 0.0
            cell.pulse_count = 0
            cell.delay_buffer = [0.0] * 6

    def snapshot(self) -> Dict:
        """Serialize for checkpointing."""
        cells_data = {}
        for (q, r), cell in self.cells.items():
            cells_data[f"{q},{r}"] = {
                "gates": [int(g) for g in cell.gates],
                "energy": cell.energy,
                "pulse_count": cell.pulse_count,
                "delay_buffer": cell.delay_buffer,
            }
        return {
            "radius": self.radius,
            "energy_decay": self.energy_decay,
            "step_count": self.step_count,
            "cells": cells_data,
        }

    @classmethod
    def from_snapshot(cls, data: Dict) -> "HexLogicGrid":
        """Restore from checkpoint."""
        grid = cls(radius=data["radius"], energy_decay=data["energy_decay"])
        grid.step_count = data["step_count"]
        for key, cell_data in data["cells"].items():
            q, r = map(int, key.split(","))
            cell = grid.cells.get((q, r))
            if cell is not None:
                cell.gates = [GateType(g) for g in cell_data["gates"]]
                cell.energy = cell_data["energy"]
                cell.pulse_count = cell_data["pulse_count"]
                cell.delay_buffer = cell_data.get("delay_buffer", [0.0] * 6)
        return grid
