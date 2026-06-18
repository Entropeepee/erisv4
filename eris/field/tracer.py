"""
Symbolic Jet Tracer
====================

After pulse propagation, the lattice contains patterns of energy
distributed across cells. The jet tracer extracts *contiguous paths*
of high-energy cells — these are the "symbolic jets" that represent
coherent information flows through the computational substrate.

Each jet carries:
- Path: sequence of (q, r) coordinates
- Mean torsion: average torsion along the path
- Total energy: sum of cell energies along the path
- Metadata: accumulated from pulses that created this path

Hotspot detection finds cells where multiple jets converge —
these are natural "attention" points in the lattice.

Usage:
    from eris.field.tracer import SymbolicJetTracer

    tracer = SymbolicJetTracer(lattice=grid)
    jets = tracer.extract_jets(min_energy=0.1)
    hotspots = tracer.find_convergence_hotspots(jets)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
from eris.config import to_numpy, xp

from eris.field.lattice import HexLogicGrid, hex_neighbor


@dataclass
class SymbolicJet:
    """A contiguous path of active cells in the hex lattice.

    Represents a coherent information flow — the lattice equivalent
    of a "stream of thought" or an "attentional beam."

    Attributes
    ----------
    path : list of (q, r) tuples
        Ordered sequence of cell coordinates.
    total_energy : float
        Sum of cell energies along this path.
    mean_energy : float
        Average energy per cell.
    torsion_signature : float
        Encoded from the energy gradient along the path.
        High torsion = path is curving (changing direction in BFECDS space).
        Low torsion = path is straight (consistent information flow).
    """
    path: List[Tuple[int, int]] = field(default_factory=list)
    total_energy: float = 0.0
    mean_energy: float = 0.0
    torsion_signature: float = 0.0

    @property
    def length(self) -> int:
        return len(self.path)

    @property
    def head(self) -> Optional[Tuple[int, int]]:
        """Where this jet starts."""
        return self.path[0] if self.path else None

    @property
    def tail(self) -> Optional[Tuple[int, int]]:
        """Where this jet ends."""
        return self.path[-1] if self.path else None


class SymbolicJetTracer:
    """Extracts jets (contiguous active paths) from the hex lattice.

    Parameters
    ----------
    lattice : HexLogicGrid
        The lattice to trace.
    min_energy : float
        Minimum cell energy to be considered "active."
    min_jet_length : int
        Minimum path length to be considered a jet.
    """

    def __init__(
        self,
        lattice: HexLogicGrid,
        min_energy: float = 0.05,
        min_jet_length: int = 3,
    ):
        self.lattice = lattice
        self.min_energy = min_energy
        self.min_jet_length = min_jet_length

    def extract_jets(self) -> List[SymbolicJet]:
        """Extract all jets from the current lattice state.

        Algorithm:
        1. Find all active cells (energy > min_energy)
        2. Starting from highest-energy unvisited cell, greedily
           follow the highest-energy neighbor chain
        3. Record each chain as a jet
        4. Repeat until all active cells are visited
        """
        # Collect active cells sorted by energy (descending)
        active: List[Tuple[Tuple[int, int], float]] = []
        for (q, r), cell in self.lattice.cells.items():
            if cell.energy >= self.min_energy:
                active.append(((q, r), cell.energy))

        if not active:
            return []

        active.sort(key=lambda x: x[1], reverse=True)
        visited: Set[Tuple[int, int]] = set()
        jets: List[SymbolicJet] = []

        for start_coord, start_energy in active:
            if start_coord in visited:
                continue

            # Trace a path greedily following highest-energy neighbors
            path = [start_coord]
            energies = [start_energy]
            visited.add(start_coord)

            current = start_coord
            while True:
                best_neighbor = None
                best_energy = 0.0

                q, r = current
                for d in range(6):
                    nq, nr = hex_neighbor(q, r, d)
                    ncoord = (nq, nr)
                    if ncoord in visited:
                        continue
                    ncell = self.lattice.get_cell(nq, nr)
                    if ncell is None or ncell.energy < self.min_energy:
                        continue
                    if ncell.energy > best_energy:
                        best_energy = ncell.energy
                        best_neighbor = ncoord

                if best_neighbor is None:
                    break  # Dead end

                path.append(best_neighbor)
                energies.append(best_energy)
                visited.add(best_neighbor)
                current = best_neighbor

            # Only keep jets above minimum length
            if len(path) < self.min_jet_length:
                continue

            # Compute torsion signature from energy gradient along path
            energy_arr = np.array(energies, dtype=np.float32)
            if len(energy_arr) > 2:
                gradient = np.diff(energy_arr)
                # Torsion = second derivative (curvature of energy along path)
                second_deriv = np.diff(gradient)
                torsion = float(np.sqrt(xp.mean(second_deriv ** 2)))
            else:
                torsion = 0.0

            jets.append(SymbolicJet(
                path=path,
                total_energy=float(xp.sum(energy_arr)),
                mean_energy=float(xp.mean(energy_arr)),
                torsion_signature=torsion,
            ))

        # Sort jets by total energy (most significant first)
        jets.sort(key=lambda j: j.total_energy, reverse=True)
        return jets

    def find_convergence_hotspots(
        self,
        jets: List[SymbolicJet],
        top_n: int = 5,
    ) -> List[Tuple[Tuple[int, int], int, float]]:
        """Find cells where multiple jets converge.

        Returns list of (coord, jet_count, total_energy) sorted by jet_count.
        These are natural "attention nodes" — places where multiple
        information flows meet.
        """
        coord_counts: Dict[Tuple[int, int], int] = {}
        coord_energy: Dict[Tuple[int, int], float] = {}

        for jet in jets:
            for coord in jet.path:
                coord_counts[coord] = coord_counts.get(coord, 0) + 1
                cell = self.lattice.get_cell(*coord)
                if cell is not None:
                    coord_energy[coord] = cell.energy

        # Only cells where 2+ jets overlap
        hotspots = [
            (coord, count, coord_energy.get(coord, 0.0))
            for coord, count in coord_counts.items()
            if count >= 2
        ]
        hotspots.sort(key=lambda x: x[1], reverse=True)
        return hotspots[:top_n]
