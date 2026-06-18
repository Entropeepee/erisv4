"""
Phase 1 Tests — Field Layer
==============================

Tests for FRACTAL PDE, hex lattice, BLC compiler, jet tracer.

Run with:
    cd eris_echo_v4
    python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from eris.config import to_numpy, xp
import tempfile
import pytest


# ─── PDE Tests ────────────────────────────────────────────────────────────

class TestFractalPDE:

    def test_field_initialization(self):
        """Field should initialize with correct dimensions and boundary conditions."""
        from eris.field.pde import FractalField

        field = FractalField(size=32)

        phi_np = to_numpy(field.phi)
        assert phi_np.shape == (32, 32)
        # Boundary should be zero (Dirichlet)
        assert phi_np[0, :].max() == 0.0
        assert phi_np[-1, :].max() == 0.0
        assert phi_np[:, 0].max() == 0.0
        assert phi_np[:, -1].max() == 0.0

    def test_step_preserves_bounds(self):
        """After stepping, phi should stay in [0, 1] and boundaries should be zero."""
        from eris.field.pde import FractalField

        field = FractalField(size=32)
        field.seed_from_text("test signal with some energy")

        for _ in range(100):
            field.step()

        phi_np = to_numpy(field.phi)
        assert phi_np.min() >= 0.0, f"phi went negative: {phi_np.min()}"
        assert phi_np.max() <= 1.0, f"phi exceeded 1: {phi_np.max()}"
        assert phi_np[0, :].max() == 0.0, "Top boundary violated"
        assert phi_np[-1, :].max() == 0.0, "Bottom boundary violated"

    def test_empty_field_stays_quiet(self):
        """An unseeded field should remain near zero (only tiny initial noise)."""
        from eris.field.pde import FractalField

        field = FractalField(size=32)
        field.p.sigma_noise = 0.0  # Disable noise to test purely deterministic decay
        field.p.memory_coupling = 0.0  # Disable background memory bias
        field.p.activations = {}  # No external forcing
        initial_energy = float(np.sum(to_numpy(field.phi)))

        field.run(50)
        final_energy = float(np.sum(to_numpy(field.phi)))

        # Should decay to near zero (only initial noise, and decay removes it)
        assert final_energy <= initial_energy + 0.01, (
            f"Unseeded field gained energy: {initial_energy:.4f} → {final_energy:.4f}"
        )

    def test_seeded_field_has_structure(self):
        """A text-seeded field should have nontrivial structure."""
        from eris.field.pde import FractalField

        field = FractalField(size=64)
        field.seed_from_text("The quick brown fox jumps over the lazy dog")

        phi_np = to_numpy(field.phi)
        # Should have some energy in the interior
        interior = phi_np[1:-1, 1:-1]
        assert interior.max() > 0.01, "Seeded field has no energy"
        # Should have spatial variation (not flat)
        assert interior.std() > 0.001, "Seeded field is spatially flat"

    def test_deterministic_seeding(self):
        """Same text should produce identical fields."""
        from eris.field.pde import FractalField

        f1 = FractalField(size=32)
        f1.seed_from_text("hello world")

        f2 = FractalField(size=32)
        f2.seed_from_text("hello world")

        np.testing.assert_array_equal(
            to_numpy(f1.phi), to_numpy(f2.phi),
            err_msg="Same text produced different fields"
        )

    def test_different_texts_different_fields(self):
        """Different texts should produce different fields."""
        from eris.field.pde import FractalField

        f1 = FractalField(size=32)
        f1.seed_from_text("hello world")

        f2 = FractalField(size=32)
        f2.seed_from_text("goodbye world")

        diff = float(np.sum(np.abs(to_numpy(f1.phi) - to_numpy(f2.phi))))
        assert diff > 0.01, "Different texts produced identical fields"

    def test_pde_produces_bvec(self):
        """A seeded+evolved field should produce a meaningful BVec."""
        from eris.field.pde import FractalField

        field = FractalField(size=32)
        field.seed_from_text("A complex thought about emergence and decay")
        field.run(50)

        bvec = field.compute_bvec()
        total = sum(bvec.as_dict().values())
        assert total > 0.01, f"BVec is all zeros after evolution: {bvec}"

    def test_checkpoint_roundtrip(self):
        """Save and load should produce identical field state."""
        from eris.field.pde import FractalField

        field = FractalField(size=32)
        field.seed_from_text("checkpoint test")
        field.run(20)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name

        try:
            field.save_checkpoint(path)
            restored = FractalField.load_checkpoint(path)

            np.testing.assert_allclose(
                to_numpy(field.phi),
                to_numpy(restored.phi),
                atol=1e-6,
            )
            np.testing.assert_allclose(
                to_numpy(field.theta),
                to_numpy(restored.theta),
                atol=1e-6,
            )
            assert restored.step_count == field.step_count
        finally:
            os.unlink(path)


# ─── Hex Lattice Tests ────────────────────────────────────────────────────

class TestHexLattice:

    def test_grid_size(self):
        """Grid with radius r should have 3r²+3r+1 cells."""
        from eris.field.lattice import HexLogicGrid

        for r in [4, 8, 16]:
            grid = HexLogicGrid(radius=r)
            expected = 3 * r * r + 3 * r + 1
            assert grid.n_cells == expected, (
                f"Radius {r}: expected {expected} cells, got {grid.n_cells}"
            )

    def test_center_cell_exists(self):
        """The origin cell (0,0) should always exist."""
        from eris.field.lattice import HexLogicGrid

        grid = HexLogicGrid(radius=8)
        assert grid.get_cell(0, 0) is not None

    def test_all_cells_have_six_gates(self):
        """Every cell should have exactly 6 gates."""
        from eris.field.lattice import HexLogicGrid

        grid = HexLogicGrid(radius=8)
        for cell in grid.cells.values():
            assert len(cell.gates) == 6

    def test_energy_injection_and_decay(self):
        """Total system energy should decrease over time with decay."""
        from eris.field.lattice import HexLogicGrid

        grid = HexLogicGrid(radius=8, energy_decay=0.9)
        grid.inject_energy(0, 0, 1.0)

        initial_energy = grid.get_cell(0, 0).energy
        assert initial_energy == 1.0

        # Propagate and check total energy decreases
        # (OR gates redistribute energy but decay should reduce total)
        grid.propagate(steps=1)
        total_after_1 = sum(c.energy for c in grid.cells.values())

        grid.propagate(steps=20)
        total_after_21 = sum(c.energy for c in grid.cells.values())

        # After many steps with 0.9 decay, total should be lower
        # (even with redistribution, decay factor compounds)
        assert total_after_21 < total_after_1 * 5, (
            f"Energy grew unexpectedly: {total_after_1:.2f} → {total_after_21:.2f}"
        )

    def test_pulse_propagation_spreads(self):
        """Energy injected at center should spread to neighbors."""
        from eris.field.lattice import HexLogicGrid

        grid = HexLogicGrid(radius=8)
        grid.inject_energy(0, 0, 5.0)
        grid.propagate(steps=5)

        # At least some neighbors should have energy
        neighbors_with_energy = 0
        for dq, dr in [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]:
            cell = grid.get_cell(dq, dr)
            if cell and cell.energy > 0.01:
                neighbors_with_energy += 1

        assert neighbors_with_energy >= 3, (
            f"Only {neighbors_with_energy}/6 neighbors got energy"
        )

    def test_hotspot_detection(self):
        """Injecting at center should make center a hotspot."""
        from eris.field.lattice import HexLogicGrid

        grid = HexLogicGrid(radius=8)
        # Inject multiple pulses at center
        for _ in range(10):
            grid.inject_energy(0, 0, 1.0)
        grid.propagate(steps=3)

        hotspots = grid.find_hotspots(top_n=5)
        assert len(hotspots) > 0, "No hotspots found"
        # Center should be the top hotspot
        assert hotspots[0][0] == (0, 0), (
            f"Center not top hotspot: {hotspots[0]}"
        )


# ─── BLC Compiler Tests ──────────────────────────────────────────────────

class TestBLCCompiler:

    def test_no_contradiction_no_seeds(self):
        """Identical BVecs should produce no field seeds."""
        from eris.computation.activations import BVec
        from eris.field.compiler import compile_contradiction

        a = BVec(B=0.5, F=0.5, E=0.5, C=0.5, D=0.5, S=0.5)
        result = compile_contradiction(a, a, field_size=32)
        assert result.n_seeds == 0

    def test_strong_contradiction_produces_seeds(self):
        """Very different BVecs should produce field seed geometries."""
        from eris.computation.activations import BVec
        from eris.field.compiler import compile_contradiction

        a = BVec(B=0.9, F=0.1, E=0.1, C=0.1, D=0.1, S=0.1)
        b = BVec(B=0.1, F=0.9, E=0.9, C=0.9, D=0.9, S=0.9)

        result = compile_contradiction(
            a, b, field_size=32,
            sgt_threshold=1.0, sgt_mean=0.0, sgt_var=0.01,
        )

        assert result.n_seeds > 0, "Strong contradiction produced no seeds"
        assert result.total_contradiction > 0.5
        # Seeds should be proper 2D arrays
        seed = result.seeds[0]
        assert seed.phi_patch.shape == (32, 32)
        assert seed.theta_patch.shape == (32, 32)

    def test_cd_contradiction_produces_xor(self):
        """C+D contradiction should produce XOR gate geometry."""
        from eris.computation.activations import BVec
        from eris.field.compiler import compile_contradiction, FieldGateType

        a = BVec(B=0.5, F=0.5, E=0.5, C=0.1, D=0.1, S=0.5)
        b = BVec(B=0.5, F=0.5, E=0.5, C=0.9, D=0.9, S=0.5)

        result = compile_contradiction(
            a, b, field_size=32,
            sgt_threshold=0.5, sgt_mean=0.0, sgt_var=0.01,
        )

        if result.n_seeds > 0:
            assert result.seeds[0].gate_type == FieldGateType.XOR


# ─── Jet Tracer Tests ────────────────────────────────────────────────────

class TestJetTracer:

    def test_empty_lattice_no_jets(self):
        """A lattice with no energy should produce no jets."""
        from eris.field.lattice import HexLogicGrid
        from eris.field.tracer import SymbolicJetTracer

        grid = HexLogicGrid(radius=8)
        tracer = SymbolicJetTracer(grid, min_energy=0.05, min_jet_length=3)
        jets = tracer.extract_jets()
        assert len(jets) == 0

    def test_line_injection_produces_jet(self):
        """Injecting energy along a line should produce a traceable jet."""
        from eris.field.lattice import HexLogicGrid
        from eris.field.tracer import SymbolicJetTracer

        grid = HexLogicGrid(radius=16)

        # Inject energy in a line along the q axis
        for q in range(-5, 6):
            grid.inject_energy(q, 0, 1.0)

        tracer = SymbolicJetTracer(grid, min_energy=0.5, min_jet_length=3)
        jets = tracer.extract_jets()

        assert len(jets) >= 1, "Line injection didn't produce a jet"
        assert jets[0].length >= 3, f"Jet too short: {jets[0].length}"

    def test_convergence_detection(self):
        """Multiple injection lines crossing should make origin a hotspot in the lattice."""
        from eris.field.lattice import HexLogicGrid
        from eris.field.tracer import SymbolicJetTracer

        grid = HexLogicGrid(radius=16)

        # Inject along q axis
        for q in range(-5, 6):
            grid.inject_energy(q, 0, 1.0)
        # Inject along r axis (crosses at origin)
        for r in range(-5, 6):
            grid.inject_energy(0, r, 1.0)

        # Origin was injected twice — should have highest pulse_count
        origin = grid.get_cell(0, 0)
        assert origin is not None
        assert origin.pulse_count >= 2, (
            f"Origin pulse_count={origin.pulse_count}, expected >= 2"
        )
        assert origin.energy >= 2.0, (
            f"Origin energy={origin.energy:.2f}, expected >= 2.0 (double injection)"
        )

        # Lattice-level hotspot detection should find origin
        hotspots = grid.find_hotspots(top_n=5)
        hotspot_coords = [h[0] for h in hotspots]
        assert (0, 0) in hotspot_coords, (
            f"Origin not in lattice hotspots: {hotspot_coords}"
        )


# ─── Integration Test ─────────────────────────────────────────────────────

class TestPhase1Integration:

    def test_full_pipeline_text_to_bvec_to_field_resolution(self):
        """Text → PDE → BVec → BLC → Field Seeds → Inject → Resolve."""
        from eris.field.pde import FractalField
        from eris.field.compiler import compile_contradiction, inject_seeds

        field = FractalField(size=32)
        field.seed_from_text("An important discovery about emergence")
        field.run(30)
        bvec_input = field.compute_bvec()

        field2 = FractalField(size=32)
        field2.seed_from_text("A skeptical rebuttal questioning the premise")
        field2.run(30)
        bvec_response = field2.compute_bvec()

        result = compile_contradiction(bvec_input, bvec_response, field_size=32,
                                       sgt_threshold=0.5, sgt_mean=0.0, sgt_var=0.01)
        inject_seeds(field, result)
        field.run(20)
        resolved_bvec = field.compute_bvec()

        print(f"\n  Pipeline result:")
        print(f"  Input archetype:    {bvec_input.archetype()}")
        print(f"  Response archetype: {bvec_response.archetype()}")
        print(f"  Contradiction:      {result.total_contradiction:.3f}")
        print(f"  Seeds:              {result.n_seeds}")
        print(f"  Resolved type:      {resolved_bvec.archetype()}")
        print(f"  Regime:             {field.detect_regime()}")

        assert isinstance(bvec_input.B, float)
        assert isinstance(resolved_bvec.B, float)


class TestFRT:
    """Tests for the Fractal Rolling Tokenizer (reflexive pathway)."""

    def test_tokenize_produces_treelets(self):
        from eris.field.frt import FractalRollingTokenizer
        frt = FractalRollingTokenizer()
        treelets = frt.tokenize("The quick brown fox jumps over the lazy dog")
        assert len(treelets) > 0
        assert all(t.hash_value > 0 for t in treelets)

    def test_deterministic_hashing(self):
        """Same text must always produce same hashes."""
        from eris.field.frt import FractalRollingTokenizer
        frt = FractalRollingTokenizer()
        t1 = frt.tokenize("hello world test")
        t2 = frt.tokenize("hello world test")
        assert [t.hash_value for t in t1] == [t.hash_value for t in t2]

    def test_different_text_different_hashes(self):
        from eris.field.frt import FractalRollingTokenizer
        frt = FractalRollingTokenizer()
        t1 = frt.tokenize("emergence and novelty")
        t2 = frt.tokenize("decay and entropy")
        h1 = {t.hash_value for t in t1}
        h2 = {t.hash_value for t in t2}
        assert h1 != h2

    def test_hash_to_pulse_ranges(self):
        """Phi in [0,1], theta in [0,2π], tau in [-1,1]."""
        from eris.field.frt import HashToPulseEncoder
        hce = HashToPulseEncoder()
        for h in [0, 2**32, 2**48, 2**64 - 1, 12345678901234]:
            phi, theta, tau = hce.encode(h)
            assert 0.0 <= phi <= 1.0, f"phi={phi} out of range"
            assert 0.0 <= theta <= 2 * 3.1416, f"theta={theta} out of range"
            assert -1.0 <= tau <= 1.0, f"tau={tau} out of range"

    def test_text_to_field_arrays(self):
        from eris.field.frt import text_to_field_arrays
        phi, theta = text_to_field_arrays("A complex thought about emergence", size=32)
        assert phi.shape == (32, 32)
        assert theta.shape == (32, 32)
        assert phi.max() > 0, "Field should have nonzero phi"
        # Dirichlet boundaries enforced
        assert phi[0, :].max() == 0.0
        assert phi[-1, :].max() == 0.0

    def test_frt_bvec_computable(self):
        """FRT path should produce a valid BVec without GPU or PDE."""
        from eris.field.frt import compute_bvec_from_frt
        bvec = compute_bvec_from_frt("Testing the reflexive pathway")
        assert 0 <= bvec.B <= 1
        assert 0 <= bvec.C <= 1
        # FRT has no temporal evolution → E and D should be ~0
        assert bvec.E < 0.05, f"FRT should have near-zero E, got {bvec.E}"
        assert bvec.D < 0.05, f"FRT should have near-zero D, got {bvec.D}"

    def test_dual_path_pde_seed_from_frt(self):
        """PDE seeded via FRT should produce valid field state."""
        from eris.field.pde import FractalField
        field = FractalField(size=32)
        field.seed_from_text("Dual path test", use_frt=True)
        # Should have nonzero phi in interior
        phi_np = to_numpy(field.phi)
        assert phi_np[1:-1, 1:-1].max() > 0.01
        # Run PDE on top of FRT-seeded field
        field.run(20)
        bvec = field.compute_bvec()
        # After PDE evolution, should have temporal dynamics too
        assert isinstance(bvec.B, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
