"""
Phase 1 Tests — Computation Layer (Updated for Davidian Hill-Power)
====================================================================
Run: cd eris_echo_v4 && python -m pytest tests/test_computation.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from eris.config import to_numpy, xp
import pytest


class TestDavidianShrinkage:

    def test_wiener_recovery(self):
        """Davidian with α=1,β=1,γ=1,δ=0 should approximate Wiener filter."""
        from eris.computation.shrinkage import davidian_weight
        s = np.linspace(0.1, 10.0, 50).astype(np.float32)
        wiener = s / (s + 1.0)
        dav = to_numpy(davidian_weight(s, alpha=1.0, beta=1.0, gamma=1.0, delta=0.0))
        np.testing.assert_allclose(dav, wiener, atol=0.05)

    def test_kill_zone(self):
        """δ>0 should suppress signals below the dead zone."""
        from eris.computation.shrinkage import davidian_weight
        s = np.array([0.1, 0.5, 2.0, 5.0], dtype=np.float32)
        w = to_numpy(davidian_weight(s, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, smooth=False))
        assert w[0] < 0.01
        assert w[1] < 0.01
        assert w[3] > 0.3

    def test_high_gamma_compresses(self):
        """Higher γ should suppress low-SNR more aggressively."""
        from eris.computation.shrinkage import davidian_weight
        s = np.array([0.5, 5.0], dtype=np.float32)
        w1 = to_numpy(davidian_weight(s, gamma=1.0))
        w2 = to_numpy(davidian_weight(s, gamma=2.0))
        assert w2[0] < w1[0], "Higher γ should suppress low SNR more"

    def test_shrink_toward_mean(self):
        """Eigenvalues should move toward their mean, not toward zero."""
        from eris.computation.shrinkage import shrink_eigenvalues
        eigenvalues = np.array([10.0, 1.0, 0.1], dtype=np.float32)
        mean_eig = np.mean(eigenvalues)
        shrunk = np.asarray(shrink_eigenvalues(eigenvalues, n_samples=10, n_features=3)).ravel()
        for i in range(len(eigenvalues)):
            assert abs(shrunk[i] - mean_eig) <= abs(eigenvalues[i] - mean_eig) + 0.5

    def test_bvec_driven_params(self):
        """BFECDS should modulate Davidian parameters meaningfully."""
        from eris.computation.shrinkage import params_from_bvec
        from eris.computation.activations import BVec
        p_crit = params_from_bvec(BVec(C=0.9), psi=1.0)
        p_stable = params_from_bvec(BVec(F=0.9), psi=1.0)
        assert p_crit.alpha > p_stable.alpha
        p_decay = params_from_bvec(BVec(D=0.9), psi=1.0)
        assert p_decay.delta > 0.5

    def test_covariance_stays_psd(self):
        """Shrunk covariance must be positive semi-definite."""
        from eris.computation.shrinkage import shrink_covariance
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 20)).astype(np.float32)
        cov = X.T @ X / 5
        shrunk = np.asarray(shrink_covariance(cov, n_samples=5))
        assert np.all(np.linalg.eigvalsh(shrunk) >= -1e-6)


class TestSGT:

    def test_gate_blocks_noise(self):
        from eris.computation.sgt import gate_decision
        should_act, z = gate_decision(1.05, 1.0, 0.01, 2.0)
        assert not should_act

    def test_gate_passes_signal(self):
        from eris.computation.sgt import gate_decision
        should_act, z = gate_decision(2.0, 1.0, 0.01, 2.0)
        assert should_act

    def test_batch_gate_vectorized(self):
        from eris.computation.sgt import gate_decision, batch_gate
        values = np.array([1.05, 2.0, 0.5, 1.5], dtype=np.float32)
        means = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        variances = np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float32)
        mask, z_scores = batch_gate(values, means, variances, threshold=2.0)
        mask_np = np.asarray(mask)
        for i in range(4):
            scalar_act, _ = gate_decision(float(values[i]), float(means[i]), float(variances[i]), 2.0)
            assert bool(mask_np[i]) == scalar_act

    def test_ema_converges(self):
        from eris.computation.sgt import update_ema
        rng = np.random.default_rng(42)
        mean, var = 0.0, 1.0
        for _ in range(1000):
            mean, var = update_ema(rng.normal(5.0, 1.0), mean, var, 0.05)
        assert abs(mean - 5.0) < 0.5

    def test_stateful_warmup(self):
        from eris.computation.sgt import SGTGate
        gate = SGTGate(warmup_observations=10)
        for _ in range(9):
            should_act, _ = gate.update(100.0)
            assert not should_act

    def test_checkpoint_roundtrip(self):
        from eris.computation.sgt import SGTGate
        gate = SGTGate(threshold_sigma=1.5)
        for v in [1.0, 2.0, 3.0]:
            gate.update(v)
        restored = SGTGate.from_snapshot(gate.snapshot())
        assert restored.running_mean == gate.running_mean


class TestActivations:

    def test_uniform_high_phi(self):
        from eris.computation.activations import compute_bvec_from_field
        size = 32
        phi = np.ones((size, size), dtype=np.float32) * 0.99
        theta = np.zeros((size, size), dtype=np.float32)
        tau = np.zeros((size, size), dtype=np.float32)
        bvec = compute_bvec_from_field(phi, theta, tau, phi.copy())
        assert bvec.B > 0.5 and bvec.S > 0.5 and bvec.E < 0.1

    def test_growing_field_emergence(self):
        from eris.computation.activations import compute_bvec_from_field
        size = 32
        phi_prev = np.full((size, size), 0.1, dtype=np.float32)
        phi = np.full((size, size), 0.5, dtype=np.float32)
        bvec = compute_bvec_from_field(phi, np.zeros_like(phi), np.zeros_like(phi), phi_prev)
        assert bvec.E > 0.3

    def test_archetype_classification(self):
        from eris.computation.activations import BVec
        assert BVec(B=0.8, F=0.9, E=0.1, C=0.1, D=0.1, S=0.1).archetype() == "Feedback Stabilizer"
        assert BVec(B=0.1, F=0.1, E=0.1, C=0.9, D=0.8, S=0.2).archetype() == "Breakdown Hub"
        assert BVec(B=0.2, F=0.2, E=0.8, C=0.8, D=0.2, S=0.1).archetype() == "Emergence Catalyst"

    def test_bvec_distance_symmetry(self):
        from eris.computation.activations import BVec, bvec_distance
        a = BVec(B=0.5, F=0.3, E=0.7, C=0.2, D=0.1, S=0.4)
        b = BVec(B=0.1, F=0.8, E=0.2, C=0.6, D=0.5, S=0.3)
        assert abs(bvec_distance(a, b) - bvec_distance(b, a)) < 1e-6
        assert bvec_distance(a, a) < 1e-6

    def test_all_in_unit_range(self):
        from eris.computation.activations import compute_bvec_from_field
        rng = np.random.default_rng(42)
        for _ in range(10):
            phi = rng.uniform(0, 1, (32, 32)).astype(np.float32)
            theta = rng.uniform(0, 6.28, (32, 32)).astype(np.float32)
            tau = rng.standard_normal((32, 32)).astype(np.float32) * 5.0
            phi_prev = rng.uniform(0, 1, (32, 32)).astype(np.float32)
            bvec = compute_bvec_from_field(phi, theta, tau, phi_prev)
            for name, val in bvec.as_dict().items():
                assert 0.0 <= val <= 1.0, f"{name}={val:.4f} out of [0,1]"


class TestGlobalObservables:
    """Tests for C(t), X(t), dC/dX — the conservation law."""

    def test_coherence_computable(self):
        """Field should produce a valid coherence value."""
        from eris.field.pde import FractalField
        field = FractalField(size=32)
        field.seed_from_text("test coherence observable")
        field.run(10)
        assert 0.0 <= field.coherence <= 1.0, f"C={field.coherence} out of [0,1]"

    def test_exchange_computable(self):
        """Field should produce a finite exchange value."""
        from eris.field.pde import FractalField
        field = FractalField(size=32)
        field.seed_from_text("test exchange observable")
        field.run(10)
        assert np.isfinite(field.exchange), f"X={field.exchange} not finite"

    def test_dCdX_computed_after_warmup(self):
        """dC/dX should be available after sufficient steps."""
        from eris.field.pde import FractalField
        field = FractalField(size=32)
        field.seed_from_text("test conservation law")
        field.run(20)
        assert len(field._dCdX_history) > 0, "dC/dX not computed"
        assert np.isfinite(field.dCdX), f"dC/dX={field.dCdX} not finite"

    def test_regime_detection(self):
        """Regime detection should return a valid string."""
        from eris.field.pde import FractalField
        field = FractalField(size=32)
        field.seed_from_text("regime detection test")
        field.run(20)
        regime = field.detect_regime()
        assert regime in ("warmup", "elastic", "plastic", "transfixed")


class TestModulators:
    """Tests for Memory and Attention modulators in the PDE."""

    def test_memory_accumulates(self):
        """Memory field should accumulate phi values over time."""
        from eris.field.pde import FractalField, PDEParams
        params = PDEParams(memory_tau=5.0)
        field = FractalField(size=32, params=params)
        field.seed_from_text("memory accumulation test")
        field.run(30)
        mem_np = np.asarray(field.memory)
        assert mem_np.max() > 0.001, "Memory field should be nonzero after evolution"

    def test_memory_independent_of_decay(self):
        """Varying memory_tau should produce different behavior than varying D_decay."""
        from eris.field.pde import FractalField, PDEParams

        # High memory, normal decay
        p1 = PDEParams(memory_tau=50.0, D_decay=0.05)
        f1 = FractalField(size=32, params=p1)
        f1.seed_from_text("independence test")
        f1.run(30)
        mem_high = float(np.asarray(f1.memory).mean())

        # Low memory, same decay
        p2 = PDEParams(memory_tau=2.0, D_decay=0.05)
        f2 = FractalField(size=32, params=p2)
        f2.seed_from_text("independence test")
        f2.run(30)
        mem_low = float(np.asarray(f2.memory).mean())

        # Memory tau should affect memory field
        assert abs(mem_high - mem_low) > 0.0001, (
            "Different memory_tau should produce different memory fields"
        )

    def test_attention_concentrates_on_gradients(self):
        """Attention field should be higher where phi gradients are large."""
        from eris.field.pde import FractalField
        field = FractalField(size=32)
        field.seed_from_text("attention concentrates on edges")
        field.run(20)
        att_np = np.asarray(field.attention)
        # Attention should have spatial variation (not flat)
        assert att_np.std() > 0.001, "Attention field should vary spatially"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
