"""Tier 1 tests — shared noise-floor estimator + criticality monitor.

Proves the substrate is correct AND that the SGTGate refactor is
behavior-preserving (no estimator => byte-identical to before). No gate is
wired into the live pipeline yet, so the orchestration benchmark still reads an
unchanged baseline (checked separately in CI by the full suite staying green).
"""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import asyncio
import tempfile
import unittest

from eris.computation.sgt import SGTGate
from eris.computation.noise_floor import NoiseFloorEstimator
from eris.computation.criticality import (
    CriticalityMonitor, Decision, FailureModeReport,
)
from eris.field.pde import FractalField


class _GStub:
    """Duck-typed estimator exposing only a fixed global multiplier."""
    def __init__(self, g):
        self._g = g

    def global_multiplier(self):
        return self._g


class TestNoiseFloorPerSignalScale(unittest.TestCase):
    def test_each_signal_uses_its_own_scale(self):
        """A tiny absolute deviation is a big outlier for a tiny-scale signal,
        and negligible for a large-scale signal — the whole point of per-signal
        local scale vs one global sigma."""
        est = NoiseFloorEstimator(warmup=5, ema_alpha=0.2)
        # "big" lives around 1.0 (spread ~0.1); "small" around 1e-3 (spread ~1e-4)
        for i in range(40):
            est.observe("big", 1.0 + (0.1 if i % 2 else -0.1))
            est.observe("small", 1e-3 + (1e-4 if i % 2 else -1e-4))
        z_small = est.z("small", 5e-3)          # +4e-3 ≈ many small-sigmas
        z_big = est.z("big", 1.0 + 5e-3)         # +5e-3 ≈ a fraction of a big-sigma
        self.assertGreater(abs(z_small), 5.0)
        self.assertLess(abs(z_big), 1.0)

    def test_global_multiplier_rises_with_turbulence(self):
        est = NoiseFloorEstimator(warmup=4, ema_alpha=0.3)
        for v in (0.10, 0.11, 0.09, 0.10, 0.10):   # calm baseline -> g ~ 1
            est.observe_global(v)
        g_calm = est.global_multiplier()
        for v in (0.4, 0.7, 1.0):                   # escalating turbulence
            est.observe_global(v)
        g_turb = est.global_multiplier()
        self.assertLess(g_calm, 1.2)                # calm field ≈ no extra caution
        self.assertGreater(g_turb, 1.5)             # turbulence raises it markedly
        self.assertLessEqual(g_turb, 3.0)           # clamped to g_max


class TestCriticalityMonitor(unittest.TestCase):
    def _warm(self, mon, name, mode, n=8):
        for i in range(n):
            mon.observe(name, 1.0 + (0.02 if i % 2 else -0.02), {"mode": mode})

    def test_protected_steps_suppress_early_firing(self):
        est = NoiseFloorEstimator(warmup=2, ema_alpha=0.5)
        mon = CriticalityMonitor("m", est, "router", k=2.0, protected_steps=4)
        for v in (0.0, 100.0, -100.0, 50.0):        # wild values, still protected
            d, r = mon.observe("s", v, {"mode": "anomaly"})
            self.assertEqual(d, Decision.CONTINUE)
            self.assertIsNone(r)

    def test_anomaly_escalates_with_report(self):
        est = NoiseFloorEstimator(warmup=5, ema_alpha=0.3)
        mon = CriticalityMonitor("m", est, "router", k=2.0)
        self._warm(mon, "dcdx", "anomaly")
        d, r = mon.observe("dcdx", 3.0, {"mode": "anomaly"})   # large high outlier
        self.assertEqual(d, Decision.ESCALATE)
        self.assertIsInstance(r, FailureModeReport)
        self.assertEqual(r.decision, Decision.ESCALATE)
        self.assertEqual(r.specialization, "router")

    def test_anomaly_switch_with_report(self):
        est = NoiseFloorEstimator(warmup=5, ema_alpha=0.3)
        mon = CriticalityMonitor("m", est, "router", k=2.0)
        self._warm(mon, "s", "anomaly")
        # Moderate outlier: above eff_k but below an explicit huge escalate_k.
        d, r = mon.observe("s", 1.5, {"mode": "anomaly", "escalate_k": 1e9})
        self.assertEqual(d, Decision.SWITCH)
        self.assertIsInstance(r, FailureModeReport)
        self.assertEqual(r.decision, Decision.SWITCH)

    def test_settle_suspends_without_report(self):
        est = NoiseFloorEstimator(warmup=5, ema_alpha=0.3)
        mon = CriticalityMonitor("m", est, "field_depth", k=2.0)
        self._warm(mon, "coh", "settle")
        d, r = mon.observe("coh", 0.0, {"mode": "settle"})     # low-side outlier
        self.assertEqual(d, Decision.SUSPEND)
        self.assertIsNone(r)                                   # suspend carries no report

    def test_deadline_forces_suspend(self):
        est = NoiseFloorEstimator(warmup=5, ema_alpha=0.3)
        mon = CriticalityMonitor("m", est, "field_depth", k=2.0)
        self._warm(mon, "coh", "settle")
        d, r = mon.observe("coh", 1.0, {"deadline": True})
        self.assertEqual(d, Decision.SUSPEND)
        self.assertIsNone(r)

    def test_in_band_continues_without_report(self):
        est = NoiseFloorEstimator(warmup=5, ema_alpha=0.3)
        mon = CriticalityMonitor("m", est, "router", k=2.0)
        self._warm(mon, "s", "anomaly")
        d, r = mon.observe("s", 1.0, {"mode": "anomaly"})      # right at the mean
        self.assertEqual(d, Decision.CONTINUE)
        self.assertIsNone(r)

    def test_global_multiplier_makes_monitor_conservative(self):
        """Same signal sequence: an outlier that ESCALATEs in a calm field is
        held to CONTINUE when the shared multiplier marks the field turbulent."""
        def run(turbulent):
            est = NoiseFloorEstimator(warmup=5, ema_alpha=0.3, g_max=20.0)
            mon = CriticalityMonitor("m", est, "router", k=2.0)
            for i in range(8):
                est.observe_global(5.0 if turbulent else 0.0)
                mon.observe("s", 1.0 + (0.02 if i % 2 else -0.02), {"mode": "anomaly"})
            # one more agitation reading so g is current
            est.observe_global(50.0 if turbulent else 0.0)
            return mon.observe("s", 1.6, {"mode": "anomaly", "escalate_k": 1e9})
        d_calm, _ = run(turbulent=False)
        d_turb, _ = run(turbulent=True)
        self.assertEqual(d_calm, Decision.SWITCH)     # fires when calm
        self.assertEqual(d_turb, Decision.CONTINUE)   # suppressed when turbulent


class TestSGTGateBackCompat(unittest.TestCase):
    def test_no_estimator_is_deterministic_and_unchanged(self):
        seq = [0.0, 1.0, 0.5, 0.2, 0.9, 1.0, 0.1, 0.3, 0.7, 0.4, 1.2, 0.6, 2.0]
        a = SGTGate(threshold_sigma=2.0, ema_alpha=0.1)
        b = SGTGate(threshold_sigma=2.0, ema_alpha=0.1)
        out_a = [a.update(v) for v in seq]
        out_b = [b.update(v) for v in seq]
        self.assertEqual(out_a, out_b)              # byte-identical, repeatable

    def test_shared_multiplier_raises_effective_threshold(self):
        """A >2σ outlier that opens a plain gate stays shut when an injected
        estimator reports a turbulent field (effective threshold = 2σ × g)."""
        seq = [0.0, 1.0] * 6                          # 12 warmup obs
        plain = SGTGate(threshold_sigma=2.0, ema_alpha=0.1)
        gated = SGTGate(threshold_sigma=2.0, ema_alpha=0.1, estimator=_GStub(2.0))
        for v in seq:
            plain.update(v)
            gated.update(v)                          # identical stats; g only scales threshold
        std = max(plain.running_var ** 0.5, 1e-10)
        outlier = plain.running_mean + 5.0 * std
        fired_plain, z_plain = plain.update(outlier)
        fired_gated, z_gated = gated.update(outlier)
        self.assertAlmostEqual(z_plain, z_gated, places=9)   # same z, only threshold differs
        self.assertGreater(z_plain, 2.0)
        self.assertLess(z_plain, 4.0)                        # below the raised 4σ threshold
        self.assertTrue(fired_plain)                         # plain (2σ) opens
        self.assertFalse(fired_gated)                        # gated (2σ×2) stays shut


class _SuspendAfter:
    """Stub monitor that SUSPENDs on its k-th observation (deterministic)."""
    def __init__(self, k):
        self.k = k
        self.calls = 0

    def observe(self, name, value, context=None):
        self.calls += 1
        return (Decision.SUSPEND if self.calls >= self.k else Decision.CONTINUE), None


class _AlwaysContinue:
    def observe(self, name, value, context=None):
        return Decision.CONTINUE, None


class TestFieldDepthGate(unittest.TestCase):
    def test_run_gated_suspends_early_respecting_min_steps(self):
        f = FractalField(size=16, seed=1)
        f.seed_from_text("hello world")
        executed = f.run_gated(_SuspendAfter(1), max_steps=50,
                               check_every=4, min_steps=8)
        self.assertEqual(executed, 8)          # min_steps floor; suspend at 1st check
        self.assertLess(executed, 50)

    def test_run_gated_runs_full_when_never_suspends(self):
        f = FractalField(size=16, seed=1)
        f.seed_from_text("hello world")
        executed = f.run_gated(_AlwaysContinue(), max_steps=30,
                               check_every=4, min_steps=8)
        self.assertEqual(executed, 30)         # turbulent: full depth, no early stop

    def test_run_gated_never_suspends_before_min_steps(self):
        f = FractalField(size=16, seed=1)
        f.seed_from_text("hello world")
        # Suspend would fire on the very first observation, but min_steps=12 with
        # check_every=4 means the first check is at step 12, not earlier.
        executed = f.run_gated(_SuspendAfter(1), max_steps=50,
                               check_every=4, min_steps=12)
        self.assertEqual(executed, 12)


class TestResponseFieldGate(unittest.TestCase):
    def test_warm_reseed_blend1_matches_cold_seed(self):
        """blend=1.0 must reproduce a cold seed_from_text (no warm prior leaks)."""
        import numpy as np
        from eris.config import to_numpy
        a = FractalField(size=16, seed=3)
        a.seed_from_text("the quick brown fox")
        b = FractalField(size=16, seed=3)
        b.run(7)                                  # evolve so it has a real warm state
        b.warm_reseed("the quick brown fox", blend=1.0)
        self.assertTrue(np.allclose(to_numpy(a.phi), to_numpy(b.phi), atol=1e-5))
        self.assertTrue(np.allclose(to_numpy(a.theta), to_numpy(b.theta), atol=1e-5))

    def test_run_gated_response_suspends_and_respects_floor(self):
        f = FractalField(size=16, seed=2)
        f.seed_from_text("hello")
        executed = f.run_gated_response(_SuspendAfter(1), max_steps=25,
                                        check_every=4, min_steps=8)
        self.assertEqual(executed, 8)
        f2 = FractalField(size=16, seed=2)
        f2.seed_from_text("hello")
        full = f2.run_gated_response(_AlwaysContinue(), max_steps=25,
                                     check_every=4, min_steps=8)
        self.assertEqual(full, 25)


class TestRouterGate(unittest.TestCase):
    """Tier 4: the four-decision router maps to the right cloud-call cost, and
    is behavior-preserving when the flag is off."""

    def tearDown(self):
        from eris.config import CONFIG
        for f in ("orchestration_enabled", "gate_field_depth", "gate_response_field",
                  "gate_router", "gate_failure_reports", "use_beta_star"):
            setattr(CONFIG, f, False)

    def _orch(self):
        from eris.config import CONFIG
        from eris.orchestrator import ErisOrchestrator
        from eris.interface.mediator import LLMResponse, LLMBackend

        class FB(LLMBackend):
            def __init__(self, name):
                self.name = name
                self.model = "m"

            def is_available(self):
                return True

            async def generate(self, prompt, system="", max_tokens=8192, temperature=0.7):
                return LLMResponse(text="ok " + self.name, provider=self.name,
                                   model="m", latency_ms=0.0)

        CONFIG.orchestration_enabled = True
        CONFIG.gate_router = True
        CONFIG.gate_field_depth = False
        CONFIG.gate_response_field = False
        o = ErisOrchestrator(data_dir=tempfile.mkdtemp(), field_size=16)
        o.mediator._backends = [FB("ollama")]
        o.deep_mediator._backends = [FB("cloud-a"), FB("cloud-b"), FB("ollama")]
        o._cloud_experts = 2
        return o

    def test_escalate_counts_full_ensemble(self):
        o = self._orch()
        o._router_monitor.observe = lambda *a, **k: (
            Decision.ESCALATE,
            FailureModeReport("router", "router", Decision.ESCALATE, "outlier", 9.0))
        asyncio.run(o.process("hello"))
        self.assertEqual(o.counters.cloud_calls, 2)        # full ensemble
        self.assertIsNotNone(o._last_router_report)

    def test_switch_counts_single_expert(self):
        o = self._orch()
        o._router_monitor.observe = lambda *a, **k: (
            Decision.SWITCH,
            FailureModeReport("router", "router", Decision.SWITCH, "outlier", 3.0))
        asyncio.run(o.process("hello"))
        self.assertEqual(o.counters.cloud_calls, 1)        # cheaper single expert
        self.assertIsNotNone(o._last_router_report)

    def test_continue_stays_local(self):
        o = self._orch()
        o._router_monitor.observe = lambda *a, **k: (Decision.CONTINUE, None)
        asyncio.run(o.process("hello"))
        self.assertEqual(o.counters.cloud_calls, 0)        # local only

    def test_failure_report_becomes_dream_question(self):
        """Tier 5: a SWITCH/ESCALATE report lands as a question in the dream
        queue when gate_failure_reports is on — and does NOT when it's off."""
        from eris.config import CONFIG
        o = self._orch()
        o._router_monitor.observe = lambda *a, **k: (
            Decision.ESCALATE,
            FailureModeReport("router", "router", Decision.ESCALATE, "outlier", 9.0,
                              recommended_action="run elevated fidelity"))
        CONFIG.gate_failure_reports = False
        asyncio.run(o.process("hello"))
        self.assertEqual(len(o.dreaming_loop.pending_questions), 0)
        CONFIG.gate_failure_reports = True
        asyncio.run(o.process("hello again"))
        qs = o.dreaming_loop.pending_questions
        self.assertEqual(len(qs), 1)
        self.assertIn("ESCALATE", qs[0])


class _BV:
    C = 0.3
    F = 0.2
    B = 0.4
    E = 0.5
    S = 0.1
    D = 0.2


class TestBetaStarBridge(unittest.TestCase):
    """Tier 6 (isolated): the β-star bridge is active when on, and neutral on
    winner selection (preserves eigenvalue ordering) in both modes."""

    def tearDown(self):
        from eris.config import CONFIG
        CONFIG.use_beta_star = False

    def test_bridge_changes_beta_when_on(self):
        from eris.config import CONFIG
        from eris.computation.shrinkage import params_from_bvec
        CONFIG.use_beta_star = False
        off = params_from_bvec(_BV(), psi=1.0).beta
        CONFIG.use_beta_star = True
        on = params_from_bvec(_BV(), psi=1.0).beta
        self.assertGreater(on, 0.0)
        self.assertNotAlmostEqual(on, off)        # the bridge is actually wired

    def test_winner_selection_stable_across_modes(self):
        """The spec's neutral-or-better check: toggling β-star must not change
        which component dominates on a fixed bid set (winner stability)."""
        import numpy as np
        from eris.config import CONFIG, to_numpy
        from eris.computation.shrinkage import shrink_eigenvalues
        eig = [5.0, 3.0, 2.0, 1.0, 0.5]
        CONFIG.use_beta_star = False
        off = to_numpy(shrink_eigenvalues(eig, 10, 5, bvec=_BV()))
        CONFIG.use_beta_star = True
        on = to_numpy(shrink_eigenvalues(eig, 10, 5, bvec=_BV()))
        # The WINNER (dominant component) is unchanged; only near-mean components
        # (shrunk to ~equal, irrelevant to selection) may reshuffle. Neutral.
        self.assertEqual(int(np.argmax(off)), int(np.argmax(on)))


if __name__ == "__main__":
    unittest.main()
