"""Phase-2 Batch 1 — config-knob honesty (Codex #7).

Three documented knobs were exposed but never read by the code they claim to control. Each is now
wired into the constructor/gate it names (or the knob would lie). Tests prove the live path honors
the knob, not a hardcoded literal.
"""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import tempfile
import unittest
from unittest import mock

import eris.config as cfg
from eris.config import CONFIG


class TestPdeDtWired(unittest.TestCase):
    def test_fractalfield_uses_config_pde_dt(self):
        from eris.field.pde import FractalField
        with mock.patch.object(CONFIG, "pde_dt", 0.1):
            f = FractalField(size=16)
            self.assertEqual(f.p.dt, 0.1)               # the knob is honored, not PDEParams.dt

    def test_explicit_params_still_win(self):
        from eris.field.pde import FractalField, PDEParams
        with mock.patch.object(CONFIG, "pde_dt", 0.1):
            f = FractalField(size=16, params=PDEParams(dt=0.02))
            self.assertEqual(f.p.dt, 0.02)              # an explicit caller override is respected


class TestSgtThresholdWired(unittest.TestCase):
    def test_orchestrator_gates_use_config_sigma(self):
        from eris.orchestrator import ErisOrchestrator
        with mock.patch.object(CONFIG, "sgt_threshold_sigma", 3.3), \
             mock.patch.object(CONFIG, "sgt_ema_alpha", 0.07):
            o = ErisOrchestrator(data_dir=tempfile.mkdtemp(), field_size=16)
            self.assertEqual(o._dissonance_gate.threshold_sigma, 3.3)
            self.assertEqual(o._dissonance_gate.ema_alpha, 0.07)
            self.assertEqual(o._router_gate.threshold_sigma, 3.3)


class TestVramCapWired(unittest.TestCase):
    def test_cpu_always_has_room(self):
        self.assertTrue(cfg.vram_check(0.0))            # no GPU → no cap pressure

    def test_cap_is_read_from_config(self):
        # simulate a GPU at 15 GB used; the decision must follow CONFIG.vram_cap_gb.
        with mock.patch.object(cfg, "GPU_AVAILABLE", True), \
             mock.patch.object(cfg, "mempool", None), \
             mock.patch.object(cfg, "vram_used_gb", return_value=15.0):
            with mock.patch.object(CONFIG, "vram_cap_gb", 16.0):
                self.assertTrue(cfg.vram_check(0.5))    # 15.5 ≤ 16
                self.assertFalse(cfg.vram_check(2.0))   # 17.0 > 16
            with mock.patch.object(CONFIG, "vram_cap_gb", 20.0):
                self.assertTrue(cfg.vram_check(2.0))    # cap raised → now fits (knob is live)


if __name__ == "__main__":
    unittest.main()
