"""A1: GPU backend initializes lazily — importing Eris never touches the GPU,
so a misconfigured CUDA runtime can't crash imports and there's no GPU stdout
at import time.

Laziness is asserted in CLEAN subprocesses (a fresh interpreter) — never via
importlib.reload, which would corrupt the shared `xp`/`cp` references other
already-imported modules hold and break unrelated tests in the same process."""
import os
os.environ.setdefault("ERIS_EMBEDDINGS", "off")  # NB: do NOT set ERIS_GPU here

import subprocess
import sys
import unittest


def _run(code: str):
    env = dict(os.environ, ERIS_EMBEDDINGS="off")
    return subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, env=env)


class TestLazyGPU(unittest.TestCase):
    def test_import_config_does_not_resolve_gpu(self):
        r = _run("import eris.config as c; print('TRIED', c._GPU_TRIED, c.GPU_AVAILABLE)")
        self.assertIn("TRIED False None", r.stdout, msg=r.stderr[-400:])
        self.assertNotIn("[Eris GPU]", r.stdout)
        self.assertNotIn("[Eris CPU]", r.stdout)

    def test_importing_orchestrator_stays_lazy(self):
        # The whole stack must import without resolving the GPU (no module-level xp).
        r = _run("import eris.orchestrator; import eris.config as c; print('TRIED', c._GPU_TRIED)")
        self.assertIn("TRIED False", r.stdout, msg=r.stderr[-600:])
        self.assertNotIn("[Eris GPU]", r.stdout)

    def test_xp_resolves_on_first_use(self):
        r = _run("import eris.config as c; a=c.xp.zeros(4);"
                 "print('SHAPE', a.shape, 'TRIED', c._GPU_TRIED,"
                 "'DECIDED', c.GPU_AVAILABLE in (True, False))")
        self.assertIn("SHAPE (4,) TRIED True DECIDED True", r.stdout, msg=r.stderr[-400:])

    def test_to_numpy_and_vram_safe(self):
        # Functional correctness in-process (no reload): these must never raise.
        import numpy as np
        import eris.config as cfg
        self.assertEqual(list(cfg.to_numpy(np.ones(3))), [1.0, 1.0, 1.0])
        self.assertIsInstance(cfg.vram_used_gb(), float)
        self.assertTrue(cfg.vram_check(0.0))


if __name__ == "__main__":
    unittest.main()
