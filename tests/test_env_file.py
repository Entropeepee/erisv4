"""The shared .env loader (eris/env_file.py): keys + config (incl. ERIS_AUTH_TOKEN) load from one
gitignored file so they apply on a fresh shell, not just the window that exported them — and the
SERVER loads it before reading os.environ, closing the fresh-shell auth footgun."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import importlib.util
import tempfile
import unittest

from eris.env_file import load_dotenv


class TestEnvFileLoader(unittest.TestCase):
    def _write(self, body):
        d = tempfile.mkdtemp()
        p = os.path.join(d, ".env")
        with open(p, "w") as f:
            f.write(body)
        return p

    def test_loads_keys_and_auth_token(self):
        p = self._write("# secrets\nERIS_AUTH_TOKEN=s3cret\n\nERIS_TIER_FREE=qwen/qwen-2.5-72b\n"
                        'QUOTED="with spaces"\n')
        for k in ("ERIS_AUTH_TOKEN", "ERIS_TIER_FREE", "QUOTED"):
            os.environ.pop(k, None)
        n = load_dotenv(p)
        self.assertEqual(n, 3)
        self.assertEqual(os.environ["ERIS_AUTH_TOKEN"], "s3cret")     # the gate's token loads
        self.assertEqual(os.environ["ERIS_TIER_FREE"], "qwen/qwen-2.5-72b")
        self.assertEqual(os.environ["QUOTED"], "with spaces")          # quotes stripped
        for k in ("ERIS_AUTH_TOKEN", "ERIS_TIER_FREE", "QUOTED"):
            os.environ.pop(k, None)

    def test_does_not_override_explicit_env(self):
        os.environ["ERIS_AUTH_TOKEN"] = "set-by-hand"
        p = self._write("ERIS_AUTH_TOKEN=from-file\n")
        load_dotenv(p)
        self.assertEqual(os.environ["ERIS_AUTH_TOKEN"], "set-by-hand")  # explicit export still wins
        os.environ.pop("ERIS_AUTH_TOKEN", None)

    def test_missing_file_is_silent_noop(self):
        self.assertEqual(load_dotenv("/no/such/.env"), 0)

    def test_malformed_line_does_not_crash(self):
        p = self._write("not a kv line\nERIS_OK=1\n")
        os.environ.pop("ERIS_OK", None)
        self.assertEqual(load_dotenv(p), 1)            # the good line still loads, no exception
        os.environ.pop("ERIS_OK", None)

    def test_server_loads_dotenv_before_eris_imports(self):
        # Regression guard for the security property: the server MUST call load_dotenv() BEFORE the
        # eris imports (which bind config) and before create_app() reads ERIS_AUTH_TOKEN — else .env
        # auth silently does nothing on a fresh shell. Checked at source level via find_spec so the
        # heavy orchestrator import is NOT paid by this test.
        spec = importlib.util.find_spec("eris.server.app")
        self.assertIsNotNone(spec and spec.origin)
        with open(spec.origin, encoding="utf-8") as f:
            src = f.read()
        i_load = src.find("load_dotenv()")
        i_orch = src.find("from eris.orchestrator")
        i_token = src.find('os.environ.get("ERIS_AUTH_TOKEN"')
        self.assertGreater(i_load, 0, "server must call load_dotenv()")
        self.assertLess(i_load, i_orch, "load_dotenv() must run BEFORE the eris imports (config bind)")
        self.assertLess(i_load, i_token, "load_dotenv() must run BEFORE the ERIS_AUTH_TOKEN read")


class TestConflictDiagnostic(unittest.TestCase):
    """A stale shell var silently winning over .env was the root cause of three attributability
    incidents. The loader keeps defer-to-shell (an export still wins) but makes the override VISIBLE."""

    def _write(self, body):
        d = tempfile.mkdtemp()
        p = os.path.join(d, ".env")
        with open(p, "w") as f:
            f.write(body)
        return p

    def test_conflict_recorded_and_surfaced_with_secret_masked(self):
        import io
        from eris.env_file import DOTENV_CONFLICTS, report_env_sources
        os.environ["ERIS_AUTH_TOKEN"] = "shell-token-value"          # a stale shell value
        os.environ.pop("ERIS_BENCH_MODEL", None)
        p = self._write("ERIS_AUTH_TOKEN=env-token-value\nERIS_BENCH_MODEL=qwen/qwen-2.5-72b\n")
        load_dotenv(p)
        keys = {c[0] for c in DOTENV_CONFLICTS}
        self.assertIn("ERIS_AUTH_TOKEN", keys)        # the override is recorded (shell still wins)
        self.assertNotIn("ERIS_BENCH_MODEL", keys)    # loaded cleanly from .env → no conflict
        self.assertEqual(os.environ["ERIS_AUTH_TOKEN"], "shell-token-value")  # defer-to-shell kept

        buf = io.StringIO()
        report_env_sources(out=buf)
        text = buf.getvalue()
        self.assertIn("OVERRIDES .env", text)                        # the override is surfaced loudly
        self.assertIn("ERIS_BENCH_MODEL ← .env", text)               # clean source shown
        self.assertNotIn("shell-token-value", text)                  # secret VALUE never logged
        self.assertNotIn("env-token-value", text)
        self.assertIn("chars>", text)                                # shown only as a masked length
        for k in ("ERIS_AUTH_TOKEN", "ERIS_BENCH_MODEL"):
            os.environ.pop(k, None)

    def test_no_conflict_no_warning(self):
        import io
        from eris.env_file import report_env_sources
        for k in ("ERIS_AUTH_TOKEN", "ERIS_BENCH_MODEL"):
            os.environ.pop(k, None)
        load_dotenv(self._write("ERIS_BENCH_MODEL=qwen/qwen-2.5-72b\n"))
        buf = io.StringIO()
        report_env_sources(out=buf)
        text = buf.getvalue()
        self.assertNotIn("WARNING", text)                            # no false alarm when clean
        self.assertIn("ERIS_BENCH_MODEL ← .env", text)
        os.environ.pop("ERIS_BENCH_MODEL", None)


if __name__ == "__main__":
    unittest.main()
