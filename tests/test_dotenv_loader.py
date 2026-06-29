"""The benchmark runner loads a .env file so keys + config are entered ONCE (not re-typed via
`set` every window). It must not override an explicit env var, must skip comments/blanks, and must
be silent on a missing file."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import tempfile
import unittest

from eris.experiments.benchmarks.run import _load_dotenv


class TestDotenvLoader(unittest.TestCase):
    def _write(self, body):
        d = tempfile.mkdtemp()
        p = os.path.join(d, ".env")
        with open(p, "w") as f:
            f.write(body)
        return p

    def test_loads_keys_and_config(self):
        p = self._write("# my keys\nHF_TOKEN=hf_abc123\n\nERIS_TIER_FREE=qwen/qwen-2.5-72b\n"
                        'QUOTED="with spaces"\n')
        for k in ("HF_TOKEN", "ERIS_TIER_FREE", "QUOTED"):
            os.environ.pop(k, None)
        n = _load_dotenv(p)
        self.assertEqual(n, 3)
        self.assertEqual(os.environ["HF_TOKEN"], "hf_abc123")
        self.assertEqual(os.environ["ERIS_TIER_FREE"], "qwen/qwen-2.5-72b")
        self.assertEqual(os.environ["QUOTED"], "with spaces")          # quotes stripped
        for k in ("HF_TOKEN", "ERIS_TIER_FREE", "QUOTED"):
            os.environ.pop(k, None)

    def test_does_not_override_explicit_env(self):
        os.environ["ERIS_TIER_FREE"] = "already-set-by-hand"
        p = self._write("ERIS_TIER_FREE=from-file\n")
        _load_dotenv(p)
        self.assertEqual(os.environ["ERIS_TIER_FREE"], "already-set-by-hand")   # `set` wins
        os.environ.pop("ERIS_TIER_FREE", None)

    def test_missing_file_is_silent_noop(self):
        self.assertEqual(_load_dotenv("/no/such/.env"), 0)


if __name__ == "__main__":
    unittest.main()
