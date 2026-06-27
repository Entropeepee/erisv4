"""§1: the DualPath spine — each mode returns from the right path; a raising novel
in SHADOW still returns the floor and logs an error; NOVEL_PRIMARY falls back."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest

from eris.dual.path import DualPath, Mode
from eris.dual.types import RetrievalResult


class _Logger:
    def __init__(self):
        self.rows = []
        self.errors = []
    def record(self, name, query, t, n, arbiter, gold=None, kw=None):
        self.rows.append((name, query, t, n))
    def record_error(self, name, query, e):
        self.errors.append((name, query, str(e)))


def _trad(q, **kw):
    return RetrievalResult(records=["T1", "T2"], scores=[0.9, 0.5])


def _novel(q, **kw):
    return RetrievalResult(records=["N1"], scores=[0.8])


def _empty_novel(q, **kw):
    return RetrievalResult(records=[], scores=[])


def _boom(q, **kw):
    raise RuntimeError("novel exploded")


class TestModes(unittest.TestCase):
    def test_traditional_only(self):
        dp = DualPath(_novel, _trad, mode=Mode.TRADITIONAL_ONLY)
        self.assertEqual(dp.run("q").records, ["T1", "T2"])

    def test_novel_only(self):
        dp = DualPath(_novel, _trad, mode=Mode.NOVEL_ONLY)
        self.assertEqual(dp.run("q").records, ["N1"])

    def test_shadow_returns_floor_and_logs(self):
        log = _Logger()
        dp = DualPath(_novel, _trad, mode=Mode.SHADOW, logger=log, name="retrieval")
        out = dp.run("q")
        self.assertEqual(out.records, ["T1", "T2"])    # floor is authoritative
        self.assertEqual(len(log.rows), 1)             # exactly one divergence row
        self.assertEqual(log.rows[0][0], "retrieval")

    def test_shadow_novel_raises_still_returns_floor(self):
        log = _Logger()
        dp = DualPath(_boom, _trad, mode=Mode.SHADOW, logger=log)
        out = dp.run("q")
        self.assertEqual(out.records, ["T1", "T2"])    # turn never breaks
        self.assertEqual(len(log.errors), 1)
        self.assertEqual(len(log.rows), 0)

    def test_novel_primary_uses_novel_when_acceptable(self):
        dp = DualPath(_novel, _trad, mode=Mode.NOVEL_PRIMARY)
        self.assertEqual(dp.run("q").records, ["N1"])

    def test_novel_primary_falls_back_on_empty(self):
        dp = DualPath(_empty_novel, _trad, mode=Mode.NOVEL_PRIMARY)
        self.assertEqual(dp.run("q").records, ["T1", "T2"])

    def test_novel_primary_falls_back_when_novel_raises(self):
        dp = DualPath(_boom, _trad, mode=Mode.NOVEL_PRIMARY)
        self.assertEqual(dp.run("q").records, ["T1", "T2"])

    def test_mode_parse(self):
        self.assertIs(Mode.parse("shadow"), Mode.SHADOW)
        self.assertIs(Mode.parse("bogus"), Mode.TRADITIONAL_ONLY)


if __name__ == "__main__":
    unittest.main()
