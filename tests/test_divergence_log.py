"""§4: divergence log — idempotent re-run adds no duplicate rows; survives a
truncated line; verdict is by arbiter success delta, not overlap."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import tempfile
import unittest

from eris.dual.divergence_log import DivergenceLog, _verdict
from eris.dual.types import RetrievalResult


class _Rec:
    def __init__(self, rid):
        self.id = rid
        self.text = f"text-{rid}"
        self.source = "reading:x"
        self.metadata = {}


class _Arbiter:
    """Scores by membership of a 'gold' id in the result — independent of overlap."""
    def score(self, query, result, gold=None):
        ids = result.top_ids()
        succ = 1.0 if (gold and gold in ids) else 0.0
        return {"gold_at_k": succ, "success": succ}


def _res(ids):
    recs = [_Rec(i) for i in ids]
    return RetrievalResult(records=recs, scores=[1.0] * len(recs),
                           aligned=recs, tension=[], coupling=[0.5] * len(recs))


class TestDivergenceLog(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "dual", "divergence.jsonl")

    def test_records_and_is_idempotent(self):
        log = DivergenceLog(self.path)
        a = _Arbiter()
        row = log.record("retrieval", "what is X?", _res(["b"]), _res(["a"]), a, gold="a")
        self.assertIsNotNone(row)
        self.assertEqual(row["verdict"], "novel_wins")    # novel had gold, trad didn't
        self.assertTrue(row["cross_domain"])              # novel surfaced 'a' trad missed
        # Re-run with the SAME query → no duplicate row.
        again = log.record("retrieval", "what is X?", _res(["b"]), _res(["a"]), a, gold="a")
        self.assertIsNone(again)
        # A fresh instance reloads the seen-set and still de-dupes.
        log2 = DivergenceLog(self.path)
        self.assertIsNone(log2.record("retrieval", "what is X?", _res(["b"]), _res(["a"]), a, gold="a"))
        self.assertEqual(len(log2.rows()), 1)

    def test_survives_truncated_line(self):
        log = DivergenceLog(self.path)
        a = _Arbiter()
        log.record("retrieval", "q1", _res(["b"]), _res(["a"]), a, gold="a")
        # Simulate a disk-full truncated final line (no newline, broken utf-8).
        with open(self.path, "ab") as f:
            f.write(b'{"query_hash":"trunc","x\xff')
        log2 = DivergenceLog(self.path)
        new = log2.record("retrieval", "q2", _res(["b"]), _res(["c"]), a, gold="c")
        self.assertIsNotNone(new)                          # new row still lands
        hashes = {r.get("query_hash") for r in log2.rows() if "verdict" in r}
        self.assertEqual(len(hashes), 2)                   # q1 + q2 both intact

    def test_verdict_by_delta_not_overlap(self):
        self.assertEqual(_verdict(0.0, 0.0), "both_miss")
        self.assertEqual(_verdict(0.9, 0.1), "trad_wins")
        self.assertEqual(_verdict(0.1, 0.9), "novel_wins")
        self.assertEqual(_verdict(0.5, 0.5), "tie")

    def test_error_row(self):
        log = DivergenceLog(self.path)
        log.record_error("retrieval", "boom", RuntimeError("nope"))
        rows = log.rows()
        self.assertEqual(len(rows), 1)
        self.assertIn("error", rows[0])


if __name__ == "__main__":
    unittest.main()
