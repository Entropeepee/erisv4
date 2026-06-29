"""Phase 1.5 (Codex r3 #1): MemoryRecord.to_dict/from_dict omitted phi/theta_snapshot, so the field
memory was ephemeral — after a restart MTM/LTM reloaded embedding-only. Now the snapshots persist,
with dtype/shape/finite validation. (Phase-3 precondition: a physics test on embedding-only memory
isn't testing the physics.)"""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import tempfile
import unittest

import numpy as np

from eris.memory.tiers import (
    MemoryRecord, MediumTermMemory, LongTermMemory, _snapshot_from_list, _snapshot_to_list,
    _snapshot_pair_from_lists)
from eris.computation.activations import BVec


def _rec(phi, theta, text="t"):
    return MemoryRecord(text=text, bvec=BVec(B=.1, F=.1, E=.1, C=.1, D=.1, S=.1),
                        phi_snapshot=phi, theta_snapshot=theta)


class TestRoundTrip(unittest.TestCase):
    def test_to_from_dict_preserves_snapshots(self):
        phi = np.random.RandomState(0).rand(8, 8).astype(np.float32)
        theta = np.random.RandomState(1).rand(8, 8).astype(np.float32)
        r2 = MemoryRecord.from_dict(_rec(phi, theta).to_dict())
        self.assertIsNotNone(r2.phi_snapshot)
        self.assertIsNotNone(r2.theta_snapshot)
        np.testing.assert_array_almost_equal(r2.phi_snapshot, phi)
        np.testing.assert_array_almost_equal(r2.theta_snapshot, theta)

    def test_none_snapshots_stay_none(self):
        d = _rec(None, None).to_dict()
        self.assertNotIn("phi_snapshot", d)
        self.assertNotIn("theta_snapshot", d)
        self.assertIsNone(MemoryRecord.from_dict(d).phi_snapshot)


class TestTierReload(unittest.TestCase):
    def test_mtm_reload_survives_restart(self):
        phi = np.full((4, 4), 0.5, np.float32)
        theta = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
        p = os.path.join(tempfile.mkdtemp(), "mtm.jsonl")
        MediumTermMemory(storage_path=p).store(_rec(phi, theta))   # store auto-saves
        reloaded = MediumTermMemory(storage_path=p)                # fresh instance → _load from disk
        recs = [r for r in reloaded._records if r.phi_snapshot is not None]
        self.assertEqual(len(recs), 1)
        np.testing.assert_array_almost_equal(recs[0].phi_snapshot, phi)
        np.testing.assert_array_almost_equal(recs[0].theta_snapshot, theta)

    def test_ltm_reload_survives_restart(self):
        phi = np.full((4, 4), 0.25, np.float32)
        theta = np.zeros((4, 4), np.float32)
        p = os.path.join(tempfile.mkdtemp(), "ltm.jsonl")
        LongTermMemory(storage_path=p).store(_rec(phi, theta))
        reloaded = LongTermMemory(storage_path=p)
        recs = [r for r in reloaded._records if r.phi_snapshot is not None]
        self.assertEqual(len(recs), 1)
        np.testing.assert_array_almost_equal(recs[0].phi_snapshot, phi)


class TestValidation(unittest.TestCase):
    def test_nonfinite_snapshot_dropped_on_write(self):
        bad = np.full((4, 4), np.nan, np.float32)
        d = _rec(bad, np.zeros((4, 4), np.float32)).to_dict()
        self.assertNotIn("phi_snapshot", d)        # NaN field never persisted (would poison reload)
        self.assertIn("theta_snapshot", d)         # the valid one still is

    def test_from_list_validation(self):
        self.assertIsNone(_snapshot_from_list([1, 2, 3]))            # 1D rejected
        self.assertIsNone(_snapshot_from_list([[1, 2], [3]]))        # ragged rejected
        self.assertIsNone(_snapshot_from_list([[float("inf"), 0], [0, 0]]))  # inf rejected
        self.assertIsNone(_snapshot_from_list([]))                   # empty rejected
        ok = _snapshot_from_list([[1, 2], [3, 4]])                   # valid 2D accepted
        self.assertIsNotNone(ok)
        self.assertEqual(ok.dtype, np.float32)
        self.assertEqual(ok.shape, (2, 2))

    def test_to_list_validation(self):
        self.assertIsNone(_snapshot_to_list(None))
        self.assertIsNone(_snapshot_to_list(np.array([np.inf, 0.0], dtype=np.float32)))  # 1D+inf
        self.assertEqual(_snapshot_to_list(np.array([[1, 2], [3, 4]], np.float32)), [[1.0, 2.0], [3.0, 4.0]])


class TestSnapshotPairConsistency(unittest.TestCase):
    """Codex #7: phi/theta are validated independently, so a SHAPE-MISMATCHED pair both load as
    non-None and DCR would compute on inconsistent geometry. A snapshot is only meaningful as a
    matched grid — a mismatched pair must fall back to no-snapshot (embedding-only)."""

    def test_mismatched_pair_falls_back_to_none_on_reload(self):
        d = _rec(np.zeros((2, 2), np.float32), np.zeros((1, 1), np.float32)).to_dict()
        self.assertIn("phi_snapshot", d)               # each side serialized fine individually
        self.assertIn("theta_snapshot", d)
        r = MemoryRecord.from_dict(d)
        self.assertIsNone(r.phi_snapshot)              # but the mismatched PAIR is dropped
        self.assertIsNone(r.theta_snapshot)

    def test_matched_pair_survives(self):
        phi, theta = np.zeros((4, 4), np.float32), np.ones((4, 4), np.float32)
        r = MemoryRecord.from_dict(_rec(phi, theta).to_dict())
        self.assertIsNotNone(r.phi_snapshot)
        self.assertEqual(r.phi_snapshot.shape, r.theta_snapshot.shape)

    def test_pair_helper_directly(self):
        # mismatch → (None, None)
        self.assertEqual(_snapshot_pair_from_lists([[1, 2], [3, 4]], [[1]]), (None, None))
        # match → both restored
        phi, theta = _snapshot_pair_from_lists([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        self.assertEqual(phi.shape, (2, 2))
        self.assertEqual(theta.shape, (2, 2))
        # a torn pair (one side missing) is NOT this fix's concern — the present side is kept
        phi, theta = _snapshot_pair_from_lists([[1, 2], [3, 4]], None)
        self.assertIsNotNone(phi)
        self.assertIsNone(theta)


if __name__ == "__main__":
    unittest.main()
