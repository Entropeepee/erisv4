"""Step 5 — the metacognitive review. She compares her NAIVE first impression of a topic against
her CONSIDERED post-hive conclusion, measures how far her view moved (the sin/torsion revision
geometry), and writes a calibration lesson about her own assumptions. Reads nothing new, runs no
hive, answers no question — it reflects on the DELTA between two of her own prior views. Offline."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import tempfile
import unittest

import numpy as np

from eris.computation.activations import BVec
from eris.memory.autobiography import Autobiography
from eris.memory.tiers import (MemorySystem, MemoryRecord,
                               _provenance_family, _CONSOLIDATE_SKIP_FAMILIES)
from eris.memory.thought_stream import ThoughtStream
from eris.metacognition.dreaming import DreamingLoop

GOAL = BVec(B=0.4, F=0.5, E=0.4, C=0.4, D=0.2, S=0.3)


class _Resp:
    def __init__(self, text): self.text = text


class _Mediator:
    def __init__(self, text="I took the headline claim at face value; once it was checked I saw "
                            "the real novelty was elsewhere. Lesson: with patent-style sources I "
                            "should distrust the most prominent number and look for the integration."):
        self.text = text
        self.calls = 0

    async def generate(self, prompt="", system=""):
        self.calls += 1
        return _Resp(self.text)


def _loop(mediator=None):
    d = tempfile.mkdtemp()
    return DreamingLoop(
        Autobiography(path=os.path.join(d, "a.jsonl")),
        MemorySystem(data_dir=d),
        mediator=mediator,
        thought_stream=ThoughtStream(path=os.path.join(d, "t.jsonl")))


def _seed_views(loop, title="SGT", naive_emb=(1.0, 0.0), considered_emb=(0.0, 1.0)):
    # a naive 'reflection' impression + a written-back 'synthesis' conclusion on the SAME topic
    loop.memory.mtm.store(MemoryRecord(
        text="[My reflection on SGT] The 1/sqrt(N) scaling looks like the whole point.",
        bvec=GOAL, embedding=np.array(naive_emb, dtype=np.float32),
        source="reflection", metadata={"title": title}, timestamp=100.0))
    loop.memory.write_back_synthesis(
        title, "The novelty is the integration, NOT the 1/sqrt(N) scaling.",
        embedding=np.array(considered_emb, dtype=np.float32), bvec=GOAL, n_sources=6)


class TestMetacognitiveReview(unittest.TestCase):
    def test_metacognition_is_a_consolidation_skip_family(self):
        self.assertIn(_provenance_family("metacognition:sgt"), _CONSOLIDATE_SKIP_FAMILIES)

    def test_compares_naive_vs_considered_and_writes_lesson(self):
        med = _Mediator()
        loop = _loop(mediator=med)
        _seed_views(loop)                                 # orthogonal embeddings → big revision
        out = loop.metacognitive_review()
        self.assertIsNotNone(out)
        self.assertEqual(out["topic"], "SGT")
        self.assertGreater(out["revision"], 0.9)          # view moved a lot (orthogonal)
        self.assertEqual(out["moved"], "a lot")
        self.assertTrue(out["lesson"])
        self.assertEqual(out["journal"]["kind"], "metacognition")
        self.assertGreaterEqual(med.calls, 1)
        # the lesson is stored as her own voice under the metacognition family
        self.assertTrue(any(_provenance_family(r.source) == "metacognition"
                            for r in loop.memory.mtm._records))

    def test_small_revision_when_views_already_agree(self):
        loop = _loop(mediator=_Mediator())
        _seed_views(loop, naive_emb=(1.0, 0.0), considered_emb=(1.0, 0.03))   # nearly aligned
        out = loop.metacognitive_review()
        self.assertLess(out["revision"], 0.2)
        self.assertEqual(out["moved"], "only a little")

    def test_no_pair_returns_none_and_does_not_generate(self):
        med = _Mediator()
        loop = _loop(mediator=med)
        # only a naive impression, no analyzed conclusion → nothing to reconsider
        loop.memory.mtm.store(MemoryRecord(text="[My reflection on SGT] face value.",
                                           bvec=GOAL, source="reflection",
                                           metadata={"title": "SGT"}))
        self.assertIsNone(loop.metacognitive_review())
        self.assertEqual(med.calls, 0)

    def test_topic_filter_selects_the_named_pair(self):
        loop = _loop(mediator=_Mediator())
        _seed_views(loop, title="SGT")
        _seed_views(loop, title="Kuramoto", naive_emb=(0.0, 1.0), considered_emb=(1.0, 0.0))
        out = loop.metacognitive_review(topic="Kuramoto")
        self.assertEqual(out["topic"], "Kuramoto")

    def test_degrades_gracefully_without_a_mediator(self):
        loop = _loop(mediator=None)
        _seed_views(loop)
        self.assertIsNone(loop.metacognitive_review())     # no voice → no lesson → None


if __name__ == "__main__":
    unittest.main()
