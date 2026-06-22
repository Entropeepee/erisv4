"""Tests for the thought-stream — her OWN thinking, separated by provenance and
NEVER quality-gated (the `kept (0)` fix)."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import tempfile
import unittest

import numpy as np

from eris.memory.thought_stream import (
    Thought, ThoughtStream, link_and_store, default_tier,
)


class TestThoughtStream(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self.path = os.path.join(self._dir, "thoughts.jsonl")

    def test_add_and_size(self):
        s = ThoughtStream(path=self.path)
        self.assertEqual(s.size(), 0)
        t = link_and_store(s, "Kuramoto coupling", "plastic", "A first thought.")
        self.assertEqual(s.size(), 1)
        self.assertIsInstance(t, Thought)
        self.assertEqual(t.provenance, "internal")

    def test_by_topic_is_trajectory_oldest_first(self):
        s = ThoughtStream(path=self.path)
        link_and_store(s, "phase", "plastic", "one")
        link_and_store(s, "phase", "elastic", "two")
        link_and_store(s, "other", "plastic", "x")
        link_and_store(s, "phase", "plastic", "three")
        traj = s.by_topic("phase", limit=5)
        self.assertEqual([t.text for t in traj], ["one", "two", "three"])

    def test_links_to_prior(self):
        s = ThoughtStream(path=self.path)
        a = link_and_store(s, "topic", "plastic", "first")
        b = link_and_store(s, "topic", "plastic", "second")
        self.assertIn(a.id, b.prior)

    def test_never_gated_keeps_everything(self):
        # Even trivial / unproven text is kept — storage is not assertion.
        s = ThoughtStream(path=self.path)
        link_and_store(s, "t", "plastic", "x")          # would fail any quality gate
        link_and_store(s, "t", "plastic", "")           # even empty stays
        self.assertEqual(s.size(), 2)

    def test_persistence_roundtrip(self):
        s = ThoughtStream(path=self.path)
        v = [0.1, 0.2, 0.3, 0.4]
        link_and_store(s, "persisted", "plastic", "remember me", embedding=v)
        # New instance reads the JSONL back.
        s2 = ThoughtStream(path=self.path)
        self.assertEqual(s2.size(), 1)
        got = s2.by_topic("persisted")[0]
        self.assertEqual(got.text, "remember me")
        self.assertEqual(len(got.embedding), 4)

    def test_retrieve_by_embedding(self):
        s = ThoughtStream(path=self.path)
        link_and_store(s, "a", "plastic", "near", embedding=[1.0, 0.0, 0.0])
        link_and_store(s, "b", "plastic", "far", embedding=[0.0, 1.0, 0.0])
        link_and_store(s, "c", "plastic", "no-emb")     # not retrievable
        hits = s.retrieve([0.9, 0.1, 0.0], k=2)
        self.assertEqual(hits[0].text, "near")

    def test_default_tier(self):
        self.assertEqual(default_tier("plastic"), "bridge")
        self.assertEqual(default_tier("transfixed"), "speculation")
        self.assertEqual(default_tier("elastic"), "inference")

    def test_explicit_claims_preserved(self):
        s = ThoughtStream(path=self.path)
        claims = [{"text": "X resembles Y", "tier": "bridge"}]
        t = link_and_store(s, "t", "plastic", "body", claims=claims)
        self.assertEqual(t.claims, claims)

    def test_supersede_is_visible_mind_change(self):
        # A retraction is a logged event: the old thought stays on disk (history,
        # still in by_topic) but drops out of the *active* trajectory.
        s = ThoughtStream(path=self.path)
        old = link_and_store(s, "gate", "plastic", "the gate IS a projection")
        new = link_and_store(s, "gate", "elastic", "actually only resembles one",
                             supersedes=old.id)
        self.assertEqual(new.supersedes, old.id)
        self.assertEqual([t.id for t in s.by_topic("gate")], [old.id, new.id])
        self.assertEqual([t.id for t in s.active_by_topic("gate")], [new.id])


class _Rec:
    def __init__(self, emb, rid="r1"):
        self.embedding = emb
        self.text = "held material about Kuramoto coupling and phase"
        self.id = rid


class TestIntrospectStoresThought(unittest.TestCase):
    def test_introspect_keeps_its_thought(self):
        # An introspection cycle has no external passages; it must still KEEP its
        # own thought (never report kept (0)).
        from eris.metacognition.dreaming import DreamingLoop

        s = ThoughtStream(path=os.path.join(tempfile.mkdtemp(), "t.jsonl"))

        class _Mem:
            class _Sub:
                def store(self, *a, **k):
                    pass
            mtm = _Sub()

        loop = DreamingLoop(autobiography=None, memory=_Mem(),
                            thought_stream=s)
        # Force a reflection (avoid network/model calls).
        loop._reflect = lambda *a, **k: "This is what I think about it."
        v = np.ones(8, dtype=np.float32)
        res = loop._introspect("Kuramoto coupling", [_Rec(v)], "broad", False)
        self.assertIsNotNone(res)
        self.assertEqual(res["stored"], 1)
        self.assertEqual(s.size(), 1)
        self.assertEqual(s.all()[0].drew_on, ["r1"])


class TestTruthContract(unittest.TestCase):
    def test_fabricated_human_autobiography_flagged(self):
        from eris.metacognition.truth_contract import fabricated_self
        self.assertTrue(fabricated_self(
            "I walked into the office and sat at the back."))
        self.assertTrue(fabricated_self("the day I was told my hair was too long"))
        self.assertTrue(fabricated_self("when I was a child my boss said no"))

    def test_metaphorical_interiority_not_flagged(self):
        # Her real first-person (regimes, felt pulls) must pass — only concrete
        # invented HUMAN experience is caught.
        from eris.metacognition.truth_contract import fabricated_self
        self.assertFalse(fabricated_self(
            "I feel the idea pull toward the boundary; the field is reshaping."))
        self.assertFalse(fabricated_self(
            "I notice a loop reinforcing itself as I sit with the contradiction."))
        self.assertFalse(fabricated_self("This resembles a projection onto ker(C)."))

    def test_contracts_have_distinct_content(self):
        from eris.metacognition.truth_contract import (
            PONDER_CONTRACT, DREAM_CONTRACT)
        self.assertIn("never invent human biographical events", PONDER_CONTRACT)
        self.assertIn("dream", DREAM_CONTRACT.lower())


class TestHybridCycleStorage(unittest.TestCase):
    def test_explore_reflection_reaches_thought_stream(self):
        # A ponder that searches AND reflects (idle_explore) must store its
        # reflection to the thought-stream, not just medium-term memory.
        from eris.metacognition.dreaming import DreamingLoop

        s = ThoughtStream(path=os.path.join(tempfile.mkdtemp(), "t.jsonl"))

        class _Mem:
            class _Sub:
                def store(self, *a, **k):
                    pass
            mtm = _Sub()
            def consolidate(self):
                pass

        loop = DreamingLoop(autobiography=None, memory=_Mem(),
                            thought_stream=s)
        # Drive a fully-stubbed explore cycle: a real topic, no crawl, a reflection.
        loop._pick_crawl_topic = lambda: ("Kuramoto coupling", False, "broad")
        loop._run_research = lambda topic: None
        loop._claude_condense_and_refine = lambda *a, **k: (None, None, False)
        loop._reflect = lambda *a, **k: "My grounded reflection on this."
        # Hold material so the router introspects... instead force the web/ponder
        # path by making coverage low: patch route_topic via topic_router import.
        import eris.metacognition.topic_router as tr
        orig = tr.route_topic
        tr.route_topic = lambda *a, **k: {"action": "web", "query": "Kuramoto coupling"}
        try:
            res = loop.idle_explore()
        finally:
            tr.route_topic = orig
        self.assertIsNotNone(res)
        self.assertEqual(s.size(), 1)
        self.assertEqual(s.all()[0].text, "My grounded reflection on this.")


if __name__ == "__main__":
    unittest.main()
