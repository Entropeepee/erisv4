"""The discovery→study loop: the hive's UNCLOSED gaps are routed by sensitivity — OPEN work
goes into the autonomous study queue (go learn it), SOVEREIGN/IP work becomes a direct question
to the user (ask the human, never egress). Ambiguous sensitivity fails CLOSED to asking, so a
bad tag can never auto-route IP material to an online search. Offline, deterministic."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import tempfile
import unittest

from eris.memory.autobiography import Autobiography
from eris.memory.tiers import MemorySystem
from eris.metacognition.dreaming import DreamingLoop, distill_gaps


def _loop():
    return DreamingLoop(
        Autobiography(path=os.path.join(tempfile.mkdtemp(), "a.jsonl")),
        MemorySystem(data_dir=tempfile.mkdtemp()))


class TestDistillGaps(unittest.TestCase):
    def test_strips_speaker_label_and_markdown(self):
        out = distill_gaps(["Elos** argues that no side-by-side experiments isolate the gating "
                            "effect from other variables"])
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].startswith("argues"))     # 'Elos**' label removed
        self.assertNotIn("**", out[0])

    def test_strips_source_label(self):
        out = distill_gaps(["Source s:2** contrasts hard binary switching with continuous "
                            "modulation in the prior art"])
        self.assertTrue(out[0].startswith("contrasts"))  # 'Source s:2**' label removed

    def test_drops_too_short_fragments(self):
        out = distill_gaps(["ok", "n/a", "a real gap about thermal drift coefficients across runs"])
        self.assertEqual(len(out), 1)
        self.assertIn("thermal drift", out[0])

    def test_dedups_case_insensitively(self):
        out = distill_gaps(["Thermal drift remains unmeasured", "thermal drift remains unmeasured"])
        self.assertEqual(len(out), 1)

    def test_caps_to_max_items(self):
        gaps = [f"distinct unmeasured quantity number {i} across the listed domains" for i in range(8)]
        self.assertEqual(len(distill_gaps(gaps, max_items=5)), 5)

    def test_trims_long_gap_on_word_boundary(self):
        long_gap = "the gate function behavior under " + ("non-Gaussian " * 40) + "noise is untested"
        out = distill_gaps([long_gap], max_len=160)
        self.assertLessEqual(len(out[0]), 160)
        self.assertEqual(out[0], out[0].rstrip())        # no trailing space / mid-word cut


class TestEnqueueResearchGaps(unittest.TestCase):
    def test_open_routes_to_study_queue_not_notifications(self):
        loop = _loop()
        routed = loop.enqueue_research_gaps(
            ["the thermal drift coefficient is unmeasured across the listed domains"], "open")
        self.assertTrue(routed["queued"])
        self.assertFalse(routed["asked"])
        self.assertTrue(any("thermal drift" in t for t in loop.topic_queue))
        self.assertEqual(loop.pending_questions, [])     # nothing surfaced to the user

    def test_sovereign_routes_to_notifications_not_queue(self):
        loop = _loop()
        routed = loop.enqueue_research_gaps(
            ["the component wear lifetime claim lacks any cited source"], "sovereign")
        self.assertTrue(routed["asked"])
        self.assertFalse(routed["queued"])
        self.assertEqual(loop.topic_queue, [])           # IP work NEVER enters the (online) crawl
        self.assertEqual(len(loop.pending_questions), 1)
        q = loop.pending_questions[0]
        self.assertIn("wear lifetime", q)
        self.assertIn("won't search it online", q)        # the no-egress promise is explicit

    def test_unknown_sensitivity_fails_closed_to_asking(self):
        # a garbage/unknown tag must NOT auto-route IP material online — coerce fails closed to
        # SOVEREIGN, so the safe default is to ask the human, not to web-search.
        loop = _loop()
        loop.enqueue_research_gaps(["a sensitive gap about the proprietary gate function"], "???")
        self.assertEqual(loop.topic_queue, [])
        self.assertEqual(len(loop.pending_questions), 1)

    def test_dedups_against_existing_queue(self):
        loop = _loop()
        loop.topic_queue = ["thermal drift coefficient unmeasured"]
        routed = loop.enqueue_research_gaps(["Thermal drift coefficient unmeasured"], "open")
        self.assertFalse(routed["queued"])               # case-insensitive duplicate skipped
        self.assertEqual(len(loop.topic_queue), 1)

    def test_respects_max_repeat_window(self):
        loop = _loop()
        gap = "the saturation regime behavior is unverified"
        loop._record_query(gap); loop._record_query(gap)   # hit _MAX_REPEAT (2)
        routed = loop.enqueue_research_gaps([gap], "open")
        self.assertFalse(routed["queued"])               # already studied to the cap → not re-queued

    def test_empty_gaps_is_a_noop(self):
        loop = _loop()
        routed = loop.enqueue_research_gaps([], "open")
        self.assertEqual(routed, {"queued": [], "asked": []})
        self.assertEqual(loop.topic_queue, [])
        self.assertEqual(loop.pending_questions, [])

    def test_routing_is_capped(self):
        loop = _loop()
        gaps = [f"distinct unverified mechanism number {i} in the architecture" for i in range(8)]
        loop.enqueue_research_gaps(gaps, "open")
        self.assertLessEqual(len(loop.topic_queue), loop._GAP_ROUTE_CAP)


if __name__ == "__main__":
    unittest.main()
