"""Tests for the study engine's summary — esp. the empty/unreachable case that
produced the confusing 'I'm not sure which topics you'd like...' LLM reply."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest

from eris.knowledge.study import StudyEngine


class _Mediator:
    """A mediator that should NOT be called when nothing was ingested."""
    def __init__(self):
        self.called = False
    async def generate(self, prompt="", system=""):
        self.called = True
        class _R:
            text = "I'm not sure which topics you'd like summarized."
        return _R()


def _engine(mediator=None):
    return StudyEngine(extractor=None, memory=None,
                       data_dir="/tmp", mediator=mediator)


class TestStudySummary(unittest.TestCase):
    def test_empty_does_not_call_llm_and_explains_why(self):
        med = _Mediator()
        eng = _engine(med)
        read = [{"topic": "Cognitive science", "source": "Wikipedia: Cognitive science",
                 "chunks": 0, "error": "<urlopen error [Errno -3] name resolution>"}]
        summary = eng._summarize(["Cognitive science"], read, total_chunks=0)
        self.assertFalse(med.called)                         # no empty-topic LLM call
        self.assertIn("none reachable", summary)
        self.assertIn("network", summary.lower())            # diagnostic, not nonsense
        self.assertIn("name resolution", summary)            # surfaces the real cause

    def test_productive_run_still_summarizes(self):
        eng = _engine(mediator=None)
        read = [{"topic": "Self-organization", "source": "Wikipedia: Self-organization",
                 "chunks": 4}]
        summary = eng._summarize(["Self-organization"], read, total_chunks=4)
        self.assertIn("Studied 1 of 1 topics", summary)
        self.assertIn("Self-organization", summary)
        self.assertNotIn("none reachable", summary)


if __name__ == "__main__":
    unittest.main()
