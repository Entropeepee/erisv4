"""§A3/§B1 wiring: a full turn still runs (guarded, default-OFF research), the workspace
goal is set each turn, and the working-memory frame the prompt reads is bounded. The hive
research entry point exists and is async. Offline (no LLM backend)."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import asyncio
import inspect
import tempfile
import unittest


class TestHiveWiring(unittest.TestCase):
    def test_turn_sets_goal_and_bounded_frame(self):
        from eris.orchestrator import ErisOrchestrator
        orch = ErisOrchestrator(field_size=16, data_dir=tempfile.mkdtemp())
        asyncio.run(orch.process("Tell me about emergence"))
        # §B1: the workspace now carries the active goal for the working-memory frame
        self.assertEqual(orch.workspace.goal_text, "Tell me about emergence")
        frame = orch.workspace.working_set(k=3)
        self.assertIn("goal", frame)
        self.assertLessEqual(len(frame["broadcasts"]), 3)         # bounded

    def test_assemble_prompt_excludes_bid_labels(self):
        # the working-memory section must NOT inject the raw field-projection bid label
        from eris.orchestrator import ErisOrchestrator
        from eris.computation.activations import BVec
        orch = ErisOrchestrator(field_size=16, data_dir=tempfile.mkdtemp())
        orch.workspace.broadcast("Logos: 0.742 bid on B+F", BVec(B=0.6, F=0.7),
                                 source="logos", coherence=0.4)
        prompt = orch._assemble_prompt("hello", None, "", BVec(E=0.5), "elastic")
        self.assertNotIn("bid on", prompt)                        # label never injected

    def test_hive_research_entry_point_is_async(self):
        from eris.orchestrator import ErisOrchestrator
        self.assertTrue(inspect.iscoroutinefunction(ErisOrchestrator.hive_research))


if __name__ == "__main__":
    unittest.main()
