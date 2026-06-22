"""Tests for the ReAct agent tool builders (Q1/Q2 operationalization)."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import tempfile
import unittest

from eris.config import CONFIG
from eris.executive.agent_tools import (
    factual_lookup_tool, durable_memory_tools, default_tools,
)
from eris.memory.durable import LocalFactStore


class _RecMemory:
    """Minimal stand-in exposing all_records(), like MemorySystem."""
    def __init__(self, texts):
        self._recs = [type("R", (), {"text": t, "embedding": None})() for t in texts]

    def all_records(self, limit=None):
        return self._recs


class TestFactualLookupTool(unittest.TestCase):
    def test_finds_exact_token(self):
        mem = _RecMemory(["weather notes", "the sgtpatent gating proof", "a recipe"])
        tool = factual_lookup_tool(mem)
        out = tool.run("sgtpatent")
        self.assertIn("sgtpatent", out)

    def test_empty_memory_message(self):
        tool = factual_lookup_tool(_RecMemory([]))
        self.assertIn("No records", tool.run("anything"))


class TestDurableMemoryTools(unittest.TestCase):
    def test_remember_then_recall(self):
        store = LocalFactStore(os.path.join(tempfile.mkdtemp(), "f.json"))
        remember, recall = durable_memory_tools(store)
        self.assertIn("Remembered", remember.run("Eris runs on an RTX 5080"))
        self.assertIn("5080", recall.run("RTX 5080"))


class TestDefaultTools(unittest.TestCase):
    def test_flags_gate_the_toolset(self):
        class _Orch:
            memory = _RecMemory(["x"])
            data_dir = tempfile.mkdtemp()
            _durable_memory = None

        orch = _Orch()
        # Both off -> no tools.
        CONFIG.agent_tool_factual_lookup = False
        CONFIG.agent_tool_durable_memory = False
        self.assertEqual(default_tools(orch), [])
        # Turn both on -> factual_lookup + remember_fact + recall_facts.
        CONFIG.agent_tool_factual_lookup = True
        CONFIG.agent_tool_durable_memory = True
        try:
            names = {t.name for t in default_tools(orch)}
            self.assertEqual(
                names, {"factual_lookup", "remember_fact", "recall_facts"})
        finally:
            CONFIG.agent_tool_factual_lookup = False
            CONFIG.agent_tool_durable_memory = False


if __name__ == "__main__":
    unittest.main()
