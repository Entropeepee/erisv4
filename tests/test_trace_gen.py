"""Tests for the distillation trace-generation harness (roadmap 2.1)."""
import os
os.environ.setdefault("ERIS_GPU", "0")

import asyncio
import tempfile
import unittest

from eris.training.trace_gen import generate_traces, load_traces


class _Resp:
    def __init__(self, text, reasoning=""):
        self.text = text
        self.reasoning = reasoning


class _Teacher:
    def __init__(self):
        self.calls = 0

    async def generate(self, prompt, system="", temperature=0.7):
        self.calls += 1
        return _Resp(f"answer to: {prompt}", reasoning="because reasons")


class TestTraceGen(unittest.TestCase):
    def test_writes_traces(self):
        d = tempfile.mkdtemp()
        out = os.path.join(d, "traces.jsonl")
        n = asyncio.run(generate_traces(_Teacher(), ["q1", "q2", "q3"], out))
        self.assertEqual(n, 3)
        rows = load_traces(out)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["response"], "answer to: q1")
        self.assertEqual(rows[0]["reasoning"], "because reasons")

    def test_resume_skips_done(self):
        d = tempfile.mkdtemp()
        out = os.path.join(d, "traces.jsonl")
        asyncio.run(generate_traces(_Teacher(), ["q1", "q2"], out))
        t = _Teacher()
        # Re-run with one overlap + one new task; only the new one is generated.
        n = asyncio.run(generate_traces(t, ["q2", "q3"], out))
        self.assertEqual(n, 1)
        self.assertEqual(t.calls, 1)                 # q2 skipped, only q3 called
        self.assertEqual(len(load_traces(out)), 3)

    def test_dict_tasks_with_system(self):
        d = tempfile.mkdtemp()
        out = os.path.join(d, "traces.jsonl")
        asyncio.run(generate_traces(
            _Teacher(), [{"prompt": "p", "system": "be brief"}], out))
        rows = load_traces(out)
        self.assertEqual(rows[0]["system"], "be brief")

    def test_skips_empty_responses(self):
        class _Empty:
            async def generate(self, prompt, system="", temperature=0.7):
                return _Resp("")
        d = tempfile.mkdtemp()
        out = os.path.join(d, "traces.jsonl")
        n = asyncio.run(generate_traces(_Empty(), ["q1"], out))
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()
