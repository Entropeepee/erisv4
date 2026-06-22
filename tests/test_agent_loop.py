"""Tests for the grounded ReAct agent loop (roadmap 3.1)."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import asyncio
import unittest

from eris.executive.agent_loop import ReActAgent, Tool


class _Resp:
    def __init__(self, text):
        self.text = text


class _ScriptedMediator:
    """Returns a queued list of outputs, one per generate() call."""
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    async def generate(self, prompt, system="", *a, **k):
        self.prompts.append(prompt)
        return _Resp(self.outputs.pop(0) if self.outputs else "Final Answer: done")


class TestReActLoop(unittest.TestCase):
    def test_uses_tool_then_finalizes(self):
        calls = {}
        def calc(x):
            calls["arg"] = x
            return "42"
        med = _ScriptedMediator([
            "Thought: I should compute it.\nAction: calc\nAction Input: 6*7",
            "Thought: I have it.\nFinal Answer: The result is 42.",
        ])
        agent = ReActAgent(med, [Tool("calc", "evaluate math", calc)])
        out = asyncio.run(agent.run("what is 6*7"))
        self.assertTrue(out["ok"])
        self.assertEqual(calls["arg"], "6*7")
        self.assertIn("42", out["answer"])
        self.assertEqual(out["steps"], 2)

    def test_field_state_grounds_the_prompt(self):
        med = _ScriptedMediator(["Final Answer: ok"])
        agent = ReActAgent(med, [Tool("noop", "does nothing", lambda x: "")],
                           field_state_fn=lambda: {"coherence": 0.04, "regime": "transfixed"})
        asyncio.run(agent.run("goal"))
        self.assertIn("transfixed", med.prompts[0])
        self.assertIn("coherence", med.prompts[0])

    def test_reflexion_on_unparseable_then_recovers(self):
        med = _ScriptedMediator([
            "I am just rambling with no action.",          # no valid action -> reflexion
            "Final Answer: recovered.",
        ])
        agent = ReActAgent(med, [Tool("noop", "noop", lambda x: "")])
        out = asyncio.run(agent.run("goal"))
        self.assertTrue(out["ok"])
        self.assertEqual(out["answer"], "recovered.")
        self.assertTrue(any("reflection" in t for t in out["trace"]))

    def test_async_tool_and_error_reflexion(self):
        async def flaky(x):
            raise RuntimeError("boom")
        med = _ScriptedMediator([
            "Action: flaky\nAction Input: go",
            "Final Answer: handled the error.",
        ])
        agent = ReActAgent(med, [Tool("flaky", "fails", flaky)])
        out = asyncio.run(agent.run("goal"))
        self.assertTrue(out["ok"])
        self.assertTrue(any("ERROR" in str(t.get("observation", "")) for t in out["trace"]))

    def test_gives_up_after_max_steps(self):
        med = _ScriptedMediator(["no action here"] * 10)
        agent = ReActAgent(med, [Tool("noop", "noop", lambda x: "")], max_steps=3)
        out = asyncio.run(agent.run("goal"))
        self.assertFalse(out["ok"])
        self.assertEqual(out["steps"], 3)
        self.assertIsNone(out["answer"])


if __name__ == "__main__":
    unittest.main()
