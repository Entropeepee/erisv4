"""Tests for the WILLOW multi-node collective (eris/agents)."""
import os
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import asyncio
import tempfile
import unittest

from eris.memory.tiers import MemorySystem
from eris.knowledge.embeddings import get_embedding
from eris.agents.memory_view import LayeredMemory
from eris.agents.agent import Agent
from eris.agents.insights import InsightLog
from eris.agents.federation import federate
from eris.agents.dialogue import generate_dialogue
from eris.agents.budget import ConversationBudget, choose_dialogue_plan


class _Stub:
    """Minimal backend with the mediator's generate() shape."""
    def __init__(self, text="ok"):
        self._text = text

    def is_available(self):
        return True

    async def generate(self, prompt="", system="", **kw):
        class R:
            text = self._text
        R.text = self._text
        return R()


def _mk(tmp, name="willow", with_log=True):
    pool = MemorySystem(data_dir=os.path.join(tmp, "pool"))
    private = MemorySystem(data_dir=os.path.join(tmp, name))
    log = InsightLog(os.path.join(tmp, f"{name}.json")) if with_log else None
    agent = Agent(name, "You are " + name, "ollama",
                  memory=LayeredMemory(pool, private), field=None, insight_log=log)
    return pool, agent


class TestLayeredMemory(unittest.TestCase):
    def test_reads_shared_and_private(self):
        tmp = tempfile.mkdtemp()
        pool, agent = _mk(tmp)
        pool.store_text("A shared fact about coherence.", embedding=get_embedding("shared coherence"))
        agent.memory.store_experience("A private memory of mine.", embedding=get_embedding("private mine"))
        hits = agent.memory.retrieve(query_embedding=get_embedding("coherence"), top_k=5)
        texts = " ".join(h.text for h in hits)
        self.assertIn("shared fact", texts)
        self.assertIn("private memory", texts)


class TestDivergence(unittest.TestCase):
    def test_respond_writes_private_not_pool(self):
        tmp = tempfile.mkdtemp()
        pool, agent = _mk(tmp)
        before = pool.stm.size
        reply = asyncio.run(agent.respond("hello there", {"ollama": _Stub("Hi, I'm Willow.")}))
        self.assertTrue(reply)
        self.assertEqual(pool.stm.size, before)          # pool did NOT grow
        self.assertGreater(agent.memory.private.stm.size, 0)  # private grew


class TestFederation(unittest.TestCase):
    def test_distill_federate_idempotent(self):
        tmp = tempfile.mkdtemp()
        pool, agent = _mk(tmp)
        agent.memory.store_experience("User: tell me about phase locking", kind="heard")
        ins = asyncio.run(agent.distill({"ollama": _Stub("Phase locking strengthens under shared noise.")}))
        self.assertIsNotNone(ins)
        self.assertEqual(pool.ltm.size, 0)
        self.assertEqual(federate(agent.insight_log, agent.name, pool), 1)
        self.assertEqual(pool.ltm.size, 1)
        self.assertEqual(federate(agent.insight_log, agent.name, pool), 0)  # idempotent


class TestDialogue(unittest.TestCase):
    def test_parse_script(self):
        tmp = tempfile.mkdtemp()
        _, a = _mk(tmp, "willow")
        _, b = _mk(tmp, "sage")
        backend = _Stub("Willow: Hello sage.\nSage: Greetings, Willow.")
        script = asyncio.run(generate_dialogue(backend, [a, b], context="meeting", turns=2))
        self.assertEqual(len(script), 2)
        self.assertEqual(script[0]["speaker"], "willow")
        self.assertEqual(script[1]["speaker"], "sage")


class TestBudget(unittest.TestCase):
    def test_ladder(self):
        tmp = tempfile.mkdtemp()
        _, a = _mk(tmp, "willow")
        _, b = _mk(tmp, "sage")
        budget = ConversationBudget(per_hour=1)
        # both on ollama (available) -> genuine
        self.assertEqual(choose_dialogue_plan([a, b], {"ollama": _Stub()}, budget)["mode"], "genuine")
        budget.charge()
        # budget exhausted -> skip
        self.assertEqual(choose_dialogue_plan([a, b], {"ollama": _Stub()}, budget)["mode"], "skip")


if __name__ == "__main__":
    unittest.main()
