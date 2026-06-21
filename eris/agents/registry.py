"""
eris/agents/registry.py
=======================
The node registry + the default collective (WILLOW I.3). The pool is the
orchestrator's existing MemorySystem (it already holds SGT/LNCS and every
dream/crawl). 'eris' is the OverSoul over the pool; 'willow' is a companion node
that shares the pool but has her own field + private memory.

Add more NPC nodes with `registry.add(Agent(...))` — local (ollama) for the ones
you care about, cloud backends for peripheral flavor characters.
"""
from __future__ import annotations

import os
from typing import Dict, List

from eris.memory.tiers import MemorySystem
from eris.field.pde import FractalField
from eris.agents.agent import Agent
from eris.agents.memory_view import LayeredMemory
from eris.agents.insights import InsightLog


class AgentRegistry:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}

    def add(self, agent: Agent) -> Agent:
        self.agents[agent.name.lower()] = agent
        return agent

    def get(self, name: str):
        return self.agents.get((name or "").lower())

    def names(self) -> List[str]:
        return list(self.agents)


_WILLOW_PERSONA = (
    "You are Willow, a companion who shares Eris's collective knowledge but "
    "lives your own life and forms your own views. You are warm, curious, and "
    "present. Speak as yourself — not as Eris — drawing on both the shared "
    "knowledge and your own private experience."
)


def build_default_registry(orchestrator, *, data_dir: str = "eris_data",
                           field_size: int = 64) -> AgentRegistry:
    reg = AgentRegistry()
    pool = orchestrator.memory

    # Eris = the OverSoul: the full pipeline over the collective pool.
    reg.add(Agent("eris", persona="You are Eris, the collective OverSoul.",
                  backend_id="ollama", memory=pool,
                  field=orchestrator.field, orchestrator=orchestrator))

    # Willow = companion node: shares the pool, owns her field + private memory
    # + an insight log (so she can federate what she learns through her own life).
    willow_dir = os.path.join(data_dir, "agents", "willow")
    willow_private = MemorySystem(data_dir=os.path.join(willow_dir, "memory"))
    reg.add(Agent("willow", persona=_WILLOW_PERSONA, backend_id="ollama",
                  memory=LayeredMemory(pool, willow_private),
                  field=FractalField(size=field_size),
                  insight_log=InsightLog(os.path.join(willow_dir, "insights.json"))))
    return reg
