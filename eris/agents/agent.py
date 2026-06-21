"""
eris/agents/agent.py
====================
An Eris node (WILLOW I.3). persona + backend + its own field + layered memory.

The 'eris' node is the OverSoul: it delegates to the full orchestrator pipeline
(its memory IS the pool). Other nodes (Willow, NPCs) run the lighter node path:
read shared+private memory, evolve their own field on the input, answer through
their backend, and write the exchange to PRIVATE memory only (so they diverge).
"""
from __future__ import annotations

import time
from typing import Optional

from eris.knowledge.embeddings import get_embedding


def _when(ts: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
    except Exception:
        return "undated"


class Agent:
    def __init__(self, name: str, persona: str, backend_id: str, memory,
                 field=None, orchestrator=None):
        self.name = name
        self.persona = persona
        self.backend_id = backend_id
        self.memory = memory
        self.field = field
        self._orch = orchestrator        # set only for the 'eris' OverSoul node
        self.has_field = field is not None

    async def respond(self, user_text: str, backends: dict, top_k: int = 8) -> str:
        # OverSoul = the full cognitive pipeline over the pool.
        if self._orch is not None:
            result = await self._orch.process(user_text)
            return result.response_text

        emb = get_embedding(user_text)
        ctx = self.memory.retrieve(query_embedding=emb, top_k=top_k, query_text=user_text)
        if self.field is not None:
            try:
                self.field.seed_from_text(user_text)
                self.field.run(20)
            except Exception:
                pass

        ctx_text = "\n\n".join(
            f"[{_when(getattr(r, 'timestamp', 0))} · {r.source}] {r.text}" for r in ctx
        ) or "(nothing yet)"
        backend = backends.get(self.backend_id) or backends.get("ollama")
        prompt = ("Relevant memory — the collective knowledge you share with Eris, "
                  f"plus your OWN private experience:\n{ctx_text}\n\n"
                  f"User: {user_text}\n\n{self.name}:")
        reply = ""
        if backend is not None:
            try:
                resp = await backend.generate(prompt, system=self.persona)
                reply = (getattr(resp, "text", "") or "").strip()
            except Exception:
                reply = ""
        if not reply:
            reply = "(I have nothing to say to that yet.)"

        # Diverge: write this exchange to PRIVATE memory only.
        try:
            self.memory.store_experience(f"User: {user_text}", embedding=emb, kind="heard")
            self.memory.store_experience(f"{self.name}: {reply}",
                                         embedding=get_embedding(reply), kind="said")
        except Exception:
            pass
        return reply

    def regime(self) -> Optional[str]:
        try:
            return self.field.detect_regime() if self.field is not None else None
        except Exception:
            return None
