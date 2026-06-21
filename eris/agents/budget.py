"""
eris/agents/budget.py
=====================
Resource governance + the degradation ladder (WILLOW I.9). Caps how many NPC↔NPC
conversations run per hour and picks the cheapest viable dialogue mode:

    genuine (each node thinks)  ->  puppeteer (one local call)  ->  skip

so ambient chatter never starves the renderer or the API budget.
"""
from __future__ import annotations

import time
from typing import Dict, List


class ConversationBudget:
    def __init__(self, per_hour: int = 60):
        self.per_hour = per_hour
        self._t: List[float] = []

    def allow(self) -> bool:
        now = time.time()
        self._t = [x for x in self._t if now - x < 3600]
        return len(self._t) < self.per_hour

    def charge(self) -> None:
        self._t.append(time.time())


def choose_dialogue_plan(speakers, backends: Dict, budget: ConversationBudget) -> Dict:
    """genuine (all speakers' backends available) -> local puppeteer -> skip."""
    if not budget.allow():
        return {"mode": "skip"}
    if speakers and all(
        backends.get(s.backend_id) is not None
        and getattr(backends.get(s.backend_id), "is_available", lambda: True)()
        for s in speakers
    ):
        return {"mode": "genuine"}
    if "ollama" in backends:
        return {"mode": "puppeteer", "backend": "ollama"}
    return {"mode": "skip"}
