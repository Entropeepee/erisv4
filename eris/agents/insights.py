"""
eris/agents/insights.py
=======================
Per-node insight records (WILLOW I.6). A node's INSIGHT is a distilled
understanding it formed from its own experience. Federation reads recent() to
find the novel ones and push them up to the collective pool.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict, field
from typing import List


@dataclass
class Insight:
    id: str
    summary: str            # one-line distilled understanding
    embedding: List[float]  # vector, for novelty comparison
    timestamp: float
    regime: str = ""        # field regime when it settled
    trigger: str = ""       # the experience that produced it
    federated: bool = False # already pushed to the pool


class InsightLog:
    """Per-node record of distilled insights (append-only, capped, persisted)."""

    def __init__(self, path: str, cap: int = 200):
        self.path = path
        self.cap = cap
        self._items: List[Insight] = []
        self.load()

    def add(self, summary: str, embedding, regime: str = "", trigger: str = "") -> Insight:
        emb = [] if embedding is None else [float(x) for x in embedding]
        ins = Insight(uuid.uuid4().hex[:12], summary, emb,
                      time.time(), regime, trigger)
        self._items.append(ins)
        self._items = self._items[-self.cap:]
        self.save()
        return ins

    def recent(self, limit: int = 20) -> List[Insight]:
        return list(reversed(self._items[-limit:]))

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump([asdict(i) for i in self._items], f)
        except Exception:
            pass

    def load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    self._items = [Insight(**d) for d in json.load(f)]
            except Exception:
                self._items = []
