"""Thought-stream — her OWN thinking, separated from the library by provenance.

Two memory spaces, split by *provenance*, not quality:
  • LIBRARY (the existing MemorySystem) — what she TOOK IN (papers, web passages,
    ingested docs). The boilerplate quality gate (`is_useful`) belongs HERE only.
  • THOUGHT-STREAM (this module) — what she MADE (introspections, hypotheses,
    connections). NEVER quality-gated — an unproven idea is the normal, correct
    state of a new thought. Filtered only for *labeling*: each claim carries a
    tier (fact/inference/bridge/speculation) so its epistemic status travels with
    it. Storage ≠ assertion: she keeps everything she thinks; she just doesn't
    *report* an unproven thought as fact.

This fixes the `kept (0)` bug — the external-source boilerplate gate was running
on introspect cycles that have no incoming passages, so it discarded her output.
An idea generated once and thrown away has no BLECD trajectory; keeping the
linked sequence is what lets a thought move through the domains over time.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import os
import time
import uuid

import numpy as np


@dataclass
class Thought:
    id: str
    timestamp: float
    topic: str                      # normalized key, for threading
    regime: str                     # field regime it was generated in
    text: str                       # the FULL reflection — stored whole, never minimized
    claims: list = field(default_factory=list)   # [{text, tier}] from the Layer-2 critic
    provenance: str = "internal"
    drew_on: list = field(default_factory=list)   # library/source ids referenced
    prior: list = field(default_factory=list)     # ids of earlier Thoughts on this topic
    supersedes: Optional[str] = None              # if this revises/retracts a prior thought
    embedding: Optional[list] = None              # for retrieval (her own ideas are retrievable)


def _norm(topic: str) -> str:
    return " ".join((topic or "").lower().split())


class ThoughtStream:
    """Append-only, provenance-internal store of her own thoughts. Persisted as
    JSONL so promotions/retractions are logged events, not hidden overwrites."""

    def __init__(self, path: str = "eris_data/thoughts.jsonl"):
        self.path = path
        self._thoughts: List[Thought] = []
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._thoughts.append(Thought(**json.loads(line)))
                        except (json.JSONDecodeError, TypeError):
                            continue
        except FileNotFoundError:
            pass

    def _append(self, t: Thought) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
        except OSError:
            pass

    def add(self, t: Thought) -> Thought:
        self._thoughts.append(t)
        self._append(t)
        return t

    def by_topic(self, topic: str, limit: int = 5) -> List[Thought]:
        """Her prior thoughts on a topic, oldest→newest (the trajectory)."""
        key = _norm(topic)
        hits = [t for t in self._thoughts if _norm(t.topic) == key]
        return hits[-limit:]

    def active_by_topic(self, topic: str, limit: int = 5) -> List[Thought]:
        """Her *current* thinking on a topic — the trajectory with retracted /
        superseded thoughts dropped, so changing her mind is a visible event
        (the old thought stays on disk as history, just no longer 'active')."""
        key = _norm(topic)
        superseded = {t.supersedes for t in self._thoughts if t.supersedes}
        hits = [t for t in self._thoughts
                if _norm(t.topic) == key and t.id not in superseded]
        return hits[-limit:]

    def retrieve(self, query_embedding, k: int = 5) -> List[Thought]:
        """Her own ideas, ranked by semantic similarity (labeled as hers)."""
        qe = np.asarray(query_embedding, dtype=np.float32).ravel()
        qn = float(np.linalg.norm(qe))
        if qn < 1e-9:
            return []
        scored = []
        for t in self._thoughts:
            if not t.embedding:
                continue
            e = np.asarray(t.embedding, dtype=np.float32).ravel()
            en = float(np.linalg.norm(e))
            if en < 1e-9 or e.shape != qe.shape:
                continue
            scored.append((t, float(np.dot(qe, e) / (qn * en))))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[:k]]

    def all(self) -> List[Thought]:
        return list(self._thoughts)

    def size(self) -> int:
        return len(self._thoughts)


def default_tier(regime: str) -> str:
    """Coarse epistemic default from the field regime (Layer-2 field-state check):
    a thought formed while the field is reshaping (plastic) is generative — a
    bridge/speculation — until grounded; a settled (elastic) thought may be an
    inference. Per-claim tiering is the LLM critic's finer job."""
    return {"plastic": "bridge", "transfixed": "speculation",
            "warmup": "speculation"}.get(regime, "inference")


def link_and_store(stream: ThoughtStream, topic: str, regime: str, text: str,
                   *, embedding=None, drew_on=None, claims=None,
                   supersedes: Optional[str] = None) -> Thought:
    """Build a Thought, link it to her prior thoughts on the topic (trajectory),
    and store it whole. NEVER quality-gated — her own output is always kept.

    `supersedes` records a revision/retraction of an earlier thought as a logged
    event (the prior thought stays on disk as history), so changing her mind is
    inspectable rather than a hidden overwrite."""
    prior = stream.by_topic(topic, limit=5)
    t = Thought(
        id=uuid.uuid4().hex[:12], timestamp=time.time(), topic=topic, regime=regime,
        text=text,
        claims=claims if claims is not None else [{"text": text[:500],
                                                   "tier": default_tier(regime)}],
        drew_on=list(drew_on or []),
        prior=[p.id for p in prior],
        supersedes=supersedes,
        embedding=(np.asarray(embedding, dtype=np.float32).ravel().tolist()
                   if embedding is not None else None))
    return stream.add(t)
