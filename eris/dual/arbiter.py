"""The independent arbiter (§3) — the load-bearing piece.

Scores a retrieval result on TASK SUCCESS, never on agreement with the other
path. RAG is a floor, not ground truth; if we scored the field against RAG's
picks we'd train it to imitate embedding space and delete the reason it exists.

Judges (offline ones ship now; the LLM one is flagged):
  • gold_passage_at_k — is the labelled gold passage in the top-k? (needs an eval
    set with gold ids)
  • citation_resolves — does the top result actually CONTAIN the query's salient
    terms (a cheap "did retrieval help" proxy; high for a lexical near-neighbour,
    so it can't carry a wrong result on its own)
  • answer_grounded — (flagged, ERIS_ARBITER_LLM) feed context to the LLM, answer
    + cite, run the fabrication/grounding detector. Off in offline tests.

Sub-scores fuse into `success` via Hill-Power shrinkage (davidian_weight), which
KILLS below-floor partial credit — a plausible-but-wrong near-neighbour scores
low even though a naive cosine ranks it well.
"""
from __future__ import annotations
from typing import Any, Iterable, Optional
import re

import numpy as np

from eris.computation.shrinkage import davidian_weight
from eris.dual.types import record_id

_WORD = re.compile(r"[a-z0-9]{3,}")
_STOP = {"the", "and", "for", "with", "that", "this", "what", "how", "does",
         "are", "was", "were", "from", "into", "about", "which", "who", "why",
         "you", "your", "its", "his", "her", "their", "they", "can", "could",
         "would", "should", "will", "have", "has", "had"}


def _terms(text: str) -> set:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP}


def gold_passage_at_k(result, gold, k: int = 8):
    """(hit, rank): is a gold id within the top-k? rank is 1-based, or None."""
    if gold is None:
        return None, None
    golds = {gold} if isinstance(gold, str) else set(gold)
    ids = result.top_ids(k)
    for i, rid in enumerate(ids):
        if rid in golds:
            return 1.0, i + 1
    return 0.0, None


def citation_resolves(query, result, k: int = 3) -> float:
    """Fraction of the query's salient terms that actually appear in the top-k
    retrieved texts — a cheap 'would the citation resolve' proxy."""
    q = _terms(query)
    if not q or result.is_empty():
        return 0.0
    hay = set()
    for r in result.records[:k]:
        hay |= _terms(getattr(r, "text", "") or "")
    return len(q & hay) / len(q)


class Arbiter:
    """Pluggable judges → sub-scores in [0,1] + a fused `success`. None of them
    compares to the other path."""

    def __init__(self, *, k: int = 8, llm_judge=None,
                 hill=(2.0, 0.5, 1.0, 0.05)):
        self.k = k
        self.llm_judge = llm_judge        # callable(query, result)->[0,1] when ERIS_ARBITER_LLM
        self.hill = hill                  # (alpha, beta, gamma, delta) for davidian_weight

    def _fuse(self, sub: dict, has_gold: bool) -> float:
        a, b, g, d = self.hill
        if has_gold:
            # The eval-set signal dominates; the lexical proxy is a minor witness.
            raw = 0.7 * sub.get("gold_at_k", 0.0) + 0.3 * sub.get("cite", 0.0)
        else:
            vals = [v for kk, v in sub.items()
                    if kk in ("cite", "grounded") and v is not None]
            raw = float(np.mean(vals)) if vals else 0.0
        w = np.asarray(davidian_weight(np.asarray([raw], dtype=float), a, b, g, d)).ravel()
        return float(w[0])

    def score(self, query, result, gold=None) -> dict:
        sub: dict = {}
        hit, rank = gold_passage_at_k(result, gold, self.k)
        if hit is not None:
            sub["gold_at_k"] = hit
            sub["rank"] = rank
        sub["cite"] = round(citation_resolves(query, result), 4)
        if self.llm_judge is not None:
            try:
                sub["grounded"] = float(self.llm_judge(query, result))
            except Exception:
                sub["grounded"] = None
        sub["success"] = round(self._fuse(sub, has_gold=("gold_at_k" in sub)), 4)
        return sub
