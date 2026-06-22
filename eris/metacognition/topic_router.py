"""Topic reasoning + source routing (Layer 1 of the deep-reasoning discipline).

Before a learning cycle searches, decide two things: *do I already hold this?*
and *what's the real query, not the surface phrase?* This stops two failures:
web-searching the literal word the user typed ("skeleton code" → a search for
strangers' repos) and the idle loop latching onto a stray word ("learned about:
Prompt").

Routing:
  • chat residue / stray words      -> skip   (do nothing, log nothing)
  • material she already holds       -> introspect (reflect over her own memory)
  • genuinely new outward curiosity  -> web    (the existing crawl, on a real query)

Not a barrier — a deliberation. She still chooses freely; she just reasons about
the choice first.
"""
from __future__ import annotations
from typing import Callable, Optional
import re

import numpy as np

_CHAT_NOISE = re.compile(
    r"\b(i think|i fixed|you should|read my|let me|can you|could you|thanks|"
    r"okay|got it|please|sorry|lol|hey|hi|prompt)\b", re.IGNORECASE)

COVERAGE_THRESHOLD = 0.45   # how on-topic her own memory must be to introspect


def _looks_like_chat(t: str) -> bool:
    t = (t or "").strip()
    return (not t) or len(t.split()) > 8 or bool(_CHAT_NOISE.search(t)) \
        or t.endswith((".", "!", "?"))


def _coverage(hits, query_embedding) -> float:
    """Max cosine similarity between the query and her retrieved memory — how well
    she already holds material on this topic."""
    qe = np.asarray(query_embedding, dtype=np.float32).ravel()
    qn = float(np.linalg.norm(qe))
    if qn < 1e-9 or not hits:
        return 0.0
    best = 0.0
    for h in hits:
        e = getattr(h, "embedding", None)
        if e is None:
            continue
        e = np.asarray(e, dtype=np.float32).ravel()
        en = float(np.linalg.norm(e))
        if en < 1e-9 or e.shape != qe.shape:
            continue
        best = max(best, float(np.dot(qe, e) / (qn * en)))
    return best


def route_topic(intention: str, memory, embed: Callable,
                expand: Optional[Callable[[str], str]] = None,
                threshold: float = COVERAGE_THRESHOLD) -> dict:
    """Decide source + build a real query. Returns one of:
      {"action": "skip"}
      {"action": "introspect", "query": q, "seed_hits": hits}
      {"action": "web", "query": q}

    `expand` (optional) reasons the surface phrase into the real query — one cheap
    local-model call; if absent, the intention is used as-is.
    """
    if not intention or _looks_like_chat(intention):
        return {"action": "skip"}
    query = (expand(intention) if expand else intention) or intention
    try:
        qe = embed(query)
        hits = memory.retrieve(query_embedding=qe, top_k=5)
    except Exception:
        qe, hits = None, []
    if qe is not None and _coverage(hits, qe) >= threshold:
        return {"action": "introspect", "query": query, "seed_hits": hits}
    return {"action": "web", "query": query}
