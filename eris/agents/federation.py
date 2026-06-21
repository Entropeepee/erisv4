"""
eris/agents/federation.py
=========================
Novelty-gated federation (WILLOW I.7). A node distills insights from its own
private experience; an insight that is NOVEL versus the collective pool is
written to the pool — so every other node, reading the pool, now gains it.

This is the experiment: an insight one node earned through its own life can be
acted on by a *different* node. The novelty gate is what keeps the pool from
filling with near-duplicates (and makes re-federation idempotent — once an
insight is in the pool it is no longer "far from everything", so it won't be
pushed again).
"""
from __future__ import annotations


def federate(insight_log, node_name: str, pool, novelty: float = 0.30) -> int:
    """Push the node's NOVEL distilled insights into the collective pool.
    Returns the number federated this pass."""
    if insight_log is None:
        return 0
    pushed = 0
    for ins in insight_log.recent(limit=20):
        if not ins.embedding:
            continue
        # far from everything already in the pool?
        if pool.max_similarity(ins.embedding) < (1.0 - novelty):
            try:
                pool.store_text(ins.summary, embedding=ins.embedding,
                                source=f"node:{node_name}", kind="insight",
                                regime=ins.regime)
                ins.federated = True
                pushed += 1
            except Exception:
                pass
    if pushed:
        insight_log.save()
    return pushed
