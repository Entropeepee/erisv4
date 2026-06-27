"""Wrap the retrieval pair (§2) — do NOT modify retrieve_resonant or hybrid.py;
wrap them as (query, **kw) -> RetrievalResult callables for the DualPath.

  traditional = hybrid.py (BM25 + dense + RRF + optional reranker) over the same
                read-only all_records() pool the agent tools use.
  novel       = MemorySystem.retrieve_resonant → (aligned, tension), with per-record
                coupling from field_interference.R_ij when field snapshots exist,
                else an embedding-cosine proxy. The aligned set is the answer; the
                tension/coupling channels ride along for the later epistemic layer.
"""
from __future__ import annotations
from typing import Optional

import numpy as np

from eris.dual.path import DualPath, Mode
from eris.dual.types import RetrievalResult


def _cosine(a, b) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9 or a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _coupling(qe, query_phi, query_theta, rec) -> float:
    """R_ij field resonance when both sides have field snapshots; else cosine."""
    phi_s = getattr(rec, "phi_snapshot", None)
    theta_s = getattr(rec, "theta_snapshot", None)
    if query_phi is not None and query_theta is not None and phi_s is not None and theta_s is not None:
        try:
            from eris.retrieval.field_interference import field_resonance
            return float(field_resonance(query_phi, query_theta, phi_s, theta_s))
        except Exception:
            pass
    return _cosine(qe, getattr(rec, "embedding", None))


def traditional_retriever(memory, *, top_k: int = 8, pool_limit: int = 400):
    """Floor: hybrid BM25+dense+RRF (+ optional reranker) over all_records()."""
    from eris.retrieval.hybrid import hybrid_search, http_reranker
    from eris.knowledge.embeddings import get_embedding
    from eris.config import CONFIG

    def _run(query, *, query_embedding=None, **kw) -> RetrievalResult:
        records = memory.all_records(limit=pool_limit) if hasattr(memory, "all_records") else []
        if not records:
            return RetrievalResult()
        qe = query_embedding if query_embedding is not None else get_embedding(query)
        reranker = http_reranker() if getattr(CONFIG, "dual_rerank", False) else None
        hits = hybrid_search(query, records, query_embedding=qe, top_k=top_k,
                             reranker=reranker)
        # hybrid_search returns a bare ranked list — synthesize rank-descending scores.
        scores = [1.0 / (i + 1) for i in range(len(hits))]
        return RetrievalResult(records=hits, scores=scores)

    return _run


def novel_retriever(memory, *, top_k: int = 8, tension_k: int = 3):
    """On-trial: resonant retrieval (aligned answer + tension channel + coupling)."""
    from eris.knowledge.embeddings import get_embedding
    from eris.computation.activations import BVec

    def _run(query, *, query_bvec=None, query_embedding=None,
             query_phi=None, query_theta=None, **kw) -> RetrievalResult:
        qe = query_embedding if query_embedding is not None else get_embedding(query)
        bvec = query_bvec if query_bvec is not None else BVec()
        aligned, tension = memory.retrieve_resonant(
            query_bvec=bvec, query_embedding=qe, top_k=top_k,
            tension_k=tension_k, query_text=query)
        coupling = [_coupling(qe, query_phi, query_theta, r) for r in aligned]
        return RetrievalResult(records=list(aligned), scores=coupling,
                               aligned=list(aligned), tension=list(tension),
                               coupling=coupling)

    return _run


def build_retrieval_dualpath(memory, *, mode: Mode = Mode.TRADITIONAL_ONLY,
                             arbiter=None, logger=None, top_k: int = 8) -> DualPath:
    """One DualPath(name='retrieval') wiring the resonant (novel) path against the
    hybrid SOTA-RAG (traditional) floor."""
    return DualPath(
        novel=novel_retriever(memory, top_k=top_k),
        traditional=traditional_retriever(memory, top_k=top_k),
        mode=mode, arbiter=arbiter, logger=logger, name="retrieval")
