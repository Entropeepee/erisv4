"""Hybrid retrieval — dense + lexical (BM25) fused with RRF, optional rerank.

Roadmap 1.3. The research plan's "robust default": dense embeddings catch
meaning, BM25 catches the exact tokens dense retrieval misses (IDs, code symbols,
names, file titles like "sgtpatent"), and Reciprocal Rank Fusion (RRF) combines
them without tuning weights. An optional cross-encoder reranker reorders the
fused top-N.

This module is **standalone and additive**: it operates on a caller-supplied list
of records (read-only) and does NOT touch `MemorySystem.retrieve_resonant` — Eris's
resonant-memory differentiator is left exactly as-is. Wire it in deliberately
later (see WORKLOG Q1).

No new dependencies: BM25 is a small stdlib implementation here. The reranker is
an interface (a callable `(query, [texts]) -> [scores]`); a real cross-encoder is
a machine-side model download and plugs in via that callable.
"""
from __future__ import annotations
from typing import Callable, List, Optional, Sequence, Any
import math
import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> List[str]:
    return _TOKEN.findall((text or "").lower())


class BM25:
    """Okapi BM25 over a small in-memory corpus (stdlib only)."""

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = [_tok(t) for t in corpus]
        self.N = len(self.docs)
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        # document frequency per term
        df: dict[str, int] = {}
        for d in self.docs:
            for term in set(d):
                df[term] = df.get(term, 0) + 1
        # smoothed idf (always positive, avoids the classic BM25 negative-idf)
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                    for t, n in df.items()}
        self._tf = [{t: d.count(t) for t in set(d)} for d in self.docs]

    def scores(self, query: str) -> np.ndarray:
        q = _tok(query)
        out = np.zeros(self.N, dtype=np.float64)
        if not self.N:
            return out
        for i in range(self.N):
            tf, dl = self._tf[i], self.doc_len[i]
            s = 0.0
            for term in q:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += idf * (f * (self.k1 + 1)) / (denom or 1)
            out[i] = s
        return out


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> List[int]:
    """Fuse several ranked lists of item indices into one. RRF score of an item =
    sum over lists of 1/(k + rank). Returns item indices, best first."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return [i for i, _ in sorted(score.items(), key=lambda x: x[1], reverse=True)]


def _stack_normalized(embeddings: List[Optional[np.ndarray]]):
    """Stack valid embeddings into one L2-normalized (m, d) matrix + a row→record-index map,
    so dense similarity is a single matmul instead of a per-record Python loop (memory-scale
    win at n~1000s). Records with None/zero-norm/odd-dim embeddings are dropped from the matrix."""
    rows, idx, dim = [], [], None
    for i, e in enumerate(embeddings):
        if e is None:
            continue
        v = np.asarray(e, dtype=np.float64).ravel()
        if dim is None:
            dim = v.shape[0]
        if v.shape[0] != dim:
            continue
        n = np.linalg.norm(v)
        if n < 1e-12:
            continue
        rows.append(v / n)
        idx.append(i)
    if not rows:
        return None, []
    return np.asarray(rows), idx                       # (m, d) row-normalized, idx[row]=record


def _dense_ranking_matrix(query_embedding, mat, idx: List[int]) -> List[int]:
    """Vectorized dense ranking from a prebuilt normalized matrix: one (m,d)·(d,) matmul."""
    if mat is None:
        return []
    q = np.asarray(query_embedding, dtype=np.float64).ravel()
    qn = np.linalg.norm(q)
    if qn < 1e-12 or q.shape[0] != mat.shape[1]:
        return []
    sims = mat @ (q / qn)                               # cosine, all candidates at once
    order = np.argsort(-sims)
    return [idx[r] for r in order]


def _dense_ranking(query_embedding, embeddings: List[Optional[np.ndarray]]) -> List[int]:
    mat, idx = _stack_normalized(embeddings)
    return _dense_ranking_matrix(query_embedding, mat, idx)


class HybridIndex:
    """Prebuilt corpus state for hybrid_search — the BM25 index and the normalized embedding
    matrix, computed ONCE and reused across queries (Stage-3 amortization). _rag rebuilt these
    from scratch every cycle (retokenizing the whole library 2-3× per hive run); building once
    and reusing is the dominant retrieval-side win."""
    __slots__ = ("records", "texts", "bm25", "_mat", "_idx")

    def __init__(self, records, *, text_attr: str = "text", embedding_attr: str = "embedding"):
        self.records = list(records)
        self.texts = [getattr(r, text_attr, "") or "" for r in self.records]
        self.bm25 = BM25(self.texts)
        self._mat, self._idx = _stack_normalized(
            [getattr(r, embedding_attr, None) for r in self.records])

    def signature(self):
        """Cheap identity for cache validity within a run (the pool is immutable mid-run)."""
        return (len(self.records), id(self.records[0]) if self.records else 0)


def build_hybrid_index(records, *, text_attr: str = "text",
                       embedding_attr: str = "embedding") -> HybridIndex:
    return HybridIndex(records, text_attr=text_attr, embedding_attr=embedding_attr)


# A reranker is any callable: (query, [candidate_texts]) -> [scores]
Reranker = Callable[[str, List[str]], List[float]]


def http_reranker(base_url: Optional[str] = None, model: Optional[str] = None,
                  timeout: Optional[float] = None) -> Optional[Reranker]:
    """Phase 2: a Reranker backed by a local OpenAI-style /rerank service
    (NPU/iGPU). Returns None when no rerank endpoint is configured, so callers
    transparently fall back to RRF-only. On any per-call error it returns neutral
    scores (keeps the fused order) — never raises into retrieval."""
    from eris.config import CONFIG
    base = (base_url or CONFIG.rerank_base_url or "").rstrip("/")
    if not base:
        return None
    # Egress guard (r3 #10): candidate documents sent to rerank are the owner's content. A REMOTE
    # rerank URL would ship them off-box — refuse unless consented, so callers fall back to RRF-only.
    from eris.interface.accelerators import check_egress_or_warn
    if not check_egress_or_warn("rerank", base):
        return None
    mdl = model or CONFIG.rerank_model or "reranker"
    to = timeout if timeout is not None else CONFIG.accel_timeout_s

    def _rr(query: str, texts: List[str]) -> List[float]:
        from eris.knowledge import embeddings as _emb   # reuse the HTTP seam
        try:
            data = _emb._post_json(
                f"{base}/rerank",
                {"model": mdl, "query": query, "documents": list(texts)}, to)
            scores = [0.0] * len(texts)
            for r in data.get("results", []):
                i = r.get("index")
                s = r.get("relevance_score", r.get("score", 0.0))
                if isinstance(i, int) and 0 <= i < len(texts):
                    scores[i] = float(s)
            return scores
        except Exception:
            return [0.0] * len(texts)   # neutral -> RRF order preserved

    return _rr


def hybrid_search(
    query: str,
    records: Sequence[Any] = None,
    *,
    index: Optional[HybridIndex] = None,
    query_embedding=None,
    top_k: int = 8,
    text_attr: str = "text",
    embedding_attr: str = "embedding",
    reranker: Optional[Reranker] = None,
    rerank_depth: int = 24,
) -> List[Any]:
    """Rank records for `query` by fusing BM25 (lexical) and dense (embedding)
    rankings with RRF, then optionally reranking the fused top-N.

    Pass either `records` (built inline) or a prebuilt `index` (HybridIndex, reused across
    queries — Stage-3 amortization). Records are objects with a `.text` (and optional
    `.embedding`) attribute. Returns the top_k records, best first. Read-only.
    """
    if index is None:
        if not records:
            return []
        index = HybridIndex(records, text_attr=text_attr, embedding_attr=embedding_attr)
    records = index.records
    if not records:
        return []
    texts = index.texts

    rankings: List[List[int]] = [list(np.argsort(-index.bm25.scores(query)))]
    if query_embedding is not None:
        dense = _dense_ranking_matrix(query_embedding, index._mat, index._idx)
        if dense:
            rankings.append(dense)

    fused = reciprocal_rank_fusion(rankings)
    if not fused:                       # all-zero corpus edge case
        fused = list(range(len(records)))

    if reranker is not None:
        head = fused[:max(rerank_depth, top_k)]
        scores = reranker(query, [texts[i] for i in head])
        head = [i for i, _ in sorted(zip(head, scores),
                                     key=lambda x: x[1], reverse=True)]
        fused = head + [i for i in fused if i not in head]

    return [records[i] for i in fused[:top_k]]
