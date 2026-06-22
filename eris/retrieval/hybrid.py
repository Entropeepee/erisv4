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


def _dense_ranking(query_embedding, embeddings: List[Optional[np.ndarray]]) -> List[int]:
    q = np.asarray(query_embedding, dtype=np.float64).ravel()
    qn = np.linalg.norm(q)
    if qn < 1e-12:
        return []
    sims = []
    for i, e in enumerate(embeddings):
        if e is None:
            continue
        e = np.asarray(e, dtype=np.float64).ravel()
        en = np.linalg.norm(e)
        if en < 1e-12 or e.shape != q.shape:
            continue
        sims.append((i, float(np.dot(q, e) / (qn * en))))
    sims.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in sims]


# A reranker is any callable: (query, [candidate_texts]) -> [scores]
Reranker = Callable[[str, List[str]], List[float]]


def hybrid_search(
    query: str,
    records: Sequence[Any],
    *,
    query_embedding=None,
    top_k: int = 8,
    text_attr: str = "text",
    embedding_attr: str = "embedding",
    reranker: Optional[Reranker] = None,
    rerank_depth: int = 24,
) -> List[Any]:
    """Rank `records` for `query` by fusing BM25 (lexical) and dense (embedding)
    rankings with RRF, then optionally reranking the fused top-N.

    `records` are objects with a `.text` (and optional `.embedding`) attribute —
    e.g. `MemoryRecord`. Returns the top_k records, best first. Read-only; the
    records and any memory store are never modified.
    """
    records = list(records)
    if not records:
        return []
    texts = [getattr(r, text_attr, "") or "" for r in records]

    rankings: List[List[int]] = [list(np.argsort(-BM25(texts).scores(query)))]
    if query_embedding is not None:
        embs = [getattr(r, embedding_attr, None) for r in records]
        dense = _dense_ranking(query_embedding, embs)
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
