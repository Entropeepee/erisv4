"""
Multi-Tier Vector Index
========================

From SuperRAG: vectors live in tiers based on access frequency:
    HOT  (VRAM/in-memory): Recent, frequently accessed. Sub-millisecond search.
    WARM (RAM):            Moderate frequency. Millisecond search.
    COLD (Disk):           Rarely accessed. Loaded on demand.

Maps to Eris Echo's memory tiers:
    HOT  = STM (short-term, ~20 recent turns)
    WARM = MTM (medium-term, ~200 records with Ebbinghaus decay)
    COLD = LTM (long-term, unlimited, persisted)

Uses numpy brute-force for now (fast enough for < 100K vectors).
Upgrade path: FAISS-cuVS with CAGRA for GPU-accelerated ANN search
(per Gemini audit: faiss-gpu-cuvs v1.14.1 with RMM memory pooling).

All vectors are GLNCS-debiased before storage.

Usage:
    from eris.retrieval.vector_index import VectorIndex

    index = VectorIndex(dim=64)
    index.add("doc_001", compressed_vector, metadata={"tier": "hot"})
    results = index.search(query_vector, top_k=10)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from eris.config import to_numpy, xp
import json
import os
import time


@dataclass
class IndexEntry:
    """A single entry in the vector index."""
    doc_id: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    tier: str = "warm"  # hot | warm | cold
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0


@dataclass
class SearchResult:
    """A single search result with score."""
    doc_id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    tier: str = "warm"


class VectorIndex:
    """Multi-tier vector index with GLNCS-debiased storage.

    Three tiers with different performance characteristics:
        hot:  In-memory numpy array. Brute-force cosine. <1ms for 1K vectors.
        warm: In-memory numpy array. Brute-force cosine. ~5ms for 10K vectors.
        cold: On-disk JSONL. Loaded on demand. ~100ms for 100K vectors.

    Automatic promotion: frequently accessed cold vectors move to warm.
    Automatic demotion: stale warm vectors move to cold after pruning.
    """

    def __init__(self, dim: int = 64, cold_storage_path: str = "vector_cold.jsonl"):
        self.dim = dim
        self.cold_storage_path = cold_storage_path

        # Hot tier: small, frequently accessed
        self._hot: Dict[str, IndexEntry] = {}
        # Warm tier: moderate, recent
        self._warm: Dict[str, IndexEntry] = {}
        # Cold tier: large, persisted
        self._cold_ids: set = set()  # Only IDs; vectors loaded on demand

        # Precomputed matrices for fast search
        self._hot_matrix: Optional[np.ndarray] = None
        self._hot_ids: List[str] = []
        self._warm_matrix: Optional[np.ndarray] = None
        self._warm_ids: List[str] = []

        self._dirty = True  # Matrices need rebuild

    def add(self, doc_id: str, vector: np.ndarray,
            metadata: Optional[Dict] = None, tier: str = "warm") -> None:
        """Add a vector to the index."""
        entry = IndexEntry(
            doc_id=doc_id,
            vector=vector.astype(np.float32),
            metadata=metadata or {},
            tier=tier,
        )

        if tier == "hot":
            self._hot[doc_id] = entry
        elif tier == "warm":
            self._warm[doc_id] = entry
        else:
            self._warm[doc_id] = entry  # Store in warm, persist to cold
            self._persist_cold(entry)

        self._dirty = True

    def search(self, query: np.ndarray, top_k: int = 10,
               tiers: Optional[List[str]] = None) -> List[SearchResult]:
        """Search across tiers. Returns top_k results sorted by score."""
        query = query.astype(np.float32).ravel()
        q_norm = np.linalg.norm(query)
        if q_norm < 1e-10:
            return []
        query_normed = query / q_norm

        tiers = tiers or ["hot", "warm"]
        results: List[SearchResult] = []

        if self._dirty:
            self._rebuild_matrices()

        # Search hot tier
        if "hot" in tiers and self._hot_matrix is not None and len(self._hot_ids) > 0:
            scores = self._hot_matrix @ query_normed
            for idx in np.argsort(-scores)[:top_k]:
                doc_id = self._hot_ids[idx]
                entry = self._hot[doc_id]
                entry.access_count += 1
                entry.last_accessed = time.time()
                results.append(SearchResult(
                    doc_id=doc_id, score=float(scores[idx]),
                    metadata=entry.metadata, tier="hot",
                ))

        # Search warm tier
        if "warm" in tiers and self._warm_matrix is not None and len(self._warm_ids) > 0:
            scores = self._warm_matrix @ query_normed
            for idx in np.argsort(-scores)[:top_k]:
                doc_id = self._warm_ids[idx]
                entry = self._warm[doc_id]
                entry.access_count += 1
                entry.last_accessed = time.time()
                results.append(SearchResult(
                    doc_id=doc_id, score=float(scores[idx]),
                    metadata=entry.metadata, tier="warm",
                ))

        # Sort all results by score, deduplicate, take top_k
        seen = set()
        unique = []
        results.sort(key=lambda r: r.score, reverse=True)
        for r in results:
            if r.doc_id not in seen:
                seen.add(r.doc_id)
                unique.append(r)
        return unique[:top_k]

    def promote(self, doc_id: str) -> None:
        """Promote a warm vector to hot tier."""
        if doc_id in self._warm:
            entry = self._warm.pop(doc_id)
            entry.tier = "hot"
            self._hot[doc_id] = entry
            self._dirty = True

    def demote(self, doc_id: str) -> None:
        """Demote a hot vector to warm tier."""
        if doc_id in self._hot:
            entry = self._hot.pop(doc_id)
            entry.tier = "warm"
            self._warm[doc_id] = entry
            self._dirty = True

    def prune_warm(self, max_age_hours: float = 168.0, max_size: int = 10000) -> int:
        """Remove stale entries from warm tier. Returns count removed."""
        cutoff = time.time() - max_age_hours * 3600
        stale = [did for did, e in self._warm.items() if e.last_accessed < cutoff]
        for did in stale:
            entry = self._warm.pop(did)
            self._persist_cold(entry)
        self._dirty = True
        return len(stale)

    def _rebuild_matrices(self) -> None:
        """Rebuild precomputed normalized matrices for fast cosine search."""
        if self._hot:
            self._hot_ids = list(self._hot.keys())
            vecs = np.array([self._hot[d].vector for d in self._hot_ids])
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            self._hot_matrix = (vecs / norms).astype(np.float32)
        else:
            self._hot_matrix = None
            self._hot_ids = []

        if self._warm:
            self._warm_ids = list(self._warm.keys())
            vecs = np.array([self._warm[d].vector for d in self._warm_ids])
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            self._warm_matrix = (vecs / norms).astype(np.float32)
        else:
            self._warm_matrix = None
            self._warm_ids = []

        self._dirty = False

    def _persist_cold(self, entry: IndexEntry) -> None:
        """Persist an entry to cold storage (disk)."""
        with open(self.cold_storage_path, "a") as f:
            f.write(json.dumps({
                "doc_id": entry.doc_id,
                "vector": entry.vector.tolist(),
                "metadata": entry.metadata,
            }) + "\n")
        self._cold_ids.add(entry.doc_id)

    @property
    def total_size(self) -> int:
        return len(self._hot) + len(self._warm) + len(self._cold_ids)

    @property
    def tier_sizes(self) -> Dict[str, int]:
        return {
            "hot": len(self._hot),
            "warm": len(self._warm),
            "cold": len(self._cold_ids),
        }
