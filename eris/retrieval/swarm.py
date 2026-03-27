"""
Retrieval Swarm — Parallel Specialized Retrievers
===================================================

From SuperRAG: 6 specialized retrievers running in MoE parallel.
Each retriever specializes in a different aspect of relevance:

    SemanticRetriever:  Embedding cosine similarity (raw meaning)
    DomainRetriever:    BFECDS alignment (BLECD domain match)
    TemporalRetriever:  Recency-weighted (Ebbinghaus freshness)
    TorsionRetriever:   High-torsion memories (contradictions, learning)
    ResonanceRetriever: Interference-positive memories (agreement)
    ArchetypeRetriever: Same-archetype memories (regime match)

Results are fused via Reciprocal Rank Fusion (RRF) — each retriever
contributes a ranked list, RRF combines them without requiring
score normalization across different similarity metrics.

"LLM = narrator. RAG = intelligence." — SuperRAG README

Usage:
    from eris.retrieval.swarm import RetrievalSwarm

    swarm = RetrievalSwarm(memory_system=mem)
    results = swarm.search(query_bvec=bvec, query_text="emergence", top_k=10)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np

from eris.computation.activations import BVec, bvec_cosine, bvec_distance
from eris.memory.tiers import MemorySystem, MemoryRecord


@dataclass
class SwarmResult:
    """A fused search result from the retrieval swarm."""
    record: MemoryRecord
    rrf_score: float           # Reciprocal Rank Fusion score
    contributing_retrievers: List[str]  # Which retrievers found this


class RetrievalSwarm:
    """Parallel specialized retrievers with RRF fusion.

    Each retriever runs independently, produces a ranked list,
    and RRF combines them into a single ranking.
    """

    def __init__(self, memory_system: MemorySystem, rrf_k: int = 60):
        self.memory = memory_system
        self.rrf_k = rrf_k  # RRF constant (standard: 60)

    def search(self, query_bvec: Optional[BVec] = None,
               query_text: str = "",
               query_embedding: Optional[np.ndarray] = None,
               top_k: int = 10) -> List[SwarmResult]:
        """Run all retrievers in parallel and fuse results via RRF."""

        # Collect all memories from MTM + LTM for searching
        candidates = (self.memory.mtm.get_fresh(min_freshness=0.01) +
                      self.memory.ltm._records)

        if not candidates:
            return []

        # Run each retriever
        ranked_lists: Dict[str, List[MemoryRecord]] = {}

        if query_bvec:
            ranked_lists["domain"] = self._domain_retrieve(candidates, query_bvec)
            ranked_lists["archetype"] = self._archetype_retrieve(candidates, query_bvec)
            ranked_lists["torsion"] = self._torsion_retrieve(candidates, query_bvec)
            ranked_lists["resonance"] = self._resonance_retrieve(candidates, query_bvec)

        ranked_lists["temporal"] = self._temporal_retrieve(candidates)

        if query_embedding is not None:
            ranked_lists["semantic"] = self._semantic_retrieve(candidates, query_embedding)

        # RRF fusion
        return self._fuse_rrf(ranked_lists, top_k)

    def _domain_retrieve(self, candidates: List[MemoryRecord],
                         query_bvec: BVec) -> List[MemoryRecord]:
        """Rank by BFECDS cosine alignment."""
        scored = [(r, bvec_cosine(query_bvec, r.bvec)) for r in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in scored]

    def _archetype_retrieve(self, candidates: List[MemoryRecord],
                            query_bvec: BVec) -> List[MemoryRecord]:
        """Rank by archetype match — same regime memories first."""
        query_arch = query_bvec.archetype()
        match = [r for r in candidates if r.bvec.archetype() == query_arch]
        other = [r for r in candidates if r.bvec.archetype() != query_arch]
        return match + other

    def _torsion_retrieve(self, candidates: List[MemoryRecord],
                          query_bvec: BVec) -> List[MemoryRecord]:
        """Rank by criticality — high-torsion memories first (learning moments)."""
        scored = [(r, r.bvec.C) for r in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in scored]

    def _resonance_retrieve(self, candidates: List[MemoryRecord],
                            query_bvec: BVec) -> List[MemoryRecord]:
        """Rank by positive interference with query (resonance)."""
        from eris.memory.interference import _csba_coupling_geometry
        scored = []
        for r in candidates:
            result = _csba_coupling_geometry(query_bvec, r.bvec)
            scored.append((r, result.elastic_energy))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in scored]

    def _temporal_retrieve(self, candidates: List[MemoryRecord]) -> List[MemoryRecord]:
        """Rank by Ebbinghaus freshness (most recent first)."""
        scored = [(r, r.freshness(half_life_hours=168.0)) for r in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in scored]

    def _semantic_retrieve(self, candidates: List[MemoryRecord],
                           query_emb: np.ndarray) -> List[MemoryRecord]:
        """Rank by embedding cosine similarity."""
        qnorm = np.linalg.norm(query_emb)
        if qnorm < 1e-10:
            return candidates

        scored = []
        for r in candidates:
            if r.embedding is not None:
                rnorm = np.linalg.norm(r.embedding)
                if rnorm > 1e-10:
                    cos = float(np.dot(query_emb, r.embedding) / (qnorm * rnorm))
                    scored.append((r, cos))
                    continue
            scored.append((r, 0.0))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in scored]

    def _fuse_rrf(self, ranked_lists: Dict[str, List[MemoryRecord]],
                  top_k: int) -> List[SwarmResult]:
        """Reciprocal Rank Fusion across all retrievers.

        RRF score = Σ 1/(k + rank_i) for each retriever i that found the doc.

        RRF is robust because it doesn't require normalizing scores across
        different similarity metrics — it only uses rank positions.
        """
        # Map doc text → (total RRF score, contributing retrievers, record)
        fusion: Dict[str, dict] = {}

        for retriever_name, ranked in ranked_lists.items():
            for rank, record in enumerate(ranked):
                key = record.text[:200]  # Dedup by text prefix
                if key not in fusion:
                    fusion[key] = {
                        "score": 0.0,
                        "retrievers": [],
                        "record": record,
                    }
                fusion[key]["score"] += 1.0 / (self.rrf_k + rank + 1)
                if retriever_name not in fusion[key]["retrievers"]:
                    fusion[key]["retrievers"].append(retriever_name)

        # Sort by fused score
        items = sorted(fusion.values(), key=lambda x: x["score"], reverse=True)

        return [
            SwarmResult(
                record=item["record"],
                rrf_score=item["score"],
                contributing_retrievers=item["retrievers"],
            )
            for item in items[:top_k]
        ]
