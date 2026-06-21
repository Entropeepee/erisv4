"""
Three-Tier Memory System
=========================

Memory is not data storage — it is resonance patterns within dynamically
stable coherence fields (DCR White Paper, Pope 2025).

Three tiers with different persistence and capacity:

    Short-Term (STM):  ~20 turns, session only, in-memory deque
    Medium-Term (MTM): ~200 records, weeks (Ebbinghaus decay), JSONL on disk
    Long-Term (LTM):   Unlimited, permanent with decay weighting, vector index

SGT gates all transitions between tiers: a memory promotes ONLY when its
novelty exceeds the noise floor of the target tier.

Every memory carries its computed BFECDS vector — not LLM-assigned.
Memories are retrieved by a combination of:
    - Recency (Ebbinghaus freshness)
    - Semantic similarity (embedding distance)
    - BFECDS alignment (cosine in domain space)

The multi-memory interference integral R_ij = ∫φᵢ·φⱼ·cos(θᵢ−θⱼ)dx
detects when stored memories resonate or conflict.

Usage:
    from eris.memory.tiers import MemorySystem, MemoryRecord

    mem = MemorySystem()
    mem.store_turn(text="hello", bvec=bvec, embedding=emb)
    results = mem.retrieve(query_embedding=emb, query_bvec=bvec, top_k=5)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from collections import deque
import time
import json
import os
import math
import numpy as np

from eris.computation.activations import BVec, bvec_cosine, bvec_distance
from eris.memory.interference import _csba_coupling_geometry
from eris.computation.sgt import SGTGate
from eris.config import CONFIG


# ─── Memory Record ────────────────────────────────────────────────────────

@dataclass
class MemoryRecord:
    """A single memory entry with computed BFECDS and metadata.

    Information = observer-system coupling geometry (not message property).
    A memory record preserves the coupling geometry as a coherence attractor.
    """
    text: str                          # The content
    bvec: BVec                         # Computed BFECDS activation vector
    embedding: Optional[np.ndarray] = None  # Semantic embedding (for retrieval)
    timestamp: float = field(default_factory=time.time)
    source: str = "conversation"       # conversation | dream | research | metacognitive
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Field state snapshot (optional — for memories worth re-evolving)
    phi_snapshot: Optional[np.ndarray] = None
    theta_snapshot: Optional[np.ndarray] = None

    # Consolidation tracking
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    tier: str = "stm"                  # stm | mtm | ltm

    def freshness(self, half_life_hours: float = 168.0) -> float:
        """Ebbinghaus forgetting curve: freshness decays exponentially.

        f(age) = exp(-ln(2) · age / half_life)

        Default half_life = 168 hours (1 week) for MTM.
        """
        age_hours = (time.time() - self.timestamp) / 3600.0
        return math.exp(-math.log(2) * age_hours / max(half_life_hours, 0.01))

    def reinforce(self) -> None:
        """Reinforce the memory by resetting its timestamp.
        This naturally counteracts the Ebbinghaus decay curve.
        """
        self.access_count += 1
        self.last_accessed = time.time()
        # Partial reset of timestamp (e.g. 50% closer to now)
        self.timestamp += (self.last_accessed - self.timestamp) * 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSONL storage."""
        d = {
            "text": self.text,
            "bvec": self.bvec.as_dict(),
            "timestamp": self.timestamp,
            "source": self.source,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "tier": self.tier,
        }
        if self.embedding is not None:
            d["embedding"] = self.embedding.tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryRecord":
        emb = np.array(d["embedding"], dtype=np.float32) if "embedding" in d else None
        return cls(
            text=d["text"],
            bvec=BVec.from_dict(d["bvec"]),
            embedding=emb,
            timestamp=d.get("timestamp", time.time()),
            source=d.get("source", "conversation"),
            metadata=d.get("metadata", {}),
            access_count=d.get("access_count", 0),
            last_accessed=d.get("last_accessed", time.time()),
            tier=d.get("tier", "stm"),
        )


# ─── Short-Term Memory ────────────────────────────────────────────────────

class ShortTermMemory:
    """Session-only buffer of recent turns.

    Capacity: ~20 turns. No persistence to disk.
    Oldest entries drop off the end automatically.
    """

    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self._buffer: deque[MemoryRecord] = deque(maxlen=capacity)

    def store(self, record: MemoryRecord) -> None:
        record.tier = "stm"
        self._buffer.append(record)

    def get_all(self) -> List[MemoryRecord]:
        return list(self._buffer)

    def get_recent(self, n: int = 5) -> List[MemoryRecord]:
        return list(self._buffer)[-n:]

    def novelty(self, bvec: BVec) -> float:
        """BFECDS novelty of `bvec` vs the rest of STM (Remediation Tier 2.4).

        Returns ``1 - max cosine similarity`` to existing records. This is a
        DIRECTION-aware distance: two different memories that happen to share the
        same total activation no longer collide (the old `sum(as_array())`
        scalar treated them as identical). 1.0 = wholly novel, 0.0 = duplicate.
        """
        others = list(self._buffer)
        if not others:
            return 1.0
        sims = [bvec_cosine(bvec, r.bvec) for r in others]
        return float(1.0 - max(sims)) if sims else 1.0

    @property
    def size(self) -> int:
        return len(self._buffer)


# ─── Medium-Term Memory ───────────────────────────────────────────────────

class MediumTermMemory:
    """Weeks-scale memory with Ebbinghaus decay. JSONL on disk.

    Capacity: ~200 records. Entries below freshness threshold are pruned.
    """

    def __init__(self, capacity: int = 200, half_life_hours: float = 168.0,
                 storage_path: str = "memory_mtm.jsonl"):
        self.capacity = capacity
        self.half_life_hours = half_life_hours
        self.storage_path = storage_path
        self._records: List[MemoryRecord] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r") as f:
                for line in f:
                    try:
                        self._records.append(MemoryRecord.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue

    def _save(self) -> None:
        with open(self.storage_path, "w") as f:
            for rec in self._records:
                f.write(json.dumps(rec.to_dict()) + "\n")

    def store(self, record: MemoryRecord) -> None:
        record.tier = "mtm"
        self._records.append(record)
        # Prune if over capacity (remove lowest freshness)
        if len(self._records) > self.capacity:
            self._records.sort(key=lambda r: r.freshness(self.half_life_hours))
            self._records = self._records[-self.capacity:]
        self._save()

    def get_fresh(self, min_freshness: float = 0.1) -> List[MemoryRecord]:
        """Return records above minimum freshness threshold."""
        return [r for r in self._records
                if r.freshness(self.half_life_hours) >= min_freshness]

    def prune(self) -> int:
        """Remove stale records. Returns count pruned."""
        before = len(self._records)
        self._records = [r for r in self._records
                         if r.freshness(self.half_life_hours) > 0.01]
        pruned = before - len(self._records)
        if pruned > 0:
            self._save()
        return pruned

    @property
    def size(self) -> int:
        return len(self._records)


# ─── Long-Term Memory ─────────────────────────────────────────────────────

class LongTermMemory:
    """Permanent memory with vector similarity retrieval.

    Uses brute-force cosine similarity on embeddings for now.
    Will be upgraded to FAISS/cuVS when Phase 5 integrates the
    embedding model (ONNX Runtime BGE-M3, per Gemini audit).

    Records are permanent but weighted by Ebbinghaus decay with
    a much longer half-life (90 days default).
    """

    def __init__(self, half_life_hours: float = 2160.0,
                 storage_path: str = "memory_ltm.jsonl"):
        self.half_life_hours = half_life_hours
        self.storage_path = storage_path
        self._records: List[MemoryRecord] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r") as f:
                for line in f:
                    try:
                        self._records.append(MemoryRecord.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue

    def _save(self) -> None:
        with open(self.storage_path, "w") as f:
            for rec in self._records:
                f.write(json.dumps(rec.to_dict()) + "\n")

    def store(self, record: MemoryRecord) -> None:
        record.tier = "ltm"
        self._records.append(record)
        self._save()

    def search_by_bvec(self, query_bvec: BVec, top_k: int = 5) -> List[MemoryRecord]:
        """Find memories most aligned in BFECDS space."""
        scored = [
            (rec, bvec_cosine(query_bvec, rec.bvec) * rec.freshness(self.half_life_hours))
            for rec in self._records
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [rec for rec, _ in scored[:top_k]]

    def search_by_embedding(self, query_emb: np.ndarray, top_k: int = 5) -> List[MemoryRecord]:
        """Find semantically similar memories via embedding cosine similarity."""
        results = []
        qnorm = np.linalg.norm(query_emb)
        if qnorm < 1e-10:
            return []
        for rec in self._records:
            if rec.embedding is None:
                continue
            rnorm = np.linalg.norm(rec.embedding)
            if rnorm < 1e-10:
                continue
            cos = float(np.dot(query_emb, rec.embedding) / (qnorm * rnorm))
            freshness = rec.freshness(self.half_life_hours)
            results.append((rec, cos * freshness))
        results.sort(key=lambda x: x[1], reverse=True)
        return [rec for rec, _ in results[:top_k]]

    @property
    def size(self) -> int:
        return len(self._records)


# ─── Unified Memory System ────────────────────────────────────────────────

class MemorySystem:
    """Three-tier memory with unified retrieval and SGT-gated consolidation.

    Usage:
        mem = MemorySystem()
        mem.store_turn(text, bvec, embedding)
        results = mem.retrieve(query_embedding, query_bvec, top_k=5)
        mem.consolidate()  # Promote worthy STM→MTM→LTM
    """

    def __init__(self, stm_capacity: int = 20, mtm_capacity: int = 200,
                 data_dir: str = "memory"):
        os.makedirs(data_dir, exist_ok=True)
        self.stm = ShortTermMemory(capacity=stm_capacity)
        self.mtm = MediumTermMemory(capacity=mtm_capacity,
                                     storage_path=os.path.join(data_dir, "mtm.jsonl"))
        self.ltm = LongTermMemory(storage_path=os.path.join(data_dir, "ltm.jsonl"))

        # SGT gates for consolidation decisions
        self._stm_to_mtm_gate = SGTGate(threshold_sigma=1.5, ema_alpha=0.1)
        self._mtm_to_ltm_gate = SGTGate(threshold_sigma=2.0, ema_alpha=0.05)

    def store_turn(self, text: str, bvec: BVec,
                   embedding: Optional[np.ndarray] = None,
                   source: str = "conversation",
                   phi_snapshot: Optional[np.ndarray] = None,
                   theta_snapshot: Optional[np.ndarray] = None,
                   metadata: Optional[Dict] = None) -> MemoryRecord:
        """Store a new turn in short-term memory."""
        record = MemoryRecord(
            text=text, bvec=bvec, embedding=embedding,
            source=source, metadata=metadata or {},
            phi_snapshot=phi_snapshot, theta_snapshot=theta_snapshot,
        )
        self.stm.store(record)
        return record

    def retrieve(self, query_bvec: Optional[BVec] = None,
                 query_embedding: Optional[np.ndarray] = None,
                 top_k: int = 5) -> List[MemoryRecord]:
        """Retrieve from all three tiers, weighted by freshness × similarity.

        Combines BFECDS alignment and semantic similarity for ranking.
        """
        candidates: List[Tuple[MemoryRecord, float]] = []

        # STM: always include recent turns (high recency weight)
        for rec in self.stm.get_all():
            score = 1.0  # STM always relevant
            if query_bvec:
                score *= (0.5 + 0.5 * bvec_cosine(query_bvec, rec.bvec))
            candidates.append((rec, score))

        # MTM: fresh records weighted by BFECDS alignment AND semantic
        # similarity (so studied/read material in medium-term is retrievable by
        # meaning, not just field signature, before it consolidates to LTM).
        for rec in self.mtm.get_fresh(min_freshness=0.05):
            freshness = rec.freshness(self.mtm.half_life_hours)
            score = freshness
            if query_bvec:
                score *= (0.3 + 0.7 * max(0, bvec_cosine(query_bvec, rec.bvec)))
            if (query_embedding is not None and rec.embedding is not None
                    and rec.embedding.shape == query_embedding.shape):
                sim = float(np.dot(query_embedding, rec.embedding))  # L2-normalized → cosine
                score *= (0.5 + 0.5 * max(0.0, sim))
            candidates.append((rec, score))

        # LTM: semantic + BFECDS search
        if query_bvec:
            for rec in self.ltm.search_by_bvec(query_bvec, top_k=top_k):
                freshness = rec.freshness(self.ltm.half_life_hours)
                score = freshness * bvec_cosine(query_bvec, rec.bvec)
                candidates.append((rec, score))

        if query_embedding is not None:
            for rec in self.ltm.search_by_embedding(query_embedding, top_k=top_k):
                freshness = rec.freshness(self.ltm.half_life_hours)
                score = freshness * 0.8  # Slight discount vs BFECDS
                candidates.append((rec, score))

        # Deduplicate (same text = same memory)
        seen_texts = set()
        unique: List[Tuple[MemoryRecord, float]] = []
        for rec, score in candidates:
            if rec.text not in seen_texts:
                seen_texts.add(rec.text)
                unique.append((rec, score))

        # Sort by score descending
        unique.sort(key=lambda x: x[1], reverse=True)
        
        # Reinforce retrieved memories to combat Ebbinghaus decay
        results = [rec for rec, _ in unique[:top_k]]
        for rec in results:
            rec.reinforce()
            
        return results

    def retrieve_resonant(self, query_bvec: BVec,
                          query_embedding: Optional[np.ndarray] = None,
                          top_k: int = 5, tension_k: int = 2):
        """Resonant retrieval — cosine AND sine (Remediation Tier 6).

        Eris's conservation law is cos^2(theta) + sin^2(theta) = 1:
          * cos^2 (ELASTIC coupling) = what is already aligned/resolved — the
            memories that directly *answer* the query. This is ordinary
            similarity retrieval.
          * sin^2 (PLASTIC coupling) = the orthogonal, unresolved component —
            memories that are strongly *coupled* to the query but in tension
            with it. This is the Emergence channel: where new structure forms
            under orthogonal pressure. Plain cosine RAG throws this away and
            returns only redundant near-duplicates.

        Returns ``(aligned, tension)``:
          * ``aligned`` — the usual relevance set (cosine / elastic + embedding).
          * ``tension`` — coupled-but-unresolved memories ranked by PLASTIC
            energy (sin^2 * coupling, so unrelated memories score ~0 and never
            surface). Feeding these to the LLM as "related-but-unresolved" is
            how Eris learns more from the input and makes non-obvious
            connections instead of parroting the nearest neighbor.
        """
        aligned = self.retrieve(query_bvec=query_bvec,
                                query_embedding=query_embedding, top_k=top_k)
        seen_text = {r.text for r in aligned}

        pool = (self.stm.get_all()
                + self.mtm.get_fresh(min_freshness=0.05)
                + self.ltm._records)
        scored = []
        for r in pool:
            if r.text in seen_text:
                continue
            ir = _csba_coupling_geometry(query_bvec, r.bvec)
            # learning value = sin^2 coupling. Because coupling weights it, a
            # memory must actually be related (shared active domains) to score —
            # this is "productive dissonance", not noise.
            if ir.plastic_energy > 1e-6:
                scored.append((r, ir.plastic_energy))
        scored.sort(key=lambda x: x[1], reverse=True)
        tension = []
        for r, _ in scored[:tension_k]:
            if r.text not in seen_text:
                tension.append(r); seen_text.add(r.text)
        return aligned, tension

    def consolidate(self) -> Dict[str, int]:
        """SGT-gated consolidation: promote worthy memories up tiers.

        STM → MTM: if the memory's BFECDS novelty exceeds the STM noise floor
        MTM → LTM: if the memory's access count or BFECDS distance from existing
                    LTM attractors exceeds the MTM noise floor

        Returns counts of promotions.
        """
        promoted_to_mtm = 0
        promoted_to_ltm = 0

        # STM → MTM
        mtm_attractors = self.mtm.get_fresh(min_freshness=0.01)
        for rec in self.stm.get_all():
            # Novelty = BFECDS distance from the nearest existing MTM attractor
            # (Tier 2.4 — direction-aware, not a scalar activation sum). A turn
            # that points somewhere genuinely new promotes; a near-duplicate of
            # something already stored does not.
            if mtm_attractors:
                nearest_sim = max(
                    (bvec_cosine(rec.bvec, m.bvec) for m in mtm_attractors),
                    default=0.0,
                )
                novelty = 1.0 - nearest_sim
            else:
                novelty = 1.0  # MTM empty — everything is novel
            should_promote, _ = self._stm_to_mtm_gate.update(novelty)
            if should_promote:
                self.mtm.store(rec)
                promoted_to_mtm += 1

        # MTM → LTM
        for rec in self.mtm.get_fresh(min_freshness=0.3):
            # Distance from nearest LTM attractor
            if self.ltm.size > 0:
                nearest = self.ltm.search_by_bvec(rec.bvec, top_k=1)
                if nearest:
                    dist = bvec_distance(rec.bvec, nearest[0].bvec)
                else:
                    dist = 1.0  # No LTM records yet — everything is novel
            else:
                dist = 1.0

            should_promote, _ = self._mtm_to_ltm_gate.update(dist)
            if should_promote:
                self.ltm.store(rec)
                promoted_to_ltm += 1

        # Prune dead MTM memories (Tier 2.2). `prune()` existed but was never
        # invoked, so freshness<0.01 records accumulated unbounded. Run it as
        # part of every consolidation pass.
        pruned = self.mtm.prune()

        return {
            "stm_to_mtm": promoted_to_mtm,
            "mtm_to_ltm": promoted_to_ltm,
            "mtm_pruned": pruned,
        }
