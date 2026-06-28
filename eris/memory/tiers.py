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

from eris.computation.activations import BVec, bvec_cosine, bvec_distance, cosine
from eris.memory.interference import _csba_coupling_geometry
from eris.computation.sgt import SGTGate
from eris.config import CONFIG


import re as _re_mod


def _dedupe_near_records(records: List[Any], threshold: float = 0.92,
                         max_keep: Optional[int] = None) -> List[Any]:
    """Collapse near-DUPLICATE records (the SAME file ingested twice under different temp names),
    keeping the first (highest-ranked) of each cluster. Two guards keep it from eating genuinely
    distinct sections that merely share boilerplate (e.g. a patent's Claim 1 vs the dependent
    Claim 2, or chunks sharing the 'Title › Section' contextual header):
      • content-word Jaccard ≥ `threshold` (0.92 — a true re-ingest is ~identical), AND
      • a LENGTH-RATIO gate (near-identical length) — a claim and its dependent claim differ.
    Stops once `max_keep` survivors are found, so it never runs the full O(n²) over a large pool
    (the caller slices to max_chunks anyway)."""
    def _cw(t: str) -> set:
        return {w for w in _re_mod.findall(r"[a-z0-9]{4,}", (t or "").lower())}
    kept, kept_sets = [], []
    for r in records:
        if max_keep is not None and len(kept) >= max_keep:
            break
        s = _cw(getattr(r, "text", ""))
        if not s:
            kept.append(r); kept_sets.append(s); continue
        dup = False
        for ks in kept_sets:
            if not ks:
                continue
            lo, hi = sorted((len(s), len(ks)))
            if lo / hi < 0.8:                      # length differs too much to be a re-ingest
                continue
            u = s | ks
            if u and len(s & ks) / len(u) >= threshold:
                dup = True
                break
        if not dup:
            kept.append(r); kept_sets.append(s)
    return kept


def _provenance_family(source: str) -> str:
    """Consolidation group key: the source NAMESPACE (the token before the first colon). Re-ingests
    of the same document (reading:tmpA.docx, reading:tmpB.docx → 'reading') and near-duplicate web
    notes (exploration:url1, exploration:url2 → 'exploration') share a family, but distinct
    provenance classes never do — a 'reflection' can never join a 'reading'. Splitting on the FIRST
    colon only keeps it URL-safe (exploration:https://… → 'exploration', not 'exploration:https').
    Genuinely distinct same-namespace content is still protected by the Jaccard + length gate."""
    return str(source or "").split(":", 1)[0]


# Provenance families that are her own first-person voice / audit trail — NEVER merged, even if
# two reflections rhyme. Consolidation reinforces the library; it must not flatten her thinking.
_CONSOLIDATE_SKIP_FAMILIES = {"reflection", "introspection", "dream", "ponder", "expert"}


def _fold_duplicate(rep: Any, other: Any) -> None:
    """Fold a near-duplicate into its representative so the surviving trace gets STRONGER, not
    forgotten: summed reinforcement (it was genuinely seen more than once), newest timestamp (stays
    fresh, won't be pruned), and the longer/more-complete text+embedding of the pair."""
    try:
        rep.access_count = (int(getattr(rep, "access_count", 0))
                            + int(getattr(other, "access_count", 0)) + 1)
        rep.last_accessed = max(getattr(rep, "last_accessed", 0.0),
                                getattr(other, "last_accessed", 0.0))
        rep.timestamp = max(getattr(rep, "timestamp", 0.0), getattr(other, "timestamp", 0.0))
        if len(getattr(other, "text", "") or "") > len(getattr(rep, "text", "") or ""):
            rep.text = other.text
            if getattr(other, "embedding", None) is not None:
                rep.embedding = other.embedding
    except Exception:
        pass


def consolidate_records(records: List[Any], threshold: float = 0.92):
    """Offline semantic consolidation (memory REPLAY): within each provenance family, collapse
    near-duplicate traces into ONE reinforced record. NEVER merges across families and NEVER
    touches the subjective/audit families (her reflections, dreams, ponders). Same Jaccard +
    length-ratio test as _dedupe_near_records, but it FOLDS duplicates (summing reinforcement)
    instead of just dropping them. Pure (no I/O) → directly unit-testable. Returns
    (kept_records, n_merged)."""
    def _cw(t: str) -> set:
        return {w for w in _re_mod.findall(r"[a-z0-9]{4,}", (t or "").lower())}
    families: Dict[str, List[Any]] = {}
    order: List[str] = []
    for r in records:
        fam = _provenance_family(getattr(r, "source", ""))
        if fam not in families:
            families[fam] = []
            order.append(fam)
        families[fam].append(r)
    kept_all: List[Any] = []
    n_merged = 0
    for fam in order:
        recs = families[fam]
        if fam in _CONSOLIDATE_SKIP_FAMILIES or len(recs) < 2:
            kept_all.extend(recs)
            continue
        reps: List[Any] = []
        rep_sets: List[set] = []
        for r in recs:
            s = _cw(getattr(r, "text", ""))
            hit = None
            if s:
                for i, ks in enumerate(rep_sets):
                    if not ks:
                        continue
                    lo, hi = sorted((len(s), len(ks)))
                    if lo / hi < 0.8:                  # length differs too much to be a re-ingest
                        continue
                    u = s | ks
                    if u and len(s & ks) / len(u) >= threshold:
                        hit = i
                        break
            if hit is None:
                reps.append(r); rep_sets.append(s)
            else:
                _fold_duplicate(reps[hit], r); n_merged += 1
        kept_all.extend(reps)
    return kept_all, n_merged


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
        # FAISS acceleration (optional): built lazily; falls back to brute force.
        self._faiss = None
        self._faiss_recs: List[MemoryRecord] = []
        self._faiss_n = -1
        self._load()

    def _ensure_faiss(self):
        """Build/refresh a FAISS inner-product index over current embeddings.
        Returns (index, parallel_records) or (None, None) if FAISS is
        unavailable or there are no usable embeddings. Inner product over
        L2-normalized vectors == cosine similarity."""
        try:
            import faiss
        except Exception:
            return None, None
        if self._faiss is not None and self._faiss_n == len(self._records):
            return self._faiss, self._faiss_recs  # unchanged since last build
        recs, vecs, dim = [], [], None
        for rec in self._records:
            e = rec.embedding
            if e is None:
                continue
            e = np.asarray(e, dtype=np.float32).ravel()
            n = float(np.linalg.norm(e))
            if n < 1e-10:
                continue
            if dim is None:
                dim = e.shape[0]
            if e.shape[0] != dim:
                continue  # skip mismatched dims (e.g. embedding model changed)
            recs.append(rec)
            vecs.append(e / n)
        if not vecs:
            self._faiss, self._faiss_recs, self._faiss_n = None, [], len(self._records)
            return None, None
        index = faiss.IndexFlatIP(dim)
        index.add(np.vstack(vecs).astype(np.float32))
        self._faiss, self._faiss_recs, self._faiss_n = index, recs, len(self._records)
        return index, recs

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
        """Find semantically similar memories via embedding cosine similarity.

        Uses FAISS when available (scales to the full ChatGPT-history import);
        otherwise brute-force cosine. Both reweight by Ebbinghaus freshness."""
        qnorm = np.linalg.norm(query_emb)
        if qnorm < 1e-10:
            return []
        index, recs = self._ensure_faiss()
        if index is not None:
            q = (np.asarray(query_emb, dtype=np.float32).ravel() / qnorm)
            if q.shape[0] == index.d:
                k = min(len(recs), max(top_k * 4, top_k))
                sims, idxs = index.search(q.reshape(1, -1), k)
                scored = []
                for sim, i in zip(sims[0], idxs[0]):
                    if i < 0:
                        continue
                    rec = recs[i]
                    scored.append((rec, float(sim) * rec.freshness(self.half_life_hours)))
                scored.sort(key=lambda x: x[1], reverse=True)
                return [rec for rec, _ in scored[:top_k]]
        # brute-force fallback
        results = []
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

    def all_records(self, limit: Optional[int] = None) -> List[MemoryRecord]:
        """Read-only union of records across all three tiers (newest tier last).
        Used by the agent's factual-lookup tool to build a candidate pool for
        hybrid retrieval without disturbing the resonant path."""
        recs = (self.stm.get_all()
                + list(self.mtm._records) + list(self.ltm._records))
        return recs[-limit:] if limit else recs

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

    def store_text(self, text: str, embedding: Optional[np.ndarray] = None,
                   source: str = "knowledge", bvec: Optional[BVec] = None,
                   **meta) -> MemoryRecord:
        """Store a standalone text fact directly into LONG-term memory (permanent
        collective knowledge). Used by federation to write a node's novel insight
        into the shared pool."""
        if embedding is not None and not isinstance(embedding, np.ndarray):
            embedding = np.asarray(embedding, dtype=np.float32)
        record = MemoryRecord(text=text, bvec=bvec or BVec(), embedding=embedding,
                              source=source, metadata=meta or {})
        self.ltm.store(record)
        return record

    def write_back_synthesis(self, topic: str, text: str,
                             embedding: Optional[np.ndarray] = None,
                             bvec: Optional[BVec] = None,
                             n_sources: int = 0) -> MemoryRecord:
        """Consolidation write-back (v2): store a canonized, citation-grounded hive synthesis as a
        FIRST-CLASS memory that outranks the raw chunks it summarized. The 'knows what it learned'
        step — later retrieval returns her resolved conclusion instead of re-deriving it from
        scattered sources. Tagged consolidated=True (a small retrieval salience lift) under the
        'synthesis' provenance family, which is foldable: a re-run on the same topic REINFORCES
        rather than duplicates."""
        if embedding is not None and not isinstance(embedding, np.ndarray):
            embedding = np.asarray(embedding, dtype=np.float32)
        slug = _re_mod.sub(r"[^a-z0-9]+", "-", (topic or "").lower()).strip("-")[:48] or "topic"
        rec = MemoryRecord(
            text=text, bvec=bvec or BVec(), embedding=embedding,
            source=f"synthesis:{slug}",
            metadata={"title": topic, "kind": "synthesis", "grounds": int(n_sources),
                      "consolidated": True},
            access_count=3)        # reads as already-reinforced — it distills many sources into one
        self.ltm.store(rec)
        return rec

    def retrieve(self, query_bvec: Optional[BVec] = None,
                 query_embedding: Optional[np.ndarray] = None,
                 top_k: int = 5,
                 query_text: Optional[str] = None) -> List[MemoryRecord]:
        """Retrieve from all three tiers, weighted by freshness × similarity.

        Combines BFECDS alignment and semantic similarity for ranking.
        If query_text is given, records whose title/source share a word with the
        query are boosted — so naming a document ("my patent") surfaces it.
        """
        candidates: List[Tuple[MemoryRecord, float]] = []

        qtokens = set()
        if query_text:
            import re as _re
            qtokens = {w for w in _re.findall(r"[a-z0-9]{4,}", query_text.lower())}

        def _title_boost(rec: MemoryRecord) -> float:
            if not qtokens:
                return 1.0
            hay = (str((rec.metadata or {}).get("title", "")) + " "
                   + str(rec.source)).lower()
            return 2.0 if any(t in hay for t in qtokens) else 1.0

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
            if query_embedding is not None and rec.embedding is not None:
                sim = cosine(query_embedding, rec.embedding)
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

        # Boost: title/source token match (when a query names a doc) AND consolidated syntheses —
        # her canonized, citation-grounded conclusions outrank the raw chunks they summarized, so
        # retrieval returns what she LEARNED, not a re-derivation from scattered sources.
        def _salience(rec: MemoryRecord) -> float:
            b = _title_boost(rec)
            if (rec.metadata or {}).get("consolidated"):
                b *= 1.5
            return b
        candidates = [(rec, score * _salience(rec)) for rec, score in candidates]

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
                          top_k: int = 5, tension_k: int = 2,
                          query_text: Optional[str] = None):
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
                                query_embedding=query_embedding, top_k=top_k,
                                query_text=query_text)
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

    def documents_matching(self, query_text: str, max_chunks: int = 8,
                           query_embedding: Optional[np.ndarray] = None) -> List[MemoryRecord]:
        """Directly fetch chunks of DOCUMENTS the user named in `query_text`
        (by filename/title), ranked by embedding similarity when available.

        Bypasses the general top-k so a named document is never crowded out by
        conversation that merely *mentions* it (e.g. 'tell me about my patent'
        otherwise ranks the chat turns containing 'patent' above the patent)."""
        import re as _re
        # 3+ chars (was 4+) so SHORT ACRONYM doc names match — "SGT"/"FFT"/"PDE"/"API" are
        # exactly the kind of names a user types, and the old 4-char floor silently dropped them
        # (the token set went empty → returned []). Guard a small set of common 3-letter words so
        # lowering the floor doesn't make name-matching over-fire on "the"/"and"/etc.
        _DOC_STOP = {"the", "and", "for", "was", "are", "its", "not", "but", "you", "all", "any",
                     "has", "had", "her", "his", "our", "out", "who", "why", "how", "can", "may",
                     "one", "two", "use", "new", "via", "per", "off", "yet", "let", "see", "now"}
        toks = {w for w in _re.findall(r"[a-z0-9]{3,}", (query_text or "").lower())
                if not (len(w) == 3 and w in _DOC_STOP)}
        if not toks:
            return []
        pool = (list(self.stm.get_all()) + list(self.mtm._records)
                + list(self.ltm._records))

        def _is_doc(rec) -> bool:
            return bool((rec.metadata or {}).get("title")) or str(rec.source).lower().startswith(
                ("reading:", "exploration:", "research:", "ponder:", "study:", "deepread"))

        # 1) SEEDS — is_doc records whose NAME matches (title/source any token = a STRONG match,
        # or the full name appears in the body = a weak match). PDFs/docx are ingested with the
        # FILENAME as the title, so an acronym like "SGT" often lives only in the body.
        seeds = []   # (record, source, strong_name_hit)
        for rec in pool:
            if not _is_doc(rec):
                continue
            title = str((rec.metadata or {}).get("title", "")).lower()
            src = str(rec.source).lower()
            text = str(getattr(rec, "text", "") or "").lower()
            name_hit = any(t in title or t in src for t in toks)
            if name_hit or all(t in text for t in toks):
                seeds.append((rec, str(rec.source), name_hit))
        if not seeds:
            return []

        # 2) Resolve the DOCUMENT(S) the name refers to by their source-group. A whole file is
        # ingested under ONE source ("reading:<filename>"), so every section of the named
        # document shares it — that is the document id. PREFER actual files over web-note
        # ("exploration:"/"study:") records that merely MENTION the name (the self-study
        # breadcrumb + USPTO/PATENTSCOPE boilerplate that was poisoning SGT runs).
        def _is_file(src: str) -> bool:
            return src.lower().startswith(("reading:", "deepread", "research:"))
        by_src: Dict[str, Dict[str, Any]] = {}
        for rec, src, nh in seeds:
            g = by_src.setdefault(src, {"count": 0, "name": False})
            g["count"] += 1
            g["name"] = g["name"] or nh
        cand_srcs = [s for s in by_src if _is_file(s)] or list(by_src)
        # CAP the expansion: a common single token can seed many documents → ranking by
        # (strong-name-hit, seed-count) keeps the few most-likely-intended docs, so the result
        # never blows up into a top-k over the whole store (adversarial review finding #1).
        cand_srcs.sort(key=lambda s: (by_src[s]["name"], by_src[s]["count"]), reverse=True)
        doc_srcs = set(cand_srcs[:6])

        # 3) EXPAND to the whole document(s): every chunk sharing those sources — INCLUDING
        # sections that don't themselves contain the name (Summary, Distinction-from-Prior-Art,
        # Claims). This is what surfaces the conceptual body, not just the name-bearing abstract.
        members = [rec for rec in pool if str(rec.source) in doc_srcs]

        # 4) Rank by the QUESTION (not the doc name) so the most relevant sections lead.
        if query_embedding is not None:
            members.sort(key=lambda r: cosine(query_embedding, r.embedding), reverse=True)
        # 5) Collapse near-duplicate chunks (same file ingested twice), lazily to max_chunks —
        # so the O(n²) compare never runs over the whole expanded pool (review finding #2).
        return _dedupe_near_records(members, max_keep=max_chunks)

    def list_documents(self) -> List[Dict[str, Any]]:
        """The distinct INGESTED DOCUMENTS in memory (one entry per source-group), for a UI
        document picker and for store diagnostics. Each dict has a display `name`, the `source`
        id, a `chunks` count, and `kind` ('file' = an ingested document, 'note' = a web/study
        record). Sorted by chunk count. A document ingested twice under different temp names
        shows as TWO entries — which is exactly what you want to SEE when diagnosing the store."""
        pool = (list(self.stm.get_all()) + list(self.mtm._records) + list(self.ltm._records))
        groups: Dict[str, Dict[str, Any]] = {}
        for rec in pool:
            src = str(rec.source)
            if not src.lower().startswith(
                    ("reading:", "exploration:", "research:", "ponder:", "study:", "deepread")):
                continue
            g = groups.get(src)
            if g is None:
                title = str((rec.metadata or {}).get("title", "")) or src.split(":", 1)[-1]
                kind = "file" if src.lower().startswith(("reading:", "deepread", "research:")) \
                    else "note"
                g = groups[src] = {"name": title, "source": src, "kind": kind, "chunks": 0}
            g["chunks"] += 1
        return sorted(groups.values(), key=lambda x: -x["chunks"])

    def max_similarity(self, embedding) -> float:
        """Highest cosine similarity of `embedding` to anything stored, across
        all tiers. Used by federation to gate an insight as novel vs the pool."""
        if embedding is None:
            return 0.0
        best = 0.0
        for rec in (list(self.stm.get_all()) + list(self.mtm._records)
                    + list(self.ltm._records)):
            if rec.embedding is not None:
                s = cosine(embedding, rec.embedding)
                if s > best:
                    best = s
        return best

    def replay_consolidate(self) -> Dict[str, int]:
        """Memory REPLAY — the second half of sleep, complementary to consolidate()'s tier-promotion.

        Where consolidate() moves memories UP the tiers (STM→MTM→LTM) by novelty, this pass works
        WITHIN a tier: it folds near-duplicate library traces of the same provenance into ONE
        reinforced record. That is what removes re-ingest junk (the patent ingested twice) WITHOUT
        deleting content — the survivor carries it and gets stronger — and what turns a fact seen
        many times into a higher-salience memory ("knows what it learned", not just "stored it").

        Provenance-safe by construction: it never merges across families, never touches her
        subjective/audit families (reflection / dream / ponder / introspection), never touches the
        thought-stream, and leaves STM (ephemeral turns) alone. Returns merge counts per tier.
        """
        out = {"mtm_merged": 0, "ltm_merged": 0}
        mtm_kept, mtm_merged = consolidate_records(self.mtm._records)
        if mtm_merged:
            self.mtm._records = mtm_kept
            self.mtm._save()
            out["mtm_merged"] = mtm_merged
        ltm_kept, ltm_merged = consolidate_records(self.ltm._records)
        if ltm_merged:
            self.ltm._records = ltm_kept
            self.ltm._save()
            out["ltm_merged"] = ltm_merged
        return out

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
