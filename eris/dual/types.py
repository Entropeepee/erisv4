"""Shared result adapter so the novel and traditional retrievers expose the same
shape to the DualPath, the arbiter, and the divergence log.

Novel returns (aligned, tension) MemoryRecord lists plus per-record coupling and a
field-interference resonance score; traditional returns a ranked record list with
scores. Both become a RetrievalResult with `.records` (the answer set) and the
optional novel-only channels carried through for the later epistemic analysis.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional
import hashlib


def record_id(rec) -> str:
    """A stable id for a MemoryRecord across runs/logging. Prefers an explicit id,
    then the content sha in metadata, then a hash of (source + text)."""
    rid = getattr(rec, "id", None)
    if rid:
        return str(rid)
    meta = getattr(rec, "metadata", None) or {}
    if meta.get("sha256"):
        return f"sha:{meta['sha256']}"
    src = (getattr(rec, "source", "") or "")
    txt = (getattr(rec, "text", "") or "")
    return "h:" + hashlib.blake2b((src + "\n" + txt).encode("utf-8"),
                                  digest_size=8).hexdigest()


@dataclass
class RetrievalResult:
    """Uniform output for both retrieval paths."""
    records: List[Any] = field(default_factory=list)        # the answer set (ranked)
    scores: List[float] = field(default_factory=list)       # parallel to records
    # Novel-only channels (None/empty for the traditional path) — kept so the log
    # can ask later whether aligned≈settled-fact and tension≈open-question.
    aligned: Optional[List[Any]] = None
    tension: Optional[List[Any]] = None
    coupling: Optional[List[float]] = None
    meta: dict = field(default_factory=dict)

    def top_ids(self, k: Optional[int] = None) -> List[str]:
        recs = self.records if k is None else self.records[:k]
        return [record_id(r) for r in recs]

    def is_empty(self) -> bool:
        return not self.records
