"""
eris/agents/memory_view.py
==========================
LayeredMemory (WILLOW I.2). A node reads SHARED (the collective pool) + PRIVATE
(its own lived experience); writes go only to PRIVATE. That's why a node knows
everything Eris does, yet diverges through what it alone has experienced.
"""
from __future__ import annotations

from typing import List, Optional

from eris.memory.tiers import MemorySystem, MemoryRecord
from eris.computation.activations import BVec


class LayeredMemory:
    def __init__(self, shared: MemorySystem, private: MemorySystem):
        self.shared = shared
        self.private = private

    def retrieve(self, query_embedding=None, query_bvec: Optional[BVec] = None,
                 top_k: int = 8, query_text: Optional[str] = None) -> List[MemoryRecord]:
        priv = self.private.retrieve(query_embedding=query_embedding,
                                     query_bvec=query_bvec, top_k=top_k,
                                     query_text=query_text)
        shared = self.shared.retrieve(query_embedding=query_embedding,
                                      query_bvec=query_bvec, top_k=top_k,
                                      query_text=query_text)
        merged, seen = [], set()
        for r in priv + shared:           # own experience first, then collective
            if r.text not in seen:
                seen.add(r.text)
                merged.append(r)
        return merged[:top_k]

    def store_experience(self, text: str, embedding=None, bvec: Optional[BVec] = None,
                         kind: str = "experience", **meta) -> MemoryRecord:
        return self.private.store_turn(text=text, bvec=bvec or BVec(),
                                       embedding=embedding, source=kind,
                                       metadata=meta or None)

    def max_similarity_in_shared(self, embedding) -> float:
        return self.shared.max_similarity(embedding)
