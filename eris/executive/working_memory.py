"""Goal-conditioned working-memory retrieval (§B2 — GPW fault #4 done right).

Each cycle, retrieval scores candidates by COHERENCE-GAIN toward the active goal (∂C/∂X) —
how much a candidate constructively interferes with the goal field — not raw similarity,
and injects the bounded top-k as STRUCTURED working-memory context. Never string-
concatenated: that concatenation was the original runaway-loop bug, so the contract returns
a bounded list of records, and the prompt assembler lays them out deliberately.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from eris.computation.activations import BVec, bvec_resonance


def coherence_gain(candidate_bvec: BVec, goal_bvec: BVec) -> float:
    """∂C/∂X toward the goal ≈ the constructive interference of the candidate's field with
    the goal field (the shared κ/λ resonance metric). High = retrieving this RAISES
    coherence toward the goal; negative = it pulls away."""
    return bvec_resonance(candidate_bvec, goal_bvec)


def _bvec_of(c) -> Optional[BVec]:
    if isinstance(c, BVec):
        return c
    bv = getattr(c, "bvec", None)
    return bv if isinstance(bv, BVec) else None


def goal_conditioned_context(candidates: List[Any], goal_bvec: BVec, *, k: int = 5,
                             working_set: Optional[Dict] = None,
                             working_weight: float = 0.25) -> List[Dict[str, Any]]:
    """Rank `candidates` (records with a `.bvec`, or BVecs) by coherence-gain toward the
    active goal and return the bounded top-k as STRUCTURED items (never a concatenated
    string). When a `working_set` is given, a candidate that also resonates with the recent
    broadcasts gets a small continuity bonus, so the frame stays coherent turn-to-turn."""
    if goal_bvec is None:
        return []
    frame_bvecs = []
    if working_set:
        for b in working_set.get("broadcasts", []):
            bb = b.get("bvec") if isinstance(b, dict) else None
            if isinstance(bb, BVec):
                frame_bvecs.append(bb)
    scored = []
    for c in candidates:
        bv = _bvec_of(c)
        if bv is None:
            continue
        s = coherence_gain(bv, goal_bvec)
        if frame_bvecs:
            s += working_weight * max(coherence_gain(bv, fb) for fb in frame_bvecs)
        scored.append((s, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"record": c, "coherence_gain": round(float(s), 4)} for s, c in scored[:max(0, k)]]
