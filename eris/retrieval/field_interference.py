"""
eris/retrieval/field_interference.py
====================================
Retrieve by FIELD RESONANCE, not just embedding cosine (Remediation Tier 4.6).

    R_ij = integral over the grid of  phi_i * phi_j * cos(theta_i - theta_j)  dx

This is the Domain-Coupling-Resonance (DCR) integral. Two fields score high when
they constructively interfere — same magnitude pattern (phi) AND aligned phase
(theta). It is the operational form of "understanding by resonance": a query's
field is compared to stored concept attractors by how strongly they resonate,
which can capture relationships embedding cosine misses (cross-domain analogies
that share a field signature but not vocabulary).

This module is also the measurement instrument for the grokking experiment
(ERIS_V4 Tier 5): resonance_vs_cosine() reports where field resonance and
embedding cosine AGREE vs DIVERGE. It does NOT claim the field 'understands' —
it gives you the number with which to test that claim.

Reconciled to the real MemorySystem API: LTM records live in
``memory.ltm._records``; each MemoryRecord carries ``phi_snapshot`` /
``theta_snapshot`` (downsampled field) and ``embedding``.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def field_resonance(phi_q, theta_q, phi_s, theta_s) -> float:
    """Discrete DCR integral between a query field and a stored field.
    Fields are arrays on the same grid (downsampled snapshots; mismatched shapes
    are flattened+truncated to a common length)."""
    phi_q = np.asarray(phi_q, dtype=float)
    theta_q = np.asarray(theta_q, dtype=float)
    phi_s = np.asarray(phi_s, dtype=float)
    theta_s = np.asarray(theta_s, dtype=float)
    if phi_q.shape != phi_s.shape:
        n = min(phi_q.size, phi_s.size)
        phi_q, theta_q = phi_q.flat[:n], theta_q.flat[:n]
        phi_s, theta_s = phi_s.flat[:n], theta_s.flat[:n]
    integrand = phi_q * phi_s * np.cos(theta_q - theta_s)
    return float(np.mean(integrand))   # mean = discrete integral / area


def _ltm_records(memory):
    """Real accessor for stored LTM records on this codebase."""
    return list(getattr(memory.ltm, "_records", []))


class FieldInterferenceRetriever:
    """Rank stored attractors by how strongly they resonate with the query field."""

    def __init__(self, memory):
        self.memory = memory

    def retrieve(self, phi_q, theta_q, k: int = 5) -> List[Tuple[float, object]]:
        scored: List[Tuple[float, object]] = []
        for rec in _ltm_records(self.memory):
            phi_s = getattr(rec, "phi_snapshot", None)
            theta_s = getattr(rec, "theta_snapshot", None)
            if phi_s is None or theta_s is None:
                continue
            R = field_resonance(phi_q, theta_q, phi_s, theta_s)
            scored.append((R, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]


# --------------------------------------------------------------------------- analysis
def resonance_vs_cosine(memory, probes) -> dict:
    """Diagnostic for the grokking experiment (Tier 5A).

    For each probe (dict with 'phi', 'theta', 'embedding', optional 'title'),
    retrieve the top neighbor by field resonance R_ij and by embedding cosine,
    and report how often they AGREE vs DIVERGE. High divergence with sensible
    field-only neighbors = the field carries non-redundant relational structure
    (evidence for grok). If R_ij just tracks cosine, the field is decorative for
    retrieval.
    """
    records = _ltm_records(memory)
    retr = FieldInterferenceRetriever(memory)

    def cos(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(a @ b / d)

    agree = 0
    diverge_examples = []
    for p in probes:
        top_field = retr.retrieve(p["phi"], p["theta"], k=1)
        emb = p.get("embedding")
        cos_ranked = sorted(
            (r for r in records if getattr(r, "embedding", None) is not None),
            key=lambda r: cos(emb, r.embedding), reverse=True,
        ) if emb is not None else []
        top_cos = cos_ranked[0] if cos_ranked else None
        if top_field and top_cos is not None and top_field[0][1] is top_cos:
            agree += 1
        else:
            def _title(r):
                return (getattr(r, "metadata", {}) or {}).get("title", getattr(r, "text", "?")[:40])
            diverge_examples.append({
                "probe": p.get("title", ""),
                "field_neighbor": _title(top_field[0][1]) if top_field else None,
                "cosine_neighbor": _title(top_cos) if top_cos else None,
            })
    n = max(1, len(probes))
    return {
        "agreement_rate": agree / n,
        "divergence_rate": 1 - agree / n,
        "divergent_examples": diverge_examples[:20],
    }
