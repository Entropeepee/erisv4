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


# ── Symbol contract (KLSE / Pope-Filter; keep these distinct, do not conflate) ──
#   κ (curvature) = the COSINE / aligned channel → φ (coherence; magnitude; *what*).
#   λ ("torsion", KLSE sense) = the SINE / perpendicular channel → θ (phase; context;
#       *how* it's said). THIS is "the sine that is usually discarded."
#   τ (tau) = the twist/discrepancy between the κ and λ channels (NOT computed here; the
#       field-PDE τ is the vorticity ∇ρ×∇θ, defined in knowledge/frontends.py).
#   tan = sin/cos = λ/κ = the torsion-to-curvature RATIO — a derived DIAGNOSTIC, not
#       torsion itself (tan<1 curvature-dominated, tan>1 torsion-dominated, tan≈1 the 45°
#       critical transition). "Is torsion the tangent?" No — torsion is the sine channel λ;
#       the tangent is λ/κ, its ratio to curvature.

def _common(phi_q, theta_q, phi_s, theta_s):
    phi_q = np.asarray(phi_q, dtype=float); theta_q = np.asarray(theta_q, dtype=float)
    phi_s = np.asarray(phi_s, dtype=float); theta_s = np.asarray(theta_s, dtype=float)
    if phi_q.shape != phi_s.shape:
        n = min(phi_q.size, phi_s.size)
        phi_q, theta_q = phi_q.flat[:n], theta_q.flat[:n]
        phi_s, theta_s = phi_s.flat[:n], theta_s.flat[:n]
    return phi_q, theta_q, phi_s, theta_s


def field_resonance(phi_q, theta_q, phi_s, theta_s) -> float:
    """Discrete DCR integral (the κ / COSINE / aligned channel only):
        R_cos = mean( φ_q·φ_s·cos(θ_q − θ_s) ).
    Kept as a scalar for backward compatibility; new ranking uses the 2D form below,
    which also surfaces the λ/sine torsion channel that this scalar discards."""
    phi_q, theta_q, phi_s, theta_s = _common(phi_q, theta_q, phi_s, theta_s)
    return float(np.mean(phi_q * phi_s * np.cos(theta_q - theta_s)))


def _analytic(phi, theta) -> np.ndarray:
    """The analytic field z = φ·e^{iθ} — the complex representation in which resonance is a
    single inner product and the nullspace/common-mode projection (S1.8 GLNCS) is one line."""
    return np.asarray(phi, dtype=np.float64) * np.exp(1j * np.asarray(theta, dtype=np.float64))


def field_resonance_2d(phi_q, theta_q, phi_s, theta_s) -> dict:
    """Full two-channel resonance — the "never just cosine" correction (§B3), written as the
    exact complex-exponential form (S1.1 FFT-family): the κ and signed-λ channels are the real
    and imaginary parts of one Hermitian inner product, not two separate trig reductions.

      R = mean( φ_q e^{iθ_q} · conj(φ_s e^{iθ_s}) ) = mean( φ_q·φ_s·e^{iΔθ} )
      R_cos = Re R = mean(φ_q·φ_s·cos Δθ)   — κ / alignment (== the scalar field_resonance)
      R_sin = Im R = mean(φ_q·φ_s·sin Δθ)   — λ / SIGNED torsion channel (the discarded sine)
      magnitude   = |R| = √(R_cos² + R_sin²) — total resonance, the ranking score
      mixing_angle= arg R = atan2(R_sin, R_cos) — torsion diagnostic
    """
    phi_q, theta_q, phi_s, theta_s = _common(phi_q, theta_q, phi_s, theta_s)
    R = complex(np.mean(_analytic(phi_q, theta_q) * np.conj(_analytic(phi_s, theta_s))))
    return {"R_cos": R.real, "R_sin": R.imag,
            "magnitude": float(abs(R)),
            "mixing_angle": float(np.angle(R))}


def _ltm_records(memory):
    """Real accessor for stored LTM records on this codebase."""
    return list(getattr(memory.ltm, "_records", []))


class FieldInterferenceRetriever:
    """Rank stored attractors by how strongly they resonate with the query field."""

    def __init__(self, memory):
        self.memory = memory

    def retrieve(self, phi_q, theta_q, k: int = 5) -> List[Tuple[float, object]]:
        """Rank by the 2D resonance MAGNITUDE √(R_cos²+R_sin²) — so a neighbor that couples
        through the torsion (sine) channel is found even when its κ/cosine alignment is
        modest (the relationships plain cosine misses). (R_cos alone remains via
        field_resonance for back-compat callers.)"""
        scored: List[Tuple[float, object]] = []
        for rec in _ltm_records(self.memory):
            phi_s = getattr(rec, "phi_snapshot", None)
            theta_s = getattr(rec, "theta_snapshot", None)
            if phi_s is None or theta_s is None:
                continue
            R = field_resonance_2d(phi_q, theta_q, phi_s, theta_s)["magnitude"]
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
    torsion_mags = []          # |R_sin| of the chosen field neighbor — how much the
    mixing_angles = []         # sine channel carried (vs pure cosine)
    for p in probes:
        top_field = retr.retrieve(p["phi"], p["theta"], k=1)
        if top_field:
            r2 = field_resonance_2d(p["phi"], p["theta"],
                                    getattr(top_field[0][1], "phi_snapshot", None),
                                    getattr(top_field[0][1], "theta_snapshot", None))
            torsion_mags.append(abs(r2["R_sin"])); mixing_angles.append(r2["mixing_angle"])
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
        # The torsion-channel evidence: if these are ~0 the sine half is decorative; if
        # meaningfully non-zero on divergent picks, the sine carries non-redundant structure.
        "mean_torsion_magnitude": float(np.mean(torsion_mags)) if torsion_mags else 0.0,
        "mean_abs_mixing_angle": float(np.mean(np.abs(mixing_angles))) if mixing_angles else 0.0,
    }
