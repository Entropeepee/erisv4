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
#       field-PDE τ is the vorticity ∇ρ×∇θ, defined in knowledge/frontends.py, and now computed
#       by the PDE/FRT by default — set ERIS_TAU_VORTICITY=0 to reach the legacy amplitude-Laplacian
#       proxy for comparison).
#   tan = sin/cos = λ/κ = the torsion-to-curvature RATIO — a derived DIAGNOSTIC, not
#       torsion itself (tan<1 curvature-dominated, tan>1 torsion-dominated, tan≈1 the 45°
#       critical transition). "Is torsion the tangent?" No — torsion is the sine channel λ;
#       the tangent is λ/κ, its ratio to curvature.

def circular_mean(values) -> float:
    """Mean of angles done on the CIRCLE — mean of the unit vectors e^{iθ}, then the angle of the
    resultant. θ is a PHASE: arithmetic-averaging it is wrong at the branch cut (mean of 0.01 and
    2π−0.01 is ≈π, the OPPOSITE phase; the circular mean is ≈0, correct). Returns 0.0 for an empty
    or all-zero-resultant input (no defined mean direction)."""
    a = np.asarray(values, dtype=np.float64).ravel()
    if a.size == 0:
        return 0.0
    z = np.exp(1j * a).mean()
    return float(np.angle(z)) if abs(z) > 1e-12 else 0.0


def _block_or_nearest(arr, th: int, tw: int):
    """Downsample `arr` (2D, real or complex) to (th, tw). Exact block-MEAN when the target evenly
    divides the source (the power-of-two field case); else average over linearly-spaced index
    ranges (handles non-divisor shapes / upsample). dtype-agnostic so it serves the circular path."""
    sh, sw = arr.shape
    if sh == th and sw == tw:
        return arr
    if th and tw and sh % th == 0 and sw % tw == 0:
        fh, fw = sh // th, sw // tw
        return arr.reshape(th, fh, tw, fw).mean(axis=(1, 3))
    ri = np.linspace(0, sh, th + 1).astype(int)
    ci = np.linspace(0, sw, tw + 1).astype(int)
    out = np.empty((th, tw), dtype=arr.dtype)
    for i in range(th):
        a0, a1 = ri[i], max(ri[i] + 1, ri[i + 1])
        for j in range(tw):
            b0, b1 = ci[j], max(ci[j] + 1, ci[j + 1])
            out[i, j] = arr[a0:a1, b0:b1].mean()
    return out


def resample_field(field, shape, *, circular: bool = False):
    """Resample a 2D field to `shape`. φ (magnitude) is block-averaged directly; θ (phase) MUST be
    `circular=True` so it is averaged on the circle (e^{iθ} block-mean → angle), never arithmetically
    across the 2π branch cut."""
    field = np.asarray(field, dtype=np.float64)
    th, tw = int(shape[0]), int(shape[1])
    if field.shape == (th, tw):
        return field
    if circular:
        return np.angle(_block_or_nearest(np.exp(1j * field), th, tw))
    return _block_or_nearest(field, th, tw)


def _common(phi_q, theta_q, phi_s, theta_s):
    """Bring two fields onto a COMMON grid before the DCR integral. Codex #4: the old path
    flatten-truncated to the first N row-major cells, comparing the WRONG spatial region (a 96×96
    signal in the lower-right of a 128×128 query lands in the truncated tail). Now both fields are
    resampled to the common (element-wise minimum) shape — φ block-averaged, θ averaged CIRCULARLY."""
    phi_q = np.asarray(phi_q, dtype=float); theta_q = np.asarray(theta_q, dtype=float)
    phi_s = np.asarray(phi_s, dtype=float); theta_s = np.asarray(theta_s, dtype=float)
    if phi_q.shape == phi_s.shape:
        return phi_q, theta_q, phi_s, theta_s
    if phi_q.ndim != 2 or phi_s.ndim != 2:        # non-2D safety: fall back to flatten-truncate
        n = min(phi_q.size, phi_s.size)
        return phi_q.flat[:n], theta_q.flat[:n], phi_s.flat[:n], theta_s.flat[:n]
    th = min(phi_q.shape[0], phi_s.shape[0])
    tw = min(phi_q.shape[1], phi_s.shape[1])
    return (resample_field(phi_q, (th, tw)), resample_field(theta_q, (th, tw), circular=True),
            resample_field(phi_s, (th, tw)), resample_field(theta_s, (th, tw), circular=True))


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


def analytic_resonance_magnitudes(query_field, cand_fields, *, denoise: bool = True):
    """Pool-level field-resonance magnitudes with optional COMMON-MODE (nullspace) removal —
    the GLNCS / S1.8 constraint-projection idea applied to retrieval ranking.

    φ is a non-negative coherence amplitude, so every candidate's analytic field z=φ·e^{iθ}
    shares a large common-mode (DC) component; that shared baseline inflates the resonance of
    ALL candidates roughly equally and compresses the discriminative part the rerank sorts on.
    Removing the rank-1 common mode (the pool mean field) projects onto the bias-annihilated
    subspace, so candidates are ranked by how they resonate DIFFERENTLY — a coherence win, not
    a speed trade. Both κ and signed-λ channels are preserved (the subtraction is linear on the
    complex field, so the torsion channel is kept and in fact sharpened).

    Returns a list of magnitudes aligned with `cand_fields` (None where a field is None)."""
    if not query_field:
        return [None] * len(cand_fields)
    zq = _analytic(*query_field).ravel()
    zs = []
    for f in cand_fields:
        zs.append(None if f is None else _analytic(*f).ravel())
    valid = [z for z in zs if z is not None]
    if not valid:
        return [None] * len(cand_fields)
    # Estimating a common mode from <3 candidates is degenerate (a 2-element pool collapses to
    # ±the same residual → all magnitudes tie). Only project when the pool can support it.
    denoise = denoise and len(valid) >= 3
    L = min([zq.size] + [z.size for z in valid])        # common grid length (defensive)
    zq = zq[:L]
    if denoise:
        zbar = np.mean(np.stack([z[:L] for z in valid]), axis=0)   # rank-1 common mode
        zq = zq - zbar
    out = []
    for z in zs:
        if z is None:
            out.append(None); continue
        ze = z[:L] - zbar if denoise else z[:L]
        out.append(float(abs(np.mean(zq * np.conj(ze)))))   # |R| with both channels retained
    return out


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
