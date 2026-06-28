"""Confidence as resonance geometry, not a scalar.

David's framing: cosine is the MATCH, but the sine/torsion define the AMOUNT and QUALITY of what
stays unresolved. A claim that rests on evidence has three numbers, not one:

  match      = cos  — how well the claim aligns with the evidence centroid (how sure)
  unresolved = sin  = sqrt(1 - match^2) — the orthogonal residual the evidence does NOT cover
  coherence  = mean pairwise cosine among the evidence — do the supports AGREE (so the residual
               is a clean, namable gap) or scatter (so the support itself is noisy)?
  torsion    = atan2(unresolved, match) — the lean into the unresolved channel, in [0, pi/2]
  confidence = match * (0.5 + 0.5*coherence) — well-supported AND coherently so

This decomposes "how sure am I" into "how much is unexplained" and "is the unexplained part a
real tension or just noise" — the calibration signal the metacognitive loop needs. The bvec
analogue is bvec_resonance_2d() (per-domain elastic/plastic, R_cos/R_sin); this is the semantic
(embedding) analogue, operating over a claim and the set of sources it draws on."""
import math

import numpy as np


def _unit(v):
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float64).ravel()
    n = float(np.linalg.norm(a))
    return (a / n) if n > 1e-12 and a.size else None


def resonance_confidence(claim_vec, evidence_vecs) -> dict:
    """Return the confidence geometry of a claim against the evidence it rests on. Pure, no I/O.
    Degrades to zero-confidence/fully-unresolved when the claim or evidence vectors are missing."""
    units = [u for u in (_unit(e) for e in (evidence_vecs or [])) if u is not None]
    c = _unit(claim_vec)
    if c is None or not units:
        return {"match": 0.0, "unresolved": 1.0, "coherence": 0.0,
                "torsion": math.pi / 2, "confidence": 0.0}
    # MATCH — alignment with the evidence centroid (agreement among sources is rewarded).
    centroid = _unit(np.mean(units, axis=0))
    match = max(0.0, float(np.dot(c, centroid))) if centroid is not None else 0.0
    # UNRESOLVED — the orthogonal residual (the conservation law cos^2 + sin^2 = 1).
    unresolved = math.sqrt(max(0.0, 1.0 - match * match))
    # COHERENCE — how much the supports agree among themselves (1.0 for a single source).
    if len(units) == 1:
        coherence = 1.0
    else:
        sims = [float(np.dot(units[i], units[j]))
                for i in range(len(units)) for j in range(i + 1, len(units))]
        coherence = max(0.0, sum(sims) / len(sims)) if sims else 1.0
    torsion = math.atan2(unresolved, match)
    confidence = match * (0.5 + 0.5 * coherence)
    return {"match": match, "unresolved": unresolved, "coherence": coherence,
            "torsion": torsion, "confidence": confidence}
