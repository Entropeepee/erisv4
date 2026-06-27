"""Cross-modal field resonance (§4) — the GVE thesis test, NO shared training.

Does an IMAGE field and the WORD field for the same concept resonate with no
learned cross-modal parameters (RULE 1)? The word is encoded through the REAL
FRT+PDE text path to a (mag, θ) field of the same size as the image field; each
image is classified by the word it couples to most. Image physics and word physics
meet ONLY at the coupling/overlap — no projection head, no contrastive loss.

THE ABLATION (the κ+Λ receipt on real data): classify three ways —
  aligned-only (elastic — what bare field_resonance gives), tension-only (plastic),
  both — and report all three accuracies. If both > aligned-only, keeping the
sine/tension half measurably helps. Significance is earned: accuracy vs chance, a
permutation-null p-value, and an effect size.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np

from eris.knowledge.frontends import TextFrontend
from eris.vision.laf import laf_signature, kappa_overlap
from eris.vision.coupling import field_coupling
from eris.experiments.zero_weight_vision import Signature, VisionConfig


def word_signature(word: str, cfg: VisionConfig,
                   exemplars: Optional[List[str]] = None) -> Signature:
    """Encode a class word (optionally averaged with fixed exemplar sentences —
    still zero learned params) through the real text path to a (mag, θ) field of
    the SAME size/normalization as an image field, plus its (κ, λ)."""
    tf = TextFrontend()
    texts = [word] + list(exemplars or [])
    mags, thetas = [], []
    for t in texts:
        m, th = tf.to_field(t, size=cfg.size)
        mags.append(np.asarray(m, dtype=np.float64))
        thetas.append(np.asarray(th, dtype=np.float64))
    mag = np.mean(mags, axis=0)
    theta = np.arctan2(np.mean(np.sin(thetas), axis=0), np.mean(np.cos(thetas), axis=0))
    kappa, lam, _ = laf_signature(mag, theta, cfg.laf)
    return Signature(mag=mag, theta=theta, kappa=kappa, lam=lam, tau_stats=np.zeros(3))


def cross_modal_score(img: Signature, word: Signature, mode: str = "both",
                      use_kappa: bool = False) -> float:
    """Coupling of an image field to a word field. mode ∈ {aligned, tension, both}:
    aligned = elastic (in-phase), tension = −plastic (less out-of-phase = better),
    both = elastic − plastic (RULE 2)."""
    e, p, _ = field_coupling(img.mag, img.theta, word.mag, word.theta)
    s = {"aligned": e, "tension": -p, "both": e - p}.get(mode, e - p)
    if use_kappa:
        a, em = kappa_overlap(img.kappa, word.kappa)
        s += 0.3 * ({"aligned": a, "tension": -em, "both": a - em}.get(mode, a - em))
    return float(s)


def classify(img: Signature, words: Dict[str, Signature], mode: str = "both",
             use_kappa: bool = False) -> str:
    return max(words, key=lambda w: cross_modal_score(img, words[w], mode, use_kappa))


def _accuracy(preds: List[str], gold: List[str]) -> float:
    return float(np.mean([p == g for p, g in zip(preds, gold)])) if gold else 0.0


def run_ablation(imgs: List[Signature], gold: List[str],
                 words: Dict[str, Signature], use_kappa: bool = False) -> Dict[str, dict]:
    """3-way ablation: accuracy for aligned-only, tension-only, both."""
    out = {}
    for mode in ("aligned", "tension", "both"):
        preds = [classify(s, words, mode, use_kappa) for s in imgs]
        out[mode] = {"accuracy": _accuracy(preds, gold), "preds": preds}
    out["chance"] = 1.0 / max(1, len(words))
    return out


def permutation_null(preds: List[str], gold: List[str], n_perm: int = 2000,
                     seed: int = 0) -> Tuple[float, float]:
    """(real_accuracy, p_value). Shuffle the gold labels many times → null
    distribution of accuracy; p = P(null ≥ real). (Math.random is unavailable in
    workflow scripts but this is a normal test/CLI process — np RandomState is fine.)"""
    real = _accuracy(preds, gold)
    rng = np.random.RandomState(seed)
    g = np.array(gold, dtype=object)
    null = np.array([_accuracy(preds, list(rng.permutation(g))) for _ in range(n_perm)])
    p = float((np.sum(null >= real) + 1) / (n_perm + 1))
    return real, p


def effect_size(imgs: List[Signature], gold: List[str],
                words: Dict[str, Signature]) -> float:
    """(mean correct-coupling − mean wrong-coupling) / pooled std."""
    corr, wrong = [], []
    for s, g in zip(imgs, gold):
        for w, ws in words.items():
            (corr if w == g else wrong).append(cross_modal_score(s, ws, "both"))
    if not corr or not wrong:
        return 0.0
    pooled = np.std(np.array(corr + wrong)) + 1e-9
    return float((np.mean(corr) - np.mean(wrong)) / pooled)
