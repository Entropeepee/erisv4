"""Cross-modal GROUNDING COST (§4) — the honest multimodal question.

Zero-shot cross-modal resonance is the NULL, not the headline (a meow does not
acoustically resemble a cat's visual structure; "cat" through the PDE resembles
neither). The real, stronger question is GROUNDING COST: when the field already
separates concepts WITHIN a modality, how few labeled pairs N does it take to align a
NEW modality? A good representation makes grounding cheap.

RULE 1 — two arms, two parameter budgets, BOTH channels in the fit:
  • Zero-shot arm: ZERO cross-modal parameters. Raw two-channel field resonance. ≈chance.
  • Grounding arm: UNITARY alignment only — a single unitary map U from N labeled pairs via
    closed-form COMPLEX Procrustes (M=AᴴB; W,_,Vᴴ=svd(M); U=W Vᴴ). The fit operates on the
    COMPLEX descriptors, so both the real (cos) and imaginary (sin) parts inform U, and
    scoring via aᴴb then yields both elastic (Re) and tension (|Im|) channels. The
    cosine-only ablation instead takes Re(·) BEFORE fitting — discarding the imaginary/sin
    half entirely — then fits a real-orthogonal map on the reduced real data. So for
    phase-dependent structure (e.g. real-part collisions) the unitary arm separates classes
    the cosine-only arm cannot: the win comes from fitting on complex data, not from any
    magic phase "preservation" of the unitary constraint itself.

ADAPT (signature layout): per-example κ coordinates live in each field's OWN SVD gauge,
so cross-example Procrustes over κ is ill-posed (and the N<d overfit regime the spec
flags would dominate). We therefore align in the commensurable COMPLEX COHERENCE
descriptor — the spinor √ρ·e^{iθ} reduced to a fixed small grid (shared coordinate frame,
fixed low dim, phase-bearing). The κ/λ/τ structure still drives the within-modal
classifier (§3) and the two-channel scoring; the alignment preserves phase via the
complex inner product. Real/cosine-only Procrustes is the §4-2b ablation.

Torch-free: complex SVD is native NumPy.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from eris.knowledge.frontends import _resize
from eris.vision.coupling import field_coupling

_EPS = 1e-12


@dataclass
class GroundingConfig:
    descriptor_grid: int = 4          # m: spinor reduced to m×m → d = m² complex dims
    Ns: Tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)   # spec grid (keep N=1 — the small-N regime is the point)
    n_repeat: int = 20                # subsamples per N for the CI
    seed: int = 0
    # NOTE on the d>N regime: a unitary U on ℂ^d has d² real DOF, so for small N the fit is
    # underdetermined and could in principle absorb noise. The SHUFFLED control (random
    # pairings) is the guard the spec mandates — it stays at chance ∀N — and the held-out
    # source items the curve classifies are never in the fit set. Read low-N points against
    # the shuffled row, not in isolation.


# ── complex coherence descriptor (shared frame, phase-bearing) ──────────────
def field_descriptor(mag, theta, grid: int = 4) -> np.ndarray:
    """Reduce a (mag=√ρ, θ) field to a fixed-dim COMPLEX vector: form the spinor
    √ρ·e^{iθ}, downsample its real & imag parts to grid×grid (bilinear — never resize an
    angle directly), recombine, L2-normalize. Shared frame ⇒ cross-example Procrustes is
    well-posed; complex ⇒ the unitary fit has phase to preserve."""
    mag = np.asarray(mag, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    spinor = mag * np.exp(1j * theta)
    re = _resize(spinor.real, grid)
    im = _resize(spinor.imag, grid)
    v = (re + 1j * im).ravel()
    n = np.linalg.norm(v)
    return v / (n + _EPS)


def descriptors_by_class(fields_by_class: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
                         grid: int = 4) -> Dict[str, List[np.ndarray]]:
    return {c: [field_descriptor(m, t, grid) for (m, t) in items]
            for c, items in fields_by_class.items()}


# ── Procrustes: unitary (complex, two-channel) and real (cosine-only ablation) ──
def unitary_procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Closed-form UNITARY map U (d×d) minimizing ‖A U − B‖ over unitary U, for paired
    rows A,B (N×d complex). M = Aᴴ B; W,_,Vᴴ = svd(M); U = W Vᴴ. Phase-preserving."""
    M = A.conj().T @ B
    W, _, Vh = np.linalg.svd(M)
    return W @ Vh


def real_procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Closed-form REAL-orthogonal map on the REAL PARTS only (the cosine-only ablation:
    the imaginary/sin half is dropped — cosine geometry). R (d×d real)."""
    Ar, Br = np.real(A), np.real(B)
    M = Ar.T @ Br
    W, _, Vh = np.linalg.svd(M)
    return W @ Vh


def _vecs(descs: List[np.ndarray]) -> np.ndarray:
    return np.asarray(descs, dtype=np.complex128)


def fit_map(src: List[np.ndarray], tgt: List[np.ndarray], kind: str = "unitary"):
    """Fit the alignment from paired source→target descriptors. kind ∈ {unitary, cosine}."""
    A, B = _vecs(src), _vecs(tgt)
    if A.shape[0] == 0:
        return None
    return unitary_procrustes(A, B) if kind == "unitary" else real_procrustes(A, B)


def _apply(desc: np.ndarray, M, kind: str) -> np.ndarray:
    if M is None:
        return desc if kind == "unitary" else np.real(desc)
    return (desc @ M) if kind == "unitary" else (np.real(desc) @ M)


def pair_score(a: np.ndarray, b: np.ndarray, kind: str = "unitary",
               channels: str = "both") -> float:
    """Two-channel similarity of aligned descriptor a to target b.
    unitary: s = aᴴb (complex) → elastic=Re(s) (cos), plastic=|Im(s)| (sin/tension).
    cosine : real inner product only (the sin half is gone by construction)."""
    if kind == "cosine":
        ar = np.real(a); br = np.real(b)
        ar = ar / (np.linalg.norm(ar) + _EPS); br = br / (np.linalg.norm(br) + _EPS)
        return float(ar @ br)
    a = a / (np.linalg.norm(a) + _EPS); b = b / (np.linalg.norm(b) + _EPS)
    s = np.vdot(a, b)                      # aᴴ b (complex)
    elastic, plastic = float(s.real), float(abs(s.imag))
    return {"aligned": elastic, "tension": -plastic}.get(channels, elastic - plastic)


def _protos(tgt_by_class: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    """Target class prototype = complex mean descriptor (deterministic, no fitting)."""
    return {c: np.mean(_vecs(v), axis=0) for c, v in tgt_by_class.items() if len(v)}


def _classify(q: np.ndarray, protos: Dict[str, np.ndarray], M, kind: str,
              channels: str) -> str:
    qa = _apply(q, M, kind)
    return max(protos, key=lambda c: pair_score(qa, protos[c], kind, channels))


# ── §2b grounding curve (the headline) ──────────────────────────────────────
def _sample_pairs(src_by_class, tgt_by_class, n, rng, shuffle=False):
    """N labeled source→target pairs (one source + one target instance of the SAME
    concept). shuffle=True pairs a source with a RANDOM target (control). Sampling is WITH
    replacement, so N may exceed the instances-per-class. Classes with no instances on
    either side are skipped (an empty/degenerate dataset yields no pairs, never a crash)."""
    classes = [c for c in sorted(src_by_class)
               if src_by_class.get(c) and tgt_by_class.get(c)]
    src, tgt = [], []
    if not classes:
        return src, tgt
    for _ in range(n):
        c = classes[rng.randint(len(classes))]
        s = src_by_class[c][rng.randint(len(src_by_class[c]))]
        tc = classes[rng.randint(len(classes))] if shuffle else c
        t = tgt_by_class[tc][rng.randint(len(tgt_by_class[tc]))]
        src.append(s); tgt.append(t)
    return src, tgt


def grounding_curve(src_by_class, tgt_by_class, cfg: GroundingConfig = None, *,
                    kind: str = "unitary", channels: str = "both",
                    shuffle: bool = False) -> Dict[int, Dict[str, float]]:
    """Accuracy vs N: fit the map on N labeled pairs, classify held-out source items in
    the aligned space against target prototypes. Repeated subsamples → mean + std (CI).
    N=0 ⇒ no map (raw cross-modal descriptor coupling — the descriptor-space zero-shot)."""
    cfg = cfg or GroundingConfig()
    rng = np.random.RandomState(cfg.seed)
    classes = sorted(src_by_class)
    out: Dict[int, Dict[str, float]] = {}
    for n in cfg.Ns:
        accs = []
        for _ in range(cfg.n_repeat):
            M = None
            if n > 0:
                src, tgt = _sample_pairs(src_by_class, tgt_by_class, n, rng, shuffle)
                M = fit_map(src, tgt, kind)
            protos = _protos(tgt_by_class)
            correct = total = 0
            for c in classes:
                for q in src_by_class[c]:
                    pred = _classify(q, protos, M, kind, channels)
                    correct += (pred == c); total += 1
            accs.append(correct / max(1, total))
        a = np.asarray(accs)
        out[n] = {"acc": float(a.mean()), "std": float(a.std()),
                  "ci95": float(1.96 * a.std() / np.sqrt(len(a)))}
    return out


def within_modal_ceiling(by_class, cfg: GroundingConfig = None,
                         channels: str = "both") -> float:
    """The ceiling: within-modal descriptor classification (same modality, no alignment
    needed) — how well the descriptors separate concepts inside one modality."""
    cfg = cfg or GroundingConfig()
    protos = _protos(by_class)
    correct = total = 0
    for c in sorted(by_class):
        for q in by_class[c]:
            pred = max(protos, key=lambda k: pair_score(q, protos[k], "unitary", channels))
            correct += (pred == c); total += 1
    return correct / max(1, total)


# ── §2a zero-shot (the NULL) + low-level probe ──────────────────────────────
def zero_shot_accuracy(src_fields_by_class, tgt_fields_by_class) -> float:
    """Raw TWO-CHANNEL field resonance (RULE 2), ZERO cross-modal params: classify each
    source field by the target concept it couples to most — scoring against the MEAN
    resonance over all of that concept's target fields (not a single exemplar). Framed as
    the null (≈chance)."""
    classes = sorted(c for c, items in tgt_fields_by_class.items() if items)
    correct = total = 0
    for c in sorted(src_fields_by_class):
        for (mg, th) in src_fields_by_class[c]:
            best = max(classes, key=lambda k: _resonance(mg, th, tgt_fields_by_class[k]))
            correct += (best == c); total += 1
    return correct / max(1, total)


def _resonance(mg, th, tgt_items) -> float:
    return float(np.mean([(lambda e, p, _: e - p)(*field_coupling(mg, th, tm, tt))
                          for (tm, tt) in tgt_items]))


def zero_shot_items(src_fields_by_class, tgt_fields_by_class):
    """Per-item zero-shot: (correct_flags, src_mags) so a low-level probe can regress
    success on a modality-specific low-level feature (transient/spiky), separating a
    bouba/kiki correspondence from any concept transfer."""
    classes = sorted(c for c, items in tgt_fields_by_class.items() if items)
    flags, mags = [], []
    for c in sorted(src_fields_by_class):
        for (mg, th) in src_fields_by_class[c]:
            best = max(classes, key=lambda k: _resonance(mg, th, tgt_fields_by_class[k]))
            flags.append(1.0 if best == c else 0.0); mags.append(mg)
    return flags, mags


def permutation_null(accuracy: float, n_items: int, n_classes: int,
                     n_perm: int = 2000, seed: int = 0) -> float:
    """p-value of a zero-shot accuracy under random labelling (is it above chance?)."""
    rng = np.random.RandomState(seed)
    chance = np.array([np.mean(rng.randint(0, n_classes, n_items)
                               == rng.randint(0, n_classes, n_items))
                       for _ in range(n_perm)])
    return float((np.sum(chance >= accuracy) + 1) / (n_perm + 1))


def transient_score(mag) -> float:
    """Low-level 'transient/percussive' proxy: temporal (column) energy burstiness of an
    audio field (high = clicky/percussive). Used to detect bouba/kiki, not concept."""
    m = np.asarray(mag, dtype=np.float64)
    col_energy = (m ** 2).sum(axis=0)
    return float(col_energy.std() / (col_energy.mean() + _EPS))


def spatial_busyness(mag) -> float:
    """Low-level 'high-spatial-frequency busyness' proxy for an image field (spiky)."""
    m = np.asarray(mag, dtype=np.float64)
    gy, gx = np.gradient(m)
    return float(np.hypot(gx, gy).mean())


def lowlevel_probe(accuracy_per_item: List[float], feature_per_item: List[float]
                   ) -> Dict[str, float]:
    """Regress per-item zero-shot success on a low-level feature; return the residual
    concept-effect (mean accuracy with the low-level trend removed). A bouba/kiki
    correspondence shows up as a high slope and a near-zero residual."""
    a = np.asarray(accuracy_per_item, dtype=np.float64)
    f = np.asarray(feature_per_item, dtype=np.float64)
    if a.size < 2 or f.std() < _EPS:
        return {"slope": 0.0, "residual": float(a.mean() - 1.0 / max(1, a.size))}
    fc = (f - f.mean()) / (f.std() + _EPS)
    slope = float(np.cov(fc, a)[0, 1] / (np.var(fc) + _EPS))
    resid = a - slope * fc
    return {"slope": slope, "residual_mean": float(resid.mean()),
            "explained_by_lowlevel": float(abs(slope) * fc.std())}


# ── §2c transitivity ─────────────────────────────────────────────────────────
def transitivity(audio_by_class, image_by_class, word_by_class,
                 cfg: GroundingConfig = None, *, n_pairs: int = 8,
                 channels: str = "both") -> Dict[str, float]:
    """Ground audio→word and image→word (never audio→image directly); does audio↔image
    follow for free? Map both into word space, classify audio by image-in-word prototypes."""
    cfg = cfg or GroundingConfig()
    rng = np.random.RandomState(cfg.seed)
    aw_s, aw_t = _sample_pairs(audio_by_class, word_by_class, n_pairs, rng)
    iw_s, iw_t = _sample_pairs(image_by_class, word_by_class, n_pairs, rng)
    U_aw = fit_map(aw_s, aw_t, "unitary")
    U_iw = fit_map(iw_s, iw_t, "unitary")
    # image prototypes pushed into word space
    img_in_word = {c: np.mean(_vecs([d @ U_iw for d in image_by_class[c]]), axis=0)
                   for c in sorted(image_by_class) if image_by_class[c]}
    correct = total = 0
    for c in sorted(audio_by_class):
        for q in audio_by_class[c]:
            pred = max(img_in_word, key=lambda k: pair_score(q @ U_aw, img_in_word[k],
                                                             "unitary", channels))
            correct += (pred == c); total += 1
    return {"audio_image_via_word_acc": correct / max(1, total),
            "chance": 1.0 / max(1, len(audio_by_class))}
