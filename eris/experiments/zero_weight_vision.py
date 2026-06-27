"""Zero-weight κ/λ/τ image classifier (RULES 1-4) + baselines.

No learned parameters, no backprop (RULE 1): each image becomes a coherence field
(ImageFrontend), summarized by the LAF signature (κ, λ) and torsion τ-stats. A class
"prototype" is the deterministic centroid of its signatures. A query is scored by
zero-weight geometry only:

  score = field_coupling (elastic − plastic, RULE 2)  +  κ-overlap (aligned − emergent)
          −  λ-distance,   Hill-Power shrunk across the class scores.

Headline features are grayscale-field-only (RULE 4); RGB/textness live in a separate
ablation arm so a positive geometry number is attributable to geometry, not colour.
Torch-free (NumPy/CuPy); baselines (SVM/PCA/kNN) are sklearn-guarded.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from eris.knowledge.frontends import ImageFrontend, torsion
from eris.vision.laf import laf_signature, kappa_overlap, lambda_distance, LAFConfig
from eris.vision.coupling import field_coupling, FieldDebias, UnknownGate
from eris.computation.shrinkage import davidian_weight


@dataclass
class VisionConfig:
    size: int = 32
    laf: LAFConfig = field(default_factory=lambda: LAFConfig(patch=8, n_scales=4, n_modes=12))
    w_field: float = 1.0          # field_coupling is the PRIMARY term
    w_kappa: float = 0.4
    w_lambda: float = 0.3
    hill: Tuple[float, float, float, float] = (1.0, 0.5, 1.0, 0.1)  # cross-class shrink
    two_channel: bool = True      # RULE 2: elastic − plastic. False = cos-only (ablation)
    use_rgb: bool = False         # RULE 4: RGB/textness only in the ablation arm


@dataclass
class Signature:
    mag: np.ndarray
    theta: np.ndarray
    kappa: np.ndarray
    lam: np.ndarray
    tau_stats: np.ndarray
    rgb: Optional[np.ndarray] = None        # ablation only
    textness: float = 0.0                   # ablation only


def compute_signature(image, cfg: VisionConfig, debias: Optional[FieldDebias] = None) -> Signature:
    mag, theta = ImageFrontend().to_field(image, size=cfg.size)
    mag = np.asarray(mag, dtype=np.float64)
    if debias is not None:
        mag = debias.apply(mag)
    rho = np.clip(mag ** 2, 0.0, None)
    tau = torsion(rho, theta)
    tau_stats = np.array([float(tau.mean()), float(tau.std()), float(np.abs(tau).mean())])
    kappa, lam, _ = laf_signature(mag, theta, cfg.laf)
    sig = Signature(mag=mag, theta=np.asarray(theta, dtype=np.float64),
                    kappa=kappa, lam=lam, tau_stats=tau_stats)
    if cfg.use_rgb and isinstance(image, np.ndarray) and image.ndim == 3:
        a = image.astype(np.float64)
        sig.rgb = np.array([a[..., c].mean() for c in range(3)]
                           + [a[..., c].std() for c in range(3)])
    return sig


@dataclass
class ClassPrototype:
    label: str
    mag: np.ndarray
    theta: np.ndarray
    kappa: np.ndarray
    lam: np.ndarray
    tau_stats: np.ndarray
    active: bool = True


def build_prototype(label: str, sigs: List[Signature], cfg: VisionConfig) -> ClassPrototype:
    """Deterministic centroid — NO fitting. Circular-mean phase, mean κ-subspace
    (SVD of the stacked modes), mean λ and τ-stats."""
    mags = np.stack([s.mag for s in sigs])
    thetas = np.stack([s.theta for s in sigs])
    mean_mag = mags.mean(axis=0)
    mean_theta = np.arctan2(np.sin(thetas).mean(axis=0), np.cos(thetas).mean(axis=0))
    K = np.concatenate([s.kappa for s in sigs], axis=1)
    U, _, _ = np.linalg.svd(K, full_matrices=False)
    r = min(cfg.laf.n_modes, U.shape[1])
    mean_kappa = U[:, :r]
    mean_lam = np.stack([s.lam for s in sigs]).mean(axis=0)
    mean_tau = np.stack([s.tau_stats for s in sigs]).mean(axis=0)
    return ClassPrototype(label, mean_mag, mean_theta, mean_kappa, mean_lam, mean_tau)


def raw_score(sig: Signature, proto: ClassPrototype, cfg: VisionConfig) -> float:
    e, p, _ = field_coupling(sig.mag, sig.theta, proto.mag, proto.theta)
    fc = (e - p) if cfg.two_channel else e          # RULE 2 vs cos-only ablation
    aligned, emergent = kappa_overlap(sig.kappa, proto.kappa)
    kappa_term = (aligned - emergent) if cfg.two_channel else aligned
    ld = lambda_distance(sig.lam, proto.lam)
    return cfg.w_field * fc + cfg.w_kappa * kappa_term - cfg.w_lambda * ld


class ZeroWeightClassifier:
    def __init__(self, cfg: VisionConfig = None):
        self.cfg = cfg or VisionConfig()
        self.protos: Dict[str, ClassPrototype] = {}

    def fit(self, class_sigs: Dict[str, List[Signature]]) -> "ZeroWeightClassifier":
        for label, sigs in class_sigs.items():
            if sigs:
                self.protos[label] = build_prototype(label, sigs, self.cfg)
        return self

    def deactivate(self, label: str) -> None:
        """Blade control: a deactivated prototype contributes EXACTLY zero (geometric
        absence, not a post-hoc filter) — it is never scored and never predicted."""
        if label in self.protos:
            self.protos[label].active = False

    def scores(self, sig: Signature) -> Dict[str, float]:
        active = [l for l, p in self.protos.items() if p.active]
        return {l: raw_score(sig, self.protos[l], self.cfg) for l in active}

    def predict(self, sig: Signature,
                unknown_gate: Optional[UnknownGate] = None) -> Tuple[str, Dict[str, float]]:
        sc = self.scores(sig)
        if not sc:
            return "unknown", {}
        labels = list(sc.keys())
        raws = np.array([sc[l] for l in labels], dtype=np.float64)
        # Hill-Power shrink across the class-score spectrum (kills below-floor credit).
        a, b, g, d = self.cfg.hill
        shifted = raws - raws.min()
        norm = shifted / (shifted.max() + 1e-9)
        w = np.asarray(davidian_weight(norm, a, b, g, d)).ravel()
        shrunk = dict(zip(labels, raws * (0.5 + 0.5 * w)))   # shrink, don't annihilate the top
        best = max(shrunk, key=shrunk.get)
        if unknown_gate is not None and not unknown_gate.is_known(shrunk[best]):
            return "unknown", shrunk
        return best, shrunk


# ── baselines (RULE 4 control arm; sklearn-guarded — they run on the machine) ──
def _sklearn():
    try:
        import sklearn  # noqa: F401
        return True
    except Exception:
        return False


def baseline_random(labels: List[str], n: int, seed: int = 0) -> List[str]:
    rng = np.random.RandomState(seed)
    return [labels[i] for i in rng.randint(0, len(labels), size=n)]


def baseline_color_hist_svm(train_imgs, train_y, test_imgs):
    """Colour-histogram + linear SVM (RULE 4 ablation control). None if no sklearn."""
    if not _sklearn():
        return None
    from sklearn.svm import LinearSVC

    def feat(im):
        a = np.asarray(im, dtype=np.float64)
        if a.ndim == 3:
            return np.concatenate([np.histogram(a[..., c], bins=16, range=(0, 255))[0]
                                   for c in range(min(3, a.shape[-1]))]).astype(float)
        return np.histogram(a, bins=16)[0].astype(float)

    Xtr = np.array([feat(i) for i in train_imgs]); Xte = np.array([feat(i) for i in test_imgs])
    clf = LinearSVC(max_iter=5000).fit(Xtr, train_y)
    return list(clf.predict(Xte))
