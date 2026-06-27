"""§4: cross-modal grounding — the RULE-1 receipt. Unitary (phase-preserving) complex
Procrustes recovers a planted phase-including alignment that real-orthogonal (cosine-only)
misses; both recover a pure real rotation; shuffled pairs never improve with N; the
low-level probe separates a bouba/kiki effect from a concept effect; the image↔word arm
runs on the real #41 frontends. Offline, synthetic, deterministic."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest
import numpy as np

from eris.experiments.grounding import (
    GroundingConfig, field_descriptor, descriptors_by_class, grounding_curve,
    unitary_procrustes, real_procrustes, within_modal_ceiling, lowlevel_probe,
    transitivity,
)


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def _phase_classes(d=6, per=8, noise=0.02, seed=0, U_true=None):
    """Three classes distinguished ONLY by a global phase on a SHARED real magnitude
    pattern u: class c = u·e^{iφc}, φ = 0, 2π/3, 4π/3. Then Re = u·cos φ collides for the
    last two classes (cos 2π/3 = cos 4π/3) → cosine-only cannot tell them apart, but the
    phase (sin) can. Target = U_true · source, so an alignment is genuinely needed."""
    rng = np.random.RandomState(seed)
    u = _unit(rng.randn(d))                       # REAL shared magnitude pattern
    phis = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
    src, tgt = {}, {}
    for ci, phi in enumerate(phis):
        base = u * np.exp(1j * phi)
        S, T = [], []
        for _ in range(per):
            s = _unit(base + noise * (rng.randn(d) + 1j * rng.randn(d)))
            t = s @ U_true if U_true is not None else s
            S.append(s); T.append(_unit(t))
        src[f"c{ci}"] = S; tgt[f"c{ci}"] = T
    return src, tgt


def _diag_unitary(d, seed=1):
    rng = np.random.RandomState(seed)
    return np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, d)))


class TestGrounding(unittest.TestCase):
    def test_unitary_procrustes_recovers_planted_map(self):
        rng = np.random.RandomState(0)
        d, n = 5, 30
        A = rng.randn(n, d) + 1j * rng.randn(n, d)
        U_true = np.linalg.qr(rng.randn(d, d) + 1j * rng.randn(d, d))[0]   # random unitary
        B = A @ U_true
        U = unitary_procrustes(A, B)
        self.assertLess(float(np.linalg.norm(A @ U - B)) / np.linalg.norm(B), 1e-6)

    def test_unitary_grounds_phase_classes_cosine_misses(self):
        # THE RULE-1 RECEIPT.
        U_true = _diag_unitary(6, seed=2)
        src, tgt = _phase_classes(d=6, per=10, U_true=U_true, seed=3)
        cfg = GroundingConfig(Ns=(8,), n_repeat=8, seed=5)
        uni = grounding_curve(src, tgt, cfg, kind="unitary", channels="both")[8]["acc"]
        cos = grounding_curve(src, tgt, cfg, kind="cosine")[8]["acc"]
        self.assertGreater(uni, 0.9)               # phase-preserving fit separates all 3
        self.assertLess(cos, 0.75)                 # cosine-only confuses the phase-twins
        self.assertGreater(uni, cos + 0.2)

    def test_both_recover_a_pure_real_rotation(self):
        # When the true relationship is a REAL rotation (no phase), unitary AND cosine fit.
        rng = np.random.RandomState(7)
        d, per = 6, 10
        centers = [_unit(rng.randn(d)) for _ in range(3)]      # REAL, well-separated
        R_true = np.linalg.qr(rng.randn(d, d))[0]              # real orthogonal
        src, tgt = {}, {}
        for ci, ctr in enumerate(centers):
            S = [_unit(ctr + 0.02 * rng.randn(d)).astype(np.complex128) for _ in range(per)]
            src[f"c{ci}"] = S
            tgt[f"c{ci}"] = [_unit(np.real(s) @ R_true).astype(np.complex128) for s in S]
        cfg = GroundingConfig(Ns=(8,), n_repeat=8, seed=1)
        uni = grounding_curve(src, tgt, cfg, kind="unitary")[8]["acc"]
        cos = grounding_curve(src, tgt, cfg, kind="cosine")[8]["acc"]
        self.assertGreater(uni, 0.9)
        self.assertGreater(cos, 0.9)

    def test_shuffled_pairs_do_not_improve_with_N(self):
        U_true = _diag_unitary(6, seed=2)
        src, tgt = _phase_classes(d=6, per=10, U_true=U_true, seed=3)
        cfg = GroundingConfig(Ns=(2, 8, 32), n_repeat=8, seed=9)
        curve = grounding_curve(src, tgt, cfg, kind="unitary", shuffle=True)
        chance = 1.0 / 3
        for n in (2, 8, 32):
            self.assertLess(curve[n]["acc"], chance + 0.25)    # never grounds on noise
        # and it does NOT trend up from N=2 to N=32
        self.assertLess(curve[32]["acc"], curve[2]["acc"] + 0.25)

    def test_grounding_beats_zero_shot_baseline(self):
        # N>0 unitary grounding clears the N=0 (no-alignment) descriptor coupling.
        U_true = _diag_unitary(6, seed=2)
        src, tgt = _phase_classes(d=6, per=10, U_true=U_true, seed=3)
        cfg = GroundingConfig(Ns=(0, 16), n_repeat=8, seed=4)
        curve = grounding_curve(src, tgt, cfg, kind="unitary")
        self.assertGreater(curve[16]["acc"], curve[0]["acc"] + 0.2)

    def test_lowlevel_probe_separates_structure_from_concept(self):
        # bouba/kiki: accuracy fully explained by a low-level feature → near-zero residual.
        feat = list(np.linspace(0, 1, 12))
        acc_lowlevel = [1.0 if f > 0.5 else 0.0 for f in feat]   # purely tracks the feature
        acc_concept = [1.0] * 12                                 # concept: high regardless
        p_low = lowlevel_probe(acc_lowlevel, feat)
        p_con = lowlevel_probe(acc_concept, feat)
        self.assertGreater(abs(p_low["slope"]), abs(p_con["slope"]))     # low-level: steep
        self.assertGreater(p_con["residual_mean"], 0.9)                  # concept survives
        self.assertGreater(p_low["explained_by_lowlevel"], 0.2)

    def test_image_word_arm_runs_on_real_frontends(self):
        # The image↔word arm grounds above chance using the real #41 frontends.
        from eris.knowledge.frontends import ImageFrontend
        from eris.experiments.cross_modal import word_signature
        from eris.experiments.zero_weight_vision import VisionConfig
        vcfg = VisionConfig(size=24)
        rng = np.random.RandomState(0)

        def stripe(orient, s):
            base = np.zeros((24, 24))
            base[::3, :] = 1.0 if orient == "h" else 0.0
            base[:, ::3] = 1.0 if orient == "v" else 0.0
            return np.clip(base + 0.05 * np.random.RandomState(s).randn(24, 24), 0, 1)

        img_fields = {"h": [ImageFrontend().to_field(stripe("h", s), size=24) for s in range(6)],
                      "v": [ImageFrontend().to_field(stripe("v", s + 9), size=24) for s in range(6)]}
        words = {"h": word_signature("horizontal", vcfg), "v": word_signature("vertical", vcfg)}
        word_fields = {c: [(words[c].mag, words[c].theta)] * 6 for c in ("h", "v")}
        gi = descriptors_by_class(img_fields)
        gw = descriptors_by_class(word_fields)
        cfg = GroundingConfig(Ns=(8,), n_repeat=6, seed=0)
        acc = grounding_curve(gi, gw, cfg, kind="unitary")[8]["acc"]
        self.assertGreaterEqual(acc, 0.5)          # grounds at/above chance on real fields
        self.assertTrue(0.0 <= within_modal_ceiling(gi) <= 1.0)


if __name__ == "__main__":
    unittest.main()
