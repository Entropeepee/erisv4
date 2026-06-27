"""Zero-weight κ/λ/τ AUDIO classifier (§3) — reuses the #41 field machinery.

An audio clip becomes a coherence field (AudioFrontend: STFT → ρ,θ), summarized by the
SAME LAF signature (κ, λ) and torsion τ-stats as an image, and classified by the SAME
deterministic-prototype, two-channel-coupling, Hill-Power, SGT-gate, blade-control
machinery (ZeroWeightClassifier). Nothing here is audio-special except the frontend and
the RULE-4 ablation features — the field is modality-agnostic by design, so we REUSE
zero_weight_vision rather than fork it.

RULE 4: the thesis features are field-only (κ/λ/τ + two-channel coupling). MFCC /
log-mel / spectral-PCA are the "RGB of audio" → a SEPARATE ablation arm (sklearn-guarded),
never the headline. Torch-free: the MFCC/log-mel features are pure NumPy.
"""
from __future__ import annotations
from typing import List, Optional

import numpy as np

from eris.knowledge.frontends import AudioFrontend, audio_density_phase, torsion, _np_stft
from eris.vision.laf import laf_signature
from eris.vision.coupling import FieldDebias
# Reuse the modality-agnostic classifier core verbatim (do NOT fork).
from eris.experiments.zero_weight_vision import (
    VisionConfig as FieldConfig, Signature, ClassPrototype, build_prototype,
    raw_score, ZeroWeightClassifier, baseline_random, _sklearn,
)

# A field classifier config is modality-agnostic; alias the #41 config so the audio
# call sites read clearly (size, laf, w_field/w_kappa/w_lambda, hill, two_channel).
AudioConfig = FieldConfig


def compute_audio_signature(samples, cfg: FieldConfig,
                            debias: Optional[FieldDebias] = None) -> Signature:
    """Audio clip → field Signature (mag=√ρ, θ, κ, λ, τ-stats). Mirrors
    compute_signature but through AudioFrontend; identical downstream contract."""
    af = AudioFrontend()
    mag, theta = af.to_field(np.asarray(samples), size=cfg.size)
    mag = np.asarray(mag, dtype=np.float64)
    if debias is not None:
        mag = debias.apply(mag)
    rho = np.clip(mag ** 2, 0.0, None)
    tau = torsion(rho, theta)
    tau_stats = np.array([float(tau.mean()), float(tau.std()), float(np.abs(tau).mean())])
    kappa, lam, _ = laf_signature(mag, theta, cfg.laf)
    return Signature(mag=mag, theta=np.asarray(theta, dtype=np.float64),
                     kappa=kappa, lam=lam, tau_stats=tau_stats)


# ── RULE 4 ablation features (the "RGB of audio") — pure NumPy ───────────────
def _mel_filterbank(n_fft: int, sr: int, n_mels: int = 26) -> np.ndarray:
    """Triangular mel filterbank (n_mels × n_freq) — NumPy, no librosa."""
    n_freq = n_fft // 2 + 1
    fmax = sr / 2.0
    def hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    mels = np.linspace(hz2mel(0.0), hz2mel(fmax), n_mels + 2)
    hz = mel2hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, n_freq - 1)
    fb = np.zeros((n_mels, n_freq))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(1, c - l)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(1, r - c)
    return fb


def logmel_features(samples, sr: int = 16000, n_fft: int = 256, n_mels: int = 26):
    """Mean+std log-mel energies over frames (a fixed ablation feature vector)."""
    Z = _np_stft(samples, n_fft=n_fft, hop=n_fft // 2)
    power = np.abs(Z) ** 2                                  # (n_freq, n_frames)
    fb = _mel_filterbank(n_fft, sr, n_mels)
    mel = fb @ power                                       # (n_mels, n_frames)
    logmel = np.log(mel + 1e-9)
    return np.concatenate([logmel.mean(axis=1), logmel.std(axis=1)])


def mfcc_features(samples, sr: int = 16000, n_fft: int = 256, n_mels: int = 26,
                  n_mfcc: int = 13):
    """Mean+std MFCC (DCT-II of log-mel) over frames — NumPy, no librosa."""
    Z = _np_stft(samples, n_fft=n_fft, hop=n_fft // 2)
    power = np.abs(Z) ** 2
    fb = _mel_filterbank(n_fft, sr, n_mels)
    logmel = np.log(fb @ power + 1e-9)                     # (n_mels, n_frames)
    nm = logmel.shape[0]
    k = np.arange(nm)
    basis = np.cos(np.pi / nm * (k[:, None] + 0.5) * np.arange(n_mfcc)[None, :])  # (n_mels, n_mfcc)
    mfcc = basis.T @ logmel                                # (n_mfcc, n_frames)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def _svm_baseline(featfn, train_samples, train_y, test_samples):
    if not _sklearn():
        return None
    from sklearn.svm import LinearSVC
    Xtr = np.array([featfn(s) for s in train_samples])
    Xte = np.array([featfn(s) for s in test_samples])
    clf = LinearSVC(max_iter=5000).fit(Xtr, train_y)
    return list(clf.predict(Xte))


def baseline_mfcc_svm(train_samples, train_y, test_samples):
    """MFCC + linear SVM (RULE 4 ablation control). None if no sklearn."""
    return _svm_baseline(mfcc_features, train_samples, train_y, test_samples)


def baseline_logmel_svm(train_samples, train_y, test_samples):
    """Log-mel + linear SVM (RULE 4 ablation control). None if no sklearn."""
    return _svm_baseline(logmel_features, train_samples, train_y, test_samples)
