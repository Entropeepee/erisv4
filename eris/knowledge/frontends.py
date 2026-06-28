"""
Modality Frontends — Any Input → BLECD Field State
=====================================================

The interface contract for translating any modality into the
universal field representation. Once data is in the field, the
PDE doesn't know or care whether it started as text, sound,
image, or sensor voltage. It just evolves under the same physics.

Current implementations:
    TextFrontend:   FRT (reflexive) + PDE (deliberative) — COMPLETE
    AudioFrontend:  Stub — spectrogram → formant → field
    ImageFrontend:  Stub — spatial frequency → field
    SensorFrontend: Stub — time series → BLECD domain mapping → field

The 15D canonical state vector (per the GVE conversation):
    6 BFECDS + φ + θ + τ = 9D minimum
    + 6 additional dimensions TBD (frequency bands? temporal windows?)

See VISION_ROADMAP.md for the full multimodal progression.

Usage:
    from eris.knowledge.frontends import TextFrontend

    frontend = TextFrontend(field_size=64)
    phi, theta = frontend.to_field("Hello world")
    bvec = frontend.to_bvec("Hello world")
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional
import numpy as np
# scipy is imported lazily inside the audio/sensor stubs that use it, so the
# image and text frontends (pure NumPy) import even where scipy isn't installed.

from eris.computation.activations import BVec


class ModalityFrontend(ABC):
    """Abstract base: any modality → field state.

    All modalities produce the SAME output format:
        (phi, theta) — 2D field arrays, same size, same physics.

    Once in the field, the FRACTAL PDE processes everything identically.
    Resonance, interference, conservation laws — all modality-agnostic.
    """

    @abstractmethod
    def to_field(self, data: Any, size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """Convert input data to (phi, theta) field arrays.

        Returns
        -------
        phi : ndarray (size, size) — coherence amplitude field
        theta : ndarray (size, size) — phase field
        """
        ...

    def to_bvec(self, data: Any, size: int = 64, pde_steps: int = 50) -> BVec:
        """Convert input to computed BFECDS via field evolution.

        Default: seed field from to_field(), evolve PDE, compute BVec.
        Override if a modality has a faster path.
        """
        from eris.field.pde import FractalField
        from eris.config import to_gpu

        phi, theta = self.to_field(data, size=size)
        field = FractalField(size=size)
        field.phi = to_gpu(phi)
        field.theta = to_gpu(theta)
        field.phi_prev = field.phi.copy()
        field.run(pde_steps)
        return field.compute_bvec()

    def to_field_evolved(self, data: Any, size: int = 64, pde_steps: int = 50,
                         settle: bool = True):
        """The EVOLVED (phi, theta) attractor — same seed→run pass as to_bvec(), but returns the
        field itself instead of measuring it to a BVec. This is what field_resonance_2d ranks on:
        the signed phase θ carries the torsion (λ) channel that the 6-vector bvec coarse-grains
        away. Returns (phi, theta) as CPU float32 arrays.

        `settle=True` (Stage-2 convergence early-stop): stop once the field reaches its attractor
        instead of always running the full `pde_steps` — the field is only being measured for
        resonance, so more steps past settling don't change the ranking. A min-steps floor keeps
        it from under-evolving."""
        from eris.field.pde import FractalField
        from eris.config import to_gpu, to_numpy

        phi, theta = self.to_field(data, size=size)
        field = FractalField(size=size)
        field.phi = to_gpu(phi)
        field.theta = to_gpu(theta)
        field.phi_prev = field.phi.copy()
        if settle:
            field.run_settled(max_steps=pde_steps)
        else:
            field.run(pde_steps)
        return (to_numpy(field.phi).astype(np.float32).copy(),
                to_numpy(field.theta).astype(np.float32).copy())


class TextFrontend(ModalityFrontend):
    """Text → field state. Already fully implemented.

    Two paths (dual-process):
        FRT (reflexive): blake2b treelets → bit-slice → field. Instant, CPU.
        PDE (deliberative): character stats → seed → evolve. Precise, GPU.
    """

    def __init__(self, use_frt: bool = False):
        self.use_frt = use_frt

    def to_field(self, data: str, size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_frt:
            from eris.field.frt import text_to_field_arrays
            return text_to_field_arrays(data, size=size)
        else:
            from eris.field.pde import FractalField
            from eris.config import to_numpy
            field = FractalField(size=size)
            field.seed_from_text(data)
            phi = to_numpy(field.phi).copy()
            theta = to_numpy(field.theta).copy()
            return phi.astype(np.float32), theta.astype(np.float32)


def _np_stft(samples: np.ndarray, n_fft: int = 256, hop: int = 128) -> np.ndarray:
    """Torch-free, scipy-free STFT (np.fft.rfft over Hann-windowed frames).
    Returns the complex spectrogram (n_freq × n_frames)."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    win = np.hanning(n_fft)
    n_frames = 1 + (x.size - n_fft) // hop
    frames = np.stack([x[i * hop:i * hop + n_fft] * win for i in range(n_frames)], axis=1)
    return np.fft.rfft(frames, axis=0)            # (n_fft//2+1, n_frames), complex


def audio_density_phase(samples: np.ndarray, n_fft: int = 256, hop: int = 128,
                        size: Optional[int] = None):
    """ρ, θ for audio per the canonical glossary: ρ = |STFT| min-max to [0,1] (energy
    at each time–freq point), θ = ∠STFT (local phase). The acoustic analog of the
    image gradient-spinor; same (ρ, θ) contract, so τ/κ/λ and coupling apply identically.

    When `size` is given, ρ=|STFT| and θ=∠STFT are resized to size×size and ρ is min-max
    normalized AFTER the resize, so its max is exactly 1.0 — matching ImageFrontend (which
    derives ρ,θ from the already-resized gray). Normalizing AFTER resize is the real
    commensurability fix (before-resize normalization left audio's max ≈ 0.9, not 1.0).
    θ is resized componentwise; the bilinear-on-a-wrapped-angle caveat is immaterial here
    (STFT phase is effectively noise) and, measured, the plain resize preserves class
    structure better than a cos/sin angle resize — so we keep it simple and faithful."""
    Z = _np_stft(samples, n_fft=n_fft, hop=hop)
    rho = np.abs(Z)
    theta = np.angle(Z)
    if size is not None:
        rho = _resize(rho, size)
        theta = _resize(theta, size)
    lo, hi = float(rho.min()), float(rho.max())
    rho = (rho - lo) / (hi - lo + 1e-9)           # min-max AFTER resize → max 1.0, commensurable
    return rho, theta


class AudioFrontend(ModalityFrontend):
    """Audio → coherence field, ZERO learned weights (RULE 1), torch-free.

    Pipeline (the acoustic sibling of ImageFrontend's gradient-spinor):
        waveform → NumPy STFT → ρ = |STFT| (min-max [0,1]) , θ = ∠STFT
        → resize the (freq × frame) grid to size×size with the SHARED normalization
          (commensurability — load-bearing: every modality on the same grid/scale or
          no cross-modal comparison is valid)
        → return (mag = √ρ, θ) matching the (phi, theta) field contract, so the #41
          two-channel coupling, τ = ∇ρ×∇θ, and LAF (κ, λ) operate on it IDENTICALLY to
          an image or word field.

    From the CFC conversation: "Sound waves have ∇ρ (density gradient) and ∇θ (phase
    gradient). You can literally speak torsion into existence." The STFT makes both
    explicit on the time–frequency plane.
    """

    def __init__(self, n_fft: int = 256, hop: int = 128):
        self.n_fft = n_fft
        self.hop = hop

    def to_field(self, data: Any, size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(data, np.ndarray):
            # deterministic mock tone (no RNG — keeps tests reproducible)
            t = np.linspace(0, 1, 16000, endpoint=False)
            data = np.sin(2 * np.pi * 220.0 * t)
        # resize the COMPLEX spectrogram inside audio_density_phase (no angle-wrap), with
        # ρ normalized AFTER resize → max 1.0, commensurable with the image field.
        rho, theta = audio_density_phase(data, n_fft=self.n_fft, hop=self.hop, size=size)
        mag = np.sqrt(np.clip(rho, 0.0, 1.0))        # phi = √ρ per the field API
        return mag.astype(np.float32), theta.astype(np.float32)


def _to_gray(data: np.ndarray) -> np.ndarray:
    """Any image array → float grayscale in [0,1]."""
    a = np.asarray(data, dtype=np.float64)
    if a.ndim == 3:
        a = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
             if a.shape[-1] >= 3 else a.mean(axis=-1))
    if a.size and a.max() > 1.5:        # looks like 0-255
        a = a / 255.0
    return a


def _resize(gray: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize to (size, size) with pure NumPy (torch-free)."""
    H, W = gray.shape
    if (H, W) == (size, size):
        return gray
    yi = np.linspace(0, H - 1, size)
    xi = np.linspace(0, W - 1, size)
    y0 = np.floor(yi).astype(int); x0 = np.floor(xi).astype(int)
    y1 = np.minimum(y0 + 1, H - 1); x1 = np.minimum(x0 + 1, W - 1)
    wy = (yi - y0)[:, None]; wx = (xi - x0)[None, :]
    top = gray[np.ix_(y0, x0)] * (1 - wx) + gray[np.ix_(y0, x1)] * wx
    bot = gray[np.ix_(y1, x0)] * (1 - wx) + gray[np.ix_(y1, x1)] * wx
    return top * (1 - wy) + bot * wy


def image_density_phase(gray: np.ndarray):
    """ρ, θ per the canonical glossary: ρ = 0.65·gray + 0.35·‖∇gray‖ (min-max to
    [0,1]); θ = arctan2(∂gray/∂y, ∂gray/∂x). The spatial-gradient spinor, NOT FFT."""
    gy, gx = np.gradient(np.asarray(gray, dtype=np.float64))
    grad_mag = np.hypot(gx, gy)
    rho = 0.65 * gray + 0.35 * grad_mag
    lo, hi = float(rho.min()), float(rho.max())
    rho = (rho - lo) / (hi - lo + 1e-9)
    theta = np.arctan2(gy, gx)
    return rho, theta


def torsion(rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """τ = ∇ρ × ∇θ on UNWRAPPED θ (RULE 3) — the z-component of the cross product
    (∂ρ/∂x)(∂θ/∂y) − (∂ρ/∂y)(∂θ/∂x). NOT the Laplacian. tanh-stabilized.

    This is where the density and phase landscapes twist relative to each other —
    true structural coupling/vorticity. erisv4 found this gives ~28× better class
    discrimination than the Laplacian τ the Chimera prototype used."""
    th = np.unwrap(np.unwrap(np.asarray(theta, dtype=np.float64), axis=0), axis=1)
    ry, rx = np.gradient(np.asarray(rho, dtype=np.float64))   # (∂/∂y, ∂/∂x)
    ty, tx = np.gradient(th)
    tau = rx * ty - ry * tx                                   # ∇ρ × ∇θ
    return np.tanh(tau)


class ImageFrontend(ModalityFrontend):
    """Image → coherence field, ZERO learned weights (RULE 1).

    Pipeline (v2, Chimera-aligned spatial gradient spinor, NOT FFT):
        image → grayscale → ρ (density) , θ (phase) per glossary
        → return (mag = √ρ, θ) matching the (phi, theta) field contract so the
          two-channel coupling and LAF signature operate on it identically to a
          word field. τ = ∇ρ×∇θ is available via `torsion(rho, theta)`.

    The 2025 prototype proved BLECD field analysis extracts classifiable structure
    from images with no neural-network training (~70-80% cats-vs-dogs, first pass).
    """

    def to_field(self, data: Any, size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(data, np.ndarray):
            y, x = np.ogrid[:size, :size]
            data = np.sin(x / 5) * np.cos(y / 5)
        gray = _resize(_to_gray(data), size)
        rho, theta = image_density_phase(gray)
        mag = np.sqrt(np.clip(rho, 0.0, 1.0))        # phi = √ρ per the field API
        return mag.astype(np.float32), theta.astype(np.float32)


class SensorFrontend(ModalityFrontend):
    """Sensor data → field state. STUB for robotics/IoT integration.

    Planned pipeline:
        Time series → sliding window → compute per-window:
            Mean → Saturation (current level relative to range)
            Variance → Emergence (novelty of fluctuations)
            Autocorrelation → Feedback (self-reinforcing patterns)
            Rate of change → Criticality (threshold crossings)
            Decay rate → Decay (how fast signal fades)
            Range utilization → Boundary (how much of sensor range used)
        → map to 2D field via Hilbert curve or spatial arrangement
        → φ field from signal magnitude
        → θ field from signal phase (Hilbert transform)
    """

    def to_field(self, data: Any, size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(data, np.ndarray):
            # Mock sensor time-series data
            data = np.cumsum(np.random.randn(size * size))
            
        # Flatten and truncate/pad to size*size
        data = data.flatten()
        if len(data) < size * size:
            data = np.pad(data, (0, size * size - len(data)), 'edge')
        else:
            data = data[:size * size]
            
        # Reshape to 2D field using a geometric wrap
        field_data = data.reshape((size, size))
        
        # Hilbert transform to get phase
        import scipy.signal
        analytic_signal = scipy.signal.hilbert(field_data, axis=1)
        
        phi = np.abs(analytic_signal)
        theta = np.angle(analytic_signal)
        
        phi = np.clip(phi / (np.max(phi) + 1e-10), 0.0, 1.0).astype(np.float32)
        theta = ((theta + np.pi) % (2 * np.pi)).astype(np.float32)
        
        return phi, theta
