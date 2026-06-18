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
import scipy.signal
import scipy.fft

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


class AudioFrontend(ModalityFrontend):
    """Audio → field state. STUB for v6.0 GVE integration.

    Planned pipeline:
        Audio waveform → spectrogram (STFT)
        → formant extraction (F1, F2, F3 frequencies)
        → map formants to BLECD domains:
            F1 (mouth openness) → Boundary (spatial extent)
            F2 (tongue position) → Feedback (articulatory control)
            F3 (lip rounding) → Saturation (spectral density)
            Spectral tilt → Decay (energy dissipation rate)
            Onset transients → Criticality (abrupt transitions)
            Harmonic richness → Emergence (structure from overtones)
        → seed φ field from spectral energy distribution
        → seed θ field from phase structure (instantaneous frequency)

    From the CFC conversation: "Sound waves have ∇ρ (density gradient)
    and ∇θ (phase gradient). You can literally speak torsion into existence."
    """

    def to_field(self, data: Any, size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        # Ensure we have a 1D numpy array
        if not isinstance(data, np.ndarray):
            data = np.random.randn(44100)  # Mock signal if not properly passed
        data = data.flatten()
        
        # Extract spectrogram (STFT)
        f, t, Zxx = scipy.signal.stft(data, nperseg=min(256, len(data)))
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        # Resize to (size, size) field
        from scipy.ndimage import zoom
        zoom_factors = (size / mag.shape[0], size / mag.shape[1]) if mag.shape[1] > 0 else (1, 1)
        
        phi = zoom(mag, zoom_factors, mode='nearest') if mag.shape[1] > 0 else np.zeros((size, size))
        theta = zoom(phase, zoom_factors, mode='nearest') if phase.shape[1] > 0 else np.zeros((size, size))
        
        # Normalize phi to [0, 1] amplitude and theta to [0, 2pi]
        phi = np.clip(phi / (np.max(phi) + 1e-10), 0.0, 1.0).astype(np.float32)
        theta = ((theta + np.pi) % (2 * np.pi)).astype(np.float32)
        
        return phi, theta


class ImageFrontend(ModalityFrontend):
    """Image → field state. STUB for v6.0 integration.

    Planned pipeline:
        Image → grayscale + edge detection
        → 2D FFT → spatial frequency spectrum
        → map to BLECD:
            Low frequencies → Boundary (large-scale structure)
            High frequencies → Emergence (fine detail/novelty)
            Edge density → Criticality (transitions between regions)
            Texture entropy → Decay (information dissipation)
            Symmetry score → Feedback (self-similar patterns)
            Pixel saturation → Saturation (intensity limits)
        → φ field = amplitude of spatial frequency components
        → θ field = phase of spatial frequency components

    The zero-weight prototype used a version of this for cats vs dogs:
    70-80% accuracy on a tiny dataset, no learned weights, first pass.
    That prototype proved the BLECD field analysis extracts classifiable
    structure from images without any neural network training.
    """

    def to_field(self, data: Any, size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(data, np.ndarray):
            # Mock 2D image data if not provided correctly
            y, x = np.ogrid[:size, :size]
            data = np.sin(x/5) * np.cos(y/5)
            
        # 2D FFT to extract spatial frequency spectrum
        fft_data = scipy.fft.fft2(data)
        fft_shifted = scipy.fft.fftshift(fft_data)
        
        mag = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)
        
        from scipy.ndimage import zoom
        zoom_factors = (size / mag.shape[0], size / mag.shape[1])
        
        phi = zoom(mag, zoom_factors, mode='nearest')
        theta = zoom(phase, zoom_factors, mode='nearest')
        
        phi = np.clip(phi / (np.max(phi) + 1e-10), 0.0, 1.0).astype(np.float32)
        theta = ((theta + np.pi) % (2 * np.pi)).astype(np.float32)
        
        return phi, theta


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
        analytic_signal = scipy.signal.hilbert(field_data, axis=1)
        
        phi = np.abs(analytic_signal)
        theta = np.angle(analytic_signal)
        
        phi = np.clip(phi / (np.max(phi) + 1e-10), 0.0, 1.0).astype(np.float32)
        theta = ((theta + np.pi) % (2 * np.pi)).astype(np.float32)
        
        return phi, theta
