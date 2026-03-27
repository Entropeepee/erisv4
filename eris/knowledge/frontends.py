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
        raise NotImplementedError(
            "AudioFrontend is a stub for v6.0 GVE integration. "
            "See VISION_ROADMAP.md section 6.0 for the planned pipeline."
        )


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
        raise NotImplementedError(
            "ImageFrontend is a stub for v6.0 integration. "
            "The zero-weight prototype code (cats vs dogs) was corrupted "
            "during Gemini iteration. See VISION_ROADMAP.md."
        )


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
        raise NotImplementedError(
            "SensorFrontend is a stub for robotics/IoT integration. "
            "See VISION_ROADMAP.md."
        )
