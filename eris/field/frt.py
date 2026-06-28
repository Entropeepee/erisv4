"""
Fractal Rolling Tokenizer (FRT)
================================

The FAST text-to-field injection path. No GPU required. No PDE needed.
Deterministic: same text always produces the same field state.

Pipeline:
    raw text → variable-length treelets (3-5 token windows)
    → blake2b 64-bit hash per treelet
    → bit-slice each hash into SymbolicPulse(φ, θ, τ)
    → inject into hex lattice or accumulate into field arrays

This is the REFLEXIVE pathway (System 1):
    - Instant (microseconds per treelet)
    - Deterministic (reproducible)
    - CPU-only (runs on a potato)
    - Approximate (static hash, not dynamic evolution)

The PDE is the DELIBERATIVE pathway (System 2):
    - Slower (milliseconds to seconds)
    - Physics-based (real field dynamics)
    - GPU-accelerated (CuPy on RTX 5080)
    - Precise (computed activations from field statistics)

Use FRT when:
    - Running on low-power hardware (no GPU)
    - Need instant response (real-time chat)
    - Bootstrapping lattice state before PDE takes over
    - Processing bulk corpus (3-year ChatGPT history)

Use PDE when:
    - Running on capable hardware (RTX 5080)
    - Need accurate BFECDS activations
    - Processing important/novel input worth careful analysis
    - Dreaming loop / metacognitive processing

The system can run BOTH: FRT for immediate lattice injection,
PDE in background for refined activations. When PDE finishes,
its computed BFECDS replace the FRT approximation.

Originally built for the Asus ROG G20 (GTX 970, 4GB VRAM).
Works on anything with Python 3.11+.

Copyright 2026 Terminus IP Group LLC.

Usage:
    from eris.field.frt import FractalRollingTokenizer, HashToPulseEncoder

    frt = FractalRollingTokenizer()
    treelets = frt.tokenize("The quick brown fox jumps over the lazy dog")
    # [Treelet(tokens=['The', 'quick', 'brown'], hash=0x3a7f...), ...]

    hce = HashToPulseEncoder()
    pulses = [hce.encode(t.hash_value) for t in treelets]
    # [SymbolicPulse(phi=0.42, theta=2.71, tau=0.18), ...]

    # Or all-in-one:
    from eris.field.frt import text_to_pulses, text_to_field_arrays
    pulses = text_to_pulses("Hello world")
    phi, theta = text_to_field_arrays("Hello world", size=64)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import hashlib
import math
import numpy as np
from eris.config import to_numpy, xp


@dataclass
class Treelet:
    """A variable-length token window with its rolling hash.

    In the full spaCy version, these are POS-tagged: (NP red ball).
    In the lightweight version (no spaCy dependency), they're
    simple sliding windows of 3-5 tokens.
    """
    tokens: List[str]
    hash_value: int  # 64-bit blake2b hash
    position: int    # Start position in original text

    @property
    def text(self) -> str:
        return " ".join(self.tokens)


class FractalRollingTokenizer:
    """Extract variable-length treelets from text with rolling hashes.

    Sliding window of size 3-5 tokens, each hashed with blake2b
    for a deterministic 64-bit fingerprint. The variable window
    sizes capture different scales of linguistic structure:
        3-token: local phrase fragments
        4-token: clause-level patterns
        5-token: inter-clause relationships

    This multi-scale extraction is the "fractal" part — structure
    at multiple scales simultaneously, like the BLECD framework
    operating across scales.
    """

    def __init__(self, min_window: int = 3, max_window: int = 5):
        self.min_window = min_window
        self.max_window = max_window

    def tokenize(self, text: str) -> List[Treelet]:
        """Extract treelets from text at multiple window sizes."""
        # Simple whitespace tokenization (no spaCy dependency)
        # For production: swap in spaCy POS-tagged treelets
        words = text.split()
        if len(words) < self.min_window:
            # Text too short — hash the whole thing as one treelet
            h = self._hash_tokens(words)
            return [Treelet(tokens=words, hash_value=h, position=0)]

        treelets = []
        for window_size in range(self.min_window, self.max_window + 1):
            for i in range(len(words) - window_size + 1):
                window = words[i:i + window_size]
                h = self._hash_tokens(window)
                treelets.append(Treelet(
                    tokens=window,
                    hash_value=h,
                    position=i,
                ))

        return treelets

    def _hash_tokens(self, tokens: List[str]) -> int:
        """Blake2b 64-bit hash of a token sequence.

        Deterministic: same tokens always produce the same hash.
        The 64-bit output gives us enough bits to slice into
        phi (16 bits), theta (16 bits), tau (16 bits) with
        16 bits reserved for future use.
        """
        text = " ".join(tokens).encode("utf-8")
        digest = hashlib.blake2b(text, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big")


class HashToPulseEncoder:
    """Bit-slice a 64-bit hash into a SymbolicPulse(φ, θ, τ).

    Hash bit layout:
        bits  0-15: φ (coherence amplitude) — quadratic-scaled to [0, 1]
        bits 16-31: θ (phase) — linear-scaled to [0, 2π]
        bits 32-47: τ (torsion) — linear-scaled to [-1, 1]
        bits 48-63: reserved (metadata, future use)

    The quadratic scaling on φ concentrates more values near zero
    (most things have low coherence) with fewer near one (strong
    coherence is rare). This matches the natural distribution of
    signal strength in text.
    """

    def encode(self, hash_value: int) -> Tuple[float, float, float]:
        """Convert a 64-bit hash to (phi, theta, tau).

        Returns
        -------
        phi : float in [0, 1] — coherence amplitude (quadratic-scaled)
        theta : float in [0, 2π] — phase angle
        tau : float in [-1, 1] — torsion
        """
        # Extract 16-bit fields
        phi_bits = (hash_value >> 0) & 0xFFFF
        theta_bits = (hash_value >> 16) & 0xFFFF
        tau_bits = (hash_value >> 32) & 0xFFFF

        # Quadratic scaling for phi: concentrate near zero
        phi_raw = phi_bits / 65535.0  # [0, 1]
        phi = phi_raw * phi_raw       # Quadratic: more values near 0

        # Linear scaling for theta: full circle
        theta = (theta_bits / 65535.0) * 2.0 * math.pi  # [0, 2π]

        # Signed linear for tau: symmetric around zero
        tau = (tau_bits / 65535.0) * 2.0 - 1.0  # [-1, 1]

        return phi, theta, tau

    def encode_to_dict(self, hash_value: int) -> dict:
        """Encode and return as a dict matching SymbolicPulse fields."""
        phi, theta, tau = self.encode(hash_value)
        return {"phi": phi, "theta": theta, "tau": tau}


# ─── Convenience Functions ────────────────────────────────────────────────

_frt = FractalRollingTokenizer()
_hce = HashToPulseEncoder()


def text_to_pulses(text: str) -> List[Tuple[float, float, float]]:
    """All-in-one: text → list of (phi, theta, tau) tuples.

    Each tuple represents one treelet's contribution to the field.
    Fast, deterministic, no GPU needed.
    """
    treelets = _frt.tokenize(text)
    return [_hce.encode(t.hash_value) for t in treelets]


def text_to_field_arrays(text: str, size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Convert text to 2D phi and theta field arrays via FRT.

    This is the fast alternative to FractalField.seed_from_text().
    Instead of PDE evolution, it directly stamps treelet pulses
    onto a grid using their position and hash-derived amplitudes.

    Parameters
    ----------
    text : str
        Input text.
    size : int
        Grid resolution (NxN).

    Returns
    -------
    phi : ndarray (size, size) — coherence field
    theta : ndarray (size, size) — phase field
    """
    phi = np.zeros((size, size), dtype=np.float32)
    theta = np.zeros((size, size), dtype=np.float32)

    pulses = text_to_pulses(text)
    if not pulses:
        return phi, theta

    n_pulses = len(pulses)

    for idx, (p_phi, p_theta, p_tau) in enumerate(pulses):
        # Map pulse index to grid position
        # Distribute pulses across the grid in a Hilbert-like pattern
        # (simple version: row-major with wrapping)
        grid_idx = idx % (size * size)
        gy = grid_idx // size
        gx = grid_idx % size

        # Stamp a small Gaussian centered at (gx, gy)
        radius = max(2, size // 16)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny = (gy + dy) % size
                nx = (gx + dx) % size
                dist2 = dx * dx + dy * dy
                weight = math.exp(-dist2 / max(radius * radius * 0.5, 1))
                phi[ny, nx] += p_phi * weight * 0.3
                theta[ny, nx] += p_theta * weight

    # Clamp phi to [0, 1], wrap theta to [0, 2π]
    phi = np.clip(phi, 0.0, 1.0)
    theta = theta % (2.0 * math.pi)

    # Enforce Dirichlet boundary (match PDE convention)
    phi[0, :] = 0.0
    phi[-1, :] = 0.0
    phi[:, 0] = 0.0
    phi[:, -1] = 0.0

    return phi, theta


def compute_bvec_from_frt(text: str, size: int = 64):
    """Fast approximate BFECDS from FRT (no PDE, no GPU).

    Uses the FRT field arrays to compute BFECDS via the same
    activation formulas as the PDE path. Less accurate than
    PDE-computed activations (no field evolution means no
    temporal dynamics), but instant and deterministic.

    Returns a BVec.
    """
    from eris.computation.activations import compute_bvec_from_field

    phi, theta = text_to_field_arrays(text, size=size)
    # τ: canonical vorticity ∇ρ×∇θ when ERIS_TAU_VORTICITY is on (matches the PDE + the symbol
    # contract; frontends.torsion is the unwrap-safe NumPy implementation), else the legacy
    # amplitude-Laplacian proxy. Single flag, defined once in pde.py.
    from eris.field.pde import _TAU_VORTICITY
    if _TAU_VORTICITY:
        from eris.knowledge.frontends import torsion as _torsion
        tau = _torsion(phi, theta)
    else:
        tau = (
            np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) +
            np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1) -
            4.0 * phi
        )
    # phi_prev = phi (no evolution → no temporal dynamics)
    # This means E and D will be ~0, which is correct:
    # FRT captures spatial structure, not temporal evolution
    return compute_bvec_from_field(phi, theta, tau, phi)
