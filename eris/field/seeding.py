"""Modular semantic-seed substrate + the 2×2 ablation (Phase-2, Step 4).

Two swappable axes, so the seed and the read-out can be ablated independently:

  SeedProjector   : embedding → (phi, theta) field
      • PlaneWaveProjector   (CONTROL) — the existing coupling_angles + plane-wave construction, but
        fed the REAL semantic get_embedding(text) (bge-m3 when available, deterministic fallback
        otherwise) instead of the in-PDE hash get_bge_m3_embedding.
      • RandomFourierProjector (RUNG-B) — a FROZEN near-isometric random projection of the embedding
        into the lowest-M torus Fourier modes (independent amplitude + phase per mode) → iFFT2 → the
        SAME phi=|Psi| / theta=angle read-out. DC mode is fixed (phi mean from the +offset, not the
        embedding).

  FieldDescriptor : evolved field → signature vector
      • BVecDescriptor   (CONTROL) — the 6-number BFECDS BVec.
      • LambdaDescriptor (+λ)      — appends the perpendicular channel λ (the "sine usually
        discarded"); κ already lives in F, τ in C, so λ is the non-degenerate 7th.

The harness runs any (SeedProjector × FieldDescriptor) pairing and the control-crossed 2×2; the
diagonal is the headline, the off-diagonals attribute a gain to the seed vs the descriptor.

NOT here: the field→word decoder (the parked endgame). This module is only the modular substrate +
the ablation. Everything is deterministic and offline-testable; the real semantic numbers come when
run with bge-m3 (ERIS_EMBEDDINGS=on) on the GPU box.
"""
from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
import numpy as np

from eris.config import xp, to_gpu, to_numpy


# ─────────────────────────── λ — the perpendicular channel ───────────────────────────
def field_lambda(phi, theta, eps: float = 1e-6) -> float:
    """λ = mean( |∇φ × ∇θ| / (|∇φ|·|∇θ| + ε) ) — the normalized PERPENDICULAR coupling (the sine of
    the angle between the amplitude and phase gradients), i.e. "the sine usually discarded". κ (their
    aligned/cosine coupling) already feeds F and τ (∇ρ×∇θ vorticity) feeds C, so λ is a distinct,
    non-degenerate channel. Torus-consistent (xp.roll) and wrap-safe in θ. Returns a scalar in [0,1]."""
    from eris.field.pde import wrap_diff
    phi = xp.asarray(phi); theta = xp.asarray(theta)
    gpx = phi - xp.roll(phi, 1, axis=1); gpy = phi - xp.roll(phi, 1, axis=0)
    gtx = wrap_diff(theta, xp.roll(theta, 1, axis=1))
    gty = wrap_diff(theta, xp.roll(theta, 1, axis=0))
    cross = xp.abs(gpx * gty - gpy * gtx)                 # |∇φ × ∇θ| (2D scalar cross)
    gpm = xp.sqrt(gpx ** 2 + gpy ** 2); gtm = xp.sqrt(gtx ** 2 + gty ** 2)
    return float(to_numpy(xp.mean(cross / (gpm * gtm + eps))))


# ─────────────────────────── SeedProjector axis ───────────────────────────
class SeedProjector:
    """embedding → (phi, theta). `seed_text` fetches the real semantic embedding; `project` does the
    geometry, so the two stages are testable apart."""
    name = "abstract"

    def project(self, embedding: np.ndarray, size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def seed_text(self, text: str, size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        from eris.knowledge.embeddings import get_embedding
        emb = np.asarray(get_embedding(text), dtype=np.float64).ravel()
        return self.project(emb, size)


class PlaneWaveProjector(SeedProjector):
    """CONTROL — the live plane-wave construction (coupling_angles → Σ c² e^{i(2π(kx·x+ky·y)/N + a)}),
    fed the REAL get_embedding instead of the hash. Identical geometry to encode_text."""
    name = "planewave"

    def __init__(self, n_channels: int = 12, amp: float = 0.6, b_max: float = 1.0, dc_offset: float = 0.12):
        self.n_channels = n_channels
        self.amp = amp
        self.b_max = b_max
        self.dc_offset = dc_offset

    def project(self, embedding: np.ndarray, size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        from eris.field.pde import coupling_angles
        th_v, c2_v = coupling_angles(np.asarray(embedding, dtype=np.float64).ravel(), self.n_channels)
        y, x = np.mgrid[0:size, 0:size].astype(np.float64)
        Psi = np.zeros((size, size), dtype=complex)
        for i, (a, c2) in enumerate(zip(th_v, c2_v)):
            kx, ky = 1 + (i % 3), 1 + (i // 3)
            Psi += c2 * np.exp(1j * (2 * np.pi * (kx * x + ky * y) / size + a))
        phi = np.abs(Psi)
        phi = phi / (phi.max() + 1e-10) * self.amp + self.dc_offset
        phi = np.clip(phi, 0.02, self.b_max - 0.02)
        theta = np.angle(Psi) % (2 * np.pi)
        return phi.astype(np.float32), theta.astype(np.float32)


class RandomFourierProjector(SeedProjector):
    """RUNG-B — a FROZEN near-isometric random projection of the embedding into the lowest-M torus
    Fourier modes (M complex modes = M amplitudes + M phases), then iFFT2 to the complex field Psi,
    read out as phi=|Psi| / theta=angle. The projection matrix is generated once from a fixed seed
    and row-orthonormalized (near-isometric). The DC mode is NOT driven by the embedding — phi's mean
    is pinned to `dc_offset` afterwards (the +0.12 convention)."""
    name = "randomfourier"

    def __init__(self, n_modes: int = 64, amp: float = 0.6, b_max: float = 1.0,
                 dc_offset: float = 0.12, seed: int = 1234):
        self.n_modes = n_modes
        self.amp = amp
        self.b_max = b_max
        self.dc_offset = dc_offset
        self.seed = seed
        self._R = None          # frozen (2M × D), lazily built once D is known
        self._D = None

    def _matrix(self, d: int) -> np.ndarray:
        if self._R is None or self._D != d:
            rng = np.random.default_rng(self.seed)
            g = rng.standard_normal((2 * self.n_modes, d))
            # row-orthonormalize (near-isometric) via QR on the transpose
            q, _ = np.linalg.qr(g.T)                       # (d × 2M), orthonormal columns
            self._R = q.T                                 # (2M × d), orthonormal rows
            self._D = d
        return self._R

    @staticmethod
    def _lowest_modes(size: int, m: int) -> List[Tuple[int, int]]:
        """The m non-DC (ky, kx) grid frequencies with the smallest |k| (lowest spatial modes)."""
        freqs = np.fft.fftfreq(size) * size               # integer mode indices
        cand = []
        for iy, ky in enumerate(freqs):
            for ix, kx in enumerate(freqs):
                if iy == 0 and ix == 0:
                    continue                              # skip DC (handled by dc_offset)
                cand.append((ky * ky + kx * kx, iy, ix))
        cand.sort(key=lambda t: t[0])
        return [(iy, ix) for _, iy, ix in cand[:m]]

    def project(self, embedding: np.ndarray, size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        emb = np.asarray(embedding, dtype=np.float64).ravel()
        coeffs = self._matrix(emb.size) @ emb             # 2M: M amplitudes ‖ M phases
        m = self.n_modes
        amps = np.abs(coeffs[:m])
        phases = coeffs[m:] * np.pi                       # spread the projected scalars over phase
        spec = np.zeros((size, size), dtype=complex)
        for (iy, ix), a_mode, ph in zip(self._lowest_modes(size, m), amps, phases):
            spec[iy, ix] = a_mode * np.exp(1j * ph)
        Psi = np.fft.ifft2(spec)
        phi = np.abs(Psi)
        mx = phi.max()
        phi = (phi / mx * self.amp) if mx > 1e-12 else phi
        phi = phi - phi.mean() + self.dc_offset           # DC fixed to the offset, not the embedding
        phi = np.clip(phi, 0.02, self.b_max - 0.02)
        theta = np.angle(Psi) % (2 * np.pi)
        return phi.astype(np.float32), theta.astype(np.float32)


# ─────────────────────────── FieldDescriptor axis ───────────────────────────
class FieldDescriptor:
    """evolved FractalField → signature vector."""
    name = "abstract"

    def describe(self, field) -> np.ndarray:
        raise NotImplementedError


class BVecDescriptor(FieldDescriptor):
    """CONTROL — the 6-number BFECDS signature."""
    name = "bvec"

    def describe(self, field) -> np.ndarray:
        return field.compute_bvec().as_array().astype(np.float64)


class LambdaDescriptor(FieldDescriptor):
    """+λ — the 6 BFECDS channels plus the perpendicular λ channel (a 7-vector)."""
    name = "bvec+lambda"

    def describe(self, field) -> np.ndarray:
        bv = field.compute_bvec().as_array().astype(np.float64)
        lam = field_lambda(field.phi, field.theta)
        return np.concatenate([bv, [lam]])


# ─────────────────────────── Harness ───────────────────────────
def seed_and_evolve(projector: SeedProjector, text: str, *, size: int = 64, steps: int = 30):
    """Seed a FractalField from `text` via `projector`, evolve `steps`, return the field."""
    from eris.field.pde import FractalField
    phi, theta = projector.seed_text(text, size)
    field = FractalField(size=size)
    field.phi = to_gpu(np.asarray(phi, dtype=np.float32))
    field.theta = to_gpu(np.asarray(theta, dtype=np.float32))
    field.phi_prev = xp.copy(field.phi)
    field.theta_prev = xp.copy(field.theta)
    field.tau = field.tau * 0.0
    field.run(steps)
    return field


def signature(projector: SeedProjector, descriptor: FieldDescriptor, text: str,
              *, size: int = 64, steps: int = 30) -> np.ndarray:
    return descriptor.describe(seed_and_evolve(projector, text, size=size, steps=steps))


def _separation(sigs: np.ndarray, group_ids: Sequence[int]) -> float:
    """Surface-invariant separation: 1 − (mean within-group distance / mean between-group distance).
    Paraphrases (same group) should be CLOSE and different meanings FAR → higher is better; ~0 means
    the signature can't tell meanings apart. Cosine distance on L2-normalized signatures."""
    g = np.asarray(group_ids)
    x = np.asarray(sigs, dtype=np.float64)
    n = x.shape[0]
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    xn = x / np.where(norm < 1e-12, 1.0, norm)
    within, between = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = 1.0 - float(np.dot(xn[i], xn[j]))
            (within if g[i] == g[j] else between).append(d)
    mw = float(np.mean(within)) if within else 0.0
    mb = float(np.mean(between)) if between else 1e-9
    return 1.0 - mw / max(mb, 1e-9)


def run_pairing(projector: SeedProjector, descriptor: FieldDescriptor,
                texts: Sequence[str], group_ids: Sequence[int],
                *, size: int = 64, steps: int = 30) -> dict:
    """Run one (SeedProjector × FieldDescriptor) cell over a labelled text set."""
    sigs = np.stack([signature(projector, descriptor, t, size=size, steps=steps) for t in texts])
    return {"seed": projector.name, "descriptor": descriptor.name,
            "separation": _separation(sigs, group_ids), "signatures": sigs}


def run_ablation_2x2(texts: Sequence[str], group_ids: Sequence[int], *,
                     size: int = 64, steps: int = 30,
                     seed_control: Optional[SeedProjector] = None,
                     seed_rung_b: Optional[SeedProjector] = None,
                     desc_control: Optional[FieldDescriptor] = None,
                     desc_lambda: Optional[FieldDescriptor] = None) -> dict:
    """The control-crossed 2×2: {control, B-seed} × {control, +λ-descriptor}. Returns the separation
    of each cell. Diagonal (control/control vs B-seed/+λ) is the headline; the off-diagonals
    attribute any gain to the seed axis vs the descriptor axis."""
    sc = seed_control or PlaneWaveProjector()
    sb = seed_rung_b or RandomFourierProjector()
    dc = desc_control or BVecDescriptor()
    dl = desc_lambda or LambdaDescriptor()
    cells = {}
    for sp in (sc, sb):
        for de in (dc, dl):
            r = run_pairing(sp, de, texts, group_ids, size=size, steps=steps)
            cells[f"{sp.name} × {de.name}"] = round(r["separation"], 4)
    return {"steps": steps, "size": size, "n_texts": len(texts), "cells": cells}
