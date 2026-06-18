"""
Eris Echo v4 — Central Configuration
=====================================

GPU auto-detection: imports CuPy if available, falls back to NumPy.
Every module imports `xp` from here instead of importing numpy/cupy directly.

Usage in any module:
    from eris.config import xp, GPU_AVAILABLE, VRAM_CAP_GB, to_numpy

Hardware target: Alienware Aurora, Intel Ultra 9, 64GB RAM, RTX 5080 (16GB VRAM)
CuPy is the primary compute backend. PyTorch is NOT used (sm_120 Blackwell unsupported).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import os

# ─── GPU Setup ───────────────────────────────────────────────────────────
VRAM_CAP_GB: float = 13.5  # Hard cap — leave headroom below 16GB

try:
    import cupy as cp
    from cupy import fft as cupyfft

    GPU_AVAILABLE = True
    xp = cp  # "array library" — use xp.array(), xp.zeros(), etc. everywhere

    # Configure memory pool
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    def vram_used_gb() -> float:
        """Current VRAM usage in GB."""
        return mempool.used_bytes() / (1024 ** 3)

    def vram_free_gb() -> float:
        """Approximate free VRAM in GB (against our cap, not physical)."""
        return max(0.0, VRAM_CAP_GB - vram_used_gb())

    def vram_check(needed_gb: float = 0.5) -> bool:
        """Check if we have room for `needed_gb` more VRAM. Frees pool first if tight."""
        if vram_used_gb() + needed_gb > VRAM_CAP_GB:
            mempool.free_all_blocks()
        return vram_used_gb() + needed_gb <= VRAM_CAP_GB

    _device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"[Eris GPU] {_device_name} — "
          f"Total: {cp.cuda.runtime.memGetInfo()[1] / (1024**3):.1f} GB, "
          f"Cap: {VRAM_CAP_GB} GB")

except ImportError:
    GPU_AVAILABLE = False
    xp = np  # Fallback: everything runs on CPU with NumPy

    def vram_used_gb() -> float:
        return 0.0

    def vram_free_gb() -> float:
        return float("inf")

    def vram_check(needed_gb: float = 0.0) -> bool:
        return True

    print("[Eris CPU] CuPy not found — running on CPU (install cupy-cuda12x for GPU)")


def to_numpy(arr) -> np.ndarray:
    """Convert any array (CuPy or NumPy) to a NumPy array on CPU.

    Use this whenever you need to:
    - Save to disk (checkpoints, JSONL)
    - Pass to non-GPU libraries (FAISS, matplotlib, JSON serialization)
    - Print/log values
    """
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def to_gpu(arr):
    """Convert a NumPy array to CuPy (GPU). No-op if GPU unavailable."""
    if GPU_AVAILABLE:
        return cp.asarray(arr)
    return arr


# ─── System-wide Constants ────────────────────────────────────────────────

@dataclass
class ErisConfig:
    """Top-level configuration. Modify these to tune the whole system."""

    # PDE field
    field_size: int = 64             # NxN grid for FRACTAL PDE
    pde_dt: float = 0.01             # PDE timestep
    pde_steps_per_input: int = 50    # How many PDE steps per text input

    # SGT gating
    sgt_threshold_sigma: float = 2.0  # Default gate threshold (2σ)
    sgt_ema_alpha: float = 0.1        # EMA smoothing for running stats

    # Memory
    stm_capacity: int = 20            # Short-term memory: recent turns
    mtm_capacity: int = 200           # Medium-term: Ebbinghaus-decayed records
    mtm_half_life_hours: float = 168.0  # 1 week half-life for medium-term
    ltm_half_life_hours: float = 2160.0  # 90 days for long-term attractors

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval_seconds: float = 300.0  # Every 5 minutes

    # VRAM budget
    vram_cap_gb: float = VRAM_CAP_GB


# Singleton config — import and modify before system init
CONFIG = ErisConfig()
