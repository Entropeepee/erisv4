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
VRAM_CAP_GB: float = float(os.environ.get("ERIS_VRAM_CAP_GB", "2.0"))  # Eris's own CuPy pool; keep small — a ~13GB local LLM shares the 16GB card. Override with ERIS_VRAM_CAP_GB.

# GPU is ON by default. Set ERIS_GPU=0 to force CPU (use this if CuPy ever
# JIT-hangs on your driver — it reverts cleanly to NumPy).
try:
    if os.environ.get("ERIS_GPU", "1").strip().lower() in ("0", "off", "false", "no"):
        raise ImportError("GPU disabled via ERIS_GPU=0")
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

    # Warm the JIT once (tiny kernel) so the FIRST real turn doesn't stall
    # while CuPy compiles. If this ever hangs, relaunch with ERIS_GPU=0.
    _ = (cp.zeros((8, 8), dtype=cp.float32) + 1.0).sum()
    cp.cuda.Stream.null.synchronize()

    _device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"[Eris GPU] {_device_name} — "
          f"Total: {cp.cuda.runtime.memGetInfo()[1] / (1024**3):.1f} GB, "
          f"Cap: {VRAM_CAP_GB} GB")

except Exception as _gpu_err:  # ImportError, CUDA/driver errors, JIT issues…
    GPU_AVAILABLE = False
    xp = np  # Fallback: everything runs on CPU with NumPy

    def vram_used_gb() -> float:
        return 0.0

    def vram_free_gb() -> float:
        return float("inf")

    def vram_check(needed_gb: float = 0.0) -> bool:
        return True

    print(f"[Eris CPU] GPU off ({_gpu_err}); running on CPU — "
          f"install cupy-cuda13x and leave ERIS_GPU unset to use the RTX 5080")


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

    # ── Orchestration (CIP cross-stage gates) — DEFAULT OFF ──────────
    # See ERIS_ORCHESTRATION_REMEDIATION.md. Every gate defaults to the current
    # behavior; with `orchestration_enabled=False` the pipeline is byte-for-byte
    # the old path. Gates land tier-by-tier — until a tier ships, its flag is
    # inert (nothing reads it), so these are safe to carry from Tier 0 onward.
    orchestration_enabled: bool = False   # master kill switch
    gate_field_depth: bool = False        # Tier 2 — field-evolution depth gate
    gate_response_field: bool = False     # Tier 3 — response-field warm-start
    gate_router: bool = False             # Tier 4 — formalized local↔cloud router
    gate_failure_reports: bool = False    # Tier 5 — reports → dream queue
    use_beta_star: bool = False           # Tier 6 — β-star bridge in shrinkage
    orch_k: float = 2.5                   # shared gate threshold (σ)
    orch_min_field_steps: int = 8         # protected floor for the field gate
    orch_answer_tol: float = 0.05         # bvec-distance fidelity tolerance

    # ── Test-time compute (TTC) — DEFAULT OFF ────────────────────────
    # "Smarter without a bigger model": sample several responses and return the
    # consensus (medoid), with the shared criticality monitor stopping early once
    # the answer has CONVERGED (more samples won't change it). This is the patent's
    # discipline applied where a convergent signal genuinely exists — vote-stability
    # across samples — unlike the field gates. Costs N× LLM calls when on, so it's
    # OFF by default and meant for hard/important turns.
    ttc_self_consistency: bool = False    # sample N + return consensus
    ttc_min_samples: int = 3              # never fewer than this when on
    ttc_max_samples: int = 8              # never more than this
    ttc_temperature: float = 0.7          # sampling temperature for diversity
    ttc_budget_forcing: bool = False      # force a min reasoning budget (needs a
    ttc_min_thinking_tokens: int = 0      #   thinking-capable model)
    ttc_max_extensions: int = 2           # how many "Wait" continuations to allow
    orch_resp_blend: float = 0.7          # Tier 3 warm-reseed: new-text weight (1.0 = cold)


# Singleton config — import and modify before system init
CONFIG = ErisConfig()

# ERIS_ORCHESTRATION = "off" (default) | "on". "on" flips the master switch and
# enables ONLY the gate the benchmark proved fidelity-safe — the formalized
# router (Tier 4). The two field-based gates (field_depth, response_field)
# regress the answer on this engine (see ORCHESTRATION_FINDINGS.md), so they are
# NOT auto-enabled; turn them on explicitly only if a future engine change makes
# them pass the benchmark. A misbehaving gate can thus be killed in production
# with zero code surgery, and "on" never silently degrades the answer.
_orch_env = os.environ.get("ERIS_ORCHESTRATION", "off").strip().lower()
if _orch_env in ("on", "1", "true", "yes"):
    CONFIG.orchestration_enabled = True
    CONFIG.gate_router = True

# ERIS_TTC = "off" (default) | "on". Enables self-consistency with the adaptive
# criticality early-stop. Independent of ERIS_ORCHESTRATION (it only needs the
# criticality monitor, which is always available).
_ttc_env = os.environ.get("ERIS_TTC", "off").strip().lower()
if _ttc_env in ("on", "1", "true", "yes"):
    CONFIG.ttc_self_consistency = True
