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

# ─── GPU Setup (LAZY) ──────────────────────────────────────────────────────
# CuPy import + pool config + warm-up kernel + device print are deferred into
# _ensure_gpu(), invoked on FIRST array use — so importing any Eris module
# (tests, scripts, the server) never touches the GPU and never crashes on a
# misconfigured CUDA runtime. `xp` is a transparent proxy that resolves to CuPy
# (or NumPy) on first attribute access. Mirrors the lazy-model pattern in
# embeddings.py.
VRAM_CAP_GB: float = float(os.environ.get("ERIS_VRAM_CAP_GB", "2.0"))  # Eris's own CuPy pool; keep small — a ~13GB local LLM shares the 16GB card. Override with ERIS_VRAM_CAP_GB.

_GPU_TRIED: bool = False
GPU_AVAILABLE = None          # None until first resolved; then True/False
cp = None                     # the cupy module when available, else None
_BACKEND = None               # resolved array module (cp or np)
mempool = None
pinned_mempool = None


def _ensure_gpu():
    """Resolve the array backend once (idempotent). Returns the array module."""
    global _GPU_TRIED, GPU_AVAILABLE, cp, _BACKEND, mempool, pinned_mempool
    if _GPU_TRIED:
        return _BACKEND
    _GPU_TRIED = True
    try:
        if os.environ.get("ERIS_GPU", "1").strip().lower() in ("0", "off", "false", "no"):
            raise ImportError("GPU disabled via ERIS_GPU=0")
        import cupy as _cp
        GPU_AVAILABLE = True
        cp = _cp
        _BACKEND = _cp
        mempool = _cp.get_default_memory_pool()
        pinned_mempool = _cp.get_default_pinned_memory_pool()
        # Warm the JIT once (tiny kernel) so the FIRST real field tick doesn't
        # stall while CuPy compiles. If this ever hangs, relaunch with ERIS_GPU=0.
        _ = (_cp.zeros((8, 8), dtype=_cp.float32) + 1.0).sum()
        _cp.cuda.Stream.null.synchronize()
        _name = _cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        print(f"[Eris GPU] {_name} — "
              f"Total: {_cp.cuda.runtime.memGetInfo()[1] / (1024**3):.1f} GB, "
              f"Cap: {VRAM_CAP_GB} GB")
    except Exception as _gpu_err:   # ImportError, CUDA/driver errors, JIT issues…
        GPU_AVAILABLE = False
        cp = None
        _BACKEND = np
        print(f"[Eris CPU] GPU off ({_gpu_err}); running on CPU — "
              f"install cupy-cuda13x and leave ERIS_GPU unset to use the RTX 5080")
    return _BACKEND


class _LazyXP:
    """Transparent stand-in for the array module. Forwards every attribute
    (zeros, array, float32, ndarray, linalg, fft, …) to the resolved backend,
    triggering GPU init on first use. So `from eris.config import xp` is free at
    import time; the GPU is only touched when an array op actually runs."""
    __slots__ = ()

    def __getattr__(self, name):
        return getattr(_ensure_gpu(), name)

    def __repr__(self):
        state = "cupy" if GPU_AVAILABLE else ("numpy" if _GPU_TRIED else "unresolved")
        return f"<eris.config.xp lazy → {state}>"


xp = _LazyXP()


def vram_used_gb() -> float:
    """Current VRAM usage in GB (0.0 on CPU)."""
    _ensure_gpu()
    if GPU_AVAILABLE and mempool is not None:
        return mempool.used_bytes() / (1024 ** 3)
    return 0.0


def vram_free_gb() -> float:
    """Approximate free VRAM in GB against our cap (inf on CPU)."""
    _ensure_gpu()
    if GPU_AVAILABLE:
        return max(0.0, VRAM_CAP_GB - vram_used_gb())
    return float("inf")


def _vram_cap_gb() -> float:
    """The live VRAM cap. Codex #7: read it from CONFIG.vram_cap_gb (the documented knob, settable
    at runtime / from ERIS_VRAM_CAP_GB) rather than only the import-time module constant."""
    cfg = globals().get("CONFIG")
    return float(getattr(cfg, "vram_cap_gb", VRAM_CAP_GB)) if cfg is not None else VRAM_CAP_GB


def vram_check(needed_gb: float = 0.5) -> bool:
    """Room for `needed_gb` more VRAM under the cap? Frees the pool first if tight. True on CPU."""
    _ensure_gpu()
    if not GPU_AVAILABLE:
        return True
    cap = _vram_cap_gb()
    if vram_used_gb() + needed_gb > cap and mempool is not None:
        mempool.free_all_blocks()
    return vram_used_gb() + needed_gb <= cap


def to_numpy(arr) -> np.ndarray:
    """Convert any array (CuPy or NumPy) to a NumPy array on CPU.

    Use this whenever you need to:
    - Save to disk (checkpoints, JSONL)
    - Pass to non-GPU libraries (FAISS, matplotlib, JSON serialization)
    - Print/log values
    """
    _ensure_gpu()
    if GPU_AVAILABLE and cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def to_gpu(arr):
    """Convert a NumPy array to CuPy (GPU). No-op if GPU unavailable."""
    _ensure_gpu()
    if GPU_AVAILABLE and cp is not None:
        return cp.asarray(arr)
    return arr


# ─── System-wide Constants ────────────────────────────────────────────────

@dataclass
class ErisConfig:
    """Top-level configuration. Modify these to tune the whole system."""

    # PDE field
    field_size: int = 64             # NxN grid for FRACTAL PDE
    pde_dt: float = 0.05             # PDE timestep (the live FractalField default; now actually wired)
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

    # ── Agent tools (ReAct loop) — DEFAULT OFF ───────────────────────
    # Capabilities the grounded ReAct loop (run_agent) may call. Q1 decision:
    # precise factual lookup is a TOOL the agent escalates to, NOT a per-turn cost
    # on the resonant fast path. Q2: the durable store is the built-in local one.
    agent_tool_factual_lookup: bool = False   # hybrid BM25+dense lookup over memory
    agent_tool_durable_memory: bool = False   # remember_fact / recall_facts

    # ── Web reading ──────────────────────────────────────────────────
    # Opt-in reader-proxy fallback for sites that 403/429 a normal fetch even
    # with browser headers. When ON, such URLs are re-fetched through
    # r.jina.ai, which returns clean reader markdown. PRIVACY TRADEOFF: this
    # sends the target URL to a third party. Default OFF — David's call.
    web_reader_proxy: bool = False

    # ── Accelerator services (optional local OpenAI-compatible endpoints) ──
    # Unset => current in-process behavior. Eris NEVER imports openvino; the NPU/
    # iGPU models run as separate local services (OpenArc / OpenVINO Model Server)
    # that Eris talks to over HTTP, with graceful fallback when unset/unreachable.
    embed_base_url: str = os.environ.get("ERIS_EMBED_BASE_URL", "")   # e.g. http://localhost:8013/v1
    embed_model: str = os.environ.get("ERIS_EMBED_MODEL", "")
    rerank_base_url: str = os.environ.get("ERIS_RERANK_BASE_URL", "")
    rerank_model: str = os.environ.get("ERIS_RERANK_MODEL", "")
    tts_base_url: str = os.environ.get("ERIS_TTS_BASE_URL", "")       # local iGPU TTS; else edge-tts
    tts_model: str = os.environ.get("ERIS_TTS_MODEL", "")
    stt_base_url: str = os.environ.get("ERIS_STT_BASE_URL", "")       # Whisper-on-NPU; else off
    stt_model: str = os.environ.get("ERIS_STT_MODEL", "")
    accel_timeout_s: float = float(os.environ.get("ERIS_ACCEL_TIMEOUT", "20"))

    # ── Contractor Layer: sovereignty-routed LLM gateway (contractor_gateway_layer_spec) ──
    # One control plane for NON-sensitive outcalls, behind the existing mediator. UNSET =>
    # current behavior (no gateway, direct backends only). The sovereign/IP-sensitive path
    # NEVER uses the gateway regardless of these. Mirrors the ERIS_*_BASE_URL convention.
    gateway_base_url: str = os.environ.get("ERIS_GATEWAY_BASE_URL", "")   # e.g. http://localhost:4000/v1
    gateway_api_key: str = os.environ.get("ERIS_GATEWAY_API_KEY", "sk-litellm-local")
    # Model-group names the gateway exposes (LiteLLM config.yaml). Cost-tiered.
    tier_free: str = os.environ.get("ERIS_TIER_FREE", "free-pool")        # bulk reasoning
    tier_cheap: str = os.environ.get("ERIS_TIER_CHEAP", "cheap-paid")     # overflow
    tier_synth: str = os.environ.get("ERIS_TIER_SYNTH", "synth")          # frontier synthesis
    gateway_timeout_s: float = float(os.environ.get("ERIS_GATEWAY_TIMEOUT", "120"))
    # Hive synthesis escalation (Stage 2) — DEFAULT OFF. The A/B remains its gate; flip on
    # only after the numbers confirm the hive earns a frontier-tier synthesis.
    hive_synth_cloud: bool = os.environ.get("ERIS_HIVE_SYNTH_CLOUD", "0").strip().lower() in ("1", "on", "true", "yes")
    # Optional sandboxed Hermes contractor (non-IP autonomous research) — DEFAULT OFF.
    hermes_base_url: str = os.environ.get("ERIS_HERMES_BASE_URL", "")     # e.g. http://127.0.0.1:8642
    hermes_api_key: str = os.environ.get("ERIS_HERMES_API_KEY", "")

    # Ingestion chunker: "structured" (section/paragraph-aware + contextual
    # headers — the higher-recall default) or "legacy" (naive fixed-char).
    chunker: str = os.environ.get("ERIS_CHUNKER", "structured")
    chunk_target_chars: int = int(os.environ.get("ERIS_CHUNK_CHARS", "2000"))
    chunk_overlap_chars: int = int(os.environ.get("ERIS_CHUNK_OVERLAP", "200"))
    # ── DualPath shadow comparison (resonance vs hybrid RAG) — DEFAULT OFF ──
    # Generic ERIS_<SUBSYS>_MODE convention. retrieval_mode default
    # "traditional_only" leaves process() byte-for-byte unchanged.
    retrieval_mode: str = os.environ.get("ERIS_RETRIEVAL_MODE", "traditional_only")
    arbiter_llm: bool = os.environ.get("ERIS_ARBITER_LLM", "0").strip().lower() in ("1", "on", "true", "yes")
    dual_rerank: bool = os.environ.get("ERIS_DUAL_RERANK", "0").strip().lower() in ("1", "on", "true", "yes")
    # dual_log: on by default WHEN a shadow/novel mode is active, else off.
    dual_log: bool = (os.environ.get("ERIS_DUAL_LOG",
                      "1" if os.environ.get("ERIS_RETRIEVAL_MODE", "traditional_only") != "traditional_only"
                      else "0").strip().lower() in ("1", "on", "true", "yes"))
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

# ERIS_AGENT_TOOLS = "off" (default) | "on". Enables the factual-lookup and
# durable-memory tools in the ReAct loop's default tool set.
_agent_env = os.environ.get("ERIS_AGENT_TOOLS", "off").strip().lower()
if _agent_env in ("on", "1", "true", "yes"):
    CONFIG.agent_tool_factual_lookup = True
    CONFIG.agent_tool_durable_memory = True

# ERIS_WEB_PROXY = "off" (default) | "on". Enables the r.jina.ai reader-proxy
# fallback for bot-blocked pages (sends the URL to a third party — see config).
_webproxy_env = os.environ.get("ERIS_WEB_PROXY", "off").strip().lower()
if _webproxy_env in ("on", "1", "true", "yes"):
    CONFIG.web_reader_proxy = True
