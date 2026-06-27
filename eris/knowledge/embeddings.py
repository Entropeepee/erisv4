"""
eris/knowledge/embeddings.py
============================
Semantic embeddings for memory retrieval (Remediation Tier 4.4).

The field previously seeded/retrieved from a hashed bag-of-words stub, so it
encoded *wording*, not *meaning*, and LTM.search_by_embedding() was never
meaningfully exercised. This module provides one entry point, get_embedding(),
with two backends:

  * REAL semantic model (BAAI/bge-m3 by default) when `sentence-transformers`
    is installed AND ERIS_EMBEDDINGS=on (or =model). This is the correct,
    meaning-aware path. BGE-M3 on CPU is slow per call — move to ONNX-quantized
    or GPU later, but correctness first.
  * A fast DETERMINISTIC fallback (the default): a stable hashed token-trigram
    vector, L2-normalized. It is NOT semantic, but unlike the old stub it is
    deterministic and dimension-fixed, so search_by_embedding() runs and
    same/similar wording retrieves consistently. Zero heavy dependencies, fast
    on CPU — safe to compute on every turn.

Switching on the real model:
    pip install sentence-transformers
    set ERIS_EMBEDDINGS=on            # or ERIS_EMBED_MODEL=BAAI/bge-small-en-v1.5
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Optional

import numpy as np

EMBED_DIM = int(os.environ.get("ERIS_EMBED_DIM", "256"))
_MODEL = None
_MODEL_TRIED = False


def _use_real_model() -> bool:
    """Default is AUTO: try the real semantic model, fall back to deterministic
    if sentence-transformers is not installed. Set ERIS_EMBEDDINGS=off to force
    the deterministic fallback (e.g. for fast tests)."""
    val = os.environ.get("ERIS_EMBEDDINGS", "auto").lower()
    if val in ("off", "0", "false", "none"):
        return False
    return True


def _model():
    """Lazily load the sentence-transformers model, once. Returns None if
    unavailable (falls back to the deterministic embedding)."""
    global _MODEL, _MODEL_TRIED
    if _MODEL is not None or _MODEL_TRIED:
        return _MODEL
    _MODEL_TRIED = True
    try:
        from sentence_transformers import SentenceTransformer
        name = os.environ.get("ERIS_EMBED_MODEL", "BAAI/bge-m3")
        # Default to CPU: with a ~13GB local LLM resident on a 16GB card there
        # is little VRAM left, and bge-m3 on GPU is ~2GB. Set ERIS_EMBED_DEVICE=
        # cuda to move embeddings onto the GPU if you have the headroom.
        dev = os.environ.get("ERIS_EMBED_DEVICE", "cpu").strip().lower()
        _MODEL = SentenceTransformer(name, device=(None if dev in ("auto", "") else dev))
        print(f"[embeddings] loaded semantic model: {name} (device={dev})")
    except Exception as e:
        print(f"[embeddings] semantic model unavailable ({e}); using deterministic fallback")
        _MODEL = None
    return _MODEL


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _hashed_embedding(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Deterministic hashed embedding: token + char-trigram hashing trick.

    Stable across runs and fixed-dimension, so cosine similarity is meaningful
    for same/similar surface forms. Not a substitute for real semantics — that's
    what the sentence-transformers path is for — but a correct, fast default.
    """
    vec = np.zeros(dim, dtype=np.float32)
    text = (text or "").lower()
    tokens = _TOKEN_RE.findall(text)
    feats = list(tokens)
    # char trigrams add some sub-word robustness
    joined = " ".join(tokens)
    feats += [joined[i:i + 3] for i in range(max(0, len(joined) - 2))]
    if not feats:
        return vec
    for feat in feats:
        h = int(hashlib.md5(feat.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vec[idx] += sign
    n = float(np.linalg.norm(vec))
    return vec / n if n > 1e-9 else vec


def _in_process_embedding(text: str) -> np.ndarray:
    """The local embedding: semantic model when enabled+available, else the
    deterministic hashed fallback. Always a 1-D float32 array."""
    if _use_real_model():
        m = _model()
        if m is not None:
            try:
                v = m.encode(text, normalize_embeddings=True)
                return np.asarray(v, dtype=np.float32).ravel()
            except Exception as e:
                print(f"[embeddings] encode failed ({e}); deterministic fallback")
    return _hashed_embedding(text)


# ── Provider seam (Phase 1): optional external embeddings service ──────────
# If CONFIG.embed_base_url is set, embeddings are computed by a local
# OpenAI-compatible service (e.g. an NPU/iGPU OpenArc/OVMS endpoint). Eris adds
# NO dependency and falls back to the in-process path on any error or dim
# mismatch. A small text-hash cache avoids re-embedding repeats.
from collections import OrderedDict

_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()
_CACHE_MAX = int(os.environ.get("ERIS_EMBED_CACHE", "8192"))
_PROVIDER_WARNED = False


def _warn_once(msg: str) -> None:
    global _PROVIDER_WARNED
    if not _PROVIDER_WARNED:
        print(msg)
        _PROVIDER_WARNED = True


def _cache_key(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def _cache_get(text: str):
    k = _cache_key(text)
    v = _CACHE.get(k)
    if v is not None:
        _CACHE.move_to_end(k)
    return v


def _cache_put(text: str, vec: np.ndarray) -> None:
    k = _cache_key(text)
    _CACHE[k] = vec
    _CACHE.move_to_end(k)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    """Thin HTTP seam (so tests can monkeypatch it). Synchronous; the embed
    service is local so latency is small, and batching keeps call count low."""
    import httpx
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


# A3: track whether the configured embedding provider actually WORKED on its last
# call, so is_semantic() reflects a successful probe rather than merely that a URL
# string is set (it could be down and silently falling back to hashed embeddings).
_PROVIDER_HEALTHY = None   # None = not yet attempted; True/False after a real call


def _provider_embeddings(texts):
    """POST many texts to the configured /embeddings service (OpenAI shape).
    Returns a list of float32 vectors, or None to signal 'fall back'."""
    global _PROVIDER_HEALTHY
    from eris.config import CONFIG
    base = (CONFIG.embed_base_url or "").rstrip("/")
    if not base:
        return None
    try:
        data = _post_json(
            f"{base}/embeddings",
            {"model": CONFIG.embed_model or "embedding", "input": list(texts)},
            CONFIG.accel_timeout_s)
        rows = data.get("data", [])
        vecs = [np.asarray(d["embedding"], dtype=np.float32).ravel() for d in rows]
        if len(vecs) != len(texts):
            _warn_once(f"[embeddings] provider returned {len(vecs)} of {len(texts)} "
                       f"vectors; falling back to in-process.")
            _PROVIDER_HEALTHY = False
            return None
        # Dimension guard: a different-dim model would corrupt retrieval.
        if vecs and vecs[0].shape[0] != EMBED_DIM:
            _warn_once(
                f"[embeddings] provider dim {vecs[0].shape[0]} != EMBED_DIM {EMBED_DIM}. "
                f"Falling back. Set ERIS_EMBED_DIM to match the served model AND run "
                f"reembed_memory() once — stored vectors were made by another model.")
            _PROVIDER_HEALTHY = False
            return None
        _PROVIDER_HEALTHY = True
        return vecs
    except Exception as e:
        _warn_once(f"[embeddings] provider error ({e}); falling back to in-process. "
                   "(logged once)")
        _PROVIDER_HEALTHY = False
        return None


def get_embeddings(texts):
    """Batch embedding. Uses the external provider when configured (one HTTP call
    for all cache-misses), else the in-process path. Order-preserving."""
    texts = list(texts)
    out = [None] * len(texts)
    miss_i, miss_t = [], []
    for i, t in enumerate(texts):
        c = _cache_get(t)
        if c is not None:
            out[i] = c
        else:
            miss_i.append(i)
            miss_t.append(t)
    if miss_t:
        provided = _provider_embeddings(miss_t)
        if provided is None:
            provided = [_in_process_embedding(t) for t in miss_t]
        for j, i in enumerate(miss_i):
            out[i] = provided[j]
            _cache_put(miss_t[j], provided[j])
    return out


def get_embedding(text: str) -> np.ndarray:
    """Return a normalized float32 embedding for `text` (provider or in-process)."""
    return get_embeddings([text])[0]


def is_semantic() -> bool:
    """True only if a real semantic source is ACTIVE — a working external provider
    or a loaded local model. A configured-but-down provider that has silently
    fallen back to hashed embeddings reports False (A3: honest state)."""
    from eris.config import CONFIG
    if CONFIG.embed_base_url:
        global _PROVIDER_HEALTHY
        if _PROVIDER_HEALTHY is None:
            try:                         # one real probe to learn provider health
                _provider_embeddings(["probe"])
            except Exception:
                _PROVIDER_HEALTHY = False
        return bool(_PROVIDER_HEALTHY)
    return _use_real_model() and _model() is not None


def reembed_memory(memory) -> int:
    """Maintenance: re-embed every stored memory record with the ACTIVE embedding
    source. Run once after switching embedding providers/models, since old vectors
    were produced by a different model and would otherwise mix meanings/dims.
    Returns the number of records re-embedded."""
    n = 0
    for tier in (memory.stm, memory.mtm, memory.ltm):
        recs = list(getattr(tier, "_buffer", None) or getattr(tier, "_records", []))
        if not recs:
            continue
        new = get_embeddings([r.text for r in recs])
        for rec, vec in zip(recs, new):
            rec.embedding = vec
            n += 1
        # Persist and invalidate any cached vector index so it rebuilds.
        if hasattr(tier, "_save"):
            try:
                tier._save()
            except Exception:
                pass
        if hasattr(tier, "_faiss"):
            tier._faiss = None
    return n
