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


def get_embedding(text: str) -> np.ndarray:
    """Return a normalized float32 embedding for `text`.

    Uses the semantic model when enabled+available, else the deterministic
    fallback. Always returns a 1-D float32 array.
    """
    if _use_real_model():
        m = _model()
        if m is not None:
            try:
                v = m.encode(text, normalize_embeddings=True)
                return np.asarray(v, dtype=np.float32).ravel()
            except Exception as e:
                print(f"[embeddings] encode failed ({e}); deterministic fallback")
    return _hashed_embedding(text)


def is_semantic() -> bool:
    """True if the real semantic model is active (vs the deterministic fallback)."""
    return _use_real_model() and _model() is not None
