"""
eris/knowledge/quality.py
=========================
Ingestion quality gate (Eris learning loop). Keeps nav menus, cookie banners,
ad copy and off-topic link-lists out of Eris's memory, and is what populates a
dream cycle's `stored[]` (what actually passed the filter and entered memory).

Gate every passage BEFORE storing it.
"""
from __future__ import annotations

import re
import numpy as np

_BOILERPLATE = re.compile(
    r"(cookie|sign ?up|subscribe|privacy policy|all rights reserved|"
    r"terms of (service|use)|advertisement|skip to (main )?content|"
    r"grammar checker|add to cart|browser add-?on|official website of|"
    r"create your free account|accept all cookies|log ?in to continue)", re.I)


def is_useful(text: str, topic_emb=None, passage_emb=None,
              min_chars: int = 200, min_relevance: float = 0.22) -> bool:
    """True if `text` is substantive prose worth keeping.

    - too short -> reject
    - dense with nav/ad boilerplate -> reject
    - mostly punctuation/links (low alpha ratio) -> reject
    - off-topic vs the crawl topic (embedding cosine) -> reject
    """
    t = (text or "").strip()
    if len(t) < min_chars:
        return False
    if len(_BOILERPLATE.findall(t)) >= 2:
        return False
    alpha = sum(c.isalpha() or c.isspace() for c in t) / max(1, len(t))
    if alpha < 0.6:
        return False
    if topic_emb is not None and passage_emb is not None:
        a = np.asarray(topic_emb, dtype=float).ravel()
        b = np.asarray(passage_emb, dtype=float).ravel()
        if a.shape == b.shape:
            sim = float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))
            if sim < min_relevance:
                return False
    return True
