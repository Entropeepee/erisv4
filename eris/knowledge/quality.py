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

from eris.computation.activations import cosine

_BOILERPLATE = re.compile(
    r"(cookie|sign ?up|subscribe|privacy policy|all rights reserved|"
    r"terms of (service|use)|advertisement|skip to (main )?content|"
    r"grammar checker|add to cart|browser add-?on|official website of|"
    r"create your free account|accept all cookies|log ?in to continue)", re.I)

# Strong site-chrome markers — a SINGLE hit means the passage is navigation /
# a login wall / a homepage index, not substantive prose. Catches the GitHub
# ("you signed in with another tab", "reload to refresh your session"),
# news-site ("skip to main content", "trending:"), and publisher-homepage
# ("all subjects", "news & events") junk that slipped through before.
_NAV_STRONG = re.compile(
    r"(skip to (main )?content|you signed in|signed out|reload to refresh|"
    r"switched accounts|another tab or window|all subjects|home \| |"
    r"create your free account|accept all cookies|trending:|"
    r"news (&|＆|and) events|enable javascript|turn on javascript)", re.I)


def _cjk_ratio(t: str) -> float:
    if not t:
        return 0.0
    cjk = sum(1 for c in t if "一" <= c <= "鿿")
    return cjk / len(t)


def is_useful(text: str, topic_emb=None, passage_emb=None,
              min_chars: int = 200, min_relevance: float = 0.22) -> bool:
    """True if `text` is substantive prose worth keeping.

    - too short -> reject
    - site chrome / login walls / homepage indexes -> reject
    - dense with nav/ad boilerplate -> reject
    - mostly punctuation/links (low alpha ratio) -> reject
    - mostly non-Latin script (foreign nav for an English agent) -> reject
    - off-topic vs the crawl topic (embedding cosine) -> reject
    """
    t = (text or "").strip()
    if len(t) < min_chars:
        return False
    if _NAV_STRONG.search(t):
        return False
    if len(_BOILERPLATE.findall(t)) >= 2:
        return False
    if _cjk_ratio(t) > 0.25:
        return False
    alpha = sum(c.isalpha() or c.isspace() for c in t) / max(1, len(t))
    if alpha < 0.6:
        return False
    if topic_emb is not None and passage_emb is not None:
        if cosine(topic_emb, passage_emb) < min_relevance:
            return False
    return True
