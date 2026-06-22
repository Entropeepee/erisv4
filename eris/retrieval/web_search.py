"""
eris.retrieval.web_search — Autonomous Web Research for Eris
============================================================
Provides web search and content extraction using DuckDuckGo
(no API key required). Used by the Dream Loop when cognitive
dissonance triggers the research flag.

Architecture:
  1. search()       → returns ranked search results (title, url, snippet)
  2. fetch_content() → retrieves and cleans full page text from a URL
  3. research()      → orchestrates: search → fetch top results → summarize
"""

import asyncio
import gzip
import re
import logging
import zlib
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import quote_plus, quote
from html.parser import HTMLParser

logger = logging.getLogger("eris.web_search")


# ── Browser-like request headers (Fix A) ──────────────────────────────────
# A bare or custom User-Agent gets 403'd by many sites; present as a real
# browser. Shared by web_search and web_reader so both paths look identical.
_BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
}


def _decode_body(resp, cap: int = 500_000) -> str:
    """Read a response body, transparently inflating gzip/deflate (we advertise
    Accept-Encoding, so we must handle it)."""
    raw = resp.read(cap)
    enc = (resp.headers.get("Content-Encoding") or "").lower()
    try:
        if "gzip" in enc:
            raw = gzip.decompress(raw)
        elif "deflate" in enc:
            raw = zlib.decompress(raw)
    except Exception:
        pass  # served mislabeled / already-decoded — fall through to decode
    return raw.decode("utf-8", errors="replace")


# ── Data Types ─────────────────────────────────────────────

@dataclass
class SearchResult:
    """A single web search result."""
    title: str
    url: str
    snippet: str
    relevance: float = 1.0  # Rank-based relevance (1.0 = top result)


@dataclass
class ResearchReport:
    """The output of a research cycle."""
    query: str
    results: List[SearchResult] = field(default_factory=list)
    full_texts: List[str] = field(default_factory=list)
    synthesis: str = ""  # LLM-generated summary (filled later)
    source_urls: List[str] = field(default_factory=list)


# ── HTML Cleaning ──────────────────────────────────────────

class _TextExtractor(HTMLParser):
    """Extracts the article body from HTML, skipping scripts/styles AND chrome.

    Beyond tag-name skips, it skips any subtree whose id/class/role looks like
    navigation/boilerplate (nav, menu, sidebar, footer, header, banner, cookie,
    breadcrumb, search, masthead, the Wikipedia "Jump to content" skip-link,
    table-of-contents, …). A tag stack makes this robust to nested and unclosed
    tags — so stored memory is clean article body, not site chrome (Fix A)."""

    SKIP_TAGS = {"script", "style", "noscript", "header", "nav", "footer",
                 "svg", "form", "aside", "button"}
    _BOILER = re.compile(
        r"(?:^|[\s_\-])(nav|navbar|menu|sidebar|side-bar|footer|header|banner|"
        r"cookie|consent|breadcrumb|skip|jump|search|masthead|toc|"
        r"mw-jump|mw-navigation|mw-panel|catlinks|noprint|metadata|"
        r"advert|promo|social|share|related|comment)",
        re.IGNORECASE)

    def __init__(self):
        super().__init__()
        self._parts: list = []
        self._stack: list = []  # (tag, is_skip_region)

    def _in_skip(self) -> bool:
        return any(skip for _, skip in self._stack)

    def handle_starttag(self, tag, attrs):
        skip = tag in self.SKIP_TAGS
        if not skip and not self._in_skip():
            ad = {k: (v or "") for k, v in attrs}
            hint = " ".join((ad.get("id", ""), ad.get("class", ""), ad.get("role", "")))
            if hint.strip() and self._BOILER.search(hint):
                skip = True
        self._stack.append((tag, skip))

    def handle_startendtag(self, tag, attrs):
        pass  # self-closing (img/br/hr/…): no text, no stack change

    def handle_endtag(self, tag):
        # Pop back to the matching open tag (tolerates unclosed tags).
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                del self._stack[i:]
                return

    def handle_data(self, data):
        if not self._in_skip():
            text = data.strip()
            if text:
                self._parts.append(text)

    def get_text(self) -> str:
        raw = re.sub(r"\s+", " ", " ".join(self._parts))
        return raw.strip()


def _extract_text_from_html(html: str) -> str:
    """Clean HTML to plain text."""
    extractor = _TextExtractor()
    try:
        extractor.feed(html)
    except Exception:
        pass
    return extractor.get_text()


# ── DuckDuckGo Search ─────────────────────────────────────

_DDG_URL = "https://html.duckduckgo.com/html/?q={query}"
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


def _parse_ddg_results(html: str) -> List[SearchResult]:
    """Parse DuckDuckGo HTML search results page."""
    results = []

    # Extract result blocks: each has class="result__a" for title/link
    # and class="result__snippet" for snippet
    title_pattern = re.compile(
        r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        re.DOTALL
    )
    snippet_pattern = re.compile(
        r'class="result__snippet"[^>]*>(.*?)</(?:td|div|span)',
        re.DOTALL
    )

    titles = title_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    for i, (url, raw_title) in enumerate(titles[:10]):
        # Clean HTML tags from title and snippet
        title = re.sub(r"<[^>]+>", "", raw_title).strip()
        snippet = ""
        if i < len(snippets):
            snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()

        # DDG wraps URLs in a redirect — extract the real URL
        if "uddg=" in url:
            match = re.search(r"uddg=([^&]+)", url)
            if match:
                from urllib.parse import unquote
                url = unquote(match.group(1))

        if title and url:
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                relevance=1.0 - (i * 0.1),  # Rank decay
            ))

    return results


async def search(query: str, max_results: int = 5) -> List[SearchResult]:
    """
    Search the web using DuckDuckGo.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of SearchResult objects, ranked by relevance.
    """
    encoded = quote_plus(query)
    url = _DDG_URL.format(query=encoded)

    def _fetch():
        req = Request(url, headers=_BROWSER_HEADERS)
        try:
            with urlopen(req, timeout=10) as resp:
                return _decode_body(resp)
        except Exception as e:
            logger.warning(f"Search failed for '{query}': {e}")
            return ""

    html = await asyncio.to_thread(_fetch)
    if not html:
        return []

    results = _parse_ddg_results(html)
    return results[:max_results]


# ── Content Fetching ───────────────────────────────────────

def _direct_fetch(url: str, max_chars: int) -> str:
    """Blocking direct fetch + clean extraction. Raises HTTPError on 4xx so the
    caller can decide whether to try the reader proxy. Runs off-loop (Fix B)."""
    req = Request(url, headers=_BROWSER_HEADERS)
    with urlopen(req, timeout=15) as resp:
        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return ""
        html = _decode_body(resp)
    return _extract_text_from_html(html)[:max_chars]


def _proxy_fetch(url: str, max_chars: int) -> str:
    """Blocking fetch through the r.jina.ai reader proxy → clean markdown.
    PRIVACY: sends `url` to a third party; only used when web_reader_proxy is on."""
    proxied = "https://r.jina.ai/" + url
    req = Request(proxied, headers=_BROWSER_HEADERS)
    with urlopen(req, timeout=20) as resp:
        text = _decode_body(resp)
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


async def fetch_content(url: str, max_chars: int = 8000) -> str:
    """
    Fetch and extract readable article text from a URL.

    Uses browser headers and clean (boilerplate-stripped) extraction. On a
    403/429, if the reader proxy is enabled (CONFIG.web_reader_proxy /
    ERIS_WEB_PROXY=on), retries through r.jina.ai. Always returns a string
    (empty on failure); never raises into the dream loop. Blocking work runs
    off the event loop so the cockpit `/ws` keepalive is never starved.
    """
    from eris.config import CONFIG
    try:
        return await asyncio.to_thread(_direct_fetch, url, max_chars)
    except HTTPError as e:
        if e.code in (403, 429) and CONFIG.web_reader_proxy:
            logger.info(f"[Fetch] {e.code} on {url} -> reader proxy")
            try:
                return await asyncio.to_thread(_proxy_fetch, url, max_chars)
            except Exception as ex:
                logger.warning(f"Proxy fetch failed for '{url}': {ex}")
                return ""
        logger.warning(f"Fetch failed for '{url}': HTTP {e.code}")
        return ""
    except Exception as e:
        logger.warning(f"Fetch failed for '{url}': {e}")
        return ""


# ── Research Orchestrator ──────────────────────────────────

async def research(
    query: str,
    max_results: int = 3,
    fetch_full: bool = True,
    max_chars_per_page: int = 4000,
) -> ResearchReport:
    """
    Perform autonomous web research on a topic.

    1. Search DuckDuckGo for the query.
    2. Optionally fetch full text from top results.
    3. Return a ResearchReport for the dream loop to process.

    Args:
        query: What to research.
        max_results: How many search results to process.
        fetch_full: Whether to fetch full page content.
        max_chars_per_page: Max text per page.

    Returns:
        ResearchReport with results and extracted text.
    """
    logger.info(f"[Research] Starting: '{query}'")

    results = await search(query, max_results=max_results)
    report = ResearchReport(query=query, results=results)

    if fetch_full and results:
        # Fetch pages concurrently
        tasks = [
            fetch_content(r.url, max_chars=max_chars_per_page)
            for r in results
        ]
        texts = await asyncio.gather(*tasks, return_exceptions=True)

        for i, text in enumerate(texts):
            if isinstance(text, str) and text.strip():
                report.full_texts.append(text)
                report.source_urls.append(results[i].url)
                logger.info(
                    f"[Research] Fetched {len(text)} chars from {results[i].url}"
                )

    logger.info(
        f"[Research] Complete: {len(report.results)} results, "
        f"{len(report.full_texts)} pages fetched"
    )
    return report
