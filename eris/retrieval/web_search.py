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
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.request import Request, urlopen
from urllib.parse import quote_plus
from html.parser import HTMLParser

logger = logging.getLogger("eris.web_search")


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
    """Extracts visible text from HTML, skipping scripts/styles."""

    SKIP_TAGS = {"script", "style", "noscript", "header", "nav", "footer", "svg"}

    def __init__(self):
        super().__init__()
        self._parts: list = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data):
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._parts.append(text)

    def get_text(self) -> str:
        raw = " ".join(self._parts)
        # Collapse whitespace
        raw = re.sub(r"\s+", " ", raw)
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
        req = Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            with urlopen(req, timeout=10) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Search failed for '{query}': {e}")
            return ""

    html = await asyncio.to_thread(_fetch)
    if not html:
        return []

    results = _parse_ddg_results(html)
    return results[:max_results]


# ── Content Fetching ───────────────────────────────────────

async def fetch_content(url: str, max_chars: int = 8000) -> str:
    """
    Fetch and extract readable text from a URL.

    Args:
        url: The URL to fetch.
        max_chars: Maximum characters to return (truncates).

    Returns:
        Cleaned plain text from the page.
    """
    def _fetch():
        req = Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            with urlopen(req, timeout=15) as resp:
                content_type = resp.headers.get("content-type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    return ""
                raw = resp.read(500_000)  # Cap at 500KB
                return raw.decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Fetch failed for '{url}': {e}")
            return ""

    html = await asyncio.to_thread(_fetch)
    if not html:
        return ""

    text = _extract_text_from_html(html)
    return text[:max_chars]


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
