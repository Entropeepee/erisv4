"""Multi-source fetch & parse router (Stage 1 §3) — beyond the Wikipedia reader.

Route by source type to the best available parser, always normalizing to clean
markdown + structured metadata before chunking:
  • web      → Trafilatura (main-text extraction), Playwright fallback for JS pages
  • paper_pdf→ GROBID (metadata + references / citation graph) + Docling (body,
               tables, equations, reading order)
  • local    → Docling (PDF/DOCX/PPTX) or `unstructured` for mixed folders

Every heavy parser is an OPTIONAL import: if it isn't installed, we degrade to the
existing stdlib path (web_reader.fetch_url / fetch_wikipedia / a plain file read)
so nothing breaks before the user installs them. Default-OFF via ERIS_PARSERS;
flip it on after `pip install trafilatura docling grobid-client unstructured`.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import os
import re


@dataclass
class ParsedDoc:
    markdown: str
    title: str = ""
    meta: dict = field(default_factory=dict)
    refs: List[str] = field(default_factory=list)
    kind: str = ""


_PDF = re.compile(r"\.pdf($|\?)", re.IGNORECASE)
_LOCALEXT = re.compile(r"\.(pdf|docx?|pptx?|xlsx?|html?|txt|md|json)$", re.IGNORECASE)


def classify(ref: str) -> str:
    """'web' | 'paper_pdf' | 'local_file' — the routing decision."""
    r = (ref or "").strip()
    if not r:
        return "web"
    if os.path.exists(r):
        return "local_file"
    if r.lower().startswith(("http://", "https://")):
        return "paper_pdf" if _PDF.search(r) else "web"
    if _LOCALEXT.search(r):
        return "local_file"
    return "web"


def parsers_enabled() -> bool:
    return os.environ.get("ERIS_PARSERS", "0").lower() not in ("0", "", "off", "false")


# ---- optional-parser wrappers (each degrades to None if dep missing) --------
def _trafilatura_markdown(url: str) -> Optional[str]:
    try:
        import trafilatura
    except Exception:
        return None
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        return trafilatura.extract(downloaded, output_format="markdown",
                                   include_tables=True) or None
    except Exception:
        return None


def _docling_markdown(path_or_url: str) -> Optional[str]:
    try:
        from docling.document_converter import DocumentConverter
    except Exception:
        return None
    try:
        res = DocumentConverter().convert(path_or_url)
        return res.document.export_to_markdown() or None
    except Exception:
        return None


def _grobid_meta(path: str) -> dict:
    try:
        from grobid_client.grobid_client import GrobidClient  # noqa: F401
    except Exception:
        return {}
    # Wiring point: a running GROBID service yields title/authors/abstract/refs.
    # Left as a seam (returns {} until configured) so import alone never fails.
    return {}


def _unstructured_markdown(path: str) -> Optional[str]:
    try:
        from unstructured.partition.auto import partition
    except Exception:
        return None
    try:
        els = partition(filename=path)
        return "\n\n".join(str(e) for e in els) or None
    except Exception:
        return None


# ---- the router -------------------------------------------------------------
def fetch_and_parse(ref: str) -> ParsedDoc:
    """Best-available parse for `ref`, normalized to markdown + metadata. Falls
    back to the stdlib web_reader path when richer parsers aren't installed (or
    when ERIS_PARSERS is off)."""
    kind = classify(ref)
    from eris.knowledge import web_reader

    if parsers_enabled():
        if kind == "paper_pdf":
            body = _docling_markdown(ref)
            if body:
                meta = _grobid_meta(ref)
                return ParsedDoc(markdown=body, title=meta.get("title", ""),
                                 meta=meta, refs=meta.get("refs", []), kind=kind)
        elif kind == "web":
            md = _trafilatura_markdown(ref)
            if md:
                return ParsedDoc(markdown=md, meta={"url": ref}, kind=kind)
        elif kind == "local_file":
            body = _docling_markdown(ref) or _unstructured_markdown(ref)
            if body:
                return ParsedDoc(markdown=body, title=os.path.basename(ref),
                                 meta={"path": ref}, kind=kind)

    # ---- graceful fallback: the existing stdlib paths -----------------------
    try:
        if kind == "web":
            return ParsedDoc(markdown=web_reader.fetch_url(ref),
                             meta={"url": ref}, kind=kind)
        if kind == "local_file":
            with open(ref, "r", encoding="utf-8", errors="replace") as fh:
                return ParsedDoc(markdown=fh.read(), title=os.path.basename(ref),
                                 meta={"path": ref}, kind=kind)
        # paper_pdf URL with no parser installed → best-effort text fetch
        return ParsedDoc(markdown=web_reader.fetch_url(ref), meta={"url": ref}, kind=kind)
    except Exception:
        return ParsedDoc(markdown="", meta={"ref": ref}, kind=kind)
