"""
eris/knowledge/web_reader.py
============================
Reading frontend (Remediation Tier 4.5). Turns an article (Wikipedia or any URL)
into knowledge using DUAL-TRACK ingestion (VISION_ROADMAP 4.1):

    text -> chunk -> (a) embedding -> LTM (retrievable text/semantics)
                  -> (b) PDE field -> store phi/theta snapshot + BFECDS (attractor)

So every chunk becomes BOTH retrievable text AND a resonant attractor in the
field — the substrate for the grokking experiment (Tier 5): Eris builds a
library of attractors, and new input is "understood" by which attractors it
resonates with (see eris/retrieval/field_interference.py).

Reconciled to the real APIs on this codebase:
  * KnowledgeExtractor.extract_text(text, title=...) returns a LIST of
    ErisDescriptor (one per chunk); we use the first descriptor per chunk.
  * ErisDescriptor carries .phi_snapshot / .theta_snapshot / .bvec / .sha256.
  * MemorySystem.ltm.store(MemoryRecord(...)) is the LTM write path.
  * Embeddings come from eris.knowledge.embeddings.get_embedding by default.

Uses only the standard library for fetching (urllib), consistent with
eris.retrieval.web_search — no extra dependencies.

Test on the box (Wikipedia is not reachable from every sandbox):
    python -c "from eris.knowledge.web_reader import fetch_wikipedia; print(fetch_wikipedia('Kuramoto model')[:300])"
"""
from __future__ import annotations

import json
import re
from typing import Callable, List, Optional
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from eris.knowledge.extractor import KnowledgeExtractor
from eris.computation.activations import BVec
# Share the browser headers + clean extractor with the dream path so both look
# like a real browser (fixes 403s) and both store clean body text (Fix A).
from eris.retrieval.web_search import (
    _BROWSER_HEADERS, _decode_body, _extract_text_from_html,
)

_UA = _BROWSER_HEADERS


# ----------------------------------------------------------------------------- fetch
def fetch_wikipedia(title: str, lang: str = "en") -> str:
    """Plain-text extract of a Wikipedia article. No API key (stdlib only)."""
    params = urlencode({
        "action": "query", "prop": "extracts", "explaintext": "1",
        "redirects": "1", "format": "json", "titles": title,
    })
    url = f"https://{lang}.wikipedia.org/w/api.php?{params}"
    req = Request(url, headers=_UA)
    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return ""
    return next(iter(pages.values())).get("extract", "")


def fetch_url(url: str) -> str:
    """HTML -> clean article text for arbitrary pages (stdlib only). Uses the
    shared boilerplate-stripping extractor so stored text has no nav/footer/
    "Jump to content" chrome — same cleanliness as the dream path."""
    req = Request(url, headers=_BROWSER_HEADERS)
    with urlopen(req, timeout=30) as resp:
        html = _decode_body(resp, cap=2_000_000)
    return _extract_text_from_html(html)


# ----------------------------------------------------------------------------- chunk
def chunk(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + max_chars])
        i += max(1, max_chars - overlap)
    return out


def _default_embed(text: str):
    from eris.knowledge.embeddings import get_embedding
    return get_embedding(text)


# ----------------------------------------------------------------------------- ingest
def ingest_text(text: str, *, title: str,
                extractor: KnowledgeExtractor,
                memory=None,
                embed_fn: Optional[Callable[[str], object]] = "default") -> int:
    """Dual-track ingest of a block of text. Returns number of chunks stored.

    For each chunk: run the PDE (via the extractor) to get a field snapshot +
    BFECDS, compute a semantic embedding, and store ONE field-primary
    MemoryRecord in LTM carrying both.
    """
    from eris.memory.tiers import MemoryRecord
    if embed_fn == "default":
        embed_fn = _default_embed

    chunks = chunk(text)
    for j, ch in enumerate(chunks):
        # (b) FIELD TRACK — extractor runs the PDE and returns descriptor(s)
        #     carrying phi/theta field snapshot + BFECDS for this chunk.
        descs = extractor.extract_text(ch, title=f"{title} [chunk {j+1}/{len(chunks)}]")
        desc = descs[0] if descs else None

        # (a) TEXT/EMBEDDING TRACK + FIELD-PRIMARY MEMORY RECORD
        # ALWAYS store the chunk so a document is never silently dropped — even
        # if the PDE extractor didn't return a descriptor/BFECDS for it (that
        # bug let the library report "N passages" while storing zero). The field
        # snapshot is attached when available; retrieval still works via the
        # embedding and the filename either way.
        if memory is not None:
            bvec = desc.bvec if (desc is not None and desc.bvec is not None) else BVec()
            record = MemoryRecord(
                text=ch,
                bvec=bvec,
                embedding=(embed_fn(ch) if embed_fn else None),
                source=f"reading:{title}",
                phi_snapshot=(desc.phi_snapshot if desc is not None else None),
                theta_snapshot=(desc.theta_snapshot if desc is not None else None),
                metadata={"title": title, "sha256": ((desc.sha256 if desc else "") or "")[:12]},
            )
            # Enter at MEDIUM-term, not straight to long-term: freshly read
            # material is immediately searchable/discussable, fades via the
            # Ebbinghaus curve if never used, and is promoted to LTM by
            # consolidate() when it proves novel or gets reinforced in
            # conversation. (Override target via record.tier if needed.)
            memory.mtm.store(record)
    return len(chunks)


def read_wikipedia(title: str, *, extractor: KnowledgeExtractor,
                   memory=None, embed_fn="default", lang: str = "en") -> int:
    """Read one Wikipedia article into Eris's knowledge base (dual-track)."""
    text = fetch_wikipedia(title, lang=lang)
    return ingest_text(text, title=f"Wikipedia: {title}",
                       extractor=extractor, memory=memory, embed_fn=embed_fn)


def read_url(url: str, *, extractor: KnowledgeExtractor,
             memory=None, embed_fn="default", title: Optional[str] = None) -> int:
    """Read an arbitrary web page into Eris's knowledge base (dual-track)."""
    text = fetch_url(url)
    return ingest_text(text, title=title or url,
                       extractor=extractor, memory=memory, embed_fn=embed_fn)


def read_queue(titles: List[str], *, extractor: KnowledgeExtractor,
               memory=None, embed_fn="default", sleep_s: float = 1.0) -> dict:
    """Autonomously read a list of Wikipedia titles, building the knowledge base.
    Returns {title: chunks_stored}. Polite delay between fetches."""
    import time
    stats = {}
    for t in titles:
        try:
            stats[t] = read_wikipedia(t, extractor=extractor, memory=memory, embed_fn=embed_fn)
        except Exception as e:
            print(f"[web_reader] failed on '{t}': {e}")
            stats[t] = 0
        time.sleep(sleep_s)
    return stats
