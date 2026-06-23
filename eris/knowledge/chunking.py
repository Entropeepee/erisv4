"""Structure-aware, contextual chunking for ingestion.

The 2026 SOTA's highest quality-per-complexity lever (Anthropic "Contextual
Retrieval"; cuts top-k retrieval failure substantially) without any new
dependency: instead of blind fixed-character splitting, respect the document's
structure and give every chunk the context it needs to be retrievable on its own.

  • SECTION-AWARE: split on the markdown heading hierarchy (## / ###), so a chunk
    never straddles unrelated sections, and pack paragraphs up to a token budget
    (never mid-sentence when avoidable).
  • CONTEXTUAL HEADER: prepend "Title › Section › Subsection" to each chunk, so the
    embedding AND the BM25 index carry the chunk's place in the document. This is
    the cheap (no per-chunk LLM call) form of contextual retrieval — a fragment
    like "It then projects onto the nullspace." becomes findable because its header
    says which paper and section it came from.
  • OVERLAP: carry a tail of the previous chunk so a fact split across a boundary
    is still retrievable from either side.

Pure stdlib; deterministic. The naive `web_reader.chunk` stays as the "legacy"
fallback (ERIS_CHUNKER=legacy).
"""
from __future__ import annotations
from typing import List, Tuple
import re

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_SENT = re.compile(r"(?<=[.!?])\s+")
_PARA = re.compile(r"\n\s*\n")


def _split_sections(text: str) -> List[Tuple[str, str]]:
    """(heading_path, body) pairs in document order. Text before the first
    heading (or a heading-less document) gets an empty path."""
    path: List[str] = []
    cur_path = ""
    buf: List[str] = []
    sections: List[Tuple[str, str]] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((cur_path, body))

    for ln in text.splitlines():
        m = _HEADING.match(ln.strip())
        if m:
            flush()
            buf.clear()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip()]
            cur_path = " › ".join(path)
        else:
            buf.append(ln)
    flush()
    if not sections:
        return [("", text.strip())]
    return sections


def _split_big_paragraph(p: str, target: int) -> List[str]:
    """A paragraph larger than the budget → pack by sentence; hard-split a
    sentence that is itself oversized."""
    out: List[str] = []
    cur = ""
    for s in _SENT.split(p):
        if len(s) > target:
            if cur:
                out.append(cur)
                cur = ""
            for i in range(0, len(s), target):
                out.append(s[i:i + target])
            continue
        if cur and len(cur) + len(s) + 1 > target:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}" if cur else s
    if cur:
        out.append(cur)
    return out


def _pack(body: str, target: int, overlap: int) -> List[str]:
    """Greedily pack paragraphs up to `target` chars; then add overlap tails."""
    paras = [p.strip() for p in _PARA.split(body) if p.strip()]
    chunks: List[str] = []
    cur = ""
    for p in paras:
        if len(p) > target:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(_split_big_paragraph(p, target))
            continue
        if cur and len(cur) + len(p) + 2 > target:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    if overlap > 0 and len(chunks) > 1:
        carried = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:].lstrip()
            carried.append(f"…{tail} {chunks[i]}" if tail else chunks[i])
        chunks = carried
    return chunks


def structured_chunks(text: str, *, title: str = "",
                      target_chars: int = 2000,
                      overlap_chars: int = 200) -> List[str]:
    """Section/paragraph-aware chunks, each carrying a contextual header
    ('Title › Section'). Returns chunk strings ready to embed + index."""
    text = (text or "").strip()
    if not text:
        return []
    result: List[str] = []
    for path, body in _split_sections(text):
        header = " › ".join(b for b in (title, path) if b)
        for bc in _pack(body, target_chars, overlap_chars):
            result.append(f"{header}\n\n{bc}" if header else bc)
    return result
