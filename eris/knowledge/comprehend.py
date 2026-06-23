"""Index-time comprehension — elaboration that turns 'definitions' into 'understanding'.

The cheapest, highest-ROI comprehension technique that needs no new dependency:
elaborative interrogation (self-Q&A). At ingest, the LLM generates a few questions
about a source and answers them FROM THE TEXT; those Q→A pairs are stored as extra
retrieval units. They directly improve later recall and integration — a question a
user later asks is far more likely to match a stored question than a raw passage.

Pure-LLM; degrades to no-op when no model is wired. (Propositions / triples / RAPTOR
summaries are the heavier Stage-2 elaborations; this is the Stage-1-friendly one.)
"""
from __future__ import annotations
from typing import Callable, List, Optional
import json
import re

_QA_PROMPT = (
    "From the passage below, write {n} short question-and-answer pairs that capture "
    "its most important, specific content — the kind of questions someone studying "
    "this would later ask. Answer ONLY from the passage; do not add outside facts. "
    "Return ONLY a JSON list of objects like "
    '[{{"q":"...","a":"..."}}]. Keep each answer to one or two sentences.\n\n'
    "TITLE: {title}\n\nPASSAGE:\n{text}")


def _parse_qa(raw: str) -> List[dict]:
    if not raw:
        return []
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    out = []
    for d in data if isinstance(data, list) else []:
        if isinstance(d, dict) and d.get("q") and d.get("a"):
            out.append({"q": str(d["q"]).strip(), "a": str(d["a"]).strip()})
    return out


def self_qa(text: str, title: str, generate: Callable[[str], str],
            *, n: int = 3, max_chars: int = 4000) -> List[dict]:
    """Generate up to `n` grounded Q→A pairs about `text`. `generate(prompt)->str`
    is the model call. Returns [] on any failure (comprehension is best-effort)."""
    text = (text or "").strip()
    if not text or generate is None:
        return []
    try:
        raw = generate(_QA_PROMPT.format(n=n, title=title or "(untitled)",
                                         text=text[:max_chars]))
    except Exception:
        return []
    return _parse_qa(raw or "")[:n]


def qa_units(qas: List[dict]) -> List[str]:
    """Render Q→A pairs as storable retrieval units (question first, so a later
    user question matches it directly)."""
    return [f"Q: {qa['q']}\nA: {qa['a']}" for qa in qas if qa.get("q") and qa.get("a")]


_PROP_PROMPT = (
    "Decompose the passage into atomic, self-contained factual statements "
    "(propositions): each a single fact, with pronouns/references resolved to "
    "explicit names, understandable on its own. Stay faithful to the passage; add "
    "nothing. Return ONLY a JSON list of strings, at most {n}.\n\nPASSAGE:\n{text}")


def _parse_props(raw: str) -> List[str]:
    if not raw:
        return []
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    return [str(x).strip() for x in data if isinstance(x, str) and x.strip()] \
        if isinstance(data, list) else []


def propositions(text: str, generate: Callable[[str], str], *,
                 n: int = 6, max_chars: int = 4000) -> List[str]:
    """Atomic, self-contained facts (Dense X Retrieval) as crisp, low-noise extra
    retrieval units. Keep the ORIGINAL chunk too — propositions augment, never
    replace, source text (verification needs the original). [] on failure."""
    text = (text or "").strip()
    if not text or generate is None:
        return []
    try:
        raw = generate(_PROP_PROMPT.format(n=n, text=text[:max_chars]))
    except Exception:
        return []
    return _parse_props(raw or "")[:n]
