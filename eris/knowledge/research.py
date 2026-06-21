"""
eris/knowledge/research.py
==========================
The research cascade (Remediation Tier 4.3): dissonance -> web search -> (if
still unresolved AND keyed) ask the expert. The theory wires into behavior here:
where the field has an unresolved channel, that channel names the query; web
search is the first stop, the Claude oracle (ask_expert) is the escalation, and
whatever comes back is INGESTED AS GROUNDING — verified against, never spoken in
Eris's own voice.

This is also what would have caught the 'Lindqvist 2017 resonance theorem'
fabrication: a search returns nothing supporting it, and the grounding
instruction makes the model flag the unsupported premise. Field dynamics never
needed to catch it — grounding does.

Non-blocking by construction: web search returns [] on failure; the expert path
is dormant until ANTHROPIC_API_KEY is set and returns None instantly otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from eris.retrieval.web_search import research as web_research
from eris.knowledge import ask_expert


@dataclass
class ResearchBundle:
    query: str
    grounding: str = ""              # formatted block to inject into the prompt / ingest
    sources: List[str] = field(default_factory=list)
    full_texts: List[str] = field(default_factory=list)
    used_expert: bool = False


def _format_grounding(query: str, report) -> str:
    blocks = []
    for i, r in enumerate(report.results, 1):
        snippet = (r.snippet or "").strip()
        blocks.append(f"[{i}] {r.title}\n{snippet}\nSOURCE: {r.url}")
    return "\n\n".join(blocks)


async def gather(query: str, *, max_results: int = 3,
                 max_chars_per_page: int = 3000,
                 allow_expert: bool = True) -> ResearchBundle:
    """Run the cascade for `query` and return grounding + sources.

    Caller decides what to do with the bundle (inject into the prompt, ingest
    into LTM, etc.). Never raises — failures degrade to an empty bundle.
    """
    bundle = ResearchBundle(query=query)
    try:
        report = await web_research(query, max_results=max_results,
                                    max_chars_per_page=max_chars_per_page)
    except Exception as e:
        print(f"[research] web search failed (non-fatal): {e}")
        return bundle

    bundle.grounding = _format_grounding(query, report)
    bundle.sources = list(report.source_urls)
    bundle.full_texts = list(report.full_texts)

    # Escalate to the expert only if web didn't resolve it AND a key is present.
    thin = len(report.full_texts) == 0
    if allow_expert and thin and ask_expert.is_available():
        ans = ask_expert.ask(query, context=bundle.grounding)
        if ans and ans.answer:
            bundle.grounding = (bundle.grounding + "\n\nEXPERT:\n" + ans.answer).strip()
            bundle.full_texts.append(ans.answer)
            bundle.used_expert = True
    return bundle
