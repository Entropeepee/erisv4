"""A/B harness for the restored hive + working memory (§5) — [machine].

Control  = the current hollow cycle (field-projection bids; no synthesis).
Treatment = the two-cycle hive research engine (reasoned findings, cross-pollination,
            citation-grounded canonization).

Run BOTH on the same study topic and compare grounded-study quality: the
citation-resolution rate (do canonized claims resolve to real sources?), the
domain-diversity of the synthesis (did multiple specialists contribute, or did one
dominate?), the cycles run, and the claims stripped for being unciteable. Keep the change
only if treatment demonstrably improves grounded quality WITHOUT reintroducing echo/loops
or breaking the cost budget — log the numbers in the PR.

The metric computation is pure/offline-testable; running it over the real library needs the
local model (so the actual A/B is a [machine] step).
"""
from __future__ import annotations
from typing import Any, Dict, List
import re

from eris.tribe.research import _cited_ids


def citation_resolution_rate(synthesis: str, n_sources: int) -> float:
    """Fraction of the synthesis's distinct citations that resolve to a real source. Use the
    PRE-grounding synthesis — measured on the final (post-strip) text it's ~always 1.0 (bad
    cites were already removed), which measures nothing. On the draft it shows how grounded
    the model's OWN claims were."""
    cited = _cited_ids(synthesis)
    if not cited:
        return 0.0
    return len(cited & set(range(n_sources))) / len(cited)


# short function words carry no grounding signal — exclude them so the metric measures
# CONTENT, not how many "the/of/and"s a passage happens to share.
_STOP = frozenset(
    "the a an and or but of to in on at by for with from into over under as is are was were be "
    "been being this that these those it its their there here then than thus so we you they he she "
    "i not no nor can could will would shall should may might must do does did has have had not "
    "which who whom whose what when where why how all any both each few more most other some such "
    "only own same too very s t can".split())


def _content_words(text: str) -> set:
    """Distinct content tokens (len≥4, non-stopword) — paraphrase-tolerant unit of meaning."""
    return {w for w in re.findall(r"[a-z][a-z0-9]{3,}", (text or "").lower()) if w not in _STOP}


def _trigrams(text: str) -> set:
    toks = re.findall(r"[a-z0-9]{3,}", (text or "").lower())
    return {tuple(toks[i:i + 3]) for i in range(len(toks) - 2)} if len(toks) >= 3 else set()


def source_alignment(synthesis: str, sources: List[str]) -> float:
    """Grounding signal: fraction of the synthesis's CONTENT WORDS that appear in some source.

    Deliberately uses unigram content words, NOT verbatim trigrams: a deeper synthesis
    paraphrases and integrates (it does not parrot source phrasing), so trigram overlap
    punishes comprehension and rewards copying — the opposite of what we want. Content-word
    overlap is paraphrase-tolerant: a faithful restatement reuses the same terms even when it
    rewrites the sentences, while an invented claim introduces vocabulary the sources lack."""
    syn = _content_words(synthesis)
    if not syn:
        return 0.0
    src = set()
    for s in sources:
        src |= _content_words(s)
    return len(syn & src) / len(syn)


def verbatim_overlap(synthesis: str, sources: List[str]) -> float:
    """Trigram overlap — how much the synthesis COPIES source phrasing (low is fine, even good:
    it means abstraction). Reported alongside source_alignment to separate 'grounded' from
    'extractive', so a paraphrasing hive isn't mistaken for an ungrounded one."""
    syn = _trigrams(synthesis)
    if not syn:
        return 0.0
    src = set()
    for s in sources:
        src |= _trigrams(s)
    return len(syn & src) / len(syn)


def metrics_from(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Honest grounded-quality metrics from a hive_research summary dict (works for both the
    hive treatment and the single-pass control)."""
    n = max(1, summary.get("n_sources", 0))
    pre = summary.get("synthesis_pre_ground", "") or summary.get("synthesis", "")
    final = summary.get("synthesis", "") or ""
    srcs = summary.get("sources", [])
    return {
        "citation_resolution_pre_ground": round(citation_resolution_rate(pre, n), 4),
        "source_alignment": round(source_alignment(final, srcs), 4),       # grounded (paraphrase-ok)
        "verbatim_overlap": round(verbatim_overlap(final, srcs), 4),       # how extractive/copied
        "domain_diversity": summary.get("n_contributors", 0),
        "n_active": summary.get("n_active", 0),
        "cycles": summary.get("cycles", 0),
        "stripped_claims": summary.get("stripped_claims", 0),
        "canonized": bool(summary.get("canonized")),
        "synthesis_len": len(final),
    }


async def run_ab(orchestrator, topic: str, max_specialists: int = 5, *,
                 scope: str = "memory", document: str = "") -> Dict[str, Any]:   # pragma: no cover
    """[machine] — run BOTH arms over the real library/local-model and compare honestly:
      control   = single-pass RAG summary (same retrieval/model/grounding, NO hive)
      treatment = the full multi-specialist two-cycle hive.
    Same topic, same sources, same scope — so the delta isolates what the hive actually adds.
    `scope`/`document` select what she reads: "memory" (all tiers + thought-stream), "doc"
    (only the named document), "web" (fresh arXiv/Wikipedia/web)."""
    kw = dict(max_specialists=max_specialists, scope=scope, document=document)
    treat = await orchestrator.hive_research(topic, mode="hive", **kw)
    if "error" in treat:
        return {"error": treat["error"], "traceback": treat.get("traceback")}
    ctrl = await orchestrator.hive_research(topic, mode="single", **kw)
    if "error" in ctrl:
        return {"error": "control failed: " + ctrl["error"], "traceback": ctrl.get("traceback")}

    tm, cm = metrics_from(treat), metrics_from(ctrl)
    # No-data guard: if BOTH arms retrieved 0 sources, neither had anything to work with —
    # this is a plumbing/retrieval failure, NOT a quality result. Don't emit a verdict (a
    # 0.0 >= 0.0 tie would otherwise falsely credit the hive a clean sweep). (Review #2.)
    if treat.get("n_sources", 0) == 0 and ctrl.get("n_sources", 0) == 0:
        return {
            "topic": topic,
            "verdict": "INCONCLUSIVE — both arms retrieved 0 sources (retrieval/harness "
                       "issue, not a quality signal). Check scope/document/library, then re-run.",
            "control_single_pass_rag": cm,
            "treatment_hive": tm,
            "treatment_raw": treat,
            "control_raw": ctrl,
        }
    return {
        "topic": topic,
        "control_single_pass_rag": cm,
        "treatment_hive": tm,
        "verdict": {
            # strict >: a tie is NOT a hive win (0.0 >= 0.0 used to credit it falsely)
            "hive_more_source_grounded": tm["source_alignment"] > cm["source_alignment"],
            "hive_more_diverse": tm["domain_diversity"] > cm["domain_diversity"],
            "hive_deeper": tm["cycles"] > cm["cycles"],
            "hive_synthesis_longer": tm["synthesis_len"] > cm["synthesis_len"],
        },
        "treatment_raw": treat,
        "control_raw": ctrl,
    }


def main(argv=None):   # pragma: no cover
    """[machine] CLI: run the hive A/B on a study topic, using her real library + local
    model. Usage:
      python -m eris.experiments.hive_ab "your topic here" [--size 32] [--specialists 5]
      python -m eris.experiments.hive_ab "what does BLECD claim?" --document BLECD --scope doc
      python -m eris.experiments.hive_ab "latest on phase transitions" --scope web
    Scopes:  memory (default, all tiers + thought-stream) | doc (only --document) | web."""
    import argparse
    import asyncio
    import json
    ap = argparse.ArgumentParser(description="Hive research A/B (control vs treatment)")
    ap.add_argument("topic", help="the study topic to research")
    ap.add_argument("--size", type=int, default=32, help="PDE field size (32 fast, 64 default)")
    ap.add_argument("--specialists", type=int, default=5, help="top-K active specialists")
    ap.add_argument("--scope", default="memory", choices=["memory", "doc", "web"],
                    help="memory=all tiers (default) | doc=only --document | web=arXiv/wiki/web")
    ap.add_argument("--document", default="",
                    help="title/filename to prioritise (or restrict to, with --scope doc)")
    args = ap.parse_args(argv)

    from eris.orchestrator import ErisOrchestrator
    print(f"[hive-ab] booting Eris (reads ./eris_data, talks to your local Ollama)…")
    orch = ErisOrchestrator(field_size=args.size)           # default data_dir = eris_data
    scope_note = f" [scope={args.scope}{', doc=' + args.document if args.document else ''}]"
    print(f"[hive-ab] researching: {args.topic!r}{scope_note}\n")
    result = asyncio.run(run_ab(orch, args.topic, max_specialists=args.specialists,
                                scope=args.scope, document=args.document))
    print(json.dumps(result, indent=2, default=str))
    if result.get("raw", {}).get("thought_id"):
        print(f"\n[hive-ab] canonized thought id: {result['raw']['thought_id']} "
              f"(saved to eris_data/thoughts.jsonl)")


if __name__ == "__main__":   # pragma: no cover
    main()
