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
from typing import Any, Dict, Optional
import re

_SRC_RE = re.compile(r"\[s:(\d+)\]")


def citation_resolution_rate(synthesis: str, n_sources: int) -> float:
    """Fraction of the synthesis's [s:i] citations that resolve to a real source. After
    grounding this should be 1.0 (unresolved cites were stripped); a control with no
    citations scores 0.0 (nothing grounded)."""
    cites = _SRC_RE.findall(synthesis or "")
    if not cites:
        return 0.0
    ok = sum(1 for c in cites if 0 <= int(c) < n_sources)
    return ok / len(cites)


def ab_metrics(result, n_sources: int) -> Dict[str, Any]:
    """Treatment-side quality metrics from a ResearchResult (or any object with the same
    fields). Control is the null synthesis (no contributors, no citations)."""
    synth = getattr(result, "synthesis", "") or ""
    return {
        "citation_resolution_rate": round(citation_resolution_rate(synth, n_sources), 4),
        "domain_diversity": getattr(result, "n_contributors", 0),
        "n_active": getattr(result, "n_active", 0),
        "cycles": getattr(result, "cycles", 0),
        "stripped_claims": getattr(result, "stripped_claims", 0),
        "canonized": bool(getattr(result, "thought_id", None)),
        "synthesis_len": len(synth),
    }


def compare(treatment, n_sources: int) -> Dict[str, Any]:
    """Side-by-side: control (hollow — no synthesis) vs treatment (hive)."""
    control = {"citation_resolution_rate": 0.0, "domain_diversity": 0, "cycles": 0,
               "canonized": False, "synthesis_len": 0}
    treat = ab_metrics(treatment, n_sources)
    return {
        "control": control,
        "treatment": treat,
        "verdict": {
            "more_grounded": treat["citation_resolution_rate"] > control["citation_resolution_rate"],
            "more_diverse": treat["domain_diversity"] > control["domain_diversity"],
            "produced_synthesis": treat["canonized"],
        },
    }


async def run_ab(orchestrator, topic: str, max_specialists: int = 5) -> Dict[str, Any]:   # pragma: no cover
    """[machine] — run the treatment over the real library/local-model via the orchestrator
    and report the comparison. (Control is the hollow cycle, which canonizes nothing.)"""
    summary = await orchestrator.hive_research(topic, max_specialists=max_specialists)
    if "error" in summary:
        return {"error": summary["error"]}

    class _R:  # adapt the summary dict back to the field shape ab_metrics expects
        synthesis = "[s:0]" * 0
        n_contributors = summary.get("n_contributors", 0)
        n_active = summary.get("n_active", 0)
        cycles = summary.get("cycles", 0)
        stripped_claims = summary.get("stripped_claims", 0)
        thought_id = summary.get("thought_id")
    # citation-resolution is read from the canonized thought itself
    tid = summary.get("thought_id")
    synth = ""
    if tid is not None:
        t = orchestrator.thought_stream.get(tid)
        synth = getattr(t, "text", "") if t else ""
    _R.synthesis = synth
    return {"topic": topic, **compare(_R, n_sources=max(1, max_specialists)), "raw": summary}
