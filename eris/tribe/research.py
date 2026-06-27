"""The two-cycle research engine (§A2) — the hive actually reasoning on research.

broad → synthesis → refined → canonize, composed ON TOP OF the existing SOTA RAG pipeline
(it does not replace knowledge/study.py — RAG is injected as `retriever`). The specialists
reason (make_reasoned_finding) over retrieved material instead of returning labels; Kairos
integrates, the MoEGate weights, the hub cross-pollinates (field resonance), Elos (if
active) tries to falsify; gaps drive a second targeted retrieval; the result is canonized
into the thought-stream with hard citation grounding — claims that cite a source which does
not resolve are STRIPPED, not shipped (the same discipline retrospect.py applies to [t:id]).

Cost guards (why it broke before): specialists are LOCAL-MODEL only (injected `model`),
capped at `max_specialists` (the §2 top-K), and this fires on the research trigger / an
explicit deep cycle only — NEVER per ordinary turn. Fully offline-testable: `retriever`
and `model` are injected callables; a stub drives the whole flow with no network or LLM.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Any
import re

from eris.computation.activations import BVec
from eris.tribe.specialists import (
    Specialist, SpecialistFinding, make_reasoned_finding, CrossAttentionHub,
    get_active_specialists, _text_to_bvec,
)

Retriever = Callable[[str], List[str]]      # query -> list of source texts (the RAG seam)
Model = Callable[[str], str]                # prompt -> text (local model)

_SRC_RE = re.compile(r"\[s:(\d+)\]")


@dataclass
class ResearchResult:
    topic: str
    synthesis: str
    thought_id: Optional[str] = None
    gaps: List[str] = field(default_factory=list)
    n_contributors: int = 0          # distinct specialists that contributed (diversity)
    n_active: int = 0
    stripped_claims: int = 0         # uncited/unresolved claims removed at canonize
    elos_critique: str = ""
    cycles: int = 0


def _format_sources(sources: List[str]) -> str:
    return "\n".join(f"[s:{i}] {s}" for i, s in enumerate(sources)) or "(no sources)"


def _ground_citations(text: str, n_sources: int, *, strict: bool = True) -> tuple:
    """Citation grounding for canonized research (mirrors retrospect's strip-if-unresolved
    discipline for [s:id]). Returns (grounded_text, n_stripped). Per sentence:
      • only-fabricated cites (a [s:i] that doesn't resolve, no valid cite) → DROP the sentence;
      • mixed valid+fabricated → strip just the fabricated token(s), KEEP the valid claim;
      • strict & a long uncited assertion (>10 words, no citation) → DROP it (a naked claim the
        sources don't support); short framing/transitions are kept.
    """
    text = re.sub(r"\[s:\s*(\d+)\s*\]", r"[s:\1]", text or "")   # normalize multiline cites
    allowed = {str(i) for i in range(n_sources)}
    keep, stripped = [], 0
    for sent in re.split(r"(?<=[.!?])\s+", text):
        cites = set(_SRC_RE.findall(sent))
        bad = cites - allowed
        good = cites & allowed
        if cites and not good:                       # only fabricated → drop whole sentence
            stripped += 1
            continue
        if bad:                                       # mixed → strip bad token(s), keep claim
            sent = _SRC_RE.sub(lambda m: "" if m.group(1) in bad else m.group(0), sent)
            sent = re.sub(r"\s{2,}", " ", sent).strip()
            stripped += 1
        elif not cites and strict and len(sent.split()) > 10:
            stripped += 1                             # long, unsupported, uncited → drop
            continue
        if sent.strip():
            keep.append(sent.strip())
    return " ".join(keep).strip(), stripped


def _gaps_from(text: str) -> List[str]:
    """Pull bulleted/numbered gap lines the synthesis named."""
    out = []
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line.strip())
        if line and len(line.split()) >= 2:
            out.append(line)
    return out[:5]


def run_two_cycle_research(
    topic: str, *, retriever: Retriever, model: Model,
    specialists: Optional[List[Specialist]] = None,
    hub: Optional[CrossAttentionHub] = None,
    moe_gate=None, thought_stream=None, embed_fn: Optional[Callable[[str], Any]] = None,
    goal_bvec: Optional[BVec] = None, max_specialists: int = 5,
    regime: str = "research", log: Optional[Callable[[str], None]] = None,
) -> ResearchResult:
    """Run the broad→synthesis→refined→canonize cycle. `retriever(query)->[sources]` is the
    existing RAG pipeline; `model(prompt)->text` is the local model. Specialists default to
    the top-K active for the topic's field; the result is canonized to `thought_stream` if
    given (citation-grounded)."""
    _log = log or (lambda m: None)
    goal_bvec = goal_bvec or _text_to_bvec(topic)
    active = (specialists if specialists is not None
              else get_active_specialists(goal_bvec, max_k=max_specialists))[:max_specialists]
    hub = hub if hub is not None else CrossAttentionHub()
    _log(f"active specialists: {', '.join(s.name for s in active)}")

    # ── Cycle 1 — broad: RAG → each active specialist reasons → post to hub ──
    ctx1 = list(retriever(topic) or [])
    _log(f"cycle 1 — retrieved {len(ctx1)} source(s); specialists reasoning…")
    src1 = _format_sources(ctx1)
    c1: List[SpecialistFinding] = []
    for s in active:
        _log(f"  {s.name} reasoning…")
        f = make_reasoned_finding(s, topic, src1, model)
        hub.post(f); c1.append(f)

    # ── Synthesis: Kairos integrates; MoEGate weights; hub cross-pollinates; Elos falsifies ──
    if moe_gate is not None:
        moe_gate.set_goal(goal_bvec, topic)
        winner = moe_gate.select_winner(c1) if c1 else None
    else:
        winner = max(c1, key=lambda f: f.confidence) if c1 else None
    cross = hub.query(winner.bvec, top_k=3) if winner is not None else []
    synth_prompt = (
        f"You are Kairos, integrating the Tribe's findings on: {topic}\n\n"
        f"FINDINGS:\n" + "\n".join(f"- {f.specialist_id}: {f.content}" for f in c1) +
        f"\n\nMost-resonant cross-links:\n" + "\n".join(f"- {f.content}" for f in cross) +
        f"\n\nSOURCES:\n{src1}\n\nIntegrate into a synthesis grounded in the sources "
        f"(cite [s:i]). Then on new lines list the open GAPS as bullets.")
    _log("synthesis — Kairos integrating…")
    synthesis = (model(synth_prompt) or "").strip()
    elos_critique = ""
    if any(s.id == "elos" for s in active):
        elos_critique = (model(
            f"As Elos (adversarial), try to FALSIFY this synthesis using only the sources. "
            f"Name its weakest unsupported claim.\n\nSYNTHESIS:\n{synthesis}\n\nSOURCES:\n{src1}"
        ) or "").strip()
    gaps = _gaps_from(synthesis)

    # ── Cycle 2 — refined: targeted RAG on the gaps → specialists refine ──
    ctx2, c2 = [], []
    if gaps:
        _log(f"cycle 2 — {len(gaps)} gap(s); targeted retrieval + refine…")
        ctx2 = list(retriever(" ; ".join(gaps)) or [])
        src2 = _format_sources(ctx1 + ctx2)
        for s in active:
            c2.append(make_reasoned_finding(
                s, f"Refine the synthesis of '{topic}', closing these gaps: {gaps}", src2, model))

    # ── Canonize: citation-grounded thought-stream entry (strip unresolved cites) ──
    all_sources = ctx1 + ctx2
    canon_prompt = (
        f"Write the final, defensible synthesis on '{topic}'. Ground EVERY claim in a "
        f"source citation [s:i]; do not assert anything the sources don't support"
        + (f". Address this critique: {elos_critique}" if elos_critique else "") +
        f".\n\nDRAFT:\n{synthesis}\n\nREFINEMENTS:\n" +
        "\n".join(f"- {f.content}" for f in c2) +
        f"\n\nSOURCES:\n{_format_sources(all_sources)}\n\nFinal synthesis:")
    _log("canonizing — citation-grounding the final synthesis…")
    final = (model(canon_prompt) or synthesis).strip()
    grounded, stripped = _ground_citations(final, len(all_sources))

    # diversity = distinct specialists that genuinely contributed across BOTH cycles
    contributors = {f.specialist_id for f in (c1 + c2)
                    if not f.metadata.get("echo") and not f.metadata.get("empty")}

    thought_id = None
    if thought_stream is not None and grounded:
        try:
            from eris.memory.thought_stream import link_and_store
            # store WITH a semantic embedding (else thought_stream.retrieve skips it and the
            # canonized synthesis is invisible to introspection/retrospection)
            emb = embed_fn(grounded) if embed_fn else None
            t = link_and_store(thought_stream, topic, regime, grounded, embedding=emb)
            thought_id = getattr(t, "id", None)
        except Exception:
            thought_id = None

    return ResearchResult(
        topic=topic, synthesis=grounded, thought_id=thought_id, gaps=gaps,
        n_contributors=len(contributors),
        n_active=len(active), stripped_claims=stripped,
        elos_critique=elos_critique, cycles=2 if gaps else 1)
