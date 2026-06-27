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

# Recognize source citations in the forms models actually emit: [s:0], (s:1), s:2, and
# ranges (s:1-3) / [s:1–3] with hyphen, en/em-dash, or minus.
_SRC_RE = re.compile(r"[\[(]?\bs:\s*(\d+)\s*(?:[-‐-―−]\s*(\d+))?\s*[\])]?",
                     re.IGNORECASE)


def _cited_ids(text: str) -> set:
    """All source ids cited in any recognized form (ranges expanded)."""
    ids = set()
    for m in _SRC_RE.finditer(text or ""):
        a = int(m.group(1)); b = int(m.group(2)) if m.group(2) else a
        ids.update(range(min(a, b), max(a, b) + 1))
    return ids


@dataclass
class ResearchResult:
    topic: str
    synthesis: str
    thought_id: Optional[str] = None
    gaps: List[str] = field(default_factory=list)
    n_contributors: int = 0          # distinct specialists that contributed (diversity)
    n_active: int = 0
    n_sources: int = 0               # how many DISTINCT sources the RAG retrieved
    sources: List[str] = field(default_factory=list)   # source previews (for source-alignment)
    synthesis_pre_ground: str = ""   # the synthesis BEFORE citation-stripping (for honest A/B)
    stripped_claims: int = 0         # uncited/unresolved claims removed at canonize
    elos_critique: str = ""
    cycles: int = 0


def _format_sources(sources: List[str], digester: Optional[Callable[[str], List[str]]] = None
                    ) -> str:
    """Format sources as [s:i] blocks. When a `digester` is given (the Stage-2 comprehension
    pipeline), prepend each source's atomic KEY FACTS (propositions) before its raw text —
    distilled, pronoun-resolved units a small model comprehends far better than a raw chunk,
    while keeping the original text so citations still verify."""
    if not sources:
        return "(no sources)"
    blocks = []
    for i, s in enumerate(sources):
        head = f"[s:{i}]"
        if digester is not None:
            try:
                facts = digester(s)
            except Exception:
                facts = []
            if facts:
                head += " KEY FACTS: " + " | ".join(facts)
        blocks.append(f"{head}\n{s}")
    return "\n\n".join(blocks)


def _distinct(sources: List[str]) -> List[str]:
    """De-duplicate sources by normalized-text prefix (the same paper chunk can recur)."""
    seen, out = set(), []
    for s in sources:
        key = " ".join((s or "").lower().split())[:200]
        if key and key not in seen:
            seen.add(key); out.append(s)
    return out


def resonance_rerank(goal_bvec: BVec, texts: List[str], *, top_k: Optional[int] = None,
                     blend: float = 0.5, bvec_of: Optional[Callable[[str], BVec]] = None
                     ) -> List[str]:
    """Re-rank retrieved chunks by FIELD RESONANCE — not just embedding cosine (§B3).

    Each chunk is scored by the 2D resonance MAGNITUDE √(R_cos²+R_sin²) against the goal
    (κ cosine/aligned channel AND λ sine/torsion channel — the sine is kept, not discarded),
    then BLENDED with its incoming lexical/dense rank so a torsion-coupled chunk can rise even
    when its cosine alignment is modest, without throwing away what BM25/dense already knew.
    `blend`=1.0 is pure resonance, 0.0 is the incoming order. Best-effort: any chunk whose
    bvec can't be computed keeps its incoming position. Returns the reordered texts."""
    from eris.computation.activations import bvec_resonance_2d
    if not texts or goal_bvec is None:
        return list(texts)
    to_bvec = bvec_of or _text_to_bvec
    n = len(texts)
    scored = []
    for i, t in enumerate(texts):
        prior = 1.0 - (i / max(1, n - 1)) if n > 1 else 1.0   # incoming rank as a [0,1] prior
        try:
            res = bvec_resonance_2d(goal_bvec, to_bvec(t))["magnitude"]
        except Exception:
            res = None
        scored.append([i, t, prior, res])
    mags = [s[3] for s in scored if s[3] is not None]
    if mags:
        lo, hi = min(mags), max(mags)
        span = (hi - lo) or 1.0
        for s in scored:
            r = s[2] if s[3] is None else (s[3] - lo) / span   # missing → fall back to prior
            s.append(blend * r + (1.0 - blend) * s[2])
    else:
        for s in scored:
            s.append(s[2])
    scored.sort(key=lambda s: (-s[-1], s[0]))                  # stable on the incoming order
    out = [s[1] for s in scored]
    return out[:top_k] if top_k else out


def _ground_citations(text: str, n_sources: int) -> tuple:
    """Citation grounding for canonized research — the strip-if-UNRESOLVED discipline from
    retrospect.py ([t:id]), applied to [s:id]. Returns (grounded_text, n_stripped). Per
    sentence: a sentence whose ONLY citations are fabricated (point to a source that doesn't
    exist) is dropped; a mixed sentence keeps the claim and drops just the fabricated
    token(s); UNCITED sentences are KEPT (honest framing and negative/absence findings —
    'the sources don't discuss X' — are not hallucinations and must not be nuked, which is
    what emptied a whole synthesis before). Whether to trust the result is the caller's job:
    canonization is gated on the synthesis actually resolving ≥1 citation."""
    allowed = set(range(n_sources))
    keep, stripped = [], 0
    for sent in re.split(r"(?<=[.!?])\s+", text or ""):
        cited = _cited_ids(sent)
        good, bad = cited & allowed, cited - allowed
        if cited and not good:                        # only fabricated → drop the sentence
            stripped += 1
            continue
        if bad:                                       # mixed → drop only wholly-bad cite tokens
            def _strip(m):
                a = int(m.group(1)); b = int(m.group(2)) if m.group(2) else a
                rng = set(range(min(a, b), max(a, b) + 1))
                return "" if (rng & bad and not rng & allowed) else m.group(0)
            sent = re.sub(r"\s{2,}", " ", _SRC_RE.sub(_strip, sent)).strip()
            stripped += 1
        if sent.strip():
            keep.append(sent.strip())
    return " ".join(keep).strip(), stripped


_GAP_HEADERS = {"open gaps", "gaps", "gap", "remaining gaps", "limitations",
                "open questions", "open gap", "gaps and limitations", "open gaps and questions"}


def _is_gap_header(line: str) -> bool:
    return re.sub(r"[^a-z ]", "", line.lower()).strip() in _GAP_HEADERS


def _gaps_from(text: str) -> List[str]:
    """Pull the gaps the synthesis named — ONLY the bullets that follow an 'open GAPS'
    section header, not every bullet in the document (the body bullets are findings, not
    gaps; capturing them bloated cycle-2's targeted retrieval with whole paragraphs).

    Falls back to scanning for explicit gap-language lines ('gap:', 'unclear', 'not
    established', 'missing', 'unspecified', 'unknown') if no header is present."""
    def _clean(s: str) -> str:
        return re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", s.strip()).strip("*").strip()

    lines = (text or "").splitlines()
    out, in_section = [], False
    for raw in lines:
        line = _clean(raw)
        if not line:
            continue
        if _is_gap_header(line):                 # entered (or re-entered) the gaps section
            in_section = True
            continue
        if in_section:
            # a new non-gap section header (Title Case / bold, no trailing punctuation) ends it
            if re.match(r"^[A-Z][A-Za-z ]{2,40}:?$", line) and not line.endswith((".", ";")):
                in_section = False
                continue
            if len(line.split()) >= 2:
                out.append(line)
    if out:
        return out[:5]
    # No explicit section — keep only lines that actually read like a gap.
    _GAPISH = re.compile(r"\b(gap|unclear|unknown|unspecified|not (?:yet )?(?:established|"
                         r"provided|defined)|missing|no empirical|remains? (?:to|unclear)|"
                         r"lacks?|absent)\b", re.I)
    for raw in lines:
        line = _clean(raw)
        if len(line.split()) >= 2 and not _is_gap_header(line) and _GAPISH.search(line):
            out.append(line)
    return out[:5]


def run_two_cycle_research(
    topic: str, *, retriever: Retriever, model: Model,
    specialists: Optional[List[Specialist]] = None,
    hub: Optional[CrossAttentionHub] = None,
    moe_gate=None, thought_stream=None, embed_fn: Optional[Callable[[str], Any]] = None,
    synth_model: Optional[Model] = None, digester: Optional[Callable[[str], List[str]]] = None,
    goal_bvec: Optional[BVec] = None, max_specialists: int = 5, single_pass: bool = False,
    regime: str = "research", log: Optional[Callable[[str], None]] = None,
) -> ResearchResult:
    """Run the broad→synthesis→refined→canonize cycle. `retriever(query)->[sources]` is the
    existing RAG pipeline; `model(prompt)->text` is the local model. Specialists default to
    the top-K active for the topic's field; the result is canonized to `thought_stream` if
    given (citation-grounded)."""
    _log = log or (lambda m: None)
    synth = synth_model or model                       # deep model for comprehension-heavy steps

    # ── CONTROL: a single-pass RAG summary (same retrieval/model/grounding, NO hive) — the
    # honest A/B baseline that isolates what the multi-specialist two-cycle engine adds. ──
    if single_pass:
        ctx = _distinct(list(retriever(topic) or []))
        _log(f"single-pass RAG control — {len(ctx)} source(s), one summary call…")
        pre = (synth(
            f"Summarize what the sources establish about '{topic}'. Ground every claim with "
            f"[s:i]; assert nothing the sources don't support.\n\n"
            f"SOURCES:\n{_format_sources(ctx, digester)}\n\nSummary:") or "").strip()
        g, strp = _ground_citations(pre, len(ctx))
        return ResearchResult(topic=topic, synthesis=(g or pre), synthesis_pre_ground=pre,
                              n_sources=len(ctx), sources=[c[:300] for c in ctx],
                              n_active=0, n_contributors=0, stripped_claims=strp, cycles=0)

    goal_bvec = goal_bvec or _text_to_bvec(topic)
    active = (specialists if specialists is not None
              else get_active_specialists(goal_bvec, max_k=max_specialists))[:max_specialists]
    hub = hub if hub is not None else CrossAttentionHub()
    _log(f"active specialists: {', '.join(s.name for s in active)}")

    # ── Cycle 1 — broad: RAG → each active specialist reasons → post to hub ──
    ctx1 = _distinct(list(retriever(topic) or []))
    _log(f"cycle 1 — retrieved {len(ctx1)} source(s); specialists reasoning…")
    for i, s in enumerate(ctx1):                       # show WHAT was retrieved (diagnose RAG)
        _log(f"    [s:{i}] {' '.join((s or '').split())[:80]}…")
    src1 = _format_sources(ctx1, digester)             # propositions distilled in (if available)
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
    cross = hub.query(winner.bvec, top_k=2) if winner is not None else []
    # Show the TOP findings only (context budget — a small model truncates an over-long prompt).
    top_findings = sorted(c1, key=lambda f: f.confidence, reverse=True)[:3]
    synth_prompt = (
        f"You are Kairos, integrating the Tribe's findings on: {topic}\n\n"
        f"FINDINGS:\n" + "\n".join(f"- {f.specialist_id}: {f.content}" for f in top_findings) +
        (("\n\nCross-links:\n" + "\n".join(f"- {f.content}" for f in cross)) if cross else "") +
        f"\n\nSOURCES:\n{src1}\n\nSynthesize with real comprehension: (1) state the CENTRAL "
        f"mechanism/claim the sources establish; (2) show how the findings connect or CONFLICT; "
        f"(3) mark what the sources SUPPORT vs what is inference. Ground each claim [s:i]. Then "
        f"list genuine open GAPS as bullets.")
    _log("synthesis — Kairos integrating…")
    synthesis = (synth(synth_prompt) or "").strip()
    elos_critique = ""
    if any(s.id == "elos" for s in active):
        elos_critique = (model(
            f"As Elos (adversarial), try to FALSIFY this synthesis using only the sources. "
            f"Name its weakest unsupported claim.\n\nSYNTHESIS:\n{synthesis}\n\nSOURCES:\n{src1}"
        ) or "").strip()
    gaps = _gaps_from(synthesis)

    # ── Cycle 2 — refined: targeted RAG on the gaps → specialists CLOSE the gaps ──
    # (conditioned on cycle-1 findings + the named gaps, not a blind re-run.)
    ctx2, c2 = [], []
    if gaps:
        _log(f"cycle 2 — {len(gaps)} gap(s); targeted retrieval + refine…")
        ctx2 = _distinct(list(retriever(" ; ".join(gaps)) or []))
        src2 = _format_sources(_distinct(ctx1 + ctx2), digester)
        c1_summary = "; ".join(f"{f.specialist_id}: {f.content[:120]}" for f in top_findings)
        gap_goal = (f"Close these GAPS in our understanding of '{topic}': " + " | ".join(gaps) +
                    f". (Cycle-1 already found: {c1_summary}.) Add only what the sources reveal "
                    f"about the gaps; if a gap is already covered, say so briefly.")
        for s in active:
            _log(f"  {s.name} closing gaps…")
            c2.append(make_reasoned_finding(s, gap_goal, src2, model))

    # ── Canonize: INTEGRATE cycle-2 gap-closures into the cycle-1 synthesis (not regenerate) ──
    all_sources = _distinct(ctx1 + ctx2)
    refinements = "\n".join(f"- {f.content}" for f in c2 if f.content.strip()
                            and not f.metadata.get("echo"))
    canon_prompt = (
        f"Refine this synthesis on '{topic}' into its final, defensible form. Integrate the "
        f"gap-closures below into ONE coherent passage; ground EVERY claim with [s:i] and assert "
        f"nothing the sources don't support"
        + (f". Address this critique: {elos_critique}" if elos_critique else "") +
        f".\n\nSYNTHESIS:\n{synthesis}\n\nGAP-CLOSURES:\n{refinements or '(none)'}\n\n"
        f"SOURCES:\n{_format_sources(all_sources, digester)}\n\nFinal synthesis:")
    _log("canonizing — integrating gaps + citation-grounding…")
    final = (synth(canon_prompt) or synthesis).strip()
    synthesis_pre_ground = final                       # capture BEFORE stripping (honest A/B)
    grounded, stripped = _ground_citations(final, len(all_sources))
    # The synthesis is ALWAYS visible (grounded text, or the draft if grounding emptied it) —
    # an empty result hides her reasoning. But canonization to long-term memory is gated on
    # the synthesis actually RESOLVING ≥1 real source citation, so ungrounded claims never
    # pollute the thought-stream (the §A2 discipline) — they're returned for inspection only.
    synthesis_out = grounded or final
    resolved = len(_cited_ids(grounded) & set(range(len(all_sources))))
    _log(f"grounded: {len(grounded)} chars, {resolved} resolved citation(s), "
         f"{stripped} stripped" + ("" if resolved else " — NOT canonized (no grounded support)"))

    # diversity = distinct specialists that genuinely contributed across BOTH cycles
    contributors = {f.specialist_id for f in (c1 + c2)
                    if not f.metadata.get("echo") and not f.metadata.get("empty")}

    thought_id = None
    if thought_stream is not None and grounded and resolved >= 1:
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
        topic=topic, synthesis=synthesis_out, thought_id=thought_id, gaps=gaps,
        n_contributors=len(contributors), n_sources=len(all_sources),
        sources=[c[:300] for c in all_sources],
        synthesis_pre_ground=synthesis_pre_ground,
        n_active=len(active), stripped_claims=stripped,
        elos_critique=elos_critique, cycles=2 if gaps else 1)
