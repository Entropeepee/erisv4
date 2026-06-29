"""Claim-support scorer (QUOTE-AND-VERIFY) — the SINGLE substance check that replaces resolve-only
grounding AND is the Phase-3 faithfulness metric (design once, serve both).

The weak pattern this replaces (calibration.py verify_grounding, retrospect unresolved-id strip,
research.py `resolved>=1` gate) only checks that a cited id *resolves* — a fabrication with a live
citation id is canonized as fact. Here, for each claim + its cited source, a local model returns a
LABEL and must QUOTE the verbatim sentence(s) from the source that justify it. The quote is then
VERIFIED to actually occur in the source — if it doesn't, the label is forced to UNSUPPORTED. The
model cannot earn "support" by asserting it; it must point to real text.

  SUPPORTED    — the source STATES the claim             -> tier 'fact'
  INFERRED     — the source IMPLIES it via specific spans -> tier 'inference' (kept WITH provenance:
                 the verified spans + a one-line reason)
  UNSUPPORTED  — the source does not support it           -> tier 'speculation' (never canonized)
  CONTRADICTED — the source asserts the opposite          -> tier 'speculation' (never canonized)

Phase-3 faithfulness = fraction of claims SUPPORTED/INFERRED *with a verified span*; the
UNSUPPORTED/CONTRADICTED fraction is the hallucination rate. Same scorer, two uses.

`model(prompt)->text` is injected, so the whole thing is offline-testable with a stub (like the
rest of the tribe). NO lexical-overlap or hybrid pre-gate — overlap is the wrong core; an
embedding-relevance pre-filter is a later cost optimization only.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Tuple
import difflib
import re

Model = Callable[[str], str]

LABELS = ("SUPPORTED", "INFERRED", "UNSUPPORTED", "CONTRADICTED")
_FAITHFUL = ("SUPPORTED", "INFERRED")            # count as faithful ONLY with a verified span
_TIER = {"SUPPORTED": "fact", "INFERRED": "inference",
         "UNSUPPORTED": "speculation", "CONTRADICTED": "speculation"}


@dataclass
class ClaimVerdict:
    claim: str
    label: str                                   # final label (after quote-verification)
    quoted_spans: List[str] = field(default_factory=list)    # spans the model offered
    verified_spans: List[str] = field(default_factory=list)  # the subset that ACTUALLY occur in source
    reason: str = ""
    model_label: str = ""                        # the model's raw label BEFORE verification (audit)

    @property
    def verified(self) -> bool:
        return bool(self.verified_spans)

    @property
    def tier(self) -> str:
        return _TIER.get(self.label, "speculation")

    @property
    def is_faithful(self) -> bool:
        """Faithful iff SUPPORTED/INFERRED AND a quoted span was verified in the source."""
        return self.label in _FAITHFUL and self.verified

    def provenance(self) -> dict:
        """For an INFERRED claim: the verified spans + reason kept as provenance (the 'inference tier'
        carries WHY it was inferred, not a bare assertion)."""
        return {"tier": self.tier, "spans": list(self.verified_spans), "reason": self.reason}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def span_occurs(span: str, source: str, *, fuzzy: float = 0.9) -> bool:
    """True if `span` really occurs in `source`: exact normalized substring, else a fuzzy match
    (>= ratio) against any source sentence — tolerates minor whitespace/punctuation/casing drift
    from the model's copy. Spans shorter than 8 normalized chars never count (a stray word isn't a
    quote, and would match trivially)."""
    s = _norm(span)
    if len(s) < 8:
        return False
    src = _norm(source)
    if s in src:
        return True
    for chunk in re.split(r"(?<=[.!?])\s+", source):
        c = _norm(chunk)
        if not c:
            continue
        if difflib.SequenceMatcher(None, s, c).ratio() >= fuzzy:
            return True
        if len(s) < len(c) and difflib.SequenceMatcher(None, s, c[:len(s) + 20]).ratio() >= fuzzy:
            return True
    return False


_LABEL_RE = re.compile(r"\bLABEL\s*[:=]\s*(SUPPORTED|INFERRED|UNSUPPORTED|CONTRADICTED)", re.I)
_REASON_RE = re.compile(r"\bREASON\s*[:=]\s*(.+)", re.I)


def _parse(raw: str) -> Tuple[str, List[str], str]:
    """Pull (label, spans, reason) from the model reply — robust to formatting drift."""
    text = raw or ""
    m = _LABEL_RE.search(text)
    label = m.group(1).upper() if m else "UNSUPPORTED"
    spans: List[str] = []
    qm = re.search(r"\bQUOTE\s*[:=]\s*(.+?)(?:\n\s*REASON\b|\Z)", text, re.I | re.S)
    if qm:
        blob = qm.group(1)
        quoted = re.findall(r'"([^"]{4,})"|“([^”]{4,})”', blob)
        spans = [a or b for a, b in quoted]
        if not spans:                            # no quote marks → take the line(s) verbatim
            spans = [ln.strip(' "“”') for ln in blob.splitlines() if ln.strip()]
    rm = _REASON_RE.search(text)
    reason = rm.group(1).strip() if rm else ""
    return label, [s for s in spans if s.strip()], reason


_PROMPT = (
    "You are checking whether a SOURCE supports a CLAIM. Reply in EXACTLY this format, nothing else:\n"
    "LABEL: <SUPPORTED|INFERRED|UNSUPPORTED|CONTRADICTED>\n"
    'QUOTE: "<the verbatim sentence(s) copied EXACTLY from the SOURCE that justify the label; empty if UNSUPPORTED>"\n'
    "REASON: <one short line>\n\n"
    "Definitions:\n"
    "- SUPPORTED: the SOURCE explicitly STATES the claim (quote the sentence that says it).\n"
    "- INFERRED: the SOURCE does not state it, but specific sentences clearly IMPLY it (quote those).\n"
    "- UNSUPPORTED: the SOURCE does not support the claim.\n"
    "- CONTRADICTED: the SOURCE asserts the opposite (quote it).\n"
    "Copy the QUOTE VERBATIM from the SOURCE — do NOT paraphrase. If you cannot quote supporting "
    "text, the label is UNSUPPORTED.\n\n"
    "CLAIM: {claim}\n\nSOURCE:\n{source}\n"
)


def judge_claim(claim: str, source: str, model: Model) -> ClaimVerdict:
    """Judge one claim against its cited source with quote-and-verify. A SUPPORTED/INFERRED/
    CONTRADICTED label is honored ONLY if backed by a span that actually occurs in the source;
    otherwise it is forced to UNSUPPORTED (no real quote = no support). Never raises."""
    if not (claim or "").strip() or not (source or "").strip():
        return ClaimVerdict(claim=claim, label="UNSUPPORTED", reason="empty claim or source")
    try:
        raw = model(_PROMPT.format(claim=claim, source=source)) or ""
    except Exception as e:                       # a judge failure must demote, never crash grounding
        return ClaimVerdict(claim=claim, label="UNSUPPORTED", reason=f"judge call failed: {e}")
    model_label, spans, reason = _parse(raw)
    verified = [s for s in spans if span_occurs(s, source)]
    label = model_label
    if model_label in ("SUPPORTED", "INFERRED", "CONTRADICTED") and not verified:
        label = "UNSUPPORTED"
        reason = (reason + " | forced UNSUPPORTED: quoted span not found in source").strip(" |")
    return ClaimVerdict(claim=claim, label=label, quoted_spans=spans, verified_spans=verified,
                        reason=reason, model_label=model_label)


def score_claims(claim_sources: List[Tuple[str, str]], model: Model) -> List[ClaimVerdict]:
    """Judge many (claim, source) pairs."""
    return [judge_claim(c, s, model) for (c, s) in claim_sources]


def faithfulness(verdicts: List[ClaimVerdict]) -> dict:
    """The Phase-3 faithfulness metric over judged claims. faithful = fraction SUPPORTED/INFERRED
    *with a verified span*; hallucination_rate = the UNSUPPORTED/CONTRADICTED fraction. The SAME
    scorer that gates grounding — one definition, two uses."""
    n = len(verdicts)
    if not n:
        return {"n": 0, "faithful": 0, "faithfulness": 0.0, "hallucination_rate": 0.0,
                "supported": 0, "inferred": 0, "unsupported": 0, "contradicted": 0}
    faithful = sum(1 for v in verdicts if v.is_faithful)
    by = {lab: sum(1 for v in verdicts if v.label == lab) for lab in LABELS}
    return {"n": n, "faithful": faithful,
            "faithfulness": round(faithful / n, 4),
            "hallucination_rate": round((n - faithful) / n, 4),
            "supported": by["SUPPORTED"], "inferred": by["INFERRED"],
            "unsupported": by["UNSUPPORTED"], "contradicted": by["CONTRADICTED"]}
