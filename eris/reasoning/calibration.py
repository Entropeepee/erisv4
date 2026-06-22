"""Claim calibration + attribution fidelity (Layer 2 of the deep-reasoning
discipline).

Eris's deep synthesis treated analogy as identity ("the patent IS the collapse
theorem") and fabricated a quote (attributed the *paper's* "GLNCS embodiment"
phrase to the *patent*, where it appears nowhere). The fix has two parts:

  • DETERMINISTIC (a hard guarantee): `verify_quotes` strips any quoted phrase
    that is not verbatim in the cited sources, demoting it to a marked paraphrase.
    This alone kills the fabricated-quote class of error.
  • PROMPT-GUIDED (a critic LLM pass, deep mode): the five-tier claim ladder and
    the critic instructions (verb audit, attribution, scope, framework-overfit,
    field-state, rigor gap) that rewrite the answer with calibrated language.

The goal is calibrated boldness — grounded claims stay strong; analogies are
announced as analogies, not asserted as equivalences.
"""
from __future__ import annotations
from typing import List, Tuple
import re

CLAIM_TIERS = {
    "fact":        "Directly stated in a named source / memory / file / tool result.",
    "inference":   "Logically supported by retrieved facts, but not directly stated.",
    "bridge":      "Creative analogy / cross-domain mapping. Suggestive, NOT demonstrated.",
    "speculation": "Plausible but not yet supported by available sources. A proposed direction.",
    "action":      "A recommendation / next step built on the above.",
}

# Tasks that warrant the full six-section calibrated structure.
_SYNTH_MARKERS = re.compile(
    r"\b(inform|inspire[sd]?|relate[sd]?|compare|contrast|versus|vs\.?|"
    r"map(s|ping)?\s+onto|connect|cross[-\s]?document|cross[-\s]?domain|"
    r"synthesi[sz]e|how\s+does\b.*\b(inform|relate|connect|inspire))\b",
    re.IGNORECASE)

# Identity verbs that overclaim when only an analogy is supported.
_IDENTITY_VERBS = re.compile(
    r"\b(is the|are the|is identical|equals|proves|guarantees|"
    r"manifests|instantiat(es|ed)|is\s+just\s+the|is\s+exactly)\b",
    re.IGNORECASE)

# Quoted spans long enough to be a genuine quotation (not a single word).
_QUOTE_RE = re.compile(r"[\"“”]([^\"“”]{12,240})[\"“”]")


def is_synthesis_task(query: str, named_sources: int = 0) -> bool:
    """Synthesis-across-sources tasks (patent review, cross-document mapping,
    code-vs-paper) get the full structure; simple deep questions don't."""
    return bool(_SYNTH_MARKERS.search(query or "")) or named_sources >= 2


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def verify_quotes(answer: str, sources_text: str) -> Tuple[str, List[str]]:
    """Demote any quoted phrase NOT verbatim in `sources_text` to a marked
    paraphrase. Returns (cleaned_answer, list_of_unverified_quotes). Deterministic
    — this is the hard guard against fabricated quotes."""
    if not answer:
        return answer, []
    src = _normalize(sources_text)
    flags: List[str] = []

    def _repl(m: re.Match) -> str:
        q = m.group(1)
        if _normalize(q) in src:
            return m.group(0)                       # verbatim -> keep the quote
        flags.append(q)
        return f"(paraphrase, not a verbatim quote: {q})"

    return _QUOTE_RE.sub(_repl, answer), flags


def has_identity_overreach(text: str) -> bool:
    """Cheap signal that a claim uses an identity verb (candidate for downgrade)."""
    return bool(_IDENTITY_VERBS.search(text or ""))


def calibration_system(is_synthesis: bool, regime: str = "") -> str:
    """The critic instructions for the deep-mode rewrite pass."""
    plastic = regime == "plastic"
    base = (
        "You are Eris, running a CALIBRATION CRITIC over your own draft answer. "
        "Keep your boldness and cross-domain creativity — do NOT become timid or "
        "generic — but make the epistemic status of every load-bearing claim "
        "explicit. Distinguish what you HAVE (from the sources) from what you MAKE "
        "(your synthesis).\n\n"
        "Apply this discipline and rewrite the answer:\n"
        "1) VERB AUDIT. If you wrote 'is / proves / guarantees / implements / "
        "manifests / is identical' where the support only justifies 'resembles / "
        "suggests / is inspired by / is analogous to / could be formalized as', "
        "downgrade the verb. Identity requires demonstrated equivalence, not a "
        "strong analogy.\n"
        "2) ATTRIBUTION + QUOTE FIDELITY (most important). Every 'what a source "
        "says' claim must name WHICH source. Never put document A's content in "
        "document B's mouth. Any quotation must be verbatim from the named source; "
        "if you cannot verify it there, paraphrase and say so — never present an "
        "unverified phrase as a quote. For a cross-document claim ('X informs Y'), "
        "keep X-content and Y-content separately sourced, and label the mapping "
        "itself as inference or bridge.\n"
        "3) SCOPE. Do not overstate what a source claims. A paper that 'unifies/"
        "identifies' methods has NOT proved that some other artifact IS the method.\n"
        "4) FRAMEWORK-OVERFIT. Do not project the user's own framework vocabulary "
        "onto a document that only partially supports it just because the words are "
        "familiar or it's their work.\n"
        "5) RIGOR GAP. For each analogy (bridge) or speculation, state briefly what "
        "would make it rigorous.\n")
    if plastic:
        base += ("FIELD STATE: this answer was generated in a PLASTIC (reshaping) "
                 "regime — generative. Default any cross-domain mapping to a 'bridge' "
                 "(announced analogy) unless equivalence is actually demonstrated.\n")
    if is_synthesis:
        base += (
            "\nStructure the rewrite in these sections:\n"
            "1. Core answer — the bold synthesis, stated clearly.\n"
            "2. What is directly grounded — facts, each with its named source.\n"
            "3. What I infer — inferences from those facts.\n"
            "4. What is an interpretive bridge — analogies, explicitly marked.\n"
            "5. What may be overclaiming — claims you softened, and why.\n"
            "6. Next step to make it rigorous — what would upgrade a bridge to a "
            "demonstrated equivalence.\n")
    else:
        base += ("\nKeep it a single thoughtful answer (no section scaffold), just "
                 "with calibrated verbs and honest attribution.\n")
    return base
