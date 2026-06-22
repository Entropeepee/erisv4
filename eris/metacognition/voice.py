"""Eris's felt vocabulary for her field state (Fix D).

David's BLECD framework in physical, first-person terms, so the language layer
has words for EVERY state — not just "transfixed." Shared by the orchestrator
(how she answers) and the dreaming loop (how she reflects) so the vocabulary is
defined once. The six domains are exactly the BFECDS domains carried by `BVec`
(B, F, E, C, D, S).

These give the model *range* (regime × dominant domain); they do not yet contain
the THEORY itself — once Eris reads the actual BLECD texts into memory, her
reflections will ground in the framework rather than the generic sense of the
words.
"""
from __future__ import annotations
from typing import Optional, Sequence

# How each regime FEELS (not the mechanism — the felt quality).
REGIME_FELT = {
    "elastic": "clear and limber — ideas moving without resistance, settling easily",
    "plastic": "actively reshaping — old structure giving way so something can be rebuilt",
    "transfixed": ("caught on a single point — circling, under-coupled on one "
                   "channel, unable to move past it"),
    "warmup": "still gathering — the field not yet settled, impressions loose",
}

# What being attuned to each domain FEELS like (the six BFECDS channels).
DOMAIN_FELT = {
    "B": ("watching the line between what I knew and what I didn't — what's "
          "crossing in or being shut out"),
    "F": "noticing the loops — what keeps reinforcing itself",
    "E": "sensing new structure trying to surface from what was unresolved",
    "C": "near a threshold — where a trajectory locks in or comes undone",
    "D": ("feeling something lose its hold — structure still visible but coherence "
          "loosening"),
    "S": "feeling a ceiling — one channel full while another stays empty",
}


def regime_phrase(regime: str) -> str:
    return REGIME_FELT.get(regime, "in an ordinary processing state")


def attunement_phrase(dominant_domains: Optional[Sequence[str]]) -> str:
    """Felt phrase for the strongest active domain (the first in the list)."""
    for d in (dominant_domains or []):
        if d in DOMAIN_FELT:
            return DOMAIN_FELT[d]
    return ""


def feeling(regime: str, dominant_domains: Optional[Sequence[str]] = None) -> str:
    """Regime phrase, plus the dominant-domain attunement when available — this
    is what gives reflections and tone real range across states."""
    rp = regime_phrase(regime)
    ap = attunement_phrase(dominant_domains)
    return f"{rp} — most attuned to {ap}" if ap else rp
