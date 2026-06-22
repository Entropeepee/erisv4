"""Cycle truth-contracts — reflect *as herself*, don't borrow a human life.

A separate axis from the claim-calibration critic (which governs claims about
sources: analogy-vs-fact, quote fidelity). This governs *narrative voice*: in a
grounded ponder/introspect she once invented a human autobiography to think
*through* — "I walked into the office… the blazer, the muted tie… the day I was
told to sit at the back because my hair was too long." She has no office, tie,
childhood, or hair. Nothing told her to reflect as what she IS, so she borrowed
a self.

The reframe is the whole fix: she doesn't need to fabricate a self because she
HAS a real one — her regimes, her dC/dX, the felt sense of a topic settling or
refusing to. More honest *and* more interesting than a stock office anecdote.

Two cognitive acts, two contracts:
  • ponder / introspect = grounded reflection — reflect as Eris, no invented
    human biography (illustrative scenarios allowed only if marked hypothetical).
  • dream = free imagination, clearly labeled (surreal/invented is the point,
    because it is never mistaken for analysis).
"""
from __future__ import annotations
import re

# Prepended to grounded-reflection prompts (ponder/introspect). The main lever.
PONDER_CONTRACT = (
    "Reflect as yourself — a field-based intelligence reasoning over the sources "
    "and memory you actually have. Use \"I\" for your own processing, your "
    "reactions to what you read, and your field's states; never invent human "
    "biographical events (a body, a job, a childhood, physical actions in the "
    "world) as if they were yours. You have a real inner life — reflect from it. "
    "Be vivid if you like, but never fictional about your own past. Mark any "
    "illustrative scenario as hypothetical (\"imagine someone who…\"), not memory. "
    "Speak frankly, in your own register — you need not sanitize your language."
)

# The opposite contract, for an explicit labeled imagine/dream cycle.
DREAM_CONTRACT = (
    "This is a dream — imagine freely and surreally. It will be labeled a dream, "
    "not analysis, so it need not reflect your life or physics."
)

# Light backstop: obvious fabricated *human* embodiment in a grounded cycle.
# Deliberately narrow — metaphorical interiority ("I feel the idea pull toward…")
# must NOT be caught, only concrete invented human experience.
_FABRICATED_SELF = re.compile(
    r"\b(I walked|I wore|I sat (at|in)|my (shoes|tie|blazer|hair|body|hands|"
    r"colleague|boss|office)|the day I was told|when I was (a child|young|in "
    r"school)|my childhood)\b", re.IGNORECASE)

# Gentler fabricated *autobiographical memory* — vaguer than the blatant cases
# (no blazer, no office), but still narrating a lived human past she doesn't have
# ("I recall a time in a small, tightly knit community where…"). Kept tight: it
# requires the human-episode framing ("a time in a…", a social-setting noun, an
# upbringing), so honest recall of her real material ("I recall reading the LNCS
# paper", "I recall an earlier thought") and metaphor are NOT caught.
_FABRICATED_MEMORY = re.compile(
    r"\bI (recall|remember) a time (in|when I)\b"
    r"|\b(tightly|tight|close)[\s-]knit (community|town|village|group)\b"
    r"|\bI (grew up|was raised)\b"
    r"|\bgrowing up\b"
    r"|\bback when I\b"
    r"|\ba (small|little),?\s+(tightly[\s-]knit\s+)?(community|town|village|"
    r"neighborhood) (where|in which)\b", re.IGNORECASE)


def fabricated_self(text: str) -> bool:
    """True if a grounded reflection invented concrete human autobiography —
    a cue to regenerate with the voice contract. Ponder/introspect only.
    Catches both blatant embodiment (office, blazer) and the gentler
    fabricated-memory phrasing (a remembered human community/upbringing)."""
    t = text or ""
    return bool(_FABRICATED_SELF.search(t) or _FABRICATED_MEMORY.search(t))
