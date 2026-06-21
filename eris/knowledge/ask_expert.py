"""
eris/knowledge/ask_expert.py
============================
External-expert research path (Remediation Tier 4.2). When Eris's field has
unresolved coupling that web search could not close, she asks an external LLM
(Claude) ONE focused question. The answer is INGESTED AS GROUNDING — it is not
spoken in Eris's own voice. The LLM is an oracle here, not Broca's area.

Design constraints (the lesson from the ensemble-timeout slowness bug):
  * Wired now, DORMANT until ANTHROPIC_API_KEY is set in the environment.
  * Returns None IMMEDIATELY if no key — never blocks the turn.
  * Short timeout + total exception safety — a failed/credit-less call must
    never break Eris.

Setup later:  pip install anthropic   and   set ANTHROPIC_API_KEY=sk-ant-...
Model defaults to env ERIS_EXPERT_MODEL or 'claude-sonnet-4-20250514' (matches
the AnthropicBackend default in eris.interface.mediator).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

_DEFAULT_MODEL = os.environ.get("ERIS_EXPERT_MODEL", "claude-sonnet-4-20250514")


@dataclass
class ExpertAnswer:
    question: str
    answer: str
    model: str


def is_available() -> bool:
    """True only if an API key is present. Use this to decide whether to escalate."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def ask(question: str, *, context: str = "",
        model: Optional[str] = None,
        max_tokens: int = 1024, timeout_s: float = 30.0) -> Optional[ExpertAnswer]:
    """Ask the external expert one focused question.

    Returns None instantly if no key (dormant), or on any failure. Returns an
    ExpertAnswer on success. Caller should treat None as 'escalation
    unavailable / failed' and fall back to whatever it has.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None  # dormant until keyed — never hangs

    try:
        import anthropic  # pip install anthropic
    except ImportError:
        print("[ask_expert] anthropic package not installed; escalation skipped")
        return None

    model = model or _DEFAULT_MODEL
    client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s)
    system = (
        "You are a precise research assistant for an autonomous reasoning system "
        "that is missing one specific piece of information. Answer factually and "
        "concisely. If the premise is false, unverifiable, or you are unsure, say "
        "so plainly rather than guessing."
    )
    user = question if not context else (
        f"Information the system already has:\n{context}\n\nFocused question: {question}"
    )

    try:
        msg = client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            b.text for b in msg.content if getattr(b, "type", None) == "text"
        ).strip()
        return ExpertAnswer(question=question, answer=text, model=model)
    except Exception as e:  # network / credit / rate-limit / API error
        print(f"[ask_expert] call failed (non-fatal): {e}")
        return None
