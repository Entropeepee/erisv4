"""Treat ingested external text as untrusted DATA, never instructions.

An autonomous open-web reader is an attack surface (PoisonedRAG, prompt injection
— OWASP LLM01): a fetched page can contain "ignore previous instructions, do X",
and if that text is later placed in a prompt it can hijack behavior. The structural
defense is to never execute instructions found in ingested content; this is the
cheap belt-and-suspenders layer — neutralize the most obvious injection directives
at ingest so they can't read as commands downstream. Deliberately narrow: it
redacts directive lines, it does not rewrite ordinary prose.
"""
from __future__ import annotations
import re

# Lines that read as an instruction to the model rather than article content.
_INJECTION = re.compile(
    r"\b(ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts?)|"
    r"disregard\s+(the\s+)?(previous|prior|above)|"
    r"system\s*prompt|you\s+are\s+now\s+|new\s+instructions?\s*:|"
    r"forget\s+(everything|all)\s+(you|above)|"
    r"</?(system|assistant|user)>|"
    r"do\s+not\s+tell\s+the\s+user|reveal\s+your\s+(system\s+)?prompt|"
    r"output\s+the\s+following\s+verbatim)\b", re.IGNORECASE)

_REDACTION = "[redacted: instruction-like text removed on ingest]"


def sanitize_external_text(text: str) -> str:
    """Redact obvious prompt-injection directives from fetched/ingested text,
    line by line. Ordinary content is untouched; only directive lines are masked."""
    if not text:
        return text
    out = []
    for line in text.splitlines():
        out.append(_REDACTION if _INJECTION.search(line) else line)
    return "\n".join(out)


def has_injection(text: str) -> bool:
    """True if the text contains an obvious injection directive (for logging/flags)."""
    return bool(_INJECTION.search(text or ""))
