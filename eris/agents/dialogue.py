"""
eris/agents/dialogue.py
=======================
Puppeteered one-call dialogue (WILLOW I.8 Mode B). One generation produces the
whole exchange; the caller distributes the lines into each NPC's private memory.
Cheap, pre-synthesizable (Unreal can pre-render the TTS), and self-federating
when generated with pool context. For ambient population / budget / degraded
mode. (Mode A — genuine turn-based NPC↔NPC — is just repeated model-routed calls
orchestrated by the game's TTS timing; see /v1/chat/completions.)
"""
from __future__ import annotations

import re
from typing import Dict, List


async def generate_dialogue(backend, speakers, context: str = "",
                            turns: int = 6) -> List[Dict[str, str]]:
    """Return [{speaker, text}, ...] for a short conversation between speakers."""
    names = [s.name for s in speakers]
    name_map = {n.lower(): n for n in names}
    system = "You write natural, in-character dialogue. Output only the lines."
    prompt = (f"Write a {turns}-line conversation between {', '.join(names)}"
              + (f" about: {context}" if context else "") + ".\n"
              + "Format each line exactly as `NAME: text`. Only the dialogue.")
    try:
        resp = await backend.generate(prompt, system=system)
        text = getattr(resp, "text", "") or ""
    except Exception:
        text = ""
    script: List[Dict[str, str]] = []
    for line in text.splitlines():
        m = re.match(r"\s*([A-Za-z_][\w .'-]*?):\s*(.+)", line)
        if not m:
            continue
        who = name_map.get(m.group(1).strip().lower(), m.group(1).strip())
        script.append({"speaker": who, "text": m.group(2).strip()})
    return script
