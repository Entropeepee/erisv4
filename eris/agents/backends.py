"""
eris/agents/backends.py
=======================
Per-agent LLM backends (WILLOW I.4). Reuses the existing mediator backends so a
node can run on the local Ollama (Willow + Eris — your IP, never leaves the
machine) or, for peripheral flavor NPCs, a cloud backend when its key is set.

ONE Ollama backend object is shared by all local nodes — the model is loaded
once, not per node.
"""
from __future__ import annotations

import os
from typing import Dict

from eris.interface.mediator import (
    LLMBackend, OllamaBackend, AnthropicBackend, OpenAIBackend, GeminiBackend,
)


def build_backends() -> Dict[str, LLMBackend]:
    backends: Dict[str, LLMBackend] = {
        "ollama": OllamaBackend(model=os.environ.get("ERIS_LOCAL_MODEL", "gpt-oss:20b")),
    }
    for key, cls, env in (("anthropic", AnthropicBackend, "ANTHROPIC_API_KEY"),
                          ("openai", OpenAIBackend, "OPENAI_API_KEY"),
                          ("gemini", GeminiBackend, "GEMINI_API_KEY")):
        api_key = os.environ.get(env, "")
        if api_key:
            backends[key] = cls(api_key=api_key)
    return backends
