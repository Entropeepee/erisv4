"""
LLM Mediator — Language Production Module (Broca's Area)
=========================================================

The LLM does NOT do the reasoning. The FRACTAL field dynamics, BFECDS
activations, specialist bids, MoEGate selection, and GPW broadcast —
THAT is the cognition. The LLM turns the winning thought into words.

This module is LLM-agnostic: plug in whatever language model is on
the machine. Ollama for local models, Claude API, OpenAI, Gemini,
Cerebras, or any other provider. The system doesn't care which mouth
it speaks through.

Architecture:
    GPW selects winning thought + memory context + field state
    → Mediator assembles prompt
    → Mediator sends to configured LLM backend(s)
    → If racing: first valid response wins (httpx + asyncio.FIRST_COMPLETED)
    → Response returned to post-processing pipeline

Supported backends:
    - OllamaBackend:   Local models via Ollama REST API
    - OpenAIBackend:   OpenAI API (GPT-4, etc.)
    - AnthropicBackend: Claude API
    - GeminiBackend:   Google Gemini API
    - CerebrasBackend: Cerebras fast inference
    - CustomBackend:   Any HTTP endpoint with configurable format

Usage:
    from eris.interface.mediator import LLMMediator, OllamaBackend

    mediator = LLMMediator()
    mediator.add_backend(OllamaBackend(model="llama3.2"))
    response = await mediator.generate(prompt, system_context)

    # Or race multiple:
    mediator.add_backend(OpenAIBackend(model="gpt-4o", api_key=key))
    mediator.add_backend(AnthropicBackend(model="claude-sonnet-4-20250514", api_key=key))
    response = await mediator.race(prompt, system_context)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import asyncio
import json
import os
import time

# Eris's reply length ceiling. 2000 was clipping longer thoughts; give her room
# to actually finish. Override with ERIS_MAX_TOKENS.
_DEFAULT_MAX_TOKENS = int(os.environ.get("ERIS_MAX_TOKENS", "8192"))


@dataclass
class LLMResponse:
    """Response from a language model backend."""
    text: str
    provider: str
    model: str
    latency_ms: float
    tokens_used: int = 0
    reasoning: str = ""
    raw: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        return f"LLMResponse({self.provider}/{self.model}, {self.latency_ms:.0f}ms, {len(self.text)} chars)"


class LLMBackend(ABC):
    """Abstract base for LLM backends. Implement for each provider."""

    name: str = "base"
    model: str = "unknown"

    @abstractmethod
    async def generate(self, prompt: str, system: str = "",
                       max_tokens: int = _DEFAULT_MAX_TOKENS,
                       temperature: float = 0.7) -> LLMResponse:
        """Generate text from a prompt. Must be async."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is configured and reachable."""
        ...


class OllamaBackend(LLMBackend):
    """Local Ollama models. No API key needed."""

    def __init__(self, model: str = "llama3.2",
                 base_url: str = "http://localhost:11434",
                 timeout: float = 300.0):
        self.name = "ollama"
        self.model = model
        self.base_url = base_url
        # Cold-loading a large local model (e.g. gpt-oss:20b ~13GB) on the
        # first turn can take well over a minute; 60s was too short and caused
        # the turn to fall back to the raw specialist bid. Allow a generous
        # ceiling (override with ERIS_OLLAMA_TIMEOUT).
        self.timeout = float(os.environ.get("ERIS_OLLAMA_TIMEOUT", timeout))

    async def generate(self, prompt: str, system: str = "",
                       max_tokens: int = _DEFAULT_MAX_TOKENS,
                       temperature: float = 0.7) -> LLMResponse:
        import httpx
        t0 = time.time()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
        full_text = data.get("response", "")
        reasoning = ""
        text = full_text
        import re
        match = re.search(r"<think>(.*?)</think>", full_text, flags=re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            text = full_text.replace(match.group(0), "").strip()

        return LLMResponse(
            text=text,
            provider="ollama",
            model=self.model,
            latency_ms=(time.time() - t0) * 1000,
            reasoning=reasoning,
            raw=data,
        )

    def is_available(self) -> bool:
        try:
            import httpx
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False


class OpenAIBackend(LLMBackend):
    """OpenAI API (GPT-4o, etc.)"""

    def __init__(self, model: str = "gpt-4o", api_key: str = "", base_url: str = "https://api.openai.com/v1"):
        self.name = "openai"
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    async def generate(self, prompt: str, system: str = "",
                       max_tokens: int = _DEFAULT_MAX_TOKENS,
                       temperature: float = 0.7) -> LLMResponse:
        import httpx
        t0 = time.time()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={"model": self.model, "messages": messages,
                      "max_tokens": max_tokens, "temperature": temperature},
            )
            resp.raise_for_status()
            data = resp.json()

        msg = data["choices"][0]["message"]
        text = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")
        if reasoning and not text:
            text = reasoning
            reasoning = ""
        tokens = data.get("usage", {}).get("total_tokens", 0)
        return LLMResponse(
            text=text, provider="openai", model=self.model,
            latency_ms=(time.time() - t0) * 1000,
            tokens_used=tokens, reasoning=reasoning, raw=data,
        )

    def is_available(self) -> bool:
        return bool(self.api_key)


class AnthropicBackend(LLMBackend):
    """Anthropic Claude API."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str = ""):
        self.name = "anthropic"
        self.model = model
        self.api_key = api_key

    async def generate(self, prompt: str, system: str = "",
                       max_tokens: int = _DEFAULT_MAX_TOKENS,
                       temperature: float = 0.7) -> LLMResponse:
        import httpx
        t0 = time.time()
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(http2=True, timeout=30.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": self.api_key,
                         "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        text = data["content"][0]["text"] if data.get("content") else ""
        return LLMResponse(
            text=text, provider="anthropic", model=self.model,
            latency_ms=(time.time() - t0) * 1000, raw=data,
        )

    def is_available(self) -> bool:
        return bool(self.api_key)


class GeminiBackend(LLMBackend):
    """Google Gemini API."""

    def __init__(self, model: str = "gemini-2.0-flash", api_key: str = ""):
        self.name = "gemini"
        self.model = model
        self.api_key = api_key

    async def generate(self, prompt: str, system: str = "",
                       max_tokens: int = _DEFAULT_MAX_TOKENS,
                       temperature: float = 0.7) -> LLMResponse:
        import httpx
        t0 = time.time()
        contents = []
        if system:
            contents.append({"role": "user", "parts": [{"text": system}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        url = (f"https://generativelanguage.googleapis.com/v1beta/"
               f"models/{self.model}:generateContent?key={self.api_key}")

        async with httpx.AsyncClient(http2=True, timeout=30.0) as client:
            resp = await client.post(url, json={
                "contents": contents,
                "generationConfig": {"maxOutputTokens": max_tokens,
                                     "temperature": temperature},
            })
            resp.raise_for_status()
            data = resp.json()

        text = ""
        if data.get("candidates"):
            parts = data["candidates"][0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)

        return LLMResponse(
            text=text, provider="gemini", model=self.model,
            latency_ms=(time.time() - t0) * 1000, raw=data,
        )

    def is_available(self) -> bool:
        return bool(self.api_key)


class CustomBackend(LLMBackend):
    """Any HTTP endpoint. Configure the URL and payload format."""

    def __init__(self, name: str = "custom", model: str = "custom",
                 url: str = "", headers: Dict[str, str] = None,
                 payload_template: str = '{"prompt": "{prompt}"}',
                 response_path: str = "text"):
        self.name = name
        self.model = model
        self.url = url
        self.headers = headers or {}
        self.payload_template = payload_template
        self.response_path = response_path

    async def generate(self, prompt: str, system: str = "",
                       max_tokens: int = _DEFAULT_MAX_TOKENS,
                       temperature: float = 0.7) -> LLMResponse:
        import httpx
        t0 = time.time()
        # Simple template substitution
        payload_str = (self.payload_template
                       .replace("{prompt}", prompt.replace('"', '\\"'))
                       .replace("{system}", system.replace('"', '\\"'))
                       .replace("{max_tokens}", str(max_tokens)))
        payload = json.loads(payload_str)

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(self.url, headers=self.headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # Navigate response path (e.g., "choices.0.message.content")
        text = data
        for key in self.response_path.split("."):
            if isinstance(text, list):
                text = text[int(key)]
            elif isinstance(text, dict):
                text = text.get(key, "")
            else:
                break
        text = str(text)

        return LLMResponse(
            text=text, provider=self.name, model=self.model,
            latency_ms=(time.time() - t0) * 1000, raw=data,
        )

    def is_available(self) -> bool:
        return bool(self.url)


# ─── The Mediator ─────────────────────────────────────────────────────────

class LLMMediator:
    """LLM-agnostic language production module.

    Add one or more backends. Call generate() for the first available,
    or race() to race all backends and take the fastest valid response.

    The LLM is Broca's area — it produces language from the thought
    that the GPW selected. It does not reason; it speaks.
    """

    def __init__(self):
        self._backends: List[LLMBackend] = []

    def add_backend(self, backend: LLMBackend) -> None:
        self._backends.append(backend)

    @property
    def available_backends(self) -> List[LLMBackend]:
        return [b for b in self._backends if b.is_available()]

    async def generate(self, prompt: str, system: str = "",
                       max_tokens: int = _DEFAULT_MAX_TOKENS,
                       temperature: float = 0.7) -> Optional[LLMResponse]:
        """Generate using the first available backend."""
        for backend in self._backends:
            if not backend.is_available():
                continue
            try:
                return await backend.generate(prompt, system, max_tokens, temperature)
            except Exception as e:
                print(f"[LLMMediator] Backend {backend.__class__.__name__} failed: {e}")
                import traceback
                traceback.print_exc()
                continue  # Try next backend
        return None

    async def race(self, prompt: str, system: str = "",
                   max_tokens: int = _DEFAULT_MAX_TOKENS,
                   temperature: float = 0.7) -> Optional[LLMResponse]:
        """Race all available backends. First valid response wins.

        Uses asyncio.wait with FIRST_COMPLETED. Cancels losers to
        save API costs (per Gemini audit recommendation).
        """
        available = self.available_backends
        if not available:
            return None

        async def _try_backend(backend: LLMBackend) -> LLMResponse:
            return await backend.generate(prompt, system, max_tokens, temperature)

        tasks = [asyncio.create_task(_try_backend(b)) for b in available]

        while tasks:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                try:
                    result = task.result()
                    if result and result.text.strip():
                        # Winner found — cancel all remaining tasks
                        for p in pending:
                            p.cancel()
                        return result
                except Exception:
                    pass  # This backend failed; continue waiting

            tasks = list(pending)

        return None

    async def ensemble(self, prompt: str, system: str = "",
                       max_tokens: int = _DEFAULT_MAX_TOKENS,
                       temperature: float = 0.7) -> List[LLMResponse]:
        """Fire all available backends in parallel for MoE synthesis."""
        available = self.available_backends
        if not available:
            return []

        async def _try_backend(backend: LLMBackend) -> Optional[LLMResponse]:
            try:
                return await backend.generate(prompt, system, max_tokens, temperature)
            except Exception as e:
                print(f"[Ensemble] Backend {backend.name} failed: {e}")
                return None

        tasks = [asyncio.create_task(_try_backend(b)) for b in available]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
