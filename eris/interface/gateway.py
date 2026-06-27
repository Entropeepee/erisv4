"""
eris/interface/gateway.py
=========================
The Contractor Layer's NON-sensitive control plane (contractor_gateway_layer_spec §4/§7/§8).

A thin wiring layer *behind* the existing mediator — it does NOT replace LLMMediator,
the backends, or the router. It builds cost-tiered backends that point at a self-hosted
LiteLLM gateway (one OpenAI-compatible endpoint that owns failover/caching/keys/cost),
plus the separate frontier-synthesis backend (Option A: Claude via the Agent SDK on the
subscription credit, NOT an API key).

Drift note vs spec §8.2: the spec says register the gateway as `CustomBackend(base_url=…)`,
but CustomBackend takes a `url` + crude string `payload_template` and is not OpenAI-shaped.
`OpenAIBackend` already speaks the `/chat/completions` format LiteLLM exposes and takes
`base_url` — so gateway tiers are OpenAIBackend instances (renamed `gateway-*`), matching
the existing `embed_base_url` OpenAI-compatible-service convention.

Everything here is OPT-IN: unset `ERIS_GATEWAY_BASE_URL` ⇒ `ContractorGateway.enabled` is
False and nothing changes. The sovereign path never touches any of this.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from eris.interface.mediator import LLMBackend, LLMMediator, LLMResponse, OpenAIBackend


# ─── Semantic/exact cache wrapper ──────────────────────────────────────────
class CachingBackend(LLMBackend):
    """Wrap a backend with a prompt cache so a repeated `open` prompt is served without a
    second upstream call (§10.5). LiteLLM/Redis does the real semantic cache; this in-process
    wrapper makes the behavior deterministic and testable, and helps even without Redis.

    Keyed on normalized (system, prompt). `hits`/`misses` are exposed for assertions."""

    def __init__(self, inner: LLMBackend, max_entries: int = 1024):
        self.inner = inner
        self.name = f"cached:{getattr(inner, 'name', 'backend')}"
        self.model = getattr(inner, "model", "")
        self._cache: "Dict[str, LLMResponse]" = {}
        self._order: List[str] = []
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(prompt: str, system: str) -> str:
        norm = lambda s: " ".join((s or "").lower().split())
        return norm(system) + "\x00" + norm(prompt)

    async def generate(self, prompt: str, system: str = "", max_tokens: int = 8192,
                       temperature: float = 0.7) -> LLMResponse:
        k = self._key(prompt, system)
        cached = self._cache.get(k)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        resp = await self.inner.generate(prompt, system, max_tokens, temperature)
        self._cache[k] = resp
        self._order.append(k)
        if len(self._order) > self.max_entries:
            self._cache.pop(self._order.pop(0), None)
        return resp

    def is_available(self) -> bool:
        return self.inner.is_available()


# ─── Frontier synthesis (Option A: Claude via the Agent SDK, subscription credit) ──
class ClaudeAgentSDKBackend(LLMBackend):
    """One-shot synthesis via `claude-agent-sdk` on the Max-plan Agent SDK credit — auth is
    subscription OAuth, NOT an API key (§6, Option A).

    ⚠️ Key-bypass guard: if ANTHROPIC_API_KEY is set, the SDK bills pay-as-you-go and bypasses
    the subscription credit. This backend therefore reports UNAVAILABLE (and warns loudly) when
    that var is set, and generate() raises — it never silently spends pay-go money."""

    def __init__(self, model: str = "", system_hint: str = ""):
        self.name = "claude-agent-sdk"
        self.model = model or os.environ.get("ERIS_SYNTH_MODEL", "")
        self.system_hint = system_hint
        self._warned = False

    @staticmethod
    def key_bypass_risk() -> bool:
        """True if ANTHROPIC_API_KEY is set — using the SDK now would bypass the credit."""
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _warn_key_once(self) -> None:
        if not self._warned:
            print("[gateway] WARNING: ANTHROPIC_API_KEY is set — the Agent-SDK synth path "
                  "would bill PAY-AS-YOU-GO and bypass your subscription credit. Disabling it. "
                  "Unset ANTHROPIC_API_KEY to use the credit, or route synth via the gateway "
                  "(Option B) deliberately.")
            self._warned = True

    def _sdk_available(self) -> bool:
        try:
            import importlib.util
            return importlib.util.find_spec("claude_agent_sdk") is not None
        except Exception:
            return False

    def is_available(self) -> bool:
        if self.key_bypass_risk():
            self._warn_key_once()
            return False
        return self._sdk_available()

    async def generate(self, prompt: str, system: str = "", max_tokens: int = 8192,
                       temperature: float = 0.7) -> LLMResponse:
        if self.key_bypass_risk():
            self._warn_key_once()
            raise RuntimeError(
                "Refusing Agent-SDK synth: ANTHROPIC_API_KEY is set (would bypass the "
                "subscription credit). Unset it to use the credit.")
        import time
        from claude_agent_sdk import query  # lazy — only when actually used
        t0 = time.time()
        sys_text = (system or self.system_hint or "").strip()
        full = (f"{sys_text}\n\n{prompt}" if sys_text else prompt)
        chunks: List[str] = []
        async for msg in query(prompt=full):           # one-shot synthesis agent
            text = getattr(msg, "text", None) or getattr(msg, "content", None)
            if isinstance(text, str):
                chunks.append(text)
        return LLMResponse(text="".join(chunks).strip(), provider="claude-agent-sdk",
                           model=self.model or "claude", latency_ms=(time.time() - t0) * 1000)


# ─── The gateway wiring ────────────────────────────────────────────────────
class ContractorGateway:
    """Builds the cost-tiered `open`-path backends from config. Inert unless
    ERIS_GATEWAY_BASE_URL is set. `backend_factory` is injectable for tests."""

    def __init__(self, config=None,
                 backend_factory: Optional[Callable[[str, str], LLMBackend]] = None,
                 synth_factory: Optional[Callable[[], LLMBackend]] = None):
        from eris.config import CONFIG
        self.cfg = config or CONFIG
        self._factory = backend_factory or self._default_factory
        self._synth_factory = synth_factory or (lambda: ClaudeAgentSDKBackend())
        self._cache: Dict[str, LLMBackend] = {}

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.cfg, "gateway_base_url", ""))

    def _default_factory(self, group_name: str, display_name: str) -> LLMBackend:
        """A gateway tier = an OpenAIBackend pointed at the LiteLLM endpoint, asking for a
        model-GROUP (LiteLLM resolves the group to a concrete provider + does failover)."""
        b = OpenAIBackend(model=group_name, api_key=self.cfg.gateway_api_key,
                          base_url=self.cfg.gateway_base_url)
        b.name = display_name                          # 'gateway-free'/'gateway-cheap' (non-local)
        return CachingBackend(b)

    def tier(self, which: str) -> Optional[LLMBackend]:
        """Return the backend for a tier: 'free' | 'cheap' | 'synth'. None if gateway off
        (free/cheap) — synth is independent of the gateway and may exist regardless."""
        if which == "synth":
            return self._cache.setdefault("synth", self._synth_factory())
        if not self.enabled:
            return None
        if which == "free":
            return self._cache.setdefault("free", self._factory(self.cfg.tier_free, "gateway-free"))
        if which == "cheap":
            return self._cache.setdefault("cheap", self._factory(self.cfg.tier_cheap, "gateway-cheap"))
        raise ValueError(f"unknown tier {which!r}")

    def open_mediator(self) -> LLMMediator:
        """An LLMMediator for `open` calls with the tier backends in failover order
        (free → cheap). The mediator's existing cascade IS the erisv4-side failover; LiteLLM
        adds its own on top. Empty mediator if the gateway is off (caller falls back to local)."""
        m = LLMMediator()
        for which in ("free", "cheap"):
            b = self.tier(which)
            if b is not None:
                m.add_backend(b)
        return m
