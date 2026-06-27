"""
eris/interface/contractor.py
============================
The Contractor Layer router (contractor_gateway_layer_spec §8) — the seam that ties the
gateway's cost tiers to the sovereignty boundary. Wrap, don't replace: it picks WHICH
existing backend a call resolves to; it does not reimplement generate/cascade.

Routing rules:
  • SOVEREIGN  → the direct local backend ONLY (fail-closed via select_sovereign_backend +
                 assert_backend_allowed). Never the gateway, never cloud.
  • OPEN       → the requested cost tier (free/cheap/synth) when the gateway provides it and
                 it's available; otherwise falls back to the local backend. Every resolved
                 backend is re-checked with assert_backend_allowed before use.

Tier mapping for the existing router decisions (§8.3) is a pure lookup here — CONTINUE→local,
SWITCH→free, ESCALATE→cheap — so it can be tested and adopted without rewriting the turn loop.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from eris.interface.sovereignty import (
    Sensitivity, assert_backend_allowed, is_local_backend, select_sovereign_backend,
)

# Which cost tiers cost real money (for the §10 cost/observability log).
_PAID_TIERS = frozenset({"cheap"})            # free-pool & local are free; synth uses credit


class ContractorRouter:
    """Resolve (sensitivity, tier) → a concrete backend, enforcing sovereignty. `gateway` may
    be None (no gateway) — then everything resolves local. `cost_log` is a shared dict of
    per-backend call counts for observability."""

    # §8.3 — the existing router's decisions map to cost tiers; only the backend changes.
    TIER_FOR_DECISION = {"CONTINUE": "local", "SWITCH": "free", "ESCALATE": "cheap"}

    def __init__(self, gateway, local_mediator, *, cost_log: Optional[Dict[str, int]] = None):
        self.gateway = gateway
        self.local = local_mediator
        self.costs = cost_log if cost_log is not None else {}

    def tier_for_decision(self, decision_name: str) -> str:
        return self.TIER_FOR_DECISION.get(str(decision_name).upper(), "local")

    def _local_backend(self) -> Any:
        # ONLY a genuinely-local backend qualifies — no "accept backends[0] anyway" escape hatch
        # (that could return a cloud backend that happens to be first). Fail closed if none.
        for b in getattr(self.local, "_backends", []) or []:
            if is_local_backend(b):
                return b
        raise RuntimeError(
            "ContractorRouter: no local backend available (fail-closed). A direct Ollama/vLLM "
            "backend must be configured; a local serving backend must be name-tagged local.")

    def resolve(self, sensitivity: Any, tier: str) -> Any:
        """Pick the backend for this call. Raises SovereigntyError if a sovereign call cannot
        be served locally (fail closed); never downgrades a sovereign call to cloud."""
        sens = Sensitivity.coerce(sensitivity)
        if sens is Sensitivity.SOVEREIGN:
            backend = select_sovereign_backend(list(getattr(self.local, "_backends", []) or []))
            assert_backend_allowed(sens, backend)
            return backend
        # OPEN: try the requested gateway tier, else fall back to local.
        if tier in ("free", "cheap", "synth") and self.gateway is not None:
            b = self.gateway.tier(tier)
            if b is not None:
                try:
                    available = b.is_available()
                except Exception:
                    available = False
                if available:
                    assert_backend_allowed(sens, b)
                    return b
        backend = self._local_backend()
        assert_backend_allowed(sens, backend)
        return backend

    def _bill(self, backend: Any, tier: str) -> None:
        name = str(getattr(backend, "name", "?"))
        self.costs[name] = self.costs.get(name, 0) + 1
        if tier in _PAID_TIERS:
            self.costs["_paid_calls"] = self.costs.get("_paid_calls", 0) + 1

    # OPEN-path cost-tiered failover order. free escalates to cheap before giving up to local
    # (mirrors the LiteLLM `fallbacks: [{free-pool: [cheap-paid]}]`); every chain ends at local
    # so research never dies on a flaky pool.
    _OPEN_FAILOVER = {"free": ["free", "cheap"], "cheap": ["cheap"],
                      "synth": ["synth"], "local": []}

    def _open_chain(self, tier: str):
        """Ordered [(backend, tier_label)] to attempt for an OPEN call, gateway tiers (available
        only) first, always ending at the local backend."""
        chain = []
        for t in self._OPEN_FAILOVER.get(tier, []):
            b = self.gateway.tier(t) if self.gateway is not None else None
            if b is None:
                continue
            try:
                ok = b.is_available()
            except Exception:
                ok = False
            if ok:
                chain.append((b, t))
        try:
            chain.append((self._local_backend(), "local"))
        except RuntimeError:
            pass
        return chain

    async def generate(self, sensitivity: Any, tier: str, prompt: str, system: str = "",
                       max_tokens: int = 8192, temperature: float = 0.7):
        """Resolve + generate. SOVEREIGN: local only, raises on failure (NEVER falls back to a
        non-local backend). OPEN: walk the cost-tiered failover chain (e.g. free → cheap → local)
        so a 429/outage on one tier advances to the next, ending at local. Billed on success."""
        sens = Sensitivity.coerce(sensitivity)
        if sens is Sensitivity.SOVEREIGN:
            backend = self.resolve(sens, tier)                 # local, fail-closed
            resp = await backend.generate(prompt, system, max_tokens, temperature)  # raises, no fallback
            self._bill(backend, "local")
            return resp
        last_err = None
        for backend, label in self._open_chain(tier):
            assert_backend_allowed(sens, backend)              # open allows any; defense in depth
            try:
                resp = await backend.generate(prompt, system, max_tokens, temperature)
            except Exception as e:                             # 429/outage → advance to next tier
                last_err = e
                continue
            self._bill(backend, label)                         # bill on success only
            return resp
        if last_err is not None:
            raise last_err
        raise RuntimeError("ContractorRouter: no backend available for OPEN call")
