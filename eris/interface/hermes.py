"""
eris/interface/hermes.py
========================
Optional sandboxed Hermes contractor (contractor_gateway_layer_spec §8.5) — a long-horizon
autonomous-research worker on the NON-IP side, reached over a loopback Runs API.

Hard constraints (all enforced here, fail-closed):
  • NEVER on a sovereign task — a sovereign goal raises SovereigntyError (the contractor is
    an external sandbox; IP-sensitive work must not leave the local model).
  • Loopback only — the base_url must resolve to 127.0.0.1/localhost; a non-loopback Hermes
    endpoint is refused (the sandbox is on-box by design).
  • Bearer key required — no key ⇒ disabled; a call without it raises.
  • DEFAULT OFF — inert unless ERIS_HERMES_BASE_URL and ERIS_HERMES_API_KEY are both set.

The HTTP poster is injectable so this is unit-testable with no network.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

from eris.interface.sovereignty import Sensitivity, SovereigntyError


class HermesNotConfiguredError(RuntimeError):
    """Raised when the Hermes path is invoked but not (correctly) configured."""


def _is_loopback(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return host in {"localhost", "127.0.0.1", "::1"}


class HermesContractor:
    """Thin client for the sandboxed Hermes Runs API. `poster(url, json, headers, timeout)` is
    injectable for tests; the default does a real loopback POST."""

    def __init__(self, config=None, poster: Optional[Callable[..., Dict[str, Any]]] = None):
        from eris.config import CONFIG
        self.cfg = config or CONFIG
        self._poster = poster or self._default_poster

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.cfg, "hermes_base_url", "")
                    and getattr(self.cfg, "hermes_api_key", ""))

    def assert_loopback(self) -> None:
        base = getattr(self.cfg, "hermes_base_url", "")
        if not _is_loopback(base):
            raise HermesNotConfiguredError(
                f"Hermes base_url {base!r} is not loopback — the contractor sandbox must be "
                f"on-box (127.0.0.1/localhost). Refusing (fail-closed).")

    @staticmethod
    def _default_poster(url, json, headers, timeout) -> Dict[str, Any]:
        import httpx
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=json, headers=headers)
            r.raise_for_status()
            return r.json()

    def run(self, goal: str, *, sensitivity: Any = Sensitivity.OPEN,
            model: str = "", timeout: float = 600.0) -> Dict[str, Any]:
        """Submit a research run to Hermes. Refuses sovereign goals and unconfigured/non-loopback
        setups. Returns the Runs API response dict."""
        sens = Sensitivity.coerce(sensitivity)
        if sens is Sensitivity.SOVEREIGN:
            raise SovereigntyError(
                "Hermes contractor refused: it is an external sandbox and must NEVER run a "
                "SOVEREIGN (IP-sensitive) task. Run it locally instead.")
        if not self.enabled:
            raise HermesNotConfiguredError(
                "Hermes contractor is OFF — set ERIS_HERMES_BASE_URL and ERIS_HERMES_API_KEY.")
        self.assert_loopback()
        base = self.cfg.hermes_base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {self.cfg.hermes_api_key}",
                   "Content-Type": "application/json"}
        payload = {"goal": goal}
        if model:
            payload["model"] = model
        return self._poster(f"{base}/v1/runs", payload, headers, timeout)
