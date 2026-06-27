"""Contractor Layer §5/§10.6 — sovereignty routing fails CLOSED. A sovereign call handed
a non-local backend must RAISE, never silently downgrade to cloud; and the egress self-check
reports 'blocked' when external egress is unreachable. Offline, deterministic (probe injected)."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest

from eris.interface.sovereignty import (
    Sensitivity, SovereigntyError, is_local_backend, assert_backend_allowed,
    select_sovereign_backend,
)
from eris.interface import egress_guard


class _Backend:
    def __init__(self, name, base_url=""):
        self.name = name
        self.base_url = base_url


class TestSensitivity(unittest.TestCase):
    def test_coerce_unknown_fails_closed_to_sovereign(self):
        self.assertIs(Sensitivity.coerce("garbage"), Sensitivity.SOVEREIGN)
        self.assertIs(Sensitivity.coerce(None), Sensitivity.SOVEREIGN)
        self.assertIs(Sensitivity.coerce("open"), Sensitivity.OPEN)
        self.assertIs(Sensitivity.coerce(Sensitivity.OPEN), Sensitivity.OPEN)


class TestLocalBackend(unittest.TestCase):
    def test_local_names_are_local(self):
        self.assertTrue(is_local_backend(_Backend("ollama")))
        self.assertTrue(is_local_backend(_Backend("vllm", "http://localhost:8000")))
        self.assertTrue(is_local_backend(_Backend("ollama", "http://127.0.0.1:11434")))

    def test_cloud_names_are_not_local(self):
        self.assertFalse(is_local_backend(_Backend("openai")))
        self.assertFalse(is_local_backend(_Backend("gateway-free")))
        self.assertFalse(is_local_backend(_Backend("claude-agent-sdk")))

    def test_unknown_name_is_not_local_fail_closed(self):
        self.assertFalse(is_local_backend(_Backend("mystery")))
        self.assertFalse(is_local_backend(object()))

    def test_local_name_pointed_offbox_is_rejected(self):
        # 'ollama' aimed at a remote host is NOT sovereign-safe
        self.assertFalse(is_local_backend(_Backend("ollama", "http://10.0.0.5:11434")))


class TestFailClosed(unittest.TestCase):
    def test_sovereign_to_cloud_raises(self):
        with self.assertRaises(SovereigntyError):
            assert_backend_allowed(Sensitivity.SOVEREIGN, _Backend("gateway-free"))

    def test_sovereign_to_local_ok(self):
        assert_backend_allowed(Sensitivity.SOVEREIGN, _Backend("ollama"))   # no raise

    def test_open_to_cloud_ok(self):
        assert_backend_allowed(Sensitivity.OPEN, _Backend("gateway-free"))  # no raise

    def test_unknown_tag_is_treated_as_sovereign(self):
        with self.assertRaises(SovereigntyError):
            assert_backend_allowed("nonsense", _Backend("openai"))

    def test_select_sovereign_backend_picks_local_or_raises(self):
        chosen = select_sovereign_backend([_Backend("openai"), _Backend("ollama")])
        self.assertEqual(chosen.name, "ollama")
        with self.assertRaises(SovereigntyError):
            select_sovereign_backend([_Backend("openai"), _Backend("gateway-free")])


class TestEgressGuard(unittest.TestCase):
    def test_blocked_when_probe_fails(self):
        blocked = lambda h, p, t: False        # connection refused/blocked
        self.assertFalse(egress_guard.egress_reachable(connect=blocked))
        self.assertEqual(egress_guard.isolation_status(connect=blocked), "blocked")
        egress_guard.assert_isolated(connect=blocked)   # must NOT raise

    def test_open_when_probe_succeeds_raises(self):
        reachable = lambda h, p, t: True       # egress is open → isolation failed
        self.assertTrue(egress_guard.egress_reachable(connect=reachable))
        self.assertEqual(egress_guard.isolation_status(connect=reachable), "open")
        with self.assertRaises(egress_guard.EgressNotBlockedError):
            egress_guard.assert_isolated(connect=reachable)

    def test_probe_exception_is_treated_as_blocked(self):
        def boom(h, p, t):
            raise OSError("nope")
        self.assertFalse(egress_guard.egress_reachable(connect=boom))

    def test_override_skips_check(self):
        os.environ["ERIS_SOVEREIGN_REQUIRE_ISOLATION"] = "0"
        try:
            egress_guard.assert_isolated(connect=lambda h, p, t: True)  # would raise, but skipped
        finally:
            del os.environ["ERIS_SOVEREIGN_REQUIRE_ISOLATION"]


if __name__ == "__main__":
    unittest.main()
