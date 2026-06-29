"""Sandbox hardening (Phase 1, corrected-order #2): the /sandbox endpoint is default-DENY
independent of the auth token, requires docker isolation, never leaks the host's secrets into a
subprocess, and runs docker non-root/read-only/no-network. The regex validator is NOT the boundary."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import importlib.util
import unittest

from eris.sandbox.executor import (
    endpoint_guard, ExecutionMode, SandboxExecutor, _subprocess_env)


class TestEndpointGuard(unittest.TestCase):
    def test_default_deny_independent_of_token(self):
        # nothing enabled → DENY, whether or not a token is set (the old bug: no-token left it ON)
        self.assertIsNotNone(endpoint_guard(ExecutionMode.DOCKER, env={}))
        self.assertIsNotNone(endpoint_guard(ExecutionMode.DOCKER, env={"ERIS_AUTH_TOKEN": "t"}))
        self.assertIsNotNone(endpoint_guard(ExecutionMode.SUBPROCESS, env={}))

    def test_enabled_requires_docker(self):
        e = {"ERIS_SANDBOX_ENABLED": "1"}
        self.assertIsNotNone(endpoint_guard(ExecutionMode.SUBPROCESS, env=e))   # subprocess refused
        self.assertIsNone(endpoint_guard(ExecutionMode.DOCKER, env=e))          # docker allowed

    def test_subprocess_requires_explicit_optin(self):
        e = {"ERIS_SANDBOX_ENABLED": "1", "ERIS_SANDBOX_ALLOW_SUBPROCESS": "1"}
        self.assertIsNone(endpoint_guard(ExecutionMode.SUBPROCESS, env=e))      # acknowledged opt-in

    def test_enabled_token_set_still_runs_docker(self):
        # default-deny is about the ENABLE flag, not the token — a token-protected box can still
        # enable the sandbox under docker.
        e = {"ERIS_SANDBOX_ENABLED": "1", "ERIS_AUTH_TOKEN": "t"}
        self.assertIsNone(endpoint_guard(ExecutionMode.DOCKER, env=e))


class TestSubprocessEnvScrub(unittest.TestCase):
    def test_helper_drops_secrets_keeps_path(self):
        os.environ["ERIS_AUTH_TOKEN"] = "tok"
        os.environ["OPENAI_API_KEY"] = "sk-secret"
        try:
            env = _subprocess_env("/tmp/ws")
            self.assertNotIn("ERIS_AUTH_TOKEN", env)            # secrets never reach the sandbox
            self.assertNotIn("OPENAI_API_KEY", env)
            self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
            self.assertEqual(env["HOME"], "/tmp/ws")
        finally:
            os.environ.pop("ERIS_AUTH_TOKEN", None)
            os.environ.pop("OPENAI_API_KEY", None)

    def test_subprocess_execution_cannot_read_host_secret(self):
        os.environ["ERIS_FAKE_SECRET"] = "leak-me-123"
        try:
            sb = SandboxExecutor(mode=ExecutionMode.SUBPROCESS)
            res = sb.execute("import os; print('S=' + os.environ.get('ERIS_FAKE_SECRET',''))",
                             validate=False)        # testing env isolation, not the validator
            self.assertIn("S=", res.stdout)
            self.assertNotIn("leak-me-123", res.stdout)        # the secret did NOT cross into the box
        finally:
            os.environ.pop("ERIS_FAKE_SECRET", None)


class TestDockerHardening(unittest.TestCase):
    def test_docker_command_is_isolated(self):
        # regression guard: the docker invocation must keep the isolation flags (can't run real
        # docker here, so assert the source carries them — removing one is a security regression).
        spec = importlib.util.find_spec("eris.sandbox.executor")
        with open(spec.origin, encoding="utf-8") as f:
            src = f.read()
        for flag in ("--network=none", "--read-only", "--cap-drop=ALL",
                     "--security-opt=no-new-privileges", "--user", "--pids-limit=128",
                     "/workspace:ro"):
            self.assertIn(flag, src, f"docker isolation flag missing: {flag}")


if __name__ == "__main__":
    unittest.main()
