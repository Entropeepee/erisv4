"""Dependency-free .env loader, shared by every entry point (the server, the benchmark runner) so
keys + config — API keys, ERIS_AUTH_TOKEN, ERIS_* tiers — are entered ONCE in a gitignored .env and
apply EVERYWHERE, not just the one shell that happened to export them.

This is the SINGLE loader for the whole project (the server and the benchmark runner both import it),
so there is exactly one copy of this logic — duplicated copies are how the flag-name lies start.

Why this matters for security: the server reads ERIS_AUTH_TOKEN from os.environ. If the token lives
only in a .env that nothing loads, a fresh shell starts the server with NO token — the access gate
silently does nothing. Loading the .env here, before the app factory reads the token, closes that
fresh-shell footgun.

Why it ALSO matters for attributability: the loader NEVER overrides an explicit env var (an export /
`set` still wins), so a deliberate shell override is honored — but a STALE shell var silently winning
over .env was the root cause of three prior attributability incidents. So the override is no longer
silent: load_dotenv() records every shell-vs-.env conflict, and report_env_sources() prints where each
critical var resolved from and warns on conflicts. Visible override can't recur as a mystery.

MUST run before eris.config / the app factory read os.environ. Silent on a missing file; never raises.
"""
import os
import sys

# Critical vars whose silent shell-override of .env caused the prior attributability incidents — the
# loader tracks where each resolved from so a stale override is surfaced, never silent.
CRITICAL_VARS = (
    "ERIS_AUTH_TOKEN", "ERIS_BIND_HOST", "ERIS_BENCH_MODEL",
    "ERIS_TIER_FREE", "ERIS_TIER_CHEAP", "ERIS_TIER_SYNTH",
    "ERIS_HIVE_SYNTH_CLOUD", "ERIS_BENCH_ATTRIBUTABLE", "ERIS_GATEWAY_BASE_URL",
)
_SECRET_HINT = ("TOKEN", "KEY", "SECRET", "PASSWORD")

# Populated by the most recent load_dotenv() call (cleared at the start of each call):
DOTENV_CONFLICTS = []   # (key, shell_value, file_value) where the shell overrides a DIFFERENT file value
_FILE_KEYS = set()      # every key present in the .env file
_LOADED_KEYS = set()    # keys actually set FROM the file (i.e. not already in the shell)


def load_dotenv(path: str = ".env") -> int:
    """Load KEY=VALUE lines from `path` into os.environ — skipping blanks/comments, stripping a
    single layer of surrounding quotes — WITHOUT clobbering anything already set. Records every
    shell-vs-.env conflict in DOTENV_CONFLICTS (the shell value wins, but the override is now visible).
    Returns the count of newly-set vars. Never raises (a malformed .env must not crash the server)."""
    n = 0
    DOTENV_CONFLICTS.clear()
    _FILE_KEYS.clear()
    _LOADED_KEYS.clear()
    try:
        if not os.path.exists(path):
            return 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if not k:
                    continue
                _FILE_KEYS.add(k)
                if k in os.environ:                  # explicit env wins over the file
                    if os.environ[k] != v:           # …but record the override so it is never silent
                        DOTENV_CONFLICTS.append((k, os.environ[k], v))
                    continue
                os.environ[k] = v
                _LOADED_KEYS.add(k)
                n += 1
    except Exception:
        pass
    return n


def _mask(key: str, value: str) -> str:
    """Never print a secret value to the log: a token/key shows only its length, not its bytes."""
    if any(h in key.upper() for h in _SECRET_HINT):
        return f"<set, {len(value)} chars>" if value else "<empty>"
    return value


def report_env_sources(keys=CRITICAL_VARS, *, out=None) -> None:
    """Print, for each critical var that is set or present in .env, WHICH source it resolved from
    (shell export / .env), and LOUDLY warn on any shell-vs-.env conflict. Secret values are masked.
    This is the attributability banner extended to 'where did this value come from' — the silent
    shell-over-.env override was the root cause of the prior incidents; surfacing it stops the recur."""
    out = out or sys.stderr
    conflicts = {k: (s, f) for k, s, f in DOTENV_CONFLICTS}
    for k in keys:
        if k not in os.environ and k not in _FILE_KEYS:
            continue                                 # don't spam vars that are neither set nor in .env
        if k in conflicts:
            src = "SHELL — OVERRIDES .env"
        elif k in _LOADED_KEYS:
            src = ".env"
        elif k in os.environ:
            src = "shell"
        else:
            src = "unset"
        print(f"[env] {k} ← {src}", file=out)
    if conflicts:
        print(f"[env] WARNING: {len(conflicts)} shell var(s) OVERRIDE your .env (an explicit "
              "set/export always wins):", file=out)
        for k, (s, f) in conflicts.items():
            print(f"[env]   {k} = {_mask(k, s)} (shell)  vs  {_mask(k, f)} (.env) — open a fresh "
                  f"terminal or clear it (set {k}= / unset {k}) to use the .env value", file=out)
