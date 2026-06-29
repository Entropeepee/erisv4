"""Dependency-free .env loader, shared by every entry point (the server, the benchmark runner) so
keys + config — API keys, ERIS_AUTH_TOKEN, ERIS_* tiers — are entered ONCE in a gitignored .env and
apply EVERYWHERE, not just the one shell that happened to export them.

Why this matters for security: the server reads ERIS_AUTH_TOKEN from os.environ. If the token lives
only in a .env that nothing loads, a fresh shell starts the server with NO token — the access gate
silently does nothing. Loading the .env here, before the app factory reads the token, closes that
fresh-shell footgun.

MUST run before eris.config / the app factory read os.environ. Never overrides an explicit env var
(an export / `set` still wins), so a deliberate shell override is always honored. Silent on a
missing file; never raises.
"""
import os


def load_dotenv(path: str = ".env") -> int:
    """Load KEY=VALUE lines from `path` into os.environ — skipping blanks/comments, stripping a
    single layer of surrounding quotes — WITHOUT clobbering anything already set. Returns the count
    of newly-set vars. Never raises (a malformed .env must not crash the server)."""
    n = 0
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
                if k and k not in os.environ:        # explicit env wins over the file
                    os.environ[k] = v
                    n += 1
    except Exception:
        pass
    return n
