"""
Code Validator — Safety checks before sandbox execution.

Blocks: file system destruction, network access, system calls, imports of
dangerous modules. Allows: numpy, cupy, scipy, math, json, collections,
itertools, functools, dataclasses, typing, time, hashlib.

This is NOT security theater — it's preventing accidental self-damage.
The subprocess sandbox provides the actual isolation boundary.
"""

from __future__ import annotations
import ast
import re
from typing import Tuple, Set

# Modules that are NEVER allowed in the sandbox
BLOCKED_MODULES = {
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "requests", "httpx", "aiohttp",
    "ctypes", "importlib", "runpy", "code", "codeop",
    "signal", "multiprocessing", "threading",
    "pickle", "shelve", "marshal",
    # B2: block `eris` — many eris modules import os/subprocess at their own
    # module level, so `import eris.<x>` was an INDIRECT path to the host. The
    # subprocess runs the host interpreter with Eris's libs, so this matters.
    "eris",
}

# Patterns that indicate dangerous operations
BLOCKED_PATTERNS = [
    r"__import__\s*\(",
    r"exec\s*\(",
    r"eval\s*\(",
    r"compile\s*\(",
    r"open\s*\([^)]*['\"]w",  # open(..., 'w') — writing files
    r"rmtree|unlink|remove\s*\(",
    r"system\s*\(",
]

# Modules explicitly allowed
ALLOWED_MODULES = {
    "numpy", "np", "cupy", "cp", "scipy", "math", "cmath",
    "json", "collections", "itertools", "functools",
    "dataclasses", "typing", "time", "hashlib", "random",
    "statistics", "decimal", "fractions", "operator",
}   # NB: `eris` deliberately removed — see BLOCKED_MODULES (B2)


def validate_code(code: str) -> Tuple[bool, str]:
    """Validate Python code for sandbox safety.

    Returns (is_safe, message). If not safe, message explains why.
    """
    # Check for blocked patterns via regex
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, code):
            return False, f"Blocked pattern detected: {pattern}"

    # Parse AST to check imports
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    for node in ast.walk(tree):
        # Check import statements
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BLOCKED_MODULES:
                    return False, f"Blocked import: {alias.name}"

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in BLOCKED_MODULES:
                    return False, f"Blocked import: from {node.module}"

        # Check for dangerous function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ("exec", "eval", "compile", "__import__"):
                    return False, f"Blocked builtin: {node.func.id}()"

    return True, "Code is safe for sandbox execution"
