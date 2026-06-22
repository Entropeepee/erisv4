#!/usr/bin/env python3
"""Pack the Eris source into a single review file.

Modes:
  (default)     full source — every eris/**/*.py concatenated with file headers.
  --skeleton    module docstrings + class/def SIGNATURES only (bodies omitted).
                Much smaller; sized to fit a local model's context for
                architecture-level review.

Excludes caches, archive/, data, and (by default) tests. Run from the repo root:
  python pack_codebase.py                 -> eris_codebase_full.txt
  python pack_codebase.py --skeleton      -> eris_codebase_skeleton.txt
"""
from __future__ import annotations
import argparse
import ast
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
EXCLUDE_DIRS = {"__pycache__", "archive", ".git", "eris_data", "checkpoints",
                "node_modules", ".pytest_cache", "knowledge_base", "outputs"}


def iter_py(base: str, include_tests: bool):
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
        if not include_tests and (os.sep + "tests") in (dirpath + os.sep):
            continue
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def _sig(node, indent: str = "") -> str:
    kw = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    try:
        args = ast.unparse(node.args)
        ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    except Exception:
        args, ret = "...", ""
    line = f"{indent}{kw}{node.name}({args}){ret}:"
    doc = ast.get_docstring(node)
    if doc:
        line += f"  # {doc.strip().splitlines()[0][:90]}"
    return line


def skeleton_of(src: str) -> str:
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return f"# (could not parse: {e})"
    out = []
    md = ast.get_docstring(tree)
    if md:
        out.append(f'"""{md.strip().splitlines()[0]}"""')
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(_sig(node))
        elif isinstance(node, ast.ClassDef):
            try:
                bases = ", ".join(ast.unparse(b) for b in node.bases)
            except Exception:
                bases = ""
            out.append(f"class {node.name}({bases}):")
            cd = ast.get_docstring(node)
            if cd:
                out.append(f'    """{cd.strip().splitlines()[0]}"""')
            members = [s for s in node.body
                       if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for s in members:
                out.append(_sig(s, "    "))
            if not members:
                out.append("    ...")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeleton", action="store_true",
                    help="signatures + docstrings only (small, fits local context)")
    ap.add_argument("--include-tests", action="store_true")
    ap.add_argument("--base", default=os.path.join(ROOT, "eris"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    files = list(iter_py(args.base, args.include_tests))
    parts = [
        "ERIS CODEBASE — " + ("SKELETON (signatures + docstrings only)"
                              if args.skeleton else "FULL SOURCE"),
        f"{len(files)} Python files. Paths are relative to the repo root.\n",
    ]
    for path in files:
        rel = os.path.relpath(path, ROOT)
        try:
            src = open(path, "r", encoding="utf-8").read()
        except Exception as e:
            parts.append(f"\n# ===== {rel} (unreadable: {e}) =====")
            continue
        body = skeleton_of(src) if args.skeleton else src
        parts.append(f"\n# ===================== {rel} =====================\n{body}")

    out = args.out or ("eris_codebase_skeleton.txt" if args.skeleton
                       else "eris_codebase_full.txt")
    text = "\n".join(parts)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    chars = len(text)
    print(f"wrote {out}: {len(files)} files, {text.count(chr(10))+1} lines, "
          f"{chars:,} chars (~{chars//4:,} tokens)")


if __name__ == "__main__":
    main()
