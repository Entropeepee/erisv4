"""Deep read — comprehend documents/folders/codebases larger than the context
window, via hierarchical map-reduce (RAPTOR).

  MAP:    summarize each chunk (cheap), store the chunk AND its summary in memory.
  REDUCE: recursively summarize the summaries into parent nodes…
  SYNTH:  …until one top-level synthesis remains (deep/thoughtful pass).

Every tree node (leaf chunks, internal summaries, root) is stored in memory with
an embedding, so the existing retrieval surfaces BOTH leaves (detail) and summary
nodes (gestalt) — that is the "multi-level retrieval" for Q&A afterward. The tree
gives comprehension of the whole; the retrievable leaves preserve specifics.

Reuses: web_reader.chunk(), embeddings, the memory store. Additive and
default-OFF — nothing calls it automatically.

Design notes:
  * `summarize(text, deep) -> str` is injected, so the module is fully testable
    with a fake summarizer (no LLM). The server wires it to the mediator with the
    Fast profile for MAP and the Deep profile for the final synthesis.
  * Checkpoint/resume: progress is persisted per source hash; a killed run resumes
    and skips chunks already summarized. Idempotent: a completed source returns its
    cached synthesis instead of re-doing the work.
  * Bounded: caps on files / chunks / depth, with a clear "hit cap" flag.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any, Tuple
import ast
import hashlib
import json
import os
import time

from eris.knowledge.web_reader import chunk as _char_chunk
from eris.knowledge.embeddings import get_embedding
from eris.memory.tiers import MemoryRecord
from eris.computation.activations import BVec

# summarize(text, deep) -> summary string
Summarizer = Callable[[str, bool], str]

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env",
              "dist", "build", "eris_data", "checkpoints", ".pytest_cache",
              "archive", ".mypy_cache", "site-packages"}
_SKIP_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip",
             ".gz", ".tar", ".bin", ".so", ".dll", ".dylib", ".pyc", ".lock",
             ".npz", ".pt", ".onnx", ".wav", ".mp3", ".mp4", ".woff", ".woff2",
             ".ttf", ".faiss", ".jsonl"}
_CODE_EXT = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h",
             ".hpp", ".go", ".rs", ".rb", ".cs", ".kt", ".swift"}


@dataclass
class DeepReadConfig:
    max_files: int = 400
    max_chunks: int = 2000
    max_depth: int = 4
    chunk_chars: int = 1500
    overlap: int = 150
    group_size: int = 6          # summaries per REDUCE group
    max_file_bytes: int = 400_000


# ── source adapter ─────────────────────────────────────────────────────────
def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def _allowed_roots() -> List[str]:
    """B1: the only directories a browser-supplied deep-read path may touch — the
    library, eris_data, and any extra roots in ERIS_DEEPREAD_ROOTS (os.pathsep-
    separated). Everything else (e.g. /etc/passwd, C:\\Windows) is rejected."""
    roots: List[str] = []
    try:
        from eris.knowledge.documents import library_dir
        roots.append(os.path.realpath(library_dir()))
    except Exception:
        pass
    for p in (os.path.realpath("eris_data"), os.path.realpath("checkpoints")):
        roots.append(p)
    extra = os.environ.get("ERIS_DEEPREAD_ROOTS", "")
    for part in (extra.split(os.pathsep) if extra else []):
        part = part.strip()
        if part:
            roots.append(os.path.realpath(part))
    return [r for r in roots if r]


def _within_roots(path: str, roots: List[str]) -> bool:
    try:
        rp = os.path.realpath(path)        # resolves symlinks, so they can't escape
    except Exception:
        return False
    return any(rp == root or rp.startswith(root + os.sep) for root in roots)


def _iter_segments(memory, source: str, cfg: DeepReadConfig) -> List[Tuple[str, str]]:
    """Return [(label, text)] for a file / folder / raw text / 'ltm'."""
    if source == "ltm":
        recs = memory.all_records() if hasattr(memory, "all_records") else []
        return [(f"ltm:{i}", r.text) for i, r in enumerate(recs) if (r.text or "").strip()]
    roots = _allowed_roots()
    if isinstance(source, str) and os.path.isdir(source):
        if not _within_roots(source, roots):       # B1: confine to approved roots
            return []
        segs: List[Tuple[str, str]] = []
        for dirpath, dirnames, filenames in os.walk(source):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            for fn in sorted(filenames):
                if os.path.splitext(fn)[1].lower() in _SKIP_EXT:
                    continue
                p = os.path.join(dirpath, fn)
                if not _within_roots(p, roots):     # a symlinked file can't escape
                    continue
                try:
                    if os.path.getsize(p) > cfg.max_file_bytes:
                        continue
                except OSError:
                    continue
                text = _read_text_file(p)
                if text.strip():
                    segs.append((os.path.relpath(p, source), text))
                if len(segs) >= cfg.max_files:
                    return segs
        return segs
    if isinstance(source, str) and os.path.isfile(source):
        if not _within_roots(source, roots):       # B1: reject /etc/passwd etc.
            return []
        return [(os.path.basename(source), _read_text_file(source))]
    return [("(text)", str(source))]


# ── structural chunking ────────────────────────────────────────────────────
def _py_chunks(text: str, cfg: DeepReadConfig) -> List[str]:
    """Chunk Python by top-level class/def spans (never mid-definition)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _para_chunks(text, cfg)
    lines = text.splitlines(keepends=True)
    spans, prev = [], 0
    tops = [n for n in tree.body if hasattr(n, "lineno")]
    for n in tops:
        start = n.lineno - 1
        end = getattr(n, "end_lineno", start + 1)
        if start > prev:
            spans.append((prev, start))    # module-level code between defs
        spans.append((start, end))
        prev = end
    if prev < len(lines):
        spans.append((prev, len(lines)))
    out, buf = [], ""
    for a, b in spans:
        seg = "".join(lines[a:b])
        if len(seg) > cfg.chunk_chars * 2:
            if buf:
                out.append(buf); buf = ""
            out.extend(_char_chunk(seg, cfg.chunk_chars, cfg.overlap))
        elif len(buf) + len(seg) > cfg.chunk_chars and buf:
            out.append(buf); buf = seg
        else:
            buf += seg
    if buf.strip():
        out.append(buf)
    return out or [text]


def _para_chunks(text: str, cfg: DeepReadConfig) -> List[str]:
    """Paragraph-aware chunking: accumulate blank-line-separated blocks up to the
    char budget, never cutting mid-paragraph (falls back to char split on a huge
    block)."""
    paras = [p for p in text.split("\n\n")]
    out, buf = [], ""
    for p in paras:
        if len(p) > cfg.chunk_chars * 2:
            if buf:
                out.append(buf); buf = ""
            out.extend(_char_chunk(p, cfg.chunk_chars, cfg.overlap))
        elif len(buf) + len(p) + 2 > cfg.chunk_chars and buf:
            out.append(buf); buf = p
        else:
            buf = (buf + "\n\n" + p) if buf else p
    if buf.strip():
        out.append(buf)
    return out or [text]


def _segment_chunks(label: str, text: str, cfg: DeepReadConfig) -> List[str]:
    if label.endswith(".py"):
        return _py_chunks(text, cfg)
    return _para_chunks(text, cfg)


# ── checkpointing ──────────────────────────────────────────────────────────
def _source_id(source: str) -> str:
    h = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:16]
    return h


def _ckpt_path(data_dir: str, sid: str) -> str:
    return os.path.join(data_dir, "deep_read", f"{sid}.json")


def _load_ckpt(data_dir: str, sid: str) -> Optional[dict]:
    try:
        with open(_ckpt_path(data_dir, sid), "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _save_ckpt(data_dir: str, sid: str, state: dict) -> None:
    p = _ckpt_path(data_dir, sid)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except OSError:
        pass


def _store_node(memory, *, level: int, text: str, summary: str, source: str,
                label: str, idx: int, embed_fn) -> None:
    """Store a tree node in memory. Leaves (level 0) also store the raw chunk so
    detail is retrievable; internal/root nodes store the summary."""
    try:
        memory.mtm.store(MemoryRecord(
            text=summary, bvec=BVec(), embedding=embed_fn(summary), source=source,
            metadata={"deepread_level": level, "path": label, "idx": idx,
                      "kind": "summary"}))
        if level == 0:
            memory.mtm.store(MemoryRecord(
                text=text, bvec=BVec(), embedding=embed_fn(text),
                source=source + ":leaf",
                metadata={"deepread_level": -1, "path": label, "idx": idx,
                          "kind": "chunk"}))
    except Exception:
        pass


def _windows(items: List[Any], n: int):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def deep_read(memory, summarize: Summarizer, source: str, *,
              data_dir: str = "eris_data", cfg: Optional[DeepReadConfig] = None,
              embed_fn=get_embedding, store: bool = True,
              progress: Optional[Callable[[dict], None]] = None) -> dict:
    """Comprehend `source` (file / folder / raw text / 'ltm') by map-reduce.

    Returns {synthesis, tree_id, n_chunks, levels, capped, resumed}. Resumable and
    idempotent via a per-source checkpoint. Synchronous — run it off the event
    loop (asyncio.to_thread) for large sources."""
    cfg = cfg or DeepReadConfig()
    sid = _source_id(source)
    ckpt = _load_ckpt(data_dir, sid)
    if ckpt and ckpt.get("complete"):
        return {"synthesis": ckpt.get("synthesis", ""), "tree_id": sid,
                "n_chunks": ckpt.get("n_chunks", 0), "levels": ckpt.get("levels", 0),
                "capped": ckpt.get("capped", False), "resumed": True, "cached": True}

    # 1–2. segment + structural chunk
    chunks: List[dict] = []
    for label, text in _iter_segments(memory, source, cfg):
        for i, ch in enumerate(_segment_chunks(label, text, cfg)):
            if ch.strip():
                chunks.append({"id": f"{label}#{i}", "label": label, "idx": i, "text": ch})
            if len(chunks) >= cfg.max_chunks:
                break
        if len(chunks) >= cfg.max_chunks:
            break
    capped = len(chunks) >= cfg.max_chunks
    if not chunks:
        return {"synthesis": "", "tree_id": sid, "n_chunks": 0, "levels": 0,
                "capped": False, "resumed": False, "error": "no readable text"}

    # 3. MAP — summarize each chunk (resume skips done ones)
    done: Dict[str, str] = dict((ckpt or {}).get("summaries", {}))
    for n, c in enumerate(chunks):
        if c["id"] in done:
            continue
        s = summarize(c["text"], False)
        done[c["id"]] = s
        if store:
            _store_node(memory, level=0, text=c["text"], summary=s,
                        source=f"deepread:{sid}", label=c["label"], idx=c["idx"],
                        embed_fn=embed_fn)
        _save_ckpt(data_dir, sid, {"summaries": done, "complete": False})
        if progress:
            progress({"stage": "map", "done": n + 1, "total": len(chunks)})

    # 4. REDUCE — recursive cheap summaries of summaries
    nodes = [done[c["id"]] for c in chunks]
    level = 0
    while len(nodes) > cfg.group_size and level < cfg.max_depth:
        level += 1
        parents = []
        for grp in _windows(nodes, cfg.group_size):
            psum = summarize("\n\n".join(grp), False)
            parents.append(psum)
            if store:
                _store_node(memory, level=level, text="\n\n".join(grp), summary=psum,
                            source=f"deepread:{sid}", label="(reduce)", idx=len(parents),
                            embed_fn=embed_fn)
        nodes = parents
        if progress:
            progress({"stage": "reduce", "level": level, "nodes": len(nodes)})

    # 5. SYNTH — final deep pass over the remaining top nodes
    synthesis = summarize("\n\n".join(nodes), True) if nodes else ""
    if store and synthesis:
        _store_node(memory, level=level + 1, text="\n\n".join(nodes),
                    summary=synthesis, source=f"deepread:{sid}", label="(synthesis)",
                    idx=0, embed_fn=embed_fn)
    _save_ckpt(data_dir, sid, {"summaries": done, "complete": True,
                               "synthesis": synthesis, "n_chunks": len(chunks),
                               "levels": level + 1, "capped": capped,
                               "ts": time.time()})
    return {"synthesis": synthesis, "tree_id": sid, "n_chunks": len(chunks),
            "levels": level + 1, "capped": capped, "resumed": bool(ckpt)}
