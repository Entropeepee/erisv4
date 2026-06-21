"""
eris/knowledge/documents.py
===========================
Document library ingestion (Tier 7.4). Lets you feed Eris real files — from a
browser upload or from a watched folder (~/Documents/ErisLibrary) — and have
them folded into her field-memory using the SAME dual-track physics as
everything else:

    text -> chunk -> (a) embedding  -> retrievable semantics
                  -> (b) PDE field  -> phi/theta (sin/cos) snapshot + BFECDS

i.e. every chunk becomes both searchable text AND a resonant field attractor.
Reuses eris.knowledge.web_reader.ingest_text so documents land in MEDIUM-term
memory and flow up to LTM via consolidate(), exactly like studied material.

Supported: .txt .md .pdf .docx .json
  * .pdf   via PyMuPDF (fitz)         — pip install PyMuPDF
  * .docx  via python-docx            — pip install python-docx
  * .json  generic, plus auto-detect of an OpenAI/ChatGPT conversations.json
           export (years of chat history become searchable memory).

A small manifest (eris_data/library_manifest.json) records the sha256 of every
ingested file so re-running the library skips unchanged files (cheap to call).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

SUPPORTED_EXT = (".txt", ".md", ".pdf", ".docx", ".json")


# ─────────────────────────────────────────────────────────── library location
def library_dir() -> str:
    """Where Eris's document library lives. Override with ERIS_LIBRARY_DIR."""
    env = os.environ.get("ERIS_LIBRARY_DIR")
    if env:
        return os.path.expanduser(env)
    return os.path.join(os.path.expanduser("~"), "Documents", "ErisLibrary")


# ─────────────────────────────────────────────────────────────────── parsers
def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError("PDF support needs PyMuPDF — pip install PyMuPDF")
    parts = []
    with fitz.open(path) as doc:
        for page in doc:
            parts.append(page.get_text())
    return "\n".join(parts)


def _read_docx(path: str) -> str:
    try:
        import docx  # python-docx
    except ImportError:
        raise RuntimeError("DOCX support needs python-docx — pip install python-docx")
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)


def _looks_like_chatgpt_export(data: Any) -> bool:
    """OpenAI/ChatGPT 'conversations.json' is a list of conversations, each
    with a 'mapping' of message nodes."""
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return "mapping" in data[0] or "title" in data[0] and "create_time" in data[0]
    if isinstance(data, dict) and "mapping" in data:
        return True
    return False


def _chatgpt_conversation_text(conv: Dict[str, Any]) -> str:
    """Flatten one ChatGPT conversation node-map into a readable transcript."""
    mapping = conv.get("mapping", {}) or {}
    nodes = []
    for node in mapping.values():
        msg = (node or {}).get("message") or {}
        if not msg:
            continue
        role = ((msg.get("author") or {}).get("role")) or "?"
        content = msg.get("content") or {}
        parts = content.get("parts") or []
        text = " ".join(p for p in parts if isinstance(p, str)).strip()
        if not text:
            continue
        ts = msg.get("create_time") or 0
        label = {"user": "David", "assistant": "Eris", "system": "System"}.get(role, role)
        nodes.append((ts or 0, f"{label}: {text}"))
    nodes.sort(key=lambda x: x[0])
    return "\n".join(t for _, t in nodes)


def iter_chatgpt_export(data: Any) -> Iterable[Tuple[str, str]]:
    """Yield (title, transcript) for each conversation in a ChatGPT export."""
    convs = data if isinstance(data, list) else [data]
    for i, conv in enumerate(convs):
        if not isinstance(conv, dict):
            continue
        title = conv.get("title") or f"ChatGPT conversation {i+1}"
        text = _chatgpt_conversation_text(conv)
        if text.strip():
            yield (str(title), text)


def extract_documents(path: str) -> List[Tuple[str, str]]:
    """Return a list of (title, text) blocks for a file. Most files yield one
    block; a ChatGPT export yields one per conversation."""
    ext = os.path.splitext(path)[1].lower()
    base = os.path.basename(path)
    if ext in (".txt", ".md"):
        return [(base, _read_txt(path))]
    if ext == ".pdf":
        return [(base, _read_pdf(path))]
    if ext == ".docx":
        return [(base, _read_docx(path))]
    if ext == ".json":
        raw = _read_txt(path)
        try:
            data = json.loads(raw)
        except Exception:
            return [(base, raw)]  # not valid JSON — ingest as plain text
        if _looks_like_chatgpt_export(data):
            return [(f"{base} — {t}", txt) for t, txt in iter_chatgpt_export(data)]
        # generic JSON: ingest a readable dump
        return [(base, json.dumps(data, ensure_ascii=False, indent=1)[:2_000_000])]
    # unknown extension — best-effort text
    return [(base, _read_txt(path))]


# ─────────────────────────────────────────────────────────────────── manifest
class LibraryManifest:
    def __init__(self, path: str):
        self.path = path
        self.entries: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(path):
            try:
                self.entries = json.load(open(path, encoding="utf-8"))
            except Exception:
                self.entries = {}

    def seen(self, sha: str) -> bool:
        return sha in self.entries

    def record(self, sha: str, info: Dict[str, Any]) -> None:
        self.entries[sha] = info
        try:
            json.dump(self.entries, open(self.path, "w", encoding="utf-8"))
        except Exception:
            pass

    def list(self) -> List[Dict[str, Any]]:
        rows = list(self.entries.values())
        rows.sort(key=lambda r: r.get("ingested_at", 0), reverse=True)
        return rows


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────── ingest
class DocumentLibrary:
    """Field-physics document ingestion for browser uploads and the watched
    ErisLibrary folder."""

    def __init__(self, extractor, memory, *, data_dir: str = "eris_data"):
        self.extractor = extractor
        self.memory = memory
        os.makedirs(data_dir, exist_ok=True)
        self.manifest = LibraryManifest(os.path.join(data_dir, "library_manifest.json"))

    def ingest_file(self, path: str, *, force: bool = False,
                    display_name: Optional[str] = None) -> Dict[str, Any]:
        """Ingest one file (dual-track field memory). Skips unchanged files.

        display_name overrides the title used in memory (e.g. an upload's
        original filename when `path` is a random temp file).
        """
        from eris.knowledge import web_reader

        base = display_name or os.path.basename(path)
        sha = _sha256(path)
        if self.manifest.seen(sha) and not force:
            prev = self.manifest.entries.get(sha, {})
            return {"file": base, "skipped": True, "chunks": prev.get("chunks", 0)}

        blocks = extract_documents(path)
        total = 0
        for title, text in blocks:
            if not (text or "").strip():
                continue
            # Prepend the title so the file's NAME is embedded in its chunks —
            # this is what lets you say "find my notes on X" and have Eris
            # surface it from memory by name.
            n = web_reader.ingest_text(f"{title}\n\n{text}", title=title,
                                       extractor=self.extractor, memory=self.memory)
            total += n
        info = {"file": base, "title": base, "chunks": total,
                "blocks": len(blocks), "ingested_at": time.time(),
                "path": path}
        self.manifest.record(sha, info)
        # Let freshly-read documents flow up the memory tiers immediately.
        try:
            self.memory.consolidate()
        except Exception:
            pass
        return {"file": base, "skipped": False, "chunks": total, "blocks": len(blocks)}

    def ingest_dir(self, directory: Optional[str] = None, *,
                   force: bool = False) -> Dict[str, Any]:
        """Ingest every supported file under a directory (default ErisLibrary)."""
        directory = directory or library_dir()
        results: List[Dict[str, Any]] = []
        if not os.path.isdir(directory):
            return {"dir": directory, "error": "folder not found", "files": []}
        for root, _, files in os.walk(directory):
            for fn in sorted(files):
                if fn.lower().endswith(SUPPORTED_EXT):
                    fp = os.path.join(root, fn)
                    try:
                        results.append(self.ingest_file(fp, force=force))
                    except Exception as e:
                        results.append({"file": fn, "error": str(e)})
        return {
            "dir": directory,
            "files": results,
            "ingested": sum(1 for r in results if not r.get("skipped") and not r.get("error")),
            "skipped": sum(1 for r in results if r.get("skipped")),
            "total_chunks": sum(r.get("chunks", 0) for r in results),
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        return self.manifest.list()
