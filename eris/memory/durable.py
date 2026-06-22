"""Durable fact memory (roadmap 1.4) — a thin, swappable store for atomic facts.

Distinct from Eris's resonant 3-tier conversational memory: this is an explicit
"remember this fact" store (mem0/Letta's core value prop) with self-editing of
conflicting facts. It's where the ReAct loop (3.1) can log durable lessons
("tool X failed on input shape Y") and where stable user facts live.

A `DurableMemory` protocol lets the backend swap without touching callers:
  • `LocalFactStore` — default, JSON-backed, no dependency, self-edits exact
    duplicates, searches via the stdlib BM25 from `eris.retrieval.hybrid`.
  • mem0 / Letta — plug in behind the same protocol when you choose one (Q2);
    `get_durable_memory()` selects via `ERIS_MEMORY_BACKEND` (default "local").

Kept standalone (not wired into `process()`) until you pick a backend.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Protocol, runtime_checkable
import json
import os
import time
import uuid

from eris.retrieval.hybrid import BM25


@runtime_checkable
class DurableMemory(Protocol):
    def add(self, text: str, **metadata) -> str: ...
    def search(self, query: str, k: int = 5) -> List[Dict]: ...
    def all(self) -> List[Dict]: ...


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


class LocalFactStore:
    """JSON-backed durable fact store. Adding a fact whose normalized text already
    exists UPDATES it (timestamp + metadata) instead of duplicating — a simple
    form of mem0-style conflict self-editing."""

    def __init__(self, path: str = "eris_data/durable_facts.json"):
        self.path = path
        self._facts: List[Dict] = []
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self._facts = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._facts = []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._facts, f, ensure_ascii=False, indent=2)

    def add(self, text: str, **metadata) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        key = _norm(text)
        for fact in self._facts:
            if _norm(fact["text"]) == key:          # self-edit existing fact
                fact["text"] = text
                fact["metadata"] = {**fact.get("metadata", {}), **metadata}
                fact["updated"] = time.time()
                self._save()
                return fact["id"]
        fid = uuid.uuid4().hex[:12]
        self._facts.append({"id": fid, "text": text, "metadata": metadata,
                            "created": time.time(), "updated": time.time()})
        self._save()
        return fid

    def search(self, query: str, k: int = 5) -> List[Dict]:
        if not self._facts:
            return []
        texts = [f["text"] for f in self._facts]
        scores = BM25(texts).scores(query)
        order = sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)
        return [self._facts[i] for i in order[:k] if scores[i] > 0] or \
               [self._facts[i] for i in order[:k]]

    def all(self) -> List[Dict]:
        return list(self._facts)

    def forget(self, fact_id: str) -> bool:
        n = len(self._facts)
        self._facts = [f for f in self._facts if f["id"] != fact_id]
        if len(self._facts) != n:
            self._save()
            return True
        return False


def get_durable_memory(path: Optional[str] = None) -> DurableMemory:
    """Factory: select the durable-memory backend via ERIS_MEMORY_BACKEND
    ("local" default | "mem0" | "letta"). Non-local backends are opt-in and
    require their package (machine-side); a clear error is raised otherwise."""
    backend = os.environ.get("ERIS_MEMORY_BACKEND", "local").strip().lower()
    if backend in ("local", "", "json"):
        return LocalFactStore(path or "eris_data/durable_facts.json")
    if backend == "mem0":
        try:
            from mem0 import Memory  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "ERIS_MEMORY_BACKEND=mem0 but mem0 isn't installed "
                "(`pip install mem0ai`).") from e
        raise NotImplementedError(
            "mem0 adapter is a documented seam — implement Mem0Memory(DurableMemory) "
            "once you've chosen mem0 (WORKLOG Q2).")
    if backend == "letta":
        raise NotImplementedError(
            "Letta adapter is a documented seam — implement LettaMemory(DurableMemory) "
            "once you've chosen Letta (WORKLOG Q2).")
    raise ValueError(f"Unknown ERIS_MEMORY_BACKEND={backend!r}")
