"""
eris/metacognition/dream_journal.py
===================================
A readable record of Eris's dream/metacognition activity (Tier 7), so you can
check up on what she worked through overnight (or on demand) and click into the
detail. Append-only JSONL; newest-first listing.

Each entry:
  id, timestamp, kind ('auto' | 'ponder'), prompt/topic, regime, resolved,
  question (if she needs your input), summary (one or two lines), detail
  (her fuller "thoughts": resolution, research findings, sources).
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional


class DreamJournal:
    def __init__(self, path: str = "eris_data/dream_journal.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def record(self, *, kind: str, topic: str, summary: str,
               regime: str = "", resolved: bool = False,
               question: Optional[str] = None,
               detail: str = "", sources: Optional[List] = None,
               guided: bool = False, used_claude: bool = False,
               archetype: str = "",
               stored: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        sources = sources or []
        stored = stored or []
        entry = {
            "id": uuid.uuid4().hex[:12], "timestamp": time.time(),
            "kind": kind, "topic": topic[:200], "summary": summary[:400],
            "regime": regime, "resolved": resolved, "question": question,
            "detail": detail, "sources": sources,
            "guided": guided, "used_claude": used_claude, "archetype": archetype,
            "stored": stored,
            "source_count": len(sources), "stored_count": len(stored),
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def list(self, limit: int = 50,
             before: Optional[float] = None) -> List[Dict[str, Any]]:
        rows = self._all()
        rows.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
        if before is not None:
            rows = [e for e in rows if e.get("timestamp", 0) < before]
        # listing view omits the heavy detail + stored passages
        return [{k: v for k, v in e.items() if k not in ("detail", "stored")}
                for e in rows[:limit]]

    def get(self, entry_id: str) -> Optional[Dict[str, Any]]:
        for e in self._all():
            if e.get("id") == entry_id:
                return e
        return None

    def _all(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        out = []
        for line in open(self.path, encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out
