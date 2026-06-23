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
        # Guard the write so a full/read-only disk degrades gracefully (the
        # caller still gets the entry for display) instead of raising into the
        # ponder thread and hanging the request. Crucially, if a PRIOR write was
        # truncated by a full disk and left a partial line with no trailing
        # newline, repair it first — otherwise this entry concatenates onto the
        # partial line and BOTH become one unparseable line (the lost-ponder bug).
        try:
            prefix = ""
            if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
                with open(self.path, "rb") as rf:
                    rf.seek(-1, os.SEEK_END)
                    if rf.read(1) != b"\n":
                        prefix = "\n"
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(prefix + json.dumps(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            pass
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
        # errors="replace" so a truncated line from a disk-full write (a partial
        # multibyte char) degrades to one skippable bad line instead of throwing
        # during iteration and nuking the whole journal (clicks open nothing).
        try:
            with open(self.path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
        except OSError:
            pass
        return out
