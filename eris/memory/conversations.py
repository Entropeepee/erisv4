"""
eris/memory/conversations.py
============================
Conversation-thread store for the cockpit history sidebar (Tier 7).

Each thread keeps its turns plus a condensed title + description and the dates
it was started and last accessed. Persisted as one JSON file per thread under
eris_data/conversations/ so it survives restarts and is easy to inspect.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

# B3: a conversation id becomes a filename, so a browser-supplied "../"-laced id
# could read/write outside the conversations dir. Generated ids (uuid4().hex[:12])
# already match this; this only rejects hostile input.
_CID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _now() -> float:
    return time.time()


class ConversationStore:
    def __init__(self, data_dir: str = "eris_data/conversations"):
        self.dir = data_dir
        os.makedirs(self.dir, exist_ok=True)

    # ---- paths ----
    def _path(self, cid: str) -> str:
        if not _CID_RE.match(cid or ""):
            raise ValueError(f"invalid conversation id: {cid!r}")
        return os.path.join(self.dir, f"{cid}.json")

    # ---- create / append ----
    def new_thread(self, first_user_msg: str = "") -> str:
        cid = uuid.uuid4().hex[:12]
        title = self._title_from(first_user_msg) if first_user_msg else "New conversation"
        doc = {"id": cid, "title": title, "description": (first_user_msg or "")[:140],
               "created_at": _now(), "last_accessed": _now(), "turns": []}
        self._write(doc)
        return cid

    def add_turn(self, cid: str, user: str, eris: str,
                 meta: Optional[Dict[str, Any]] = None) -> None:
        doc = self._read(cid)
        if doc is None:
            doc = {"id": cid, "title": self._title_from(user),
                   "description": user[:140], "created_at": _now(),
                   "last_accessed": _now(), "turns": []}
        doc["turns"].append({"t": _now(), "user": user, "eris": eris,
                             "meta": meta or {}})
        doc["last_accessed"] = _now()
        if len(doc["turns"]) == 1:
            doc["title"] = self._title_from(user)
            doc["description"] = user[:140]
        self._write(doc)

    # ---- read ----
    def list_threads(self) -> List[Dict[str, Any]]:
        out = []
        for fn in os.listdir(self.dir):
            if not fn.endswith(".json"):
                continue
            try:
                d = json.load(open(os.path.join(self.dir, fn), encoding="utf-8"))
                out.append({"id": d["id"], "title": d.get("title", "Conversation"),
                            "description": d.get("description", ""),
                            "created_at": d.get("created_at", 0),
                            "last_accessed": d.get("last_accessed", 0),
                            "turn_count": len(d.get("turns", []))})
            except Exception:
                continue
        out.sort(key=lambda x: x["last_accessed"], reverse=True)
        return out

    def get_thread(self, cid: str) -> Optional[Dict[str, Any]]:
        d = self._read(cid)
        if d is not None:
            d["last_accessed"] = _now()
            self._write(d)
        return d

    # ---- helpers ----
    @staticmethod
    def _title_from(text: str) -> str:
        t = " ".join((text or "").split())
        if len(t) <= 48:
            return t or "New conversation"
        return t[:46].rstrip() + "…"

    def _read(self, cid: str) -> Optional[Dict[str, Any]]:
        p = self._path(cid)
        if not os.path.exists(p):
            return None
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return None

    def _write(self, doc: Dict[str, Any]) -> None:
        tmp = self._path(doc["id"]) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        os.replace(tmp, self._path(doc["id"]))
