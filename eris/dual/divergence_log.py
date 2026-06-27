"""Divergence log (§4) — the curriculum. One durable row per shadow turn.

Append-only JSONL, idempotent by query-hash (re-runs add no duplicate rows),
flushed per row, and resilient to a disk-full truncated line (the newline-repair
+ errors="replace" pattern from PR #30). Carries the novel aligned/tension/coupling
channels through deliberately — raw material for the later epistemic-layer question,
not analysed here.

The verdict is by arbiter SUCCESS DELTA, never by overlap with the floor (RAG is a
floor, not ground truth; scoring agreement would train the field to imitate it).
"""
from __future__ import annotations
from typing import Any, List, Optional
import hashlib
import json
import os
import time

from eris.dual.types import RetrievalResult, record_id


def query_hash(query: str) -> str:
    return hashlib.blake2b((query or "").encode("utf-8"), digest_size=16).hexdigest()


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _verdict(ts: float, ns: float, eps: float = 0.02) -> str:
    """By arbiter.success delta — NOT overlap."""
    if ts < eps and ns < eps:
        return "both_miss"
    if ns - ts > eps:
        return "novel_wins"
    if ts - ns > eps:
        return "trad_wins"
    return "tie"


class DivergenceLog:
    def __init__(self, path: str = "eris_data/dual/divergence.jsonl", counters=None):
        self.path = path
        self.counters = counters        # optional DualCounters (cumulative /vitals tally)
        self._seen = set()              # query_hashes with a logged VERDICT row
        self._seen_err = set()          # query_hashes with a logged ERROR row
        self._load_seen()

    def _load_seen(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    qh = row.get("query_hash")
                    if not qh:
                        continue
                    # A verdict row blocks re-logging (resume idempotency); an error
                    # row dedups itself but must NOT block a later success retry.
                    if "verdict" in row:
                        self._seen.add(qh)
                    elif "error" in row:
                        self._seen_err.add(qh)
        except (FileNotFoundError, OSError):
            pass

    def _append(self, row: dict) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            prefix = ""
            if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
                with open(self.path, "rb") as rf:
                    rf.seek(-1, os.SEEK_END)
                    if rf.read(1) != b"\n":
                        prefix = "\n"          # repair a prior disk-full-truncated line
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(prefix + json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            pass

    @staticmethod
    def _side(result: RetrievalResult, arbiter, query, gold, *, novel: bool) -> dict:
        ids = result.top_ids()
        sub = {}
        if arbiter is not None:
            try:
                sub = arbiter.score(query, result, gold=gold)
            except Exception:
                sub = {}
        side = {"top_ids": ids, "scores": [round(float(s), 4) for s in (result.scores or [])],
                "arbiter": sub}
        if novel:
            side["aligned_ids"] = [record_id(r) for r in (result.aligned or [])]
            side["tension_ids"] = [record_id(r) for r in (result.tension or [])]
            side["coupling"] = [round(float(c), 4) for c in (result.coupling or [])]
        return side

    def record(self, subsystem: str, query: str, trad: RetrievalResult,
               novel: RetrievalResult, arbiter, *, gold=None, kw=None) -> Optional[dict]:
        qh = query_hash(query)
        if qh in self._seen:                    # idempotent — already logged
            return None
        t_side = self._side(trad, arbiter, query, gold, novel=False)
        n_side = self._side(novel, arbiter, query, gold, novel=True)
        ts = float(t_side["arbiter"].get("success", 0.0))
        ns = float(n_side["arbiter"].get("success", 0.0))
        overlap = _jaccard(t_side["top_ids"], n_side["top_ids"])
        # cross_domain: novel surfaced a hit RAG missed AND it actually helped.
        novel_extra = set(n_side["top_ids"]) - set(t_side["top_ids"])
        cross_domain = bool(novel_extra) and ns >= ts and ns > 0.02
        row = {
            "ts": round(time.time(), 3), "subsystem": subsystem,
            "query_hash": qh, "query": query[:500],
            "trad": t_side, "novel": n_side,
            "verdict": _verdict(ts, ns),
            "overlap": round(overlap, 4),       # DESCRIPTIVE ONLY — never the verdict
            "cross_domain": cross_domain,
        }
        self._append(row)
        self._seen.add(qh)
        if self.counters is not None:
            self.counters.record_verdict(row["verdict"])
        return row

    def record_error(self, subsystem: str, query: str, err: Exception) -> None:
        qh = query_hash(query)
        if qh in self._seen_err:        # idempotent — same error already logged
            return
        self._append({"ts": round(time.time(), 3), "subsystem": subsystem,
                      "query_hash": qh, "query": query[:500],
                      "error": f"{type(err).__name__}: {err}"})
        self._seen_err.add(qh)
        if self.counters is not None:
            self.counters.novel_errors += 1

    def rows(self) -> List[dict]:
        out = []
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            out.append(json.loads(line))
                        except (json.JSONDecodeError, ValueError):
                            continue
        except (FileNotFoundError, OSError):
            pass
        return out
