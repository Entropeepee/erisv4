"""Knowledge graph — entities + (subject, relation, object) triples for multi-hop
synthesis (HippoRAG-2-style associative memory), Stage 2 of the comprehension doc.

Single-hop embedding retrieval can't answer "how does X relate to Y" when X and Y
never co-occur in one passage. A triple graph can: extract relations at ingest,
then at query time walk the graph to pull in associatively-related entities no
single embedding match would surface.

Deliberately dependency-light and default-OFF (ERIS_KG): triple extraction is an
LLM call with strict-JSON parsing + one retry (small local models produce
unreliable JSON — that's why this is gated, not always-on). Graph walk is stdlib
BFS; if `networkx` is installed we additionally offer Personalized PageRank, but
nothing here requires it.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Set, Tuple
from collections import deque, defaultdict
import json
import os
import re
import time


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


_EXTRACT_PROMPT = (
    "Extract the key factual relationships from the passage as (subject, relation, "
    "object) triples. Use canonical entity names (resolve pronouns). Only relations "
    "stated or directly implied by the text. Return ONLY a JSON list like "
    '[{{"s":"...","r":"...","o":"..."}}], at most {n}.\n\nPASSAGE:\n{text}')


def parse_triples(raw: str) -> List[dict]:
    if not raw:
        return []
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    out = []
    for d in data if isinstance(data, list) else []:
        if isinstance(d, dict) and d.get("s") and d.get("r") and d.get("o"):
            out.append({"s": str(d["s"]).strip(), "r": str(d["r"]).strip(),
                        "o": str(d["o"]).strip()})
    return out


def extract_triples(text: str, generate: Callable[[str], str], *,
                    n: int = 8, max_chars: int = 4000, retries: int = 1) -> List[dict]:
    """LLM triple extraction with strict-JSON parse + retries. [] on failure."""
    text = (text or "").strip()
    if not text or generate is None:
        return []
    prompt = _EXTRACT_PROMPT.format(n=n, text=text[:max_chars])
    for attempt in range(retries + 1):
        try:
            raw = generate(prompt if attempt == 0 else
                           prompt + "\n\nReturn ONLY valid JSON, nothing else.")
        except Exception:
            return []
        trips = parse_triples(raw or "")
        if trips:
            return trips[:n]
    return []


class KnowledgeGraph:
    """Append-only triple store + in-memory adjacency for multi-hop walks."""

    def __init__(self, path: str = "eris_data/knowledge_graph.jsonl"):
        self.path = path
        self._triples: List[dict] = []
        self._adj: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)  # entity -> {(relation, other)}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._index(json.loads(line))
                    except (json.JSONDecodeError, ValueError):
                        continue
        except (FileNotFoundError, OSError):
            pass

    def _index(self, t: dict) -> None:
        self._triples.append(t)
        s, r, o = _norm(t.get("s")), t.get("r", ""), _norm(t.get("o"))
        if s and o:
            self._adj[s].add((r, o))
            self._adj[o].add((f"inv:{r}", s))      # undirected walk, direction preserved in label

    def add_triples(self, triples: List[dict], *, source: str = "") -> int:
        added = 0
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                for t in triples:
                    if not (t.get("s") and t.get("r") and t.get("o")):
                        continue
                    rec = {"s": t["s"], "r": t["r"], "o": t["o"],
                           "source": source, "ts": time.time()}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    self._index(rec)
                    added += 1
        except OSError:
            for t in triples:                       # still index in memory if disk fails
                if t.get("s") and t.get("r") and t.get("o"):
                    self._index({**t, "source": source})
                    added += 1
        return added

    def neighbors(self, entity: str) -> Set[Tuple[str, str]]:
        return set(self._adj.get(_norm(entity), set()))

    def expand(self, entities, *, hops: int = 2, limit: int = 20) -> List[str]:
        """Multi-hop BFS from seed entities → associatively related entities
        (excludes the seeds). The structural basis for 'how does X relate to Y'."""
        seeds = {_norm(e) for e in (entities or []) if _norm(e)}
        seen = set(seeds)
        out: List[str] = []
        q = deque((s, 0) for s in seeds)
        while q and len(out) < limit:
            node, d = q.popleft()
            if d >= hops:
                continue
            for _rel, other in self._adj.get(node, ()):
                if other in seen:
                    continue
                seen.add(other)
                out.append(other)
                q.append((other, d + 1))
                if len(out) >= limit:
                    break
        return out

    def size(self) -> int:
        return len(self._triples)

    def to_networkx(self):
        """Optional richer analysis (Personalized PageRank etc.) when networkx is
        installed; returns None otherwise so callers degrade to BFS."""
        try:
            import networkx as nx
        except Exception:
            return None
        g = nx.DiGraph()
        for t in self._triples:
            g.add_edge(_norm(t.get("s")), _norm(t.get("o")), relation=t.get("r", ""))
        return g
