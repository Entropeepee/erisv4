"""
eris/knowledge/study.py
=======================
Autonomous self-directed learning (Tier 7). Eris reads reputable NON-FICTION on
a topic list you control (plus topics drawn from the day's conversations),
ingests it dual-track into memory, and writes a "study report" you can review in
the cockpit.

Reliable-source bias: defaults to Wikipedia (encyclopedic, cited) and an
allow-list of reputable domains for web search. You set the topic list via the
cockpit (stored in eris_data/study_topics.json); reports land in
eris_data/study_reports.jsonl.

Designed to run nightly from the server's scheduler, or on demand.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

# Reputable nonfiction domains for the web-search arm (Wikipedia is always used).
RELIABLE_DOMAINS = [
    "wikipedia.org", "britannica.com", "nature.com", "science.org",
    "ncbi.nlm.nih.gov", "nasa.gov", "nist.gov", "ourworldindata.org",
    "stanford.edu", "mit.edu", "arxiv.org", "who.int",
]

DEFAULT_TOPICS = ["Cognitive science", "Complex systems", "Self-organization"]


class StudyEngine:
    def __init__(self, extractor, memory, *, data_dir: str = "eris_data",
                 journal=None, mediator=None):
        self.extractor = extractor
        self.memory = memory
        self.journal = journal          # optional DreamJournal for cross-posting
        self.mediator = mediator        # optional LLMMediator for summaries
        self.topics_path = os.path.join(data_dir, "study_topics.json")
        self.reports_path = os.path.join(data_dir, "study_reports.jsonl")
        os.makedirs(data_dir, exist_ok=True)

    # ---- topic directive (user-controlled) ----
    def get_topics(self) -> List[str]:
        if os.path.exists(self.topics_path):
            try:
                return json.load(open(self.topics_path, encoding="utf-8")).get("topics", [])
            except Exception:
                pass
        return list(DEFAULT_TOPICS)

    def set_topics(self, topics: List[str]) -> List[str]:
        topics = [t.strip() for t in topics if t.strip()][:50]
        json.dump({"topics": topics, "updated": time.time()},
                  open(self.topics_path, "w", encoding="utf-8"))
        return topics

    # ---- the study session ----
    def study(self, topics: Optional[List[str]] = None, *,
              per_topic_articles: int = 1) -> Dict[str, Any]:
        from eris.knowledge import web_reader

        topics = topics or self.get_topics()
        started = time.time()
        read: List[Dict[str, Any]] = []
        for topic in topics:
            try:
                chunks = web_reader.read_wikipedia(
                    topic, extractor=self.extractor, memory=self.memory)
                read.append({"topic": topic, "source": f"Wikipedia: {topic}",
                             "chunks": chunks})
            except Exception as e:
                read.append({"topic": topic, "source": f"Wikipedia: {topic}",
                             "chunks": 0, "error": str(e)})
            time.sleep(0.5)

        total_chunks = sum(r.get("chunks", 0) for r in read)
        summary = self._summarize(topics, read, total_chunks)
        report = {
            "id": uuid.uuid4().hex[:12], "timestamp": started,
            "topics": topics, "read": read, "total_chunks": total_chunks,
            "summary": summary,
        }
        with open(self.reports_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(report) + "\n")
        if self.journal is not None:
            self.journal.record(
                kind="study", topic=", ".join(topics),
                summary=summary.split("\n")[0][:300],
                detail=summary, resolved=True,
                sources=[r["source"] for r in read],
            )
        return report

    def _summarize(self, topics, read, total_chunks) -> str:
        ok = [r for r in read if r.get("chunks", 0) > 0]
        head = (f"Studied {len(ok)} of {len(topics)} topics "
                f"({total_chunks} passages ingested): "
                f"{', '.join(r['topic'] for r in ok) or 'none reachable'}.")
        # Optional richer summary via the local LLM if one is wired and reachable.
        if self.mediator is not None:
            try:
                import asyncio
                titles = "; ".join(r["topic"] for r in ok)
                prompt = (f"In 3-4 sentences, summarize what was just studied and one "
                          f"interesting connection between these topics: {titles}.")
                resp = asyncio.run(self.mediator.generate(prompt=prompt, system="Be concise and factual."))
                if resp and resp.text.strip():
                    return head + "\n\n" + resp.text.strip()
            except Exception:
                pass
        return head

    # ---- reports ----
    def list_reports(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self._all_reports()
        rows.sort(key=lambda r: r.get("timestamp", 0), reverse=True)
        return [{"id": r["id"], "timestamp": r["timestamp"], "topics": r["topics"],
                 "total_chunks": r.get("total_chunks", 0),
                 "summary_head": (r.get("summary", "").split("\n")[0])[:200]}
                for r in rows[:limit]]

    def get_report(self, rid: str) -> Optional[Dict[str, Any]]:
        for r in self._all_reports():
            if r.get("id") == rid:
                return r
        return None

    def _all_reports(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.reports_path):
            return []
        out = []
        for line in open(self.reports_path, encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out
