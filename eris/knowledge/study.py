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
                 journal=None, mediator=None, thought_stream=None, field_size=32):
        self.extractor = extractor
        self.memory = memory
        self.journal = journal          # optional DreamJournal for cross-posting
        self.mediator = mediator        # optional LLMMediator for summaries
        self.thought_stream = thought_stream   # her synthesis is HERS → thought-stream
        self.field_size = field_size
        self.topics_path = os.path.join(data_dir, "study_topics.json")
        self.reports_path = os.path.join(data_dir, "study_reports.jsonl")
        os.makedirs(data_dir, exist_ok=True)

    # ---- self-seeking topic selection (autonomous study) ----
    def _recent_topics(self, limit: int = 12) -> set:
        recent = set()
        for r in self._all_reports()[-limit:]:
            for t in (r.get("topics") or []):
                recent.add(t)
        return recent

    def choose_topics(self, n: int = 1) -> List[str]:
        """What to study when nobody clicked a button: bias toward her configured
        interests, range across the curated cross-domain seeds, rotate away from
        what she just studied."""
        from eris.knowledge.curiosity import pick_topics
        extra = list(self.get_topics())
        return pick_topics(n, recent=self._recent_topics(), extra=extra)

    def _gen(self, prompt: str, system: str = "") -> str:
        """Best-effort local-model call (sync). Returns "" if no model wired."""
        if self.mediator is None:
            return ""
        try:
            import asyncio
            resp = asyncio.run(self.mediator.generate(prompt=prompt, system=system))
            return (getattr(resp, "text", "") or "")
        except Exception:
            return ""

    def _comprehend(self, topic: str, text: str) -> int:
        """Index-time elaboration: store self-Q&A units to the library so a later
        question matches a stored question, not just a raw passage. Returns count."""
        if not text or self.mediator is None or self.memory is None:
            return 0
        from eris.knowledge.comprehend import self_qa, qa_units
        from eris.knowledge.embeddings import get_embedding
        from eris.memory.tiers import MemoryRecord
        from eris.computation.activations import BVec
        qas = self_qa(text, topic, self._gen, n=3)
        stored = 0
        for unit in qa_units(qas):
            try:
                self.memory.mtm.store(MemoryRecord(
                    text=f"[Study Q&A: {topic}] {unit}", bvec=BVec(),
                    embedding=get_embedding(unit), source=f"study:qa:{topic}",
                    metadata={"title": topic, "kind": "qa"}))
                stored += 1
            except Exception:
                pass
        return stored

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
                text = web_reader.fetch_wikipedia(topic)
                chunks = web_reader.ingest_text(
                    text, title=f"Wikipedia: {topic}",
                    extractor=self.extractor, memory=self.memory)
                qa = self._comprehend(topic, text)       # self-Q&A comprehension
                read.append({"topic": topic, "source": f"Wikipedia: {topic}",
                             "chunks": chunks, "qa": qa})
            except Exception as e:
                read.append({"topic": topic, "source": f"Wikipedia: {topic}",
                             "chunks": 0, "error": str(e)})
            time.sleep(0.5)

        # Let freshly-read material flow up the tiers: novel/important passages
        # promote MTM→LTM now; the rest stay in medium-term and fade via the
        # Ebbinghaus curve unless later conversation reinforces them.
        try:
            self.memory.consolidate()
        except Exception:
            pass

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

    # ---- autonomous variants (driven by the continuous-study scheduler) ----
    def study_one(self, topic: Optional[str] = None) -> Dict[str, Any]:
        """Single-article study on a self-chosen topic (the 15-minute cadence)."""
        topic = topic or (self.choose_topics(1) or [None])[0]
        if not topic:
            return {"error": "no topic"}
        return self.study([topic])

    def deep_dive(self, topics: Optional[List[str]] = None, *,
                  n: int = 3) -> Dict[str, Any]:
        """Multi-reference deep dive (the twice-hourly cadence): study several
        related sources, then synthesize ACROSS them through the calibration
        discipline (bridges labeled, not stated flat) and store the synthesis to
        the thought-stream — her synthesis is hers; the sources live in the
        library."""
        topics = topics or self.choose_topics(n)
        report = self.study(topics)
        report["kind"] = "deep_dive"
        synthesis = self._synthesize_calibrated(topics, report)
        if synthesis:
            report["synthesis"] = synthesis
        return report

    def _synthesize_calibrated(self, topics, report) -> str:
        """Cross-source synthesis run through the calibration critic + quote guard,
        stored to the thought-stream. Returns the labeled synthesis text (or "")."""
        ok = [r for r in report.get("read", []) if r.get("chunks", 0) > 0]
        if len(ok) < 2 or self.mediator is None:
            return ""
        # Pull what was just stored back as the grounding material.
        material = self._recall_material(topics)
        if not material.strip():
            return ""
        try:
            from eris.field.pde import FractalField
            field = FractalField(size=self.field_size)
            field.seed_from_text(" ".join(topics)); field.run(30)
            regime = field.detect_regime()
        except Exception:
            regime = "plastic"
        from eris.reasoning.calibration import calibration_system, verify_quotes
        system = calibration_system(is_synthesis=True, regime=regime)
        prompt = (f"Synthesize across these studied topics: {', '.join(topics)}.\n\n"
                  f"Grounding material (cite it; do not add outside facts):\n\n"
                  f"{material[:6000]}\n\nWrite the calibrated synthesis.")
        draft = self._gen(prompt, system)
        if not draft.strip():
            return ""
        draft, _ = verify_quotes(draft, material)        # strip fabricated quotes
        if self.thought_stream is not None:
            try:
                from eris.knowledge.embeddings import get_embedding
                from eris.memory.thought_stream import link_and_store
                link_and_store(self.thought_stream, topic=", ".join(topics),
                               regime=regime, text=draft,
                               embedding=get_embedding(draft),
                               drew_on=[f"study:{t}" for t in topics])
            except Exception:
                pass
        return draft

    def _recall_material(self, topics, per_topic: int = 3) -> str:
        """Read back a few just-stored passages per topic to ground the synthesis."""
        if self.memory is None:
            return ""
        from eris.knowledge.embeddings import get_embedding
        seen, parts = set(), []
        for t in topics:
            try:
                hits = self.memory.retrieve(query_embedding=get_embedding(t), top_k=per_topic)
            except Exception:
                hits = []
            for h in hits:
                txt = (getattr(h, "text", "") or "")
                if txt and txt not in seen:
                    seen.add(txt)
                    parts.append(txt[:800])
        return "\n\n".join(parts)

    def _summarize(self, topics, read, total_chunks) -> str:
        ok = [r for r in read if r.get("chunks", 0) > 0]
        head = (f"Studied {len(ok)} of {len(topics)} topics "
                f"({total_chunks} passages ingested): "
                f"{', '.join(r['topic'] for r in ok) or 'none reachable'}.")
        # Nothing ingested → don't ask the LLM to summarize an empty set (that
        # produced the confusing "I'm not sure which topics you'd like..." reply).
        # Surface WHY instead, so a network/proxy/block issue is diagnosable.
        if not ok:
            errs = [f"{r['topic']}: {r['error']}" for r in read if r.get("error")]
            why = ("\n\nCouldn't reach any sources. Likely a network, DNS, proxy, "
                   "or firewall issue (Wikipedia was unreachable from this machine "
                   "just now), not the topics themselves.")
            if errs:
                why += "\n\nWhat failed:\n- " + "\n- ".join(errs[:5])
            return head + why
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
        # errors="replace" so a disk-full-truncated line can't throw mid-iteration
        # and wipe the whole report history.
        try:
            with open(self.reports_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
        except OSError:
            pass
        return out
