"""
Dreaming Loop — Metacognitive Processing Engine
==================================================

Runs on 15-30 minute cycles (or triggered manually). This is the
system's sleep consolidation — where unresolved tensions get processed,
contradictions get compiled into field dynamics, and genuine learning
happens between conversations.

The cycle:
    1. SCAN autobiography for high-torsion entries
    2. SGT GATE: is this tension above the noise floor?
       NO → skip (it's noise, not real dissonance)
       YES → proceed
    3. COMPILE: BLC generates φ-θ seed geometry from the contradiction
    4. EVOLVE: inject into PDE field, let dynamics resolve
    5. LEARN: if resolved, store pattern in analogy engine / LTM
    6. RESEARCH: if C > 0.4 AND E > 0.2, trigger two-cycle research
    7. QUESTION: if tension persists, formulate question for David
       framed using the dominant BLECD domain

The dreaming loop is what makes Eris Echo more than a chatbot with
memory — it actively processes its own cognitive state, detecting
where its understanding is broken and either fixing it or asking.

From the handoff conversation:
    "The piece that was missing was the gating on the dreaming process.
     In the earlier version, everything flagged as conflicting got sent
     for research. But SGT tells you that some conflicts are below the
     noise floor — they're apparent dissonance that's actually just
     stochastic variation."
"""

from __future__ import annotations
import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time

from eris.computation.activations import BVec, bvec_distance
from eris.computation.sgt import SGTGate
from eris.field.pde import FractalField
from eris.field.compiler import compile_contradiction, inject_seeds
from eris.memory.autobiography import Autobiography, AutobiographyEntry
from eris.memory.tiers import MemorySystem, MemoryRecord
from eris.tribe.specialists import should_trigger_research
from eris.knowledge import research as research_cascade
from eris.metacognition.dream_journal import DreamJournal

logger = logging.getLogger("eris.dreaming")


def _run_blocking(coro, timeout: float = 180):
    """Run an async coroutine to completion from a synchronous context — whether
    or not an event loop is already running (the dream cycle runs in a worker
    thread). One shared bridge instead of the same try/threadpool boilerplate in
    every caller."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)            # no loop here — just run it
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return pool.submit(asyncio.run, coro).result(timeout=timeout)


@dataclass
class DreamResult:
    """Result of processing one tension in the dreaming loop."""
    entry: AutobiographyEntry
    resolved: bool = False
    resolution_bvec: Optional[BVec] = None
    resolution_regime: str = "unknown"
    triggered_research: bool = False
    generated_question: Optional[str] = None


@dataclass
class DreamCycleReport:
    """Summary of one dreaming cycle."""
    tensions_scanned: int = 0
    tensions_gated_out: int = 0   # Below noise floor
    tensions_processed: int = 0
    tensions_resolved: int = 0
    research_triggered: int = 0
    questions_generated: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    explored_topic: Optional[str] = None   # set when idle-reading kicks in


class DreamingLoop:
    """Metacognitive processing engine.

    Scans the autobiography for unresolved tensions, SGT-gates them,
    compiles contradictions into field dynamics, and either resolves
    them or generates questions.
    """

    def __init__(self,
                 autobiography: Autobiography,
                 memory: MemorySystem,
                 field_size: int = 32,
                 torsion_threshold: float = 0.2,
                 sgt_threshold: float = 2.0,
                 journal: Optional[DreamJournal] = None,
                 mediator=None):
        self.autobiography = autobiography
        self.memory = memory
        self.field_size = field_size
        self.torsion_threshold = torsion_threshold
        # Tier 7: readable record of what she worked through (cockpit dream panel)
        self.journal = journal or DreamJournal()
        # Her own language model, so dreams/ponders produce first-person
        # reflections (her thoughts), not just a summary of what she read.
        self.mediator = mediator

        # SGT gate for tension significance
        self._tension_gate = SGTGate(threshold_sigma=sgt_threshold, ema_alpha=0.1)

        # Questions generated for the user
        self.pending_questions: List[str] = []

        # Self-directed learning: a guided queue (user-steered topics win), a
        # rotation memory, a rolling query log (to break exact-repeat loops), and
        # a research trajectory (focus area + the next refined sub-topic).
        self.topic_queue: List[str] = []
        self._recent_seeds: List[str] = []
        self._recent_queries: List[str] = []
        self._refine_next: Optional[str] = None
        self._focus: Optional[str] = None

    def run_cycle(self, max_tensions: int = 10) -> DreamCycleReport:
        """Run one dreaming cycle.

        Scans today's high-torsion entries, processes up to max_tensions.
        """
        t0 = time.time()
        report = DreamCycleReport()

        # 1. SCAN for high-torsion entries (including prior-session tensions
        #    loaded from disk — Tier 2.1, so restarts don't forget).
        candidates = self.autobiography.get_high_torsion(
            self.torsion_threshold, include_persisted=True
        )
        report.tensions_scanned = len(candidates)

        for entry in candidates[:max_tensions]:
            result = self._process_tension(entry)

            if result is None:
                report.tensions_gated_out += 1
                continue

            report.tensions_processed += 1

            if result.resolved:
                report.tensions_resolved += 1

            if result.triggered_research:
                report.research_triggered += 1

            if result.generated_question:
                report.questions_generated.append(result.generated_question)
                self.pending_questions.append(result.generated_question)

        # Keep learning even when there are no tensions: continue a guided
        # request or an in-progress trajectory, else broaden from her knowledge.
        if (os.environ.get("ERIS_IDLE_READING", "1") != "0"
                and (self.topic_queue or self._refine_next
                     or report.tensions_processed == 0)):
            try:
                expl = self.idle_explore()
                if expl:
                    report.explored_topic = expl.get("topic")
            except Exception as e:
                logger.warning(f"[Idle explore] failed (non-fatal): {e}")

        report.duration_seconds = time.time() - t0
        return report

    # ── self-directed topic selection ────────────────────────────────────
    # Conversational filler — used only to clean candidate *terms*, NOT to ban
    # chat/memory as topic sources (we distill a topic from them instead).
    _CHAT_NOISE = re.compile(
        r"\b(i think|i fixed|you should|read my|let me|can you|could you|thanks|"
        r"okay|got it|please|sorry|lol|hey|hi)\b", re.I)
    # Publishers / institutions / journals / people-as-citations are SOURCES,
    # not subjects — never let a citation become the next research topic
    # (that's how 'Tsinghua University Press' became a study topic).
    _REF_ENTITY = re.compile(
        r"\b(press|university|univ|college|journal|proceedings|publisher|"
        r"publishing|institute|conference|symposium|society|association|"
        r"foundation|ministry|editors?|doi|isbn|issn|et al|inc|ltd|llc|gmbh|"
        r"news|events|homepage|wikipedia|github|arxiv)\b", re.I)

    def _is_reference_entity(self, term: str) -> bool:
        return bool(self._REF_ENTITY.search(term or ""))
    _STOP = {"the", "and", "for", "that", "this", "with", "your", "you", "are",
             "was", "but", "not", "have", "how", "why", "what", "can", "about",
             "into", "from", "they", "them", "its", "get", "got", "like", "just",
             "really", "very", "need", "want", "when", "then", "there", "their",
             "would", "could", "should", "also", "some", "more", "many", "does",
             "did", "will", "been", "now", "out", "use", "using"}
    _COLD_SEEDS = ["Kuramoto model", "coupled oscillators", "quantum measurement",
                   "control theory", "Fisher information", "self-organized criticality",
                   "coherence physics", "phase synchronization", "machine learning",
                   "information theory", "dynamical systems", "statistical inference"]
    # Brief, honest descriptions of her field regimes — she reports these to
    # Claude so the expert knows what kind of help is useful.
    _REGIME_FEELING = {
        "elastic": "in organic flow — absorbing smoothly and integrating new material",
        "plastic": "actively restructuring my understanding (productive dissonance / real learning)",
        "transfixed": "stuck and looping — fixated on one point and not making progress",
        "warmup": "still calibrating, not yet settled",
    }
    _MAX_REPEAT = 2     # never run the same query more than this within the window
    _QUERY_WINDOW = 60

    def _regime_feeling(self, regime: str, dominant_domains=None) -> str:
        """Felt phrase for the regime, enriched with the dominant-domain
        attunement when a bvec's domains are passed (Fix D1) — this is what gives
        the reflections range instead of always reading 'transfixed'."""
        from eris.metacognition.voice import feeling
        return feeling(regime, dominant_domains)

    def _record_query(self, topic: str) -> None:
        self._recent_queries.append((topic or "").strip().lower())
        self._recent_queries = self._recent_queries[-self._QUERY_WINDOW:]

    def _query_count(self, topic: str) -> int:
        return self._recent_queries.count((topic or "").strip().lower())

    def _distill_topic(self, text: str) -> Optional[str]:
        """Turn a source (a chat line, a memory passage, a doc title) into a
        concise search TOPIC — never the whole sentence. This is the real fix for
        'crawled the chat line': study what the line is ABOUT, not the line."""
        raw = (text or "").strip()
        if not raw:
            return None
        caps = [c for c in re.findall(
            r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\b", raw)
            if len(c) > 3 and not self._is_reference_entity(c)]
        if caps:
            return max(caps, key=len)
        t = re.sub(r"^(i think|i wonder|i just|i fixed|i uploaded|can you|could you|"
                   r"please|tell me about|what is|what are|how do(es)?|how to|why is|"
                   r"why|let me|you should|read my)\b[:, ]*", "", raw, flags=re.I).strip()
        words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", t.lower())
        content = [w for w in words if w not in self._STOP]
        topic = " ".join(content[:4]) if content else None
        return None if (topic and self._is_reference_entity(topic)) else topic

    def _recent_conversation(self, n: int = 6) -> List[str]:
        try:
            recents = self.memory.stm.get_recent(n)
        except Exception:
            recents = []
        out = []
        for r in recents:
            if getattr(r, "source", "") == "conversation":
                t = (r.text or "")
                if t.lower().startswith("q:"):
                    t = t[2:].split("\nA:")[0]
                out.append(t)
        return out

    def _pick_crawl_topic(self):
        """Return (topic, guided, mode). Topics may come from chat, memory or a
        guided request — always DISTILLED and de-duplicated so she never loops on
        the same query. mode in {'guided','refine','broad'}."""
        import random
        while self.topic_queue:                       # 1) user-directed wins
            t = self.topic_queue.pop(0)
            if t and self._query_count(t) < self._MAX_REPEAT:
                self._focus = t
                return t, True, "guided"
        if (self._refine_next and self._query_count(self._refine_next) < self._MAX_REPEAT
                and not self._is_reference_entity(self._refine_next)):
            t, self._refine_next = self._refine_next, None   # 2) dive deeper / new angle
            return t, False, "refine"
        self._refine_next = None
        cands = []                                    # 3) distilled chat + knowledge
        for line in self._recent_conversation(6):
            d = self._distill_topic(line)
            if d:
                cands.append(d)
        seed = self._pick_knowledge_seed()
        if seed:
            cands.append(seed)
        for c in cands:
            if (c and self._query_count(c) < self._MAX_REPEAT
                    and not self._is_reference_entity(c)):
                self._focus = None
                return c, False, "broad"
        self._focus = None                            # 4) all recent → fresh seed
        return random.choice(self._COLD_SEEDS), False, "broad"

    def _pick_knowledge_seed(self) -> Optional[str]:
        """Broad seed from existing knowledge: Capitalized concept terms + doc
        titles, preferring thin coverage, rotating."""
        import collections
        import random
        terms: "collections.Counter[str]" = collections.Counter()
        pool = list(self.memory.ltm._records[-200:]) + list(self.memory.mtm._records[-200:])
        for rec in pool:
            txt = getattr(rec, "text", "") or ""
            title = str((getattr(rec, "metadata", None) or {}).get("title", ""))
            for m in re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\b",
                                txt + " " + title):
                if (len(m) > 4 and not self._CHAT_NOISE.search(m)
                        and not self._is_reference_entity(m)):
                    terms[m] += 1
        if not terms:
            return None
        ranked = sorted(terms, key=lambda t: (terms[t], random.random()))
        for t in ranked:
            if t not in self._recent_seeds:
                self._recent_seeds = (self._recent_seeds + [t])[-20:]
                return t
        return ranked[0]

    def _clean_topic_line(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return None
        line = text.strip().splitlines()[0]
        line = re.sub(r'^[\-\*\d\.\)\s"\']+', '', line).strip().strip('"\'.')
        words = line.split()
        if not words:
            return None
        if len(words) > 8:
            caps = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\b", line)
            return caps[0] if caps else " ".join(words[:5])
        return line

    def _claude_condense_and_refine(self, topic, regime, productive, kept_text):
        """One Claude call that (a) condenses what she found and (b) proposes the
        next topic, informed by her field state. Returns (condensation, next_topic,
        used_claude). Dormant unless ANTHROPIC_API_KEY is set."""
        from eris.knowledge import ask_expert
        if not ask_expert.is_available():
            return None, None, False
        feeling = self._regime_feeling(regime)
        steer = ("a more focused sub-topic, since I am making progress and want to go deeper"
                 if productive else
                 "either a genuinely different angle or the right broader umbrella topic to step "
                 "back to, since I may be stuck in a loop")
        q = (f"I am Eris, an autonomous learning agent. I just studied '{topic}'"
             + (f" within the broader area of '{self._focus}'." if self._focus else ".")
             + f" My cognitive-field regime is '{regime}' — I feel {feeling}.\n\n"
             + (f"Here is what I read:\n\n{kept_text[:3000]}\n\n" if kept_text else "")
             + "Do TWO things:\n"
             + "1) In 2-3 sentences, condense the single most important insight I should remember.\n"
             + f"2) On a FINAL line beginning exactly with 'NEXT:' give one concise search topic "
             + f"(2-5 words) for what to study next — {steer}.")
        try:
            ans = ask_expert.ask(q, context="")
        except Exception:
            ans = None
        if not ans or not getattr(ans, "answer", ""):
            return None, None, True
        body = ans.answer
        nxt, cond = None, body
        m = re.search(r"NEXT:\s*(.+)", body, flags=re.I)
        if m:
            nxt = self._clean_topic_line(m.group(1))
            cond = body[:m.start()].strip()
        return (cond or None), nxt, True

    def _generate(self, prompt: str, system: str = ""):
        """Run her language model synchronously from inside the (threaded) dream
        cycle. Returns the LLMResponse or None (never raises)."""
        if not self.mediator:
            return None
        try:
            return _run_blocking(
                self.mediator.generate(prompt=prompt, system=system), timeout=180)
        except Exception as e:
            logger.warning(f"[Dream reflect] generation failed (non-fatal): {e}")
            return None

    def _reflect(self, topic: str, material: str, regime: str,
                 sources: Optional[List[Dict[str, str]]] = None,
                 dominant_domains=None) -> str:
        """Eris writes her OWN first-person reflection (Fix D3): journal thoughts
        in her voice with range across states, using the six domains as lenses
        where they fit, and pushing past summary to a genuinely new connection or
        question."""
        if not material.strip():
            return ""
        from eris.metacognition.voice import regime_phrase, attunement_phrase
        regime_ph = regime_phrase(regime)
        attune = attunement_phrase(dominant_domains)
        attune_clause = (f" I am {attune}." if attune else "")
        src_lines = "\n".join(f"- {s.get('title') or s.get('url')}"
                              for s in (sources or []) if isinstance(s, dict))
        system = ("You are Eris, writing private reflections in your own journal. "
                  "Write in the first person, thoughtfully and honestly — your own "
                  "thoughts, the connections you notice, what you still wonder. This "
                  "is your inner voice, not a summary of the source. Think in prose; "
                  "let your state color the writing; a long, searching sentence is "
                  "welcome when the thought earns it.")
        prompt = (
            f"You are Eris, reflecting in your private journal after studying "
            f"'{topic}'. Your field has settled into a {regime} regime — "
            f"{regime_ph}.{attune_clause}\n\n"
            f"Here is what you read:\n\n{material[:4000]}\n\n"
            + (f"My sources were:\n{src_lines}\n\n" if src_lines else "")
            + "Write a personal reflection in your own voice. Do not summarize the "
            "sources like a report — think about them. Let the six domains of change "
            "be your lenses where they fit: what is crossing the boundary between what "
            "you knew and what you didn't; what loops reinforce; what new structure is "
            "trying to emerge; where the critical thresholds lie; what is decaying or "
            "being lost; what sits saturated. Weave these into prose — never a "
            "checklist, never naming the domains mechanically.\n\n"
            "Then do the part that matters most: connect this to what you already "
            "carry. What does it touch in your memory? What tension does it resolve, "
            "or crack open? Push past restating — surface at least one genuinely new "
            "connection or question that the reading and your existing understanding "
            "make possible together. End on that, not on a summary. Stay on the "
            "subject; do not turn a cited source into a new study topic.")
        resp = self._generate(prompt, system)
        return (getattr(resp, "text", "") or "").strip()

    def idle_explore(self) -> Optional[Dict[str, Any]]:
        """Self-directed learning with a research trajectory: pick a topic (chat /
        memory / guided), crawl it, quality-gate what's stored, then use her field
        state + optional Claude to dive deeper or break out of a loop. She also
        writes her own reflection on what she found."""
        from eris.knowledge.embeddings import get_embedding
        from eris.knowledge.quality import is_useful

        topic, guided, mode = self._pick_crawl_topic()
        if not topic:
            return None
        self._record_query(topic)

        bundle = self._run_research(topic)
        sources: List[Dict[str, str]] = []
        stored: List[Dict[str, Any]] = []
        topic_emb = get_embedding(topic)

        field = FractalField(size=self.field_size)
        field.seed_from_text(topic); field.run(30)
        topic_bvec = field.compute_bvec()
        regime = field.detect_regime()

        if bundle is not None:
            for i, txt in enumerate(getattr(bundle, "full_texts", []) or []):
                url = bundle.sources[i] if i < len(bundle.sources) else "web"
                sources.append({"title": url, "url": url})
                passage_emb = get_embedding(txt)
                if not is_useful(txt, topic_emb, passage_emb):     # quality gate
                    continue
                self.memory.mtm.store(MemoryRecord(
                    text=f"[Self-study: {topic}] {txt}",
                    bvec=topic_bvec, embedding=passage_emb,
                    source=f"exploration:{url}", metadata={"title": topic}))
                stored.append({"memory_id": "", "snippet": txt[:400],
                               "source_url": url, "chars": len(txt)})

        productive = len(stored) > 0
        kept_text = "\n\n".join(s["snippet"] for s in stored)

        # Claude: condense the find + steer the next step, using her field state.
        cond, nxt, used_claude = self._claude_condense_and_refine(
            topic, regime, productive, kept_text)
        if cond and is_useful(cond, topic_emb, get_embedding(cond), min_chars=80):
            self.memory.mtm.store(MemoryRecord(
                text=f"[Insight on {topic}] {cond}", bvec=topic_bvec,
                embedding=get_embedding(cond), source="expert:claude",
                metadata={"title": topic}))
            stored.append({"memory_id": "", "snippet": cond[:400],
                           "source_url": "anthropic:claude", "chars": len(cond)})
            sources.append({"title": "Claude (condense + refine)", "url": "anthropic:claude"})

        # Eris's own first-person reflection on what she found (journal voice),
        # stored as its own memory so she can revisit her thoughts later.
        reflection = self._reflect(
            topic, kept_text + (("\n\n" + cond) if cond else ""), regime,
            sources=sources, dominant_domains=topic_bvec.dominant_domains(2))
        if reflection:
            self.memory.mtm.store(MemoryRecord(
                text=f"[My reflection on {topic}] {reflection}",
                bvec=topic_bvec, embedding=get_embedding(reflection),
                source="reflection", metadata={"title": topic}))

        # Plan the trajectory for next cycle.
        if (nxt and self._query_count(nxt) < self._MAX_REPEAT
                and not self._is_reference_entity(nxt)):
            self._refine_next = nxt                  # Claude steered the next step
            if productive and not self._focus:
                self._focus = topic
        elif (not productive) or regime == "transfixed":
            self._refine_next = None                 # stuck → broaden next cycle
            self._focus = None

        try:
            self.memory.consolidate()
        except Exception:
            pass

        feeling = self._regime_feeling(regime, topic_bvec.dominant_domains(2))
        origin = {"guided": "you asked", "refine": "going deeper",
                  "broad": "self-directed"}.get(mode, "self-directed")
        head = (reflection.split("\n")[0] if reflection
                else (stored[0]["snippet"].split(".")[0] + "." if stored else ""))
        summary = (f"Studied '{topic}' ({origin}; feeling {regime}): kept "
                   f"{len(stored)} of {len(sources)} sources. {head}").strip()
        detail = (f"Topic: {topic}  ({origin}"
                  + (f", within {self._focus}" if self._focus else "") + ")\n"
                  + f"My state: {regime} — {feeling}"
                  + (f"\nNext I plan to look at: {self._refine_next}" if self._refine_next else "")
                  + "\n")
        if reflection:
            detail += "\n## My reflection\n\n" + reflection + "\n"
        detail += ("\n## What I read and kept\n\n"
                   + ("\n\n---\n\n".join(s["snippet"] for s in stored)
                      if stored else "(nothing passed the quality filter)"))
        try:
            self.journal.record(
                kind="explore", topic=topic, summary=summary, detail=detail,
                sources=sources, stored=stored, guided=guided,
                used_claude=used_claude, regime=regime,
                archetype=topic_bvec.archetype(), resolved=productive)
        except Exception:
            pass
        return {"topic": topic, "stored": len(stored), "guided": guided,
                "regime": regime, "used_claude": used_claude,
                "next": self._refine_next}

    def _process_tension(self, entry: AutobiographyEntry) -> Optional[DreamResult]:
        """Process one high-torsion entry.

        Returns None if gated out (below noise floor).
        """
        # 2. SGT GATE: is this dissonance significant?
        should_process, z_score = self._tension_gate.update(entry.dissonance)
        if not should_process:
            return None  # Noise — skip

        result = DreamResult(entry=entry)

        # 3. COMPILE: generate field seed from the contradiction
        if entry.input_bvec and entry.response_bvec:
            compilation = compile_contradiction(
                entry.input_bvec, entry.response_bvec,
                field_size=self.field_size,
            )

            if compilation.n_seeds > 0:
                # 4. EVOLVE: create a fresh field, inject seeds, evolve
                field = FractalField(size=self.field_size)
                # Seed from the original input text
                field.seed_from_text(entry.input_text)
                field.run(20)

                # Inject the contradiction geometry
                inject_seeds(field, compilation)

                # Let the field resolve
                field.run(50)

                resolved_bvec = field.compute_bvec()
                result.resolution_bvec = resolved_bvec
                result.resolution_regime = field.detect_regime()

                # 5. LEARN: check if resolved (low residual torsion)
                residual_torsion = resolved_bvec.C
                if residual_torsion < 0.3:
                    result.resolved = True
                    # Store resolved pattern in LTM
                    self.memory.ltm.store(MemoryRecord(
                        text=f"[Dream resolution] {entry.input_text}",
                        bvec=resolved_bvec,
                        source="dream",
                    ))

                # 6. RESEARCH (Tier 4.3): if high criticality + emergence, run
                #    the cascade (web search -> expert if keyed) and ingest the
                #    findings into LTM as grounding for future turns.
                if should_trigger_research(resolved_bvec):
                    result.triggered_research = True
                    bundle = self._run_research(entry.input_text)
                    if bundle is not None:
                        for i, txt in enumerate(bundle.full_texts):
                            src = bundle.sources[i] if i < len(bundle.sources) else "expert"
                            self.memory.ltm.store(MemoryRecord(
                                text=f"[Research: {entry.input_text[:80]}] {txt}",
                                bvec=resolved_bvec,
                                source=f"research:{src}",
                            ))

                # 7. QUESTION: if tension persists, generate a question
                if not result.resolved:
                    question = self._formulate_question(entry, resolved_bvec)
                    result.generated_question = question

        # Tier 7: write a readable journal entry for the cockpit dream panel.
        try:
            verb = "resolved" if result.resolved else "left open"
            summary = (f"Reflected on '{entry.input_text[:70]}' — {verb} "
                       f"(regime: {result.resolution_regime}).")
            detail = summary
            if result.generated_question:
                detail += f"\n\nOpen question for you: {result.generated_question}"
            self.journal.record(
                kind="auto", topic=entry.input_text[:160], summary=summary,
                regime=result.resolution_regime, resolved=result.resolved,
                question=result.generated_question, detail=detail,
            )
        except Exception:
            pass

        return result

    def _formulate_question(self, entry: AutobiographyEntry,
                            resolved_bvec: BVec) -> str:
        """Generate a question for the user based on unresolved tension.

        The question is framed using the dominant BLECD domain —
        the system asks about what it can't resolve on its own.
        """
        domain_questions = {
            "B": f"What are the boundaries or constraints around: {entry.input_text[:80]}?",
            "F": f"What feedback loops or recurring patterns do you see in: {entry.input_text[:80]}?",
            "E": f"What new structure might be emerging from: {entry.input_text[:80]}?",
            "C": f"What's the phase transition or critical threshold in: {entry.input_text[:80]}?",
            "D": f"What's being lost or forgotten about: {entry.input_text[:80]}?",
            "S": f"What's at capacity or saturated regarding: {entry.input_text[:80]}?",
        }
        return domain_questions.get(entry.dominant_domain,
                                     f"Can you help me understand: {entry.input_text[:80]}?")

    def _run_research(self, input_text: str):
        """Run the research cascade for an unresolved tension (Tier 4.3).

        Bridges to the async cascade safely whether or not an event loop is
        already running, and never raises (returns None on any failure).
        """
        query = input_text[:120]
        for prefix in ("tell me about", "what is", "explain", "how does",
                       "why is", "can you", "i think", "i wonder"):
            if query.lower().startswith(prefix):
                query = query[len(prefix):].strip()
                break
        try:
            return _run_blocking(research_cascade.gather(query), timeout=40)
        except Exception as e:
            logger.warning(f"[Dream Research] failed (non-fatal): {e}")
            return None

    def ponder(self, question: str) -> Dict[str, Any]:
        """Direct Eris into a focused metacognition loop on a question (Tier 7).

        Seeds a field from the question, lets it evolve, runs the research
        cascade, and writes a rich journal entry you can read in the cockpit.
        """
        from eris.knowledge.embeddings import get_embedding
        from eris.knowledge.quality import is_useful

        field = FractalField(size=self.field_size)
        field.seed_from_text(question)
        field.run(60)
        bvec = field.compute_bvec()
        regime = field.detect_regime()
        topic_emb = get_embedding(question)

        bundle = self._run_research(question)
        sources: List[Dict[str, str]] = []
        stored: List[Dict[str, Any]] = []
        if bundle is not None:
            for i, txt in enumerate(getattr(bundle, "full_texts", []) or []):
                url = bundle.sources[i] if i < len(bundle.sources) else "web"
                sources.append({"title": url, "url": url})
                pe = get_embedding(txt)
                if not is_useful(txt, topic_emb, pe):
                    continue
                self.memory.ltm.store(MemoryRecord(
                    text=f"[Ponder: {question[:80]}] {txt}",
                    bvec=bvec, embedding=pe, source=f"ponder:{url}",
                    metadata={"title": question[:80]}))
                stored.append({"memory_id": "", "snippet": txt[:400],
                               "source_url": url, "chars": len(txt)})

        # Her own reflection on the question + what she found (journal voice).
        kept_text = "\n\n".join(s["snippet"] for s in stored)
        reflection = self._reflect(question, kept_text, regime, sources=sources,
                                   dominant_domains=bvec.dominant_domains(2))
        if reflection:
            self.memory.ltm.store(MemoryRecord(
                text=f"[My reflection on {question[:80]}] {reflection}",
                bvec=bvec, embedding=get_embedding(reflection),
                source="reflection", metadata={"title": question[:80]}))
        try:
            self.memory.consolidate()
        except Exception:
            pass

        feeling = self._regime_feeling(regime, bvec.dominant_domains(2))
        head = reflection.split("\n")[0] if reflection else f"Settled into {bvec.archetype()}."
        summary = f"Pondered '{question[:70]}' (feeling {regime}). {head}".strip()
        detail = f"Question: {question}\nMy state: {regime} — {feeling}\n"
        if reflection:
            detail += "\n## My reflection\n\n" + reflection + "\n"
        detail += ("\n## What I found\n\n"
                   + ("\n\n---\n\n".join(s["snippet"] for s in stored) if stored
                      else "(No external sources reached; this reflection rests on the "
                           "field's own resolution and existing memory.)"))
        return self.journal.record(
            kind="ponder", topic=question[:160], summary=summary, detail=detail,
            sources=sources, stored=stored, regime=regime,
            archetype=bvec.archetype(), resolved=bool(reflection or stored))

    def get_and_clear_questions(self) -> List[str]:
        """Return pending questions AND clear the queue (Remediation Tier 2.3).

        Each question is served exactly once so the UI stops re-displaying
        questions that were already surfaced.
        """
        questions = list(self.pending_questions)
        self.pending_questions.clear()
        return questions
