# Eris's cognitive functions — explicit, non-overlapping

This is the canonical map of Eris's distinct mental functions. They share *primitives*
(the field, the `_reflect`/`_dream_voice` voice, the memory tiers) but each function has a
**different job**. If two ever start doing the same job, that's a bug — split them again.

The quickest way to tell them apart is four questions:
**what triggers it · does it read anything new · does it use the hive · what does it produce.**

| Function | Trigger | Reads new external material? | Uses the hive? | Produces | Where it lives |
|---|---|---|---|---|---|
| **crawl / study** (`idle_explore`, `study_one`, `deep_dive`) | timer / a queued topic | **Yes** — web/Wikipedia | No | ingested passages (the library) | `dreaming.py`, `knowledge/study.py` |
| **hive research** (`run_two_cycle_research`) | a question/topic to analyze | only what RAG retrieves | **Yes** — it *is* the hive | a citation-grounded, falsified synthesis | `tribe/research.py` |
| **ponder** (`ponder`) | **a question you give her** | **Yes** — runs the research cascade | No | a first-person answer-reflection on that question | `dreaming.py` |
| **reflect** (`_reflect`) | called by crawl/ponder | No (operates on material already gathered) | No | first-person prose *about material she studied* | `dreaming.py` (primitive) |
| **introspect** (`_introspect`) | crawl, when a topic is already in memory | No — reads her **own** memory | No | first-person synthesis of what she already holds | `dreaming.py` |
| **tension-processing** (`_process_tension` / `run_cycle`) | timer (the dream cycle) | only if criticality+emergence fire | No | a contradiction *resolved into the field* + maybe a question | `dreaming.py` |
| **subjective dream** (`subjective_dream`) | sleep (nightly) / on demand | **No** — nothing new at all | No | a private, first-person decompression on **the day** | `dreaming.py` |
| **metacognitive review** (`metacognitive_review`) | sleep / on demand | **No** — compares two of her **own** prior views | No | a calibration lesson on how her view moved once analyzed | `dreaming.py` |
| **consolidate (promote)** (`consolidate`) | sleep | No | memories moved up tiers (STM→MTM→LTM) | `memory/tiers.py` |
| **replay (consolidate semantically)** (`replay_consolidate`) | sleep | No | near-duplicate traces folded into one *reinforced* record | `memory/tiers.py` |

## The boundaries that matter most (the ones that look similar)

**ponder vs. subjective dream.** Both speak in her first-person voice, but they are opposite in
every other way:
- *ponder* is **directed and outward** — you hand her a question, she **goes and researches it**
  (web cascade), and reflects on the *answer*. Active.
- *subjective dream* is **undirected and inward** — no question, **no research, reads nothing
  new**, and reflects on **her own day** (the conversations that mattered, the dissonance she
  felt, a new connection). It's decompression, not problem-solving. Off-duty.
- Analogy: *ponder* is "contemplate this question I gave you"; *dream* is "process your day in
  your sleep."

**subjective dream vs. tension-processing.** Both happen in "the dream cycle," but:
- *tension-processing* takes a contradiction and **resolves it through field dynamics** — it's
  mechanism, not narrative; its output is a settled `bvec` (+ optionally a question).
- *subjective dream* doesn't resolve anything — it **speaks about** what stayed with her. Its
  output is prose in her voice.

**crawl vs. hive.** This is the fast/slow split at the research level:
- *crawl/study* is **System 1** — fast ingest, take the source largely at face value.
- *hive* is **System 2** — slow, multi-specialist, falsifies, grounds every claim in a citation,
  catches over-claims (including its own).

**reflect vs. subjective dream.** Same voice primitive, different subject:
- *`_reflect`* reflects on **material she just studied** (it takes `sources`).
- *`_dream_voice`* reflects on **her day** (it takes no sources and reads nothing new).

## Provenance — who writes what (so consolidation never collapses a function)

Each function writes a distinct `source=` tag. Semantic consolidation (`replay_consolidate`)
groups by the **namespace before the first colon** and never merges across namespaces; the
subjective/voice families are never merged at all.

| Function | `source=` tag | Store | Consolidation |
|---|---|---|---|
| crawl | `exploration:<url>` | MTM | foldable (near-dup web boilerplate collapses) |
| study | `study:qa:*`, `study:prop:*` | MTM | foldable within namespace |
| named-doc / web ingest | `reading:<file>` | MTM | foldable (re-ingests of one file collapse) |
| deep read | `deepread:<id>(:leaf)` | MTM | foldable within namespace |
| hive (canonized) | thought-stream (internal) | thought-stream | **never** (audit trail) |
| ponder passages | `ponder:<url>` | LTM | **never** (skip-family) |
| reflect / introspect | `reflection`, `introspection` | MTM/LTM + thought-stream | **never** (skip-family) |
| tension resolution | `dream` | LTM | **never** (skip-family) |
| **subjective dream** | `dream:subjective` | thought-stream | **never** (skip-family) |
| conversation | `conversation` | STM | never (ephemeral) |

## How they compose (the intended metacognitive loop)

The point of keeping these separate is that they can be **chained**:

1. **crawl** ingests an article (fast, face value).
2. **reflect**/**subjective dream** gives her first naive impression (`reflection` / `dream`).
3. **hive** does the slow analysis of the same material; **write-back** stores its conclusion as a
   first-class `synthesis` memory that outranks the raw chunks.
4. the **confidence** geometry (`resonance_confidence`: cos match + sin/torsion) measures how far
   apart her naive read and the analyzed conclusion sit.
5. **metacognitive review** (`metacognitive_review`) compares #2 vs the #3 conclusion, takes the
   #4 revision magnitude, and writes a calibration lesson — *how her view shifted once it was
   analyzed*, and what to distrust next time. Stored as `metacognition:<slug>` (her own voice).

All five steps now exist. The loop runs during sleep (replay → dream → reconsider) and on demand
(`POST /api/dream/reconsider`). This is the mechanism by which she learns to question her own
first impressions.
