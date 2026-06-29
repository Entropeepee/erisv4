# Architecture audit — does the physics add value? (2026-06)

A file:line, adversarially-verified audit of every physics/architecture primitive in Eris,
prompted by QuALITY-HARD benchmark failures (both the hive AND a bare qwen-72B scored 0/3, the
hive at ~23× the tokens). The question: **do φ/θ/ρ, τ vorticity, and the sin/cos/torsion
confidence geometry actually change an output — or are they computed and discarded?**

Method: 6 subsystem tracers → per-quantity adversarial refutation → synthesis. ~40 agents.
Each verdict below survived an adversary instructed to break it with contradicting code.

## Bottom line

**As wired today, the physics adds no value on these benchmark questions — because it largely
does not run on the benchmark path, or runs but cannot select.** This is *not* evidence against
the physics; it means the benchmark never gave the physics a fair test. The thing moving the
(wrong) answer is the **citation-grounding discipline** — correctly engineered for factual QA,
exactly wrong for literary inference.

Three proven mechanisms behind the observed failures:

1. **Wrong evidence frame** ← the φ/θ resonance rerank ranks on a **hashed bag-of-words stopgap
   seed** (`pde.py:22-34`, `is_semantic()=False`; the real ONNX BGE-M3 is a TODO). It scores
   lexical surface, not meaning, so it up-ranks vocabulary-overlapping distractors over the
   emotionally-causal frame the answer needs.
2. **"must be omitted"** ← the canon/synthesis prompt + the Elos "strike the weakest claim" pass
   (`research.py:391-401, 436-446`) instruct the synthesizer to mark an inference as not-fact and
   delete the least-citable claim. A QuALITY answer *is* a supported-but-unstated inference, so the
   discipline is trained to delete exactly the right answer.
3. **Physics-vocabulary leak** ("resonant triad", "oscillations lock into phase") ← `voice.feeling()`
   regime/domain phrases (`voice.py:18-24`) on the live path; on the benchmark the specialist
   framing is the likely source. Either way it is decoration, not reasoning.

## Why the physics is inert *on the benchmark specifically*

- The benchmark arm calls `hive_research` directly, **bypassing** `_assemble_prompt` — so the
  felt-state injection, `dCdX` cloud-escalation, the dissonance gate, and the per-turn MoEGate
  **never run** on the benchmark. All of that load-bearing live-path physics is untested here.
- Where φ/θ resonance *does* run (the rerank), it is **inert because `cap == slice`**: the
  benchmark sets `ERIS_HIVE_MAX_SOURCES=50` (our "read the whole passage" fix, `eris_arm.py:177`),
  and a QuALITY story is ~17 chunks, so `(led + rest)[:50]` returns *all* chunks. The rerank only
  **permutes** a list that is never truncated — every chunk reaches every specialist regardless of
  order. The field can reorder, but it cannot **select**, which is its whole mechanism.

So our own "read the whole passage" setting neutralized the one place the field was load-bearing.
The physics has not failed the benchmark; the benchmark never exercised it.

## Scorecard (load-bearing vs decorative on the answer path)

| Primitive | Verdict | Notes (file:line) |
|---|---|---|
| Citation-grounding (canon prompt + Elos + strip) | **load-bearing — and the most HARMFUL element here** | `research.py:385,391-401,436-446` → scored answer. Deletes the required inference. |
| `gaps` (gates cycle-2 + re-query) | **load-bearing, also harmful** | `research.py:402,407,409`. Brittle regex reclassifies an inference-hedge as an open gap and chases it out of the answer. |
| φ/θ field resonance rerank | **load-bearing LIVE (6-source path); INERT on benchmark** | `orchestrator.py:874-889,983-985`; seed is hashed stopgap `pde.py:22-34`. |
| `tau_rms → C` criticality channel | **load-bearing LIVE only** | `activations.py:377-378` → `workspace.py:166`, `specialists.py:173`. Bypassed on benchmark. |
| BVec / BFECDS + regime + felt-state | **load-bearing LIVE only** | `orchestrator.py:474,630,1211-1214`. Bypassed on benchmark; source of the vocab leak. |
| `dCdX` → cloud escalation; `coherence` → PDE stop | **load-bearing LIVE only** | `orchestrator.py:532-548`; `pde.py:376-410`. Not computed on benchmark. |
| `get_active_specialists`, CrossAttentionHub | **load-bearing (hive)** | `specialists.py:160-167`; `research.py:376,382`. Runs on benchmark. |
| **resonance_confidence** (match/unresolved/coherence/**torsion**) | **DECORATIVE on every answer path** | `confidence.py:30-53`, computed AFTER the answer (`orchestrator.py:1140`), read by nothing that branches. Cosine of *hashed* embeddings. `torsion=acos(match)` is redundant with `match`. Used only by the offline dreaming loop. |
| signed-torsion retrieval channel `R_sin` | **decorative** | `field_interference.py:104` — negligible vs `R_cos`; live swarm uses cosine-only. |
| specialist_divergence, gaps_closed, elos_changed, stripped_claims (count) | **decorative** | `research.py:367,427,461,486` — logged/printed, never branch the answer. |
| per-turn MoEGate `winner` | **decorative** | `_assemble_prompt` never reads `winner` (`orchestrator.py:1159-1216`). |
| `goal_conditioned_context` / `coherence_gain` (§B1/B2) | **DEAD CODE** | `working_memory.py:15,29` — **zero production call sites**. Believed shipped (task #10); not wired. |

## The cleanest experiment to settle it — a 4-way ablation (fixed seed)

| Arm | Config | Tests |
|---|---|---|
| A | bare 72B | baseline |
| B | hive, `ERIS_HIVE_RESONANCE=0` (pure BM25/dense, no field) | physics off |
| C | hive, field resonance ON (today's default) | physics on |
| D | hive, `ERIS_HIVE_TASK=inference` (grounding permits inference, Elos skipped) | grounding fix |

**Prediction from the code:** `B == C` (the physics is decorative here — `cap==slice` means it
cannot select), and `D >> A/B/C` (the problem was strict grounding suppressing the inference, not
the physics). If that holds, the physics earns nothing on this benchmark *until* the field is
reseeded with a real semantic embedding **and** given a task where the candidate pool exceeds the
cap so the rerank can select rather than permute.

## Ranked improvements

1. **(small) Task-condition the grounding** — `ERIS_HIVE_TASK=inference`: rewrite the canon prompt
   to "state your best-supported inference, cite the passages that IMPLY it, do not omit a
   well-supported inference," and SKIP the Elos strike pass for inference items
   (`research.py:436-444, 390-401`). The single most direct lever. **Additive** — strict grounding
   stays the default; inference mode is opt-in (IP/factual work keeps strict).
2. **(medium) MC-native option scorer** — for forced-choice, score each option's support against
   the sources and pick the argmax, instead of free-synthesis-then-regex-extract
   (`eris_arm.py:279-283`). Removes the strip/Elos/extraction failure surface for MC.
3. **(small) Cut the 23× cost** — single-pass for forced-choice; gate cycle-2 to genuine factual
   holes, not inference-hedges (`research.py:248-299, 407`).
4. **(medium) Real semantic field seed** — finish the ONNX BGE-M3 TODO (`pde.py:22-34`) or gate the
   resonance rerank behind `is_semantic()`. No physics primitive on this seed can earn its keep first.
5. **(small) Suppress felt-state injection on comprehension/forced-choice** — source of the vocab
   leak + a hedge nudge (`orchestrator.py:1211-1214`, `voice.py:18-24`).
6. **(medium) Make resonance_confidence a CONTROLLER** — an `ERIS_CONFIDENCE_GATE`: when match is low
   but coherence high (a clean inference gap) PERMIT the labelled inference; when coherence is low,
   abstain/re-retrieve. Converts a decorative primitive into a load-bearing one. Needs the semantic
   seed (#4) first.
7. **(medium) Make gaps_closed / divergence gate integration** — fold a cycle-2 refinement into the
   synthesis only when it closes a gap; when divergence is high, present competing readings instead
   of collapsing to one citation-safe frame.
8. **(small) Delete/wire the dead code; relabel honestly** — remove unused `R_cos/R_sin/mixing_angle`
   returns; delete or wire `goal_conditioned_context`; relabel resonance_confidence in the docs as
   an embedding-cosine readout, not "physics confidence."

## Honest takeaways

- The benchmark failures are **not** evidence against the physics. They are evidence about (a) the
  grounding discipline, (b) a hashed-embedding stopgap seed, and (c) a `cap==slice` setting we
  introduced. The physics deserves a fair test, which the 4-way ablation provides.
- On the **live chat path**, much of the physics *is* load-bearing (specialist selection,
  dissonance gate, dCdX escalation, regime-conditioned prompting, coherence stop-criterion). The
  benchmark just doesn't route through it.
- `resonance_confidence` — the "physics confidence" — is the one celebrated primitive that is
  decorative on *every* answer path. It is the clearest candidate to either wire into a real
  decision (#6) or relabel honestly.
- One believed-shipped feature (§B1/B2 goal-conditioned retrieval, task #10) is **dead code**.
