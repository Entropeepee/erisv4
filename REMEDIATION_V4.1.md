# ERIS ECHO v4.1 — Remediation Changelog

Branch: `v4.1-remediation` (based on `main` @ `ccb5ad3`)
Scope: the **local AI instance** only. Unreal Engine world/game scripts were not
touched; the existing telemetry/bridge endpoints are left intact.

This branch implements Tiers 0–4 of `ERIS_V4_REMEDIATION`. Each tier is a
separate commit so you can diff, bisect, or revert by version. Tier 5 (the
grokking experiment) is a research protocol, not code, and is left for a later
pass — but its instruments (`field_interference.resonance_vs_cosine`,
`web_reader.read_queue`) are now in place.

## How the local instance runs with zero new dependencies
- `web_search` / `web_reader` use only the Python standard library.
- `embeddings` default to a fast **deterministic** fallback (no model download).
- `ask_expert` is **dormant** until `ANTHROPIC_API_KEY` is set.
- Cloud LLM backends are registered **only** when their API key is present.

Optional upgrades: `pip install sentence-transformers` + `set ERIS_EMBEDDINGS=on`
for real semantic embeddings; `pip install anthropic` + `set ANTHROPIC_API_KEY`
for the research oracle. `ERIS_LOCAL_MODEL` overrides the local model name
(default `gpt-oss:20b`).

---

## Tier 0 — Stop the slowness (`orchestrator.py`)
**Root cause:** the deep MoE ensemble fired whenever `coherence < 0.2`, but this
engine's global coherence is structurally ~0.04, so **every** turn took the deep
path — paying for keyless cloud-backend timeouts plus a CPU synthesis pass, and
the `[ROUTER] dCdX > 0.3` log line was hardcoded regardless of which condition
fired.
- Cloud backends (`Anthropic/OpenAI/Gemini`) are added to the deep mediator
  **only if their key is set**; keyless ones are skipped, never hang.
- The router now defaults to **one fast local generation** and escalates to the
  cloud ensemble only when `|dC/dX|` is a genuine **SGT z-score outlier** *and*
  ≥1 cloud expert is wired. The always-true `coherence < 0.2` trigger is gone.

## Tier 1 — Honest field instruments (`pde.py`, `workspace.py`, `orchestrator.py`)
- `detect_regime()` uses **self-calibrating percentiles** of this engine's own
  `dC/dX` history instead of absolute thresholds tuned for a since-replaced
  advection PDE (previously "transfixed" was unreachable; regime stayed pinned
  "elastic").
- Transfixion routes through the **SGT stagnation gate** (scale-adaptive) instead
  of a magic `dCdX_stagnation_threshold`.
- `dC/dX` is repositioned as an **internal-state** signal, not a fact-checker.
  `check_hallucination_signature` → `check_empty_confidence_signature`
  (back-compat alias kept). Prompt/system wording reframed: "transfixed" =
  stuck/under-coupled, *not* "this is a hallucination". Real factual
  hallucination is handled by grounding (Tier 4).
- Vitals expose `dCdX` and `dissonance` as **two distinct fields**.

## Tier 2 — Memory plumbing (`autobiography.py`, `tiers.py`, `dreaming.py`, …)
- Autobiography `get_high_torsion(include_persisted=True)` loads prior-session
  tensions from disk, so a restart no longer forgets them.
- `consolidate()` actually calls `MTM.prune()` each pass (dead memories were
  accumulating unbounded).
- `/questions` drains via `get_and_clear_questions()` — each question served once.
- STM→MTM promotion uses **direction-aware BFECDS distance** (+ `ShortTermMemory.
  novelty()`), replacing the scalar `sum(as_array())` that collided
  differently-meaning memories.

## Tier 3 — Specialists bid their FIELD signature (FORK 3-A)
Each specialist finding was `f"[{name}] Analysis of: {user_message}"` — an echo.
Now `make_field_finding()` projects the live field BFECDS onto each specialist's
domain sensitivity; the projected vector is the bid, its magnitude the strength,
its dominant domains the label. Free at runtime, fast on CPU, on-architecture —
so the MoEGate's wave-interference selection operates on real field projections.

## Tier 4 — Knowledge, grounding & retrieval
- `knowledge/ask_expert.py` — Claude oracle, dormant-until-keyed, never blocks.
- `knowledge/embeddings.py` — `get_embedding()`: real BGE-M3 when enabled, else a
  fast deterministic hashed embedding so `search_by_embedding()` is exercised
  (replaces the wording-only stub).
- `knowledge/web_reader.py` — dual-track reading (text→embedding→LTM **and**
  text→PDE field→φ/θ snapshot + BFECDS attractor). Reconciled to the real APIs.
- `knowledge/research.py` — the cascade: web search → escalate to the expert only
  if web is thin *and* keyed. Non-blocking; ingested as grounding.
- `retrieval/field_interference.py` — `R_ij = ∫ φ_i·φ_j·cos(θ_i−θ_j)` resonance
  retrieval + `resonance_vs_cosine()` harness for the grok experiment.
- Wiring: real embeddings on stored turns + retrieval query; the dream-loop
  research trigger runs the cascade and ingests findings; a field-resonance
  retriever is fused (weighted high) into the RRF swarm.

## Tier 5 — The grokking experiment (`run_experiment_grok.py`)
Runnable, checkpoint-safe harness for the two falsifiable tests:
- **5A (gating):** field-resonance R_ij vs embedding-cosine top-neighbor
  agreement, plus cross-domain *analogy recall* (does the field surface
  immune-system↔firewall / predator-prey↔arms-race / resonance↔consensus?).
- **5B (sharpness):** basin width vs N (1,2,4,8,…) — looks for a *sharp* jump
  (grok-as-phase-transition) vs a smooth curve.
Runs OFFLINE on a built-in near/far-domain corpus for reproducibility; `--online`
reads real Wikipedia. The offline numbers are a smoke test, **not** evidence —
a real verdict needs `ERIS_EMBEDDINGS=on` (BGE-M3) and 100+ articles on the box.

## Tier 6 — Sine-aware resonant retrieval (RAG: good → strong)
First-principles fix to memory retrieval. Eris's conservation law is
`cos²θ + sin²θ = 1`, but retrieval only ever used the **cosine** half
(`elastic_energy`) — returning redundant near-duplicates and discarding the
**sine** half (`plastic_energy`), which is the Emergence/learning channel.
- `MemorySystem.retrieve_resonant()` returns `(aligned, tension)`: `aligned` is
  ordinary cosine/embedding relevance; `tension` is ranked by `plastic_energy`
  (sin²·coupling) — memories strongly *coupled* to the query but *unresolved*.
  Coupling weighting means unrelated memories score ~0, so the sine set is
  "productive dissonance", not noise.
- The live conversational turn now feeds the LLM both sets (the tension set
  labeled "related but unresolved — look for the hidden connection"), so Eris
  connects ideas instead of parroting the nearest neighbor. This also brings
  resonance into the **live** loop (previously only in the Tier 5 experiment).
- **Real embeddings default-ON (auto):** `get_embedding()` now tries the
  semantic model first and falls back to the deterministic hash only if
  `sentence-transformers` is absent. Install it (`pip install
  sentence-transformers`) to make the field encode *meaning*; set
  `ERIS_EMBEDDINGS=off` to force the fast fallback for tests. **Throw this
  switch before running the grok Experiment 5A.**

## Pre-existing test bugs fixed (from the Opus 4.8 audit)
- `tests/test_infrastructure.py` had a literal newline inside a string literal,
  making the whole file uncollectable. Fixed → suite now collects.
- `tests/test_computation.py::test_memory_independent_of_decay` referenced
  `PDEParams(D_decay=…)`; the June refactor renamed it `d_decay`. Fixed.
- Result: full suite **120 passed, 0 failed** (with fastapi installed).

## Bug fix (pre-existing)
- `pde.clone()` copied non-existent `_C_hist` / `_X_hist`, raising
  `AttributeError`. Because `clone()` runs every turn via `probe_reactivity()`
  (the transfixion check), the committed code crashed on **every chat turn**.
  Fixed to copy the real history lists.

## Verification
- `tests/test_remediation_v41.py` — 11 tests pinning each tier (all pass).
- Existing unit suite: 85 pass; the single failure
  (`test_memory_independent_of_decay`) is a stale test referencing a `PDEParams`
  arg removed by the earlier June PDE refactor — unrelated to this branch.
- Full `orchestrator.process()` turn runs end-to-end on CPU with no LLM backend.
