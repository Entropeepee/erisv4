# Eris Echo — Consolidated Remediation Roadmap

Folds the **original Phase 0–3 plan**, the **Phase 1.5** added after Codex round 1, and every
finding from **Codex rounds 1–3** + the **attack-surface swarm** into one ordered punch-list.
Every item is file:line-verified by an adversarial process. Items re-run against the code by Claude
(chat) are marked **✓v**; the rest are Codex/swarm-verified (concrete repro, file:line) but not
independently re-run — flagged so the trust level is explicit.

**Legend:** `✓done` · `▶PR open` · `☐todo` · `⚑Phase-3 precondition`
Baseline commit for all anchors: `6ec67fa`.

### Version & maintenance — read first
**v1.3** · living document; update in place, don't fork. So nothing is lost across token windows or
between the two agents:
- **Homed in the repo** as `docs/REMEDIATION_ROADMAP.md` → git history *is* the version log (every
  check-off is a diffable, revertible commit; no manual version juggling).
- **Claude Code maintains it:** when a PR closes an item, flip its box to ✓ and add the PR # *in
  that same PR*. Doc and code move together.
- **Resume from this file alone:** a fresh agent can rebuild the full state — scope, status, anchors,
  merge gates — without re-running a single audit. This file is the insurance against a context reset.

### This document is a tool, not dogma
The roadmap is the current best *snapshot* — not the source of truth. The source of truth is the
live code and fresh adversarial findings.
- **New evidence beats this file.** If a fresh finding contradicts or supersedes an item, the finding
  wins — update the file, don't defend it.
- **Re-verify before invoking.** Confirm an item still holds against current code; the repo moves.
- **Items can be struck.** Anything here can be corrected or removed if mistaken.
- **"It's on the roadmap" is not authority.** The roadmap serves the work; the work does not serve
  the roadmap.

### Changelog
- **v1.3 (2026-06-29):** Phase 1 security MERGED — #83→#90 all landed on `main`
  (`c68d3ff`…`5928494`); 42 security tests green; the three Tier-3 items (#85/#86/#87) each passed
  the real-server exploit re-run before merge. Phase 1 boxes flipped to ✓done. Homed the doc in the
  repo.
- **v1.2 (2026-06-29):** added the not-dogma / supersession clause. Phase 1 #83–#90 all built.
- **v1.1 (2026-06-29):** added versioning + maintenance protocol.
- **v1.0 (2026-06-29):** initial consolidation.

---

## Phase 0 — Secrets · ✓ DONE
- ✓v Git history clean of any real `.env` (no key rotation needed). **Owner's local step:**
  consolidate keys to one source (kill the shell + desktop copies).

## Phase 1 — Security (attack surface) · ✓ DONE (merged)
All eight landed on `main`; each Tier-3 item passed a real-server exploit re-run before merge.
- ✓done **[#83]** Server loads `.env` so `ERIS_AUTH_TOKEN` actually applies; shared loader
  `eris/env_file.py`; visible shell-vs-`.env` override diagnostic (secrets masked); `run.py` deduped.
- ✓done **[#84]** Bind `127.0.0.1` by default; external bind refused without a token.
- ✓done **[#85]** Sandbox default-DENY independent of token + docker isolation + subprocess env scrub.
  *Tier 3 — verified: `main` leaked the canary / wrote / ran code; #85 → 403 ×6 on both endpoints.*
- ✓done **[#86]** WS auth on `/ws` + `/ws/field` (in-endpoint, close 1008 before accept) + `?agent=`
  whitelist. *Tier 3 — verified: both exploits refused before accept; non-allowlisted node → `eris`.*
- ✓done **[#87]** `system_context` merged under an immutable default at both call sites + `/v1`.
  *Tier 3 — verified: jailbreak appended under the surviving default end-to-end via `process()`.*
- ✓done **[#88]** `/ws/field` connection cap (reuse `ERIS_WS_MAX`).
- ✓done **[#89]** Request size-caps + opt-in rate limit on `/chat`,`/v1`,`/ingest`,`/api/stt`,
  `/api/tts/generate` + max TTS length + edge_tts egress-consent (off by default).
- ✓done **[#90]** Autonomous-loop cost guard — paid dream/condense path OFF by default + per-process
  ceiling + visible budget signal. *Verified by tracing the call is gated (0 paid calls / 5 default
  cycles), not just a passing test.*
- ☐ **[Codex r3 #10]** Loopback-guard / loud-warn ALL "local" accelerator URLs — embed, rerank, STT,
  VLM, *not just* edge_tts. A misconfigured remote URL exfiltrates IP content (config.py:211;
  embeddings.py:177; hybrid.py:166; vision.py:72; stt.py:21). *(Carried forward — #89 covered only
  edge_tts; the other accelerator URLs remain. Re-scope as a small follow-up.)*

## Phase 1.5 — Memory integrity & durability · ▶ IN PROGRESS
*Owner's own data — journals, research, field memory — must be trustworthy before any physics test
means anything.* **Building now: r3 #1 + r1 #2/#3. The rest HOLD for Codex round-4** (concurrency /
shared-mutable-state / long-run loop audit will widen them — do the durability cluster once, informed
by r4).
- ☐ ⚑ **[r3 #1 ✓v]** Field snapshots never serialized — `to_dict`/`from_dict` omit
  `phi/theta_snapshot`, so after restart MTM/LTM are embedding-only (tiers.py:214/231). Serialize
  with dtype/shape/finite validation; test reload survives. **Phase-3 precondition. — BUILDING.**
- ☐ **[r1 #2/#3 ✓v]** "Grounding" checks citation-ID-resolution, not claim SUPPORT — a fabrication
  with a live id is canonized as fact (calibration.py:80; research.py:468). Replace with a
  substance/entailment check; fix the 2 false-confidence tests. **Build as the Phase-3 faithfulness
  scorer (design once, serves both). — BUILDING.**
- ⏸ **[r3 #7]** Non-atomic MTM/LTM saves (tiers.py:313/405/414). *HOLD for r4.*
- ⏸ **[r1 #5]** Thought-stream write `OSError` swallowed (thought_stream.py:84-89). *HOLD for r4.*
- ⏸ **[r3 #11]** Corrupt conversation file → thread overwritten empty (conversations.py:49/106). *HOLD for r4.*
- ⏸ **[r1 #1 ✓v]** `orchestrator.field` shared singleton, no foreground lock (governor.py:38). *HOLD
  for r4. NOTE: overlaps orchestrator.py with #87 — rebase on merged state when reached. The
  localhost bind already de-risks this race to self-induced-only, so no exposure pressure.*
- ⏸ **[r3 #3]** `all_records(limit=N)` tail-slice drops the current session (tiers.py:498). *HOLD for r4.*

## Phase 2 — Physics & math correctness · ☐ NOT STARTED
*The architecture runs and the live path is load-bearing, but almost none of the physics is currently
in a state to do what the theory says. Fix before testing.*

**Seeding / field**
- ☐ ⚑ **Semantic seed** — finish BGE-M3 so the field seeds on meaning, not hashed bag-of-words
  (pde.py:21). **The precondition every review names.**
- ☐ **FRT flag ignored** — `seed_from_text` always calls `encode_text` (pde.py:211). Wire `use_frt`
  or delete the dead flag + dead tau channel.

**Resonance / DCR — multiple divergent implementations; consolidate to one**
- ☐ **[r2 ✓v]** DCR shape bug — `_common` flatten-truncates mismatched fields; **LIVE when
  field_size>64** (the 32×32 snapshot downsample at orchestrator.py:665). Resample to a common grid
  (circular θ averaging) or reject loudly (field_interference.py:44).
- ☐ **[r2 ✓v]** θ downsample uses arithmetic mean of a *phase* (orchestrator.py:669) — average
  sin/cos, recover atan2.
- ☐ **[r3 #8 ✓v]** Second DCR — `memory/interference.py:_field_integral` divides by field energy →
  normalized correlation, not the raw integral its docstring claims. Pick raw-vs-normalized and align
  ALL DCR impls + tests.
- ☐ **λ/sine channel dropped** in `_coupling` — uses cosine-only scalar `field_resonance`, not
  `field_resonance_2d` (dual/retrieval.py:37).

**Stencils / numerics**
- ☐ **[r1-noted ✓v]** BFECDS feedback not wrap-safe (raw θ roll-subtraction, activations.py:361) —
  use `wrap_diff`.
- ☐ **[r2]** PDE gradient stencils use periodic `xp.roll` while claiming Dirichlet edges
  (pde.py:56/77/94; activations.py:363).
- ☐ **[r3 #6]** NaN/inf from the field propagate into BVec/archetype/gates/JSON memory
  (activations.py:377) — finite guards + JSON `allow_nan=False`.

**Gates / config**
- ☐ **[r3 #2 ✓v]** SGT gate is two-sided `abs(value-mean)` where callers mean "exceeds" (sgt.py:69).
  Make directional per call site. **Touches the filed SGT patent's intended semantics.**
- ☐ **[r1 #4 ✓v]** `CONFIG.pde_dt` / `sgt_threshold_sigma` are no-ops (hardcoded params win) — wire
  or delete.
- ☐ `ERIS_RETRIEVAL_MODE=traditional_only` actually runs the resonant path (orchestrator.py:422) —
  rename + add a true BM25+dense mode (**Phase 3 needs a real "physics off" floor**).
- ☐ **[r3 #12]** VRAM cap (`VRAM_CAP_GB`/`vram_check`) never enforced (config.py:103) — wire on
  field/rerank/ingest loops (16GB card).

**Retrieval scoring**
- ☐ LTM discards cosine similarity, replaced by freshness×0.8 (tiers.py:608).
- ☐ Tension retrieval ranks by BVec plastic energy, not relevance (tiers.py:641).

**Checkpoint / data pipeline**
- ☐ **[r3 #9]** Field checkpoint saves only phi/theta/step — reload has stale tau/histories
  (pde.py:467). Save/recompute full state.
- ☐ **[r3 #4]** Zero-chunk (scanned/image) PDFs marked ingested then skipped forever
  (documents.py:267) — treat as error/retry; add OCR.
- ☐ **[r3 #5]** Batch embedding validates only `vec[0]` (embeddings.py:181) — validate every
  vector's dim + finiteness.
- ❓ Codex found **no live LlamaIndex/Qdrant on main** (GROBID metadata-only, Docling optional). If
  the SOTA-study pipeline is meant to be wired, confirm where it lives — it may not be on main.

## Phase 3 — Shadow harness: does the physics earn its keep? · ☐ NOT STARTED  *(the original goal)*
- ☐ Hybrid BM25+dense = authoritative floor; physics ranking in shadow; scored on task success.
- ☐ A task routing through the **live** path (not the benchmark bypass) with pool > cap so the rerank
  **selects**, not permutes.
- ☐ Real statistics: seeded, k≥5 repeats, significance test, pre-registered rejection rule, measured
  noise floor. No scalar PASS/FAIL at N=3.
- ☐ If QuALITY-MC is used: give **every** arm the options (fairness fix — see the option-blindness
  finding), and a pool-fixed rerank-only selection test.
- ⚑ **PRECONDITION STACK:** semantic seed (P2) · DCR shape + normalization fixes (P2) ·
  **field-snapshot persistence (P1.5 r3 #1)** · the grounding-substance scorer (P1.5). Until these
  land, a Phase-3 run benchmarks degraded/embedding-only retrieval and calls it "physics."

---

## Verified-good (Codex "clean" section)
- **τ vorticity matches the canon scalar curl ∇ρ×∇θ** (sign + axis) apart from the known
  boundary-stencil issue — the core torsion math is correct.
- No additional CPU/GPU math divergence beyond the known `xp.roll` boundary + the VRAM-cap no-op.

## The pattern across all reviews
Four recurring failure modes, not random bugs: **computed-but-not-persisted** (snapshots,
checkpoints), **computed-but-not-correct** (two DCRs, wrap, branch-cut, two-sided SGT, NaN),
**configured-but-not-wired** (pde_dt, VRAM cap, traditional_only, FRT flag), **tested-but-not-proven**
(the false-confidence tests). The physics is real and runs; it is not yet in a state to realize the
theory. That is now mapped — not guessed.
