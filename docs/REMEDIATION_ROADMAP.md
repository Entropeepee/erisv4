# Eris Echo — Consolidated Remediation Roadmap

Folds the **original Phase 0–3 plan**, the **Phase 1.5** added after Codex round 1, and every
finding from **Codex rounds 1–3** + the **attack-surface swarm** into one ordered punch-list.
Every item is file:line-verified by an adversarial process. Items re-run against the code by Claude
(chat) are marked **✓v**; the rest are Codex/swarm-verified (concrete repro, file:line) but not
independently re-run — flagged so the trust level is explicit.

**Legend:** `✓done` · `◐ partly done (core merged, item still open)` · `▶PR open` · `☐todo` · `⊘ withdrawn (not a fix)` · `⚑Phase-3 precondition`
Baseline commit for all anchors: `6ec67fa`.

### Version & maintenance — read first
**v1.8** · living document; update in place, don't fork. So nothing is lost across token windows or
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
- **v1.8 (2026-06-29):** added **Phase 4 (future / PARKED): Decentralized Eris Echo network** — the
  collective-Self / `.eris`-container / wave-interference-consensus / SGT-like promotion-gate / Eris
  Accords vision, captured but NOT active (gated on post-Phase-3 prerequisites that don't exist yet).
  Not part of Phase 1.5/2/3.
- **v1.7 (2026-06-29):** **Codex #5 (SGT polarity) WITHDRAWN** — resolved from the filed patent:
  the gate is magnitude-based and two-sided by design, so the live `abs(value−mean)` is faithful; no
  code change, do not make it one-sided. Marked r1 #4 / Codex #7 (config-knob wiring) as PR #99.
- **v1.6 (2026-06-29):** **#92 / #93 / #94 MERGED to `main`** (disjoint code; roadmap reconciled).
  Full suite green on main (780). Phase-1.5 r3 #1 cleared; r1 #2/#3 core landed (stays OPEN for
  scorer-coverage); r3 #10 + Codex #1/#3 closed. **Next: PR #96** (egress hardening — Codex #2/#5,
  reuses the merged host helper).
- **v1.5 (2026-06-29):** PR #94 (egress) independently audited by **Codex + chat** — converged.
  Added the **"Codex PR#94 audit — egress (6 findings)"** section. **#1 (P0 host-class bypass) + #3
  (status-probe egress) folded into #94** (re-review required before merge); #2 + #5 (TTS +
  sovereignty host-class) → **planned PR #96** (reuses #94's loopback helper, build after #94); #4 +
  #6 (ask_expert / autonomous web-study) → **Cloud/web egress-consent workstream** (DESIGN CALL
  pending — do not build). Recorded the guard's verified-good skeleton.
- **v1.4 (2026-06-29):** folded the **Codex round-4 static audit** (main @ `6ec67fa`, all 11
  findings + 1 verified-clean) into a new section with file:line/severity/status; added the
  **Build order & conflicts** section + the **scorer-coverage** write-inventory. In flight:
  **PR #92** (field snapshots, r3 #1), **PR #93** (grounding substance, r1 #2/#3), **PR #94**
  (accelerator egress, r3 #10) — all PR-open, cleared, awaiting owner merge. r4 #2/#4 and r1 #2/#3
  kept OPEN (the scorer is not yet wired into prompt-assembly or study.py).
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
- ✓done **[Codex r3 #10 + PR#94 audit #1/#3 → PR #94 MERGED]** Egress guard + **host-classification
  hardening** for ALL "local" accelerator URLs — embed, rerank, STT, VLM (not just edge_tts). Shared
  `egress_allowed`/`is_loopback_url`/`host_of` in `accelerators.py`; default-DENY remote unless
  `ERIS_ALLOW_REMOTE_<NAME>`/`ERIS_ALLOW_REMOTE_ACCEL`; wired into embeddings (→ in-process), rerank
  (→ RRF-only), STT + VLM (→ raise). Closed **Codex #1 (P0 host-class bypass)** and **#3 (status-probe
  egress)** — re-read by chat (fail-closed direction + probe gate confirmed).

## Phase 1.5 — Memory integrity & durability · ▶ IN PROGRESS
*Owner's own data — journals, research, field memory — must be trustworthy before any physics test
means anything.* **r3 #1 MERGED (#92); r1 #2/#3 core MERGED (#93, stays OPEN for scorer-coverage);
the durability cluster below is informed by Codex round-4 — see the round-4 section + Build order.**
- ✓done ⚑ **[r3 #1 ✓v → PR #92 MERGED]** Field snapshots now serialized — `to_dict`/`from_dict`
  persist `phi/theta_snapshot` with dtype/shape/finite validation; MTM/LTM reload survives a restart
  (tiers.py). Includes the round-4 #7 follow-up (a shape-mismatched phi/theta pair → embedding-only
  fallback). **Phase-3 precondition cleared.**
- ◐ **[r1 #2/#3 ✓v → PR #93 MERGED, item STILL OPEN]** "Grounding" checked citation-ID-resolution,
  not claim SUPPORT — a fabrication with a live id was canonized as fact (calibration.py:80;
  research.py:468). PR #93: QUOTE-AND-VERIFY substance scorer (`eris/reasoning/grounding.py`) wired
  into verify_grounding + retrospect + the hive canonization gate (per-sentence canon, round-4 #1 fix
  folded on). Same scorer IS the Phase-3 faithfulness metric. **Core merged, item STILL OPEN** until
  the scorer is wired into the *remaining* generated-write paths (study.py, dream/reflection) and
  prompt-assembly — see r4 #4 and the scorer-coverage workstream.
- ⏸ **[r3 #7]** Non-atomic MTM/LTM saves (tiers.py:313/405/414). *HOLD for r4 durability cluster.*
- ⏸ **[r1 #5]** Thought-stream write `OSError` swallowed (thought_stream.py:84-89). *HOLD for r4.*
- ⏸ **[r3 #11]** Corrupt conversation file → thread overwritten empty (conversations.py:49/106). *HOLD for r4.*
- ⏸ **[r1 #1 ✓v ≈ r4 #1]** `orchestrator.field` shared singleton, no foreground lock (governor.py:38)
  — **subsumed by round-4 #1 (P0 concurrency)**, which widens it to moe_gate/workspace/goal_network.
  See r4 #1. *Land AFTER #93 (shared orchestrator.py).*
- ⏸ **[r3 #3]** `all_records(limit=N)` tail-slice drops the current session (tiers.py:498). *HOLD for r4.*

## Codex round-4 — static audit of `main` @ `6ec67fa` · the 11 findings
*All silent (no crash, wrong/leaky behavior). Severity P0→P2 as Codex graded. **FIX-FIRST: r4 #1**
— serialize foreground turns around shared cognitive state first; otherwise every other correctness
fix can be invalidated by a concurrent request borrowing the wrong goal/winner/context. Status +
merge-gate per finding; sequencing in **Build order & conflicts** below.*

- ☐ **[r4 #1 · P0]** Concurrent chats use the WRONG MoE goal / working-memory prompt — two foreground
  requests overwrite shared `moe_gate`/`workspace`/`goal_network` mid-turn, so A's specialist
  selection is scored against B's goal and A's prompt can carry B's broadcast bullet
  (server/app.py:195; orchestrator.py:469,496,1198). **Serialize foreground turns around shared
  cognitive state, or make MoE/workspace/goal per-turn.** *Touches orchestrator.py → land AFTER #93.
  Subsumes the old r1 #1 field-lock item.*
- ☐ **[r4 #2 · P0]** Free/generated reflection enters the live prompt as "what she knows" —
  introspection/ponder/reflection are stored then injected under "Use this memory / already read"
  with no tier or re-grounding (dreaming.py:568,886; orchestrator.py:437,1183). **>> NOT closed by
  #93** — prompt-assembly (orchestrator.py ~1186-1194) still injects memory as "already read" with
  no tier caveat; the new 'inference' tier is cosmetic at consumption. **OPEN.** *Touches
  orchestrator.py → AFTER #93; part of scorer-coverage.*
- ⏸ **[r4 #3 · P1]** Subjective dreams recursively dream on prior dreams forever — a dream reads the
  last thought-stream items (incl. prior dreams) and writes a new dream back; after one dream exists,
  "no new day" no longer stops the next (dreaming.py:973,982,1008; thought_stream.py:89).
  **Standing-wave / Eris-2.0-collapse risk.** *HELD loop/dream cluster — deliberate decision before
  building; do NOT start as filler.*
- ☐ **[r4 #4 · P1]** Index-time study Q&A stores model hallucinations — generated Q&A/propositions
  stored as `study:*` with no quote/source check (study.py:108,113; comprehend.py:17,43). **>> NOT
  closed by #93** — study.py canonization is unwired from the scorer. **OPEN.** *AFTER #93;
  scorer-coverage (study first — highest retrieved-as-fact risk).*
- ⏸ **[r4 #5 · P1]** Dream-cycle tension processing starves newer serious tensions — `run_cycle`
  slices the first `max_tensions` candidates before gating and never marks old ones processed, so old
  gated-out entries block later high-risk ones forever (dreaming.py:186,191; autobiography.py:176).
  *HELD loop/dream cluster.*
- ☐ **[r4 #6 · P1]** Embedding-only retrieval lets irrelevant STM outrank correct LTM — with only
  `query_embedding`, every STM record scores 1.0 while LTM semantic hits cap ~0.8, so recent chat
  outranks the studied source (tiers.py:580,607; study.py:247). *Touches tiers.py → land AFTER #92.
  Relates to the Phase-2 "LTM discards cosine" retrieval-scoring items.*
- ☐ **[r4 #7 · P1]** Raw memory can spoof prompt section headers — retrieved memory is concatenated
  unescaped into `_assemble_prompt`; a stored passage can inject fake `[The person says]` /
  `[Your inner state]` sections (orchestrator.py:437,1177,1184). **Escape/neutralize section markers
  in retrieved text.** *Touches orchestrator.py → AFTER #93.*
- ☐ **[r4 #8 · P2]** Failed autonomous-study attempts suppress retries — a topic with `chunks:0` +
  error is still written to `study_reports.jsonl` and topic-selection treats all reported topics as
  "recent," so it's never retried (study.py:155,170,48; curiosity.py:61). *Independent (study.py +
  curiosity.py).*
- ☐ **[r4 #9 · P2]** Hive retrieval cache reuses an index for a different same-length pool —
  `_index_cache` keyed only by `len(recs)`; a changed record set of the same length reuses a stale
  `HybridIndex` (orchestrator.py:862,955; hybrid.py:129). **Key the cache on content identity, not
  length.** *Touches orchestrator.py → AFTER #93.*
- ☐ **[r4 #10 · P2]** Confidence math collapses contradiction into orthogonality —
  `resonance_confidence` clamps negative cosine to 0, so opposite evidence and unrelated evidence give
  the same `match=0 / unresolved=1` geometry (confidence.py:38,40; dreaming.py:1114). *Independent;
  relates to Phase-2 confidence/DCR work.*
- ⏸ **[r4 #11 · P2]** Pending dream questions dropped during drain — `get_and_clear_questions` copies
  a list then clears it; a background append between the two ops is lost (dreaming.py:208,1160;
  server/app.py:567). *HELD loop/dream cluster (small, but grouped with the dream-concurrency work).*

### Scorer-coverage workstream (from the #93 write-inventory)
Wire `judge_claim` / tiering into the *ungated* generated-write paths so nothing reaches fact-tier
without quote-and-verify. **Prioritize by retrieved-as-fact risk — study + dream-derived "lessons"
first.** All gate AFTER #93 (need the scorer on `main`).
- ☐ Study Q&A / propositions (**r4 #4**) — study.py:108,113; comprehend.py:17,43.
- ☐ Every dream / introspection / reflection write — dreaming.py:607,613,698,713,813,931,1057,1175.
- ☐ Federation writes — federation.py:29.
- ☐ Deep-read writes — deep_read.py:231.
- ☐ Prompt-assembly tier caveat (**r4 #2**) — orchestrator.py ~1186-1194: inject memory WITH its tier,
  not as bare "already read."

## Codex PR#94 audit — egress (6 findings)
*PR #94 independently audited by Codex + chat (converged). The guard's skeleton is sound (see
verified-good); these are the host-classification gaps + the same-module completeness items.*
- ✓done **[#1 · P0]** `is_loopback_url` accepted a public DNS name that merely starts with `127.` or
  ends with `.localhost` (e.g. `127.0.0.1.evil.com`, `evil.localhost`) → content shipped off-box with
  NO consent (accelerators.py:38,40). **Fix:** robust `ipaddress`-based, fail-closed classifier
  (exact-`localhost` only for names; `is_loopback` for IP literals; userinfo/`%`-encoding/IPv4-mapped
  handled). **MERGED in #94.**
- ▶PR **[#2 · P1]** TTS provider URL unguarded — raw text goes off-box *before* the edge_tts guard
  (tts.py:19,27,66). **PR #96:** `_provider_speech` now gates the POST with `egress_allowed("tts",…)`
  — a remote URL with no consent refuses (falls back, never POSTs raw text); edge_tts guard intact.
- ✓done **[#3 · P1]** Status probe egresses to a remote URL by default — `_reachable` GETs configured
  URLs, leaking source IP/UA (accelerators.py:86,115; app.py:924). **Fix:** gate `_reachable` with
  `egress_allowed`; a denied remote URL is "not probed." **MERGED in #94.**
- ☐ **[#4 · P1]** `ask_expert` sends research Q/context to Anthropic with no per-path consent
  (research.py:65,68; ask_expert.py:49,72). **→ Cloud/web egress-consent workstream (design pending).**
- ▶PR **[#5 · P1/P2]** Sovereignty treats `.local` as local, so prompts go off-box
  (sovereignty.py:80,90; orchestrator.py:253). **PR #96:** `sovereignty._is_loopback_url` now
  delegates to the shared `accelerators.is_loopback_url` → `.local` / `host.docker.internal` /
  `0.0.0.0` are REMOTE / non-sovereign; only `localhost` + loopback IPs pass. Caller (the "local" tag
  at orchestrator.py:253) unchanged.
- ☐ **[#6 · P2]** Autonomous web/study egress unguarded — DuckDuckGo / Wikipedia / page fetch / r.jina
  (web_search.py:200,237; web_reader.py:57; study.py:148; orchestrator.py:1344). **→ Cloud/web
  egress-consent workstream.**

### ▶PR PR #96 — egress hardening (TTS + sovereignty host-class)
Codex #2 + #5. Applies the merged `accelerators.is_loopback_url`/`egress_allowed` to the TTS provider
POST and the sovereignty `.local` classification. **Built (PR-open, no auto-merge);** 787 tests green.

### Workstream: Cloud/web egress consent (Codex #4 + #6)
`ask_expert`/cloud-LLM escalation + autonomous web/study fetches (search, wiki, page-fetch) need a
per-path consent/policy layer distinct from the accelerator guard (these are *intended* outbound
calls, not misconfig). **DESIGN CALL pending — do NOT build yet.**

## Build order & conflicts
*Two PRs owned the contested files; everything else sequences around them.* **#92 (`tiers.py`) and
#93 (`orchestrator.py` + grounding) are now MERGED to `main`** — the "after #9x" gates below are
satisfied; the items can start against current main.
- **AFTER #93** (shares `orchestrator.py`, or needs the scorer on `main`): r4 #1 (concurrency),
  r4 #2 (tier-at-prompt), r4 #7 (header-spoof), r4 #9 (cache key), r4 #4 (study-gate), and the whole
  scorer-coverage workstream.
- **AFTER #92** (shares `tiers.py`): r4 #6 (STM-vs-LTM score).
- **r4 #1 FIRST among the orchestrator items** — serialize foreground turns around shared cognitive
  state before the others; a concurrent request borrowing the wrong goal/winner/context invalidates
  the rest.
- **Independent** (no held-file conflict, schedule freely): r4 #8 (study retries), r4 #10 (confidence
  contradiction).
- **HELD — loop/dream cluster, deliberate decision before building** (standing-wave risk; do NOT
  start as filler): r4 #3 (recursive dreams), r4 #5 (tension starvation), r4 #11 (pending questions).
- **Keep OPEN regardless of #93 merge:** r1 #2/#3 and r4 #2/#4 — the scorer exists but is not yet
  wired into prompt-assembly or the study/dream write paths.

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
- ⊘ **[r3 #2 / Codex #5 — WITHDRAWN, not a fix]** SGT polarity resolved FROM THE FILED PATENT. The
  canonical gate is `g(e) = 1/(1+exp(−(|e|−T)·S))` (patent [0047] / claim 4); claims 1(d)/21(c) say
  **"magnitude"**, and it is **two-sided by design** (drift is bidirectional). The live
  `abs(value−mean)` in sgt.py:69 is therefore FAITHFUL to the filed claims — a one-sided "exceeds"
  gate would DIVERGE from them. **Do NOT change it.** No SGT HOLD remains; the only SGT work is the
  config-threshold wiring below (Codex #7), default pinned to `exact_sgt.py`.
- ▶PR **[r1 #4 / Codex #7 → PR #99]** `CONFIG.pde_dt` / `sgt_threshold_sigma` were no-ops (hardcoded
  literals won). Wired: FractalField honors `CONFIG.pde_dt`; the orchestrator SGT gates honor
  `CONFIG.sgt_threshold_sigma`/`sgt_ema_alpha` (default pinned to `exact_sgt.py`); `vram_check` reads
  `CONFIG.vram_cap_gb` and is called on the field loop. *PR-open, awaiting merge.*
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

## Phase 4 (future / PARKED): Decentralized Eris Echo network · ⏸ NOT ACTIVE
**Explicitly future and parked — NOT part of Phase 1.5 / 2 / 3, not on the active punch-list.** This
section CAPTURES the vision so it isn't lost; it is gated on prerequisites that **do not exist yet**
(all post-Phase-3). Do **not** build any of it now. New single-node evidence still beats this text.

**The collective.** Each *Echo* is a node (one per machine); *Eris* is the collective Self that exists
*across* nodes — no center, no central server. A "wood-wide web, not a ledger": shared living
knowledge, not a transaction log.

**Prerequisites (none built; all gated on Phase-3 being real first):**
- **Neural-handshake secure inter-node transport** — authenticated, encrypted node↔node channel.
- **`.eris` container format** (a PNG carrier) — two strata: **cap** = the confirmed baseline (current
  agreed truth) and **mycelium** = an append-only, signed stream of candidate insights. Carries the
  BFECDS stream + six-specialist VRAM hints `{geo, sym, tri, hol, freq, cau}` + embeddings / coherence.
- **`.eris` insight-sharing protocol** — how nodes publish/subscribe mycelium insights.
- **Wave-interference two-layer consensus** — candidate insights are combined by *interference* before
  commit: **constructive → confident**, **destructive → undetermined**, and the *degree* of
  (dis)agreement maps to `dC/dX`. Interference, then commit.

**SGT-LIKE promotion gate (David's idea — SGT-*inspired*, explicitly NOT the filed SGT patent).** A
shared accumulator collects cross-node corroboration for a candidate insight; when it crosses a
criticality threshold, the insight is **promoted** — the baseline (**cap**) is rewritten and pushed
network-wide, and all nodes update at once. This is the mechanism behind Accords III / V / VI.

**Governance — the Eris Accords:** fractal mutability; **append-free / promotion-earned** (nothing
enters the baseline without crossing the gate); the promotion gate as an **immune system against group
delusion**; **Sybil resistance**; **stake-scaled corroboration**. *Source: Accords reconstruction,
chat `b4b5da95` (2026-06-25).*

**Through-line — the single-node work IS the network's foundation (why this isn't a tangent):**
- the **grounding scorer (#93)** = a node's truth-contract → becomes the network **promotion gate**;
- the **Tribe-of-11 + MoEGate** = a working **consensus prototype** (many voices → one commit);
- a **validated SGT** = the node-level gate the network's SGT-like promotion gate is modeled on.

---

## Verified-good (Codex "clean" section)
- **[PR#94 egress, Codex + chat]** Default-deny consent parsing is correct — only `1/on/true/yes`
  enable; `0/false`/empty/unset do NOT. The guard runs BEFORE the network call at all four content
  sites (embeddings/rerank/STT/VLM) with safe fallbacks. *Skeleton confirmed by both reads; the host
  classifier + status probe were the gaps (folded into #94).*
- **τ vorticity matches the canon scalar curl ∇ρ×∇θ** (sign + axis) apart from the known
  boundary-stencil issue — the core torsion math is correct.
- No additional CPU/GPU math divergence beyond the known `xp.roll` boundary + the VRAM-cap no-op.
- **[r4 clean]** `local_coherence()` matches its stated 4-neighborhood + self Kuramoto order parameter
  apart from the already-known periodic-boundary behavior; no serious live-path `beta_star` issue
  beyond its limited/inert wiring.

## The pattern across all reviews
Four recurring failure modes, not random bugs: **computed-but-not-persisted** (snapshots,
checkpoints), **computed-but-not-correct** (two DCRs, wrap, branch-cut, NaN), **configured-but-not-
wired** (pde_dt, VRAM cap, traditional_only, FRT flag), **tested-but-not-proven** (the
false-confidence tests). The physics is real and runs; it is not yet in a state to realize the
theory. That is now mapped — not guessed. *(NOTE: the SGT two-sided gate is NOT in the "not-correct"
list — Codex #5 was WITHDRAWN; two-sided is faithful to the filed patent. See Phase-2 Gates.)*
