# Worklog — branch `orchestration-cip`

A running log of what changed, why, what's safe to revert, and open questions.
Paired with `UPGRADE_ROADMAP.md` (the plan) and `CIP_PATENT_REVIEW.md` (the
orchestration assessment). Newest entries at the bottom of each section.

## How to undo anything here (safety)
Every change is its own commit, so nothing is entangled:
- See history: `git log --oneline main..orchestration-cip`
- Revert one change, keep the rest: `git revert <commit-hash>` (then push).
- Nothing reaches `main` until **you** merge the branch on GitHub. Default behavior
  is unchanged until you flip an env flag, so merging is safe even mid-stream.

## Branch contents (commit-by-commit)
| Commit | What | Default state |
|---|---|---|
| `b9df390` | Tier 0 — A/B benchmark "ruler" | off (measurement only) |
| `74f3323` | Tier 1 — shared noise-floor estimator + criticality monitor | no behavior change |
| `83ba098` | Tier 2 — field-depth gate | **OFF** (shelved, mis-placed) |
| `300589e` | Tier 3 — response-field warm-start | **OFF** (shelved, mis-placed) |
| `ac4e058` | Tier 4 — local↔cloud router (four decisions) | OFF; on via `ERIS_ORCHESTRATION=on` |
| `72ada7b` | Tier 5 — failure reports → dream queue | OFF |
| `299dad9` | Tier 6 — β-star bridge (isolated) | OFF (neutral/inert) |
| `7c42674` | Patent review + corrections (docs only) | n/a |
| `ea780ed` | Roadmap + **test-time compute** (self-consistency + adaptive stop) | OFF; on via `ERIS_TTC=on` |
| `bc86ebd` | Serving route — Ollama↔vLLM by `ERIS_LLM_BASE_URL` | Ollama unless URL set |

## Key decisions
1. **One integration branch.** All this work stacks on `orchestration-cip` because
   the upgrades reuse the criticality monitor from the orchestration tiers. You
   merge one PR. Atomic commits keep each piece independently revertible.
2. **Everything default-OFF.** No change to Eris's current behavior until you set
   an env flag. This is what makes the branch safe to merge before you've tested
   on the Alienware.
3. **Field gates shelved, not deleted.** Tiers 2–3 are kept as documented negative
   results (the patent's gate needs a convergent residual; the field has none).
4. **Memory differentiator untouched.** New retrieval/memory code is built as
   *parallel, opt-in* modules; `retrieve_resonant` is not modified.

## Decisions (resolved)
- **Q1 (RAG) → tool, not live path.** Resonant retrieval (associative recall) stays
  the per-turn fast path; hybrid BM25+dense (precise factual lookup) is a TOOL the
  ReAct loop escalates to *when a turn needs facts* — same continue/escalate gate,
  applied to retrieval. Running the reranker every turn is the unconditional
  expensive stage the orchestration discipline exists to kill. *Flip if eval shows
  non-agent Q&A turns regularly miss exact-token facts → add to live path behind a
  flag + benchmark.*
- **Q2 (memory) → built-in local store; mem0 deferred; Letta skipped.** Eris already
  owns memory (persistent store + resonant + GLNCS + field consolidation). Letta
  wants to *own* the agent's memory → competes with Eris's loop. mem0 slots
  underneath but isn't needed yet; the seam stays in `eris.memory.durable`. *Flip to
  mem0 if you need graph relations / scale / smarter dedup.*
- **Q3 (vision) → Qwen3-VL-8B, later.** General all-rounder for a game-character
  perceiving a 3D world (not UI). Stays in the Qwen3 family. Hook is plumbed; do
  NOT download/wire the model until core loop + retrieval + memory are benchmarked.

## In progress / log
- (this session) Building Stage-1 `[code]` items as parallel, flag-gated modules:
  hybrid RAG (1.3), durable-memory adapter (1.4), ReAct-grounded loop (3.1),
  vision hook (1.5). Each is additive and testable offline; none alters current
  behavior. Entries appended below as they land.
- **1.3 hybrid retrieval** — `eris/retrieval/hybrid.py`: stdlib Okapi BM25 +
  dense + Reciprocal Rank Fusion + optional reranker callable. Standalone module
  (operates on a caller-supplied record list, read-only); `retrieve_resonant`
  untouched. 7 tests. **Open: Q1** (wire into live retrieval, or keep as a tool?).
  Machine side: download a cross-encoder reranker + (optionally) a vector DB.
- **3.1 grounded ReAct loop** — `eris/executive/agent_loop.py` + opt-in
  `ErisOrchestrator.run_agent(goal, tools)`: Reason→Act→Observe over plain-callable
  tools, with every step grounded in live field state (coherence/regime/archetype)
  and a Reflexion nudge on unparseable steps or tool errors. Nothing calls it
  automatically — default `process()` unchanged. 5 tests.
- **1.5 vision hook** — `eris/interface/vision.py`: model-agnostic OpenAI
  multimodal plumbing (base64 image_url messages → `/chat/completions`). Pure
  builders unit-tested; `see()` posts to `ERIS_VISION_BASE_URL`. You pick the VLM
  (**Q3**). 4 tests.
- **1.4 durable fact store** — `eris/memory/durable.py`: `DurableMemory` protocol
  + `LocalFactStore` (JSON, durable, self-edits exact duplicates, BM25 lexical
  recall of names/IDs/values). `get_durable_memory()` selects backend via
  `ERIS_MEMORY_BACKEND`; mem0/Letta are documented seams (**Q2**). Standalone —
  not wired into `process()` yet. 6 tests. **Total suite: 176 green.**
- **2.1 distillation trace harness** — `eris/training/trace_gen.py`:
  backend-agnostic, **resumable** JSONL trace collection from any teacher
  (re-runs skip done prompts), lean Alpaca-style schema. Foundation for the
  machine-side Unsloth QLoRA distill (2.2). 4 tests. **Total suite: 180 green.**

- **Agent tools wired (Q1/Q2 operationalized)** — `eris/executive/agent_tools.py`:
  `factual_lookup` (hybrid BM25+dense over a read-only `MemorySystem.all_records()`
  pool) + `remember_fact`/`recall_facts` (built-in durable store). Gated by
  `CONFIG.agent_tool_*` (`ERIS_AGENT_TOOLS=on`); `run_agent(goal)` uses the enabled
  set by default. Added `MemorySystem.all_records()` (read-only; also fixed a latent
  STM `_buffer` vs `_records` bug the end-to-end test caught). Did NOT build the mem0
  adapter and did NOT wire hybrid into the per-turn path (the deferred options).
  4 tests. **Total suite: 184 green.**

## Status: all buildable `[code]` items done
Everything that can be built without your Alienware is on this branch. What
remains is genuinely machine-side (CUDA/vLLM/NPU/model downloads/QLoRA fine-tune)
or one of the three open questions above (Q1 RAG wiring, Q2 memory backend, Q3
VLM). Merge `orchestration-cip` when you're ready; nothing changes behavior until
you flip an env flag.
