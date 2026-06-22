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

## Open questions for you (answer anytime via Claude chat)
- **Q1 (RAG):** do you want the hybrid retrieval wired *into* Eris's live retrieval
  (replacing/augmenting `retrieve_resonant`), or kept as a separate tool the agent
  can call? I'm building it standalone first so it changes nothing until you decide.
- **Q2 (memory):** mem0 vs Letta as the durable store under the field layer — or
  the simple built-in JSON store I'm scaffolding? Needs a `pip install` either way
  (your machine).
- **Q3 (vision):** which VLM — Qwen3-VL-8B (general) or InternVL3-8B (UI/screens)?
  Affects only the machine-side download; the code hook is model-agnostic.

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
