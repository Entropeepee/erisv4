# Red-team of the physics-value ablation (2026-06)

An adversarial review (~52 agents, 6 attack angles, every flaw verified) of the QuALITY ablation
spec (authored by a degraded Claude-chat, implemented verbatim on `feat/bench-ablation`). Raised 45
candidate flaws; **40 confirmed: 1 fatal, 23 major, 16 minor.** Full output:
`/tmp/.../tasks/wik75j829.output`.

## Verdict: do NOT run as-is

As written, the experiment would emit confident PASS/FAIL booleans that a non-expert reads as
settled architecture verdicts — while every verdict sits inside the noise band and several are
confounded at the root. The harness *code* is sound; the experimental *design* and what it claims
to conclude are not.

## THE FATAL FLAW — option-blindness (the big miss)

Only the **bare** arm sees the multiple-choice options. Every hive arm (B/C/D/C-sel) is handed an
**option-stripped, open-ended** question (`eris_arm.py` `_question_line` keeps only `Question: …`),
reasons over it, produces a synthesis, then a single extraction call maps that synthesis onto four
choices **the hive never saw**. QuALITY-HARD is largely answerable only by *elimination among the
four options*, so:

- "only the config differs between arms" is **false at the most basic level** — arm A reasons
  *with* the options, the hive reasons *without* them.
- Every bare-vs-hive and D-vs-A number is confounded *before the physics is even in play*.
- The benchmark code comment "both arms see the same information" is simply wrong for MC.

This is not just an ablation bug — it taints **every QuALITY result so far**, including the 0/3 run.
The hive's eloquent-but-wrong syntheses were answering a *different, harder* question (open-ended
"why does X feel Y") than the bare model (pick the best of these four). Fixing this is the single
most important change, and it is **fairness, not teaching to the test**: both arms must see the same
task.

## The other confirmed flaws that matter

**Broken positive control.** `C-sel` (cap=6) vs `C` (cap=50) is sold as "proof the field selects,"
but lowering `ERIS_HIVE_MAX_SOURCES` silently moves **four** coupled knobs (lead chunks 50→16,
`top_k`, candidate pool 50→12, final slice 50→6) plus embedding pre-truncation. So `C-sel ≠ C` is
guaranteed by **pool starvation, not selection**. The correct selection test never runs: hold the
pool fixed, toggle **only** the rerank, and compare the chosen **chunk-id sets** (not answer letters).

**No falsifiability / no power.** Every verdict is a bare scalar compare (`D > max_abc`,
`rate == 1.0`, `rate < 1.0`) with no effect-size threshold, no significance test, no seed, no
repeats, and no rejection rule. At N=3 (default) and N=10, **no outcome can reach significance**, yet
a single-item decode flip flips every headline. The all-zero outcome (the observed regime) makes
`D > max_abc` compute `0 > 0 = False` and reports "the inference fix was refuted" when the test was
simply uninformative (floor effect).

**Wrong signal for the mechanistic claims.** "B==C" and "C-sel≠C" are judged on final extracted
**letters**, not source sets — but the truth is already decidable from `retrieval_stats.truncated_any`
(which the evaluator never even receives). Two arms can ground on different chunks yet collapse to
the same letter by ~25% MC chance; thresholds are rigged opposite directions (B==C needs *perfect*
1.0; C-sel≠C passes on *any* difference), so noise fabricates the positive control.

**D bundles three levers.** `ERIS_HIVE_TASK=inference` changes the synth prompt, the canon prompt,
**and** skips Elos — all at once. A D win can't be attributed to any one, and they're different
production changes.

**Scoring can fake or hide the answer.** On extraction failure the full multi-thousand-char synthesis
is scored, and the `option_text` substring path credits the item if the gold option's wording appears
*anywhere* — biased toward D (longest, most quote-heavy syntheses). The extraction call also runs on
`orch.mediator` (local), which **bypasses the attributability guard** — the slot that actually picks
the scored letter may be a different model than the banner claims.

**Unverified foundations.** The gold-letter mapping (1-based vs 0-based) is flagged "the one real
ambiguity" in the code and never hand-checked — an off-by-one would grade **every item** against the
wrong letter. `limit=3` takes the first three questions, which in QuALITY's nested schema likely come
from **one article** — generalizes to nothing. `token_ratio_vs_bare` mixes real tokens with a
call-count proxy and can report D as ~100× cheaper when it actually spends more.

**Determinism is assumed, not achieved.** No seed is wired; temp-0 greedy is not bit-reproducible
under vLLM continuous batching; each item runs once; the self-consistency noise floor is never
measured. Every equality-based verdict rides on decode jitter.

## Corrected design (what to actually do)

1. **Give every arm the full task** — question + the A/B/C/D options — so the hive reasons
   option-conditioned, or use an MC-native option scorer for all arms (incl. bare). *Fixes the fatal
   flaw; also removes the free-text→letter extractor as a confound.*
2. **Selection test = pool-fixed, rerank-only toggle:** cap=6 resonance-OFF vs cap=6 resonance-ON,
   with `n_doc`/`top_k`/dedup-cap pinned identical, judged on chosen **chunk-id sets**.
3. **"Decorative" (B==C) judged on source SETS,** not letters: pass iff `truncated_any` is False for
   both AND the grounded chunk-id sets match. Sort the capped list deterministically so order can't
   leak into the comparison.
4. **Decompose D** into D-prompt / D-elos / D-bundle, each vs C, and add a per-item diagnostic: did
   strict-mode stripping remove the gold-supporting sentence, and did that flip the letter?
5. **Real statistics:** k≥5 seeded repeats/item; a one-sided Fisher/binomial test on the common
   graded set; an explicit rejection rule; a power-based target N (likely ≫10). At N=3 the only
   honest output is "underpowered / inconclusive."
6. **Wire a seed** + pin provider/quantization; measure the run-to-run noise floor F and judge
   equality verdicts against F, never `==1.0`/`<1.0`.
7. **Equal denominators:** retry errored items or compare only the intersection graded by every arm;
   force `ERIS_HIVE_CONCURRENCY=1`; hard-set (not `setdefault`) the temps and assert the *effective*
   extraction temp is 0.
8. **Fix the extractor confounds:** route extraction through the attributable tier (or assert+print
   its model in the banner); on extraction failure mark the item **ungraded**, don't score the full
   synthesis; restrict the substring path to the short answer and count items scored via it.
9. **Sample honestly:** seeded items across **distinct articles**, report article coverage, treat
   limit=3 as a no-verdict smoke test; hand-verify gold on 3–5 items; pin the dataset revision; add a
   closed-book control to quantify 72B memorization of public QuALITY.
10. **Cost honestly:** compute `token_ratio` only when both arms are `real_tokens`; fold the
    Eris-only extraction call into the comparison; assert a shared cost basis before any budget verdict.

## Bottom line

The degraded-model spec would NOT have produced a trustworthy answer. Its three worst mistakes:
option-blindness (every arm comparison confounded before physics), a broken positive control (C-sel
changes the pool, not the selection), and zero falsifiability/power (a one-item flip flips every
verdict at N=3). The experiment is salvageable but needs the surgery above. Even fully fixed, be
honest about the ceiling: a 72B single-node pilot on a public (possibly memorized) benchmark can, at
best and with significance, tell you whether *permitting-the-inference beats strict-grounding on this
model* and whether *the resonance rerank changes which chunks are selected when the pool exceeds the
cap*. It cannot prove the physics adds end-to-end value, and it cannot show the result transfers to
the production frontier-synth pipeline.

This compounds the architecture audit's finding: the physics largely **doesn't run on the benchmark
path at all**. Together they raise the strategic question of whether QuALITY-HARD MC is even the
right instrument for testing what this architecture is for.
