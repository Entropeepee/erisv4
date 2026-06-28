# Benchmark suite — Eris vs a bare model (two-arm, equal-budget)

Goal: test whether Eris's document-grounded multi-specialist pipeline lifts a **weak local model**
on the capabilities it *claims* (comprehension, faithfulness, multi-hop) — measured honestly
against the **same bare model** on identical datasets and scorers, at a **matched token budget**.

This package (`eris/experiments/benchmarks/`) is a self-contained two-arm runner. Its load-bearing
logic (prompt building, scoring, equal-token-budget accounting, dataset mapping) is unit-tested
offline; the live parts need a served model + the `datasets` package. It mirrors Inspect AI's
Dataset → Solver → Scorer split (optional `inspect_glue.py` wrapper) but needs no extra deps.

## The honest framing (from the benchmark research)

- **Lead with document-grounded benchmarks** — FRAMES, QuALITY-HARD, RAGTruth, MuSiQue. The
  passage is provided, so the score reflects *comprehension + faithfulness*, not the small model's
  memorized-knowledge ceiling, and contamination matters far less. **This is where Eris should
  win.**
- **Closed-book (MMLU-Pro, GPQA-Diamond) are controls only** — they measure the bare model's raw
  deficit. Expect Eris to lift them modestly; a large closed-book win over a frontier model would
  be *extraordinary* and needs a contamination audit before you believe it. (GPQA-Diamond is
  near-saturated at the top and Gemini leads it — use it to size the deficit, not as a proving
  ground.)
- **Equal token budget is mandatory.** The most common credibility failure in "scaffold beats bare
  model" claims is unequal compute. Every run reports tokens/question for both arms and flags
  whether the comparison was equal-budget. A higher-accuracy arm that also spent more tokens has
  **not** cleanly won.

## Install

```bash
pip install datasets requests          # live loaders + the bare arm
pip install inspect_ai                  # optional, only for the Inspect wrapper
```

## Serve the local model (Arm A, and Eris's language center)

Ollama or vLLM, both OpenAI-compatible. **Critical:** Ollama defaults to a 4096-token context,
which silently truncates QuALITY/FRAMES passages — raise it:

```bash
OLLAMA_CONTEXT_LENGTH=22000 ollama serve
export ERIS_BENCH_BASE_URL=http://localhost:11434/v1
export ERIS_BENCH_MODEL=<your-local-model>     # e.g. the ~13GB model you run for Eris
```

## Run

Bare arm (works once the model is served):

```bash
python -m eris.experiments.benchmarks.run --benchmark mmlu_pro --arm bare --limit 50
python -m eris.experiments.benchmarks.run --benchmark quality  --arm bare --limit 50 --hard-only
```

Both arms head-to-head — wire the Eris arm with a factory that returns an `answer_fn`:

```bash
python -m eris.experiments.benchmarks.run --benchmark frames --arm both --limit 50 \
    --eris-factory my_bench:make_eris_arm
```

### Wiring the Eris arm

`eris/experiments/benchmarks/arms.py::eris_pipeline_arm` is the intended shape. For each question
it splits the provided SOURCE back out, ingests it into a **scratch** Eris memory, then asks the
full pipeline. A minimal factory:

```python
# my_bench.py
from eris.orchestrator import ErisOrchestrator
from eris.knowledge import web_reader
from eris.experiments.benchmarks.arms import eris_pipeline_arm

def make_eris_arm():
    orch = ErisOrchestrator(data_dir="eris_bench_data")   # NOT your real eris_data
    def reset():  ...   # clear the scratch MTM/LTM between items
    def ingest(ctx): web_reader.ingest_text(orch.memory, ctx, title="bench")
    def ask(q):
        import asyncio
        r = asyncio.run(orch.hive_research(q, scope="doc", document="bench"))
        # token cost = sum of tier calls if you want strict equal-budget accounting
        return (r.get("synthesis_full") or r.get("synthesis") or ""), 0
    return eris_pipeline_arm(ingest, ask, reset)
```

Use a **separate `data_dir`** so benchmarking never writes into your live `eris_data`. The A/B
harness (`hive_ab.py`) already sets `ERIS_ROUTE_GAPS=0` and `ERIS_SYNTHESIS_WRITEBACK=0`; set the
same here if you reuse the live orchestrator.

## Suite (recommended order)

| Tier | Benchmark | `--benchmark` | Why |
|---|---|---|---|
| 1 (proving ground) | FRAMES | `frames` | multi-hop comprehension-over-extraction |
| 1 | QuALITY-HARD | `quality --hard-only` | skim-proof long-document comprehension |
| 1 | RAGTruth | `ragtruth` | span-level faithfulness ("catch own hallucinations") |
| 1 | MuSiQue | `musique` | shortcut-resistant 2–4 hop |
| 2 (control) | MMLU-Pro | `mmlu_pro` | general-reasoning bare-model ceiling |
| 2 (control) | GPQA-Diamond | `gpqa` | the one Gemini leads; sizes the deficit |

**RAGTruth is faithfulness, not accuracy** — its items are `meta["type"]=="faithfulness"` with
annotated hallucination spans. Score it with span overlap / RAGChecker / RAGAS layered on the
arm's output, not exact-match. The other five score with the built-in exact-match / multiple-choice
/ abstention scorers.

## Cross-check

Validate the bare arm's closed-book numbers (MMLU-Pro, GPQA) against EleutherAI
`lm-evaluation-harness` (`local-chat-completions`, same endpoint) to catch prompt/scoring drift.

## Reading the result

- **Eris wins on grounded tasks, not closed-book** → the *expected and honest* outcome; position
  the architecture as comprehension/faithfulness amplification, not knowledge amplification.
- **Eris loses on FRAMES/MuSiQue at equal budget** → the multi-agent layer is adding coordination
  overhead without benefit (consistent with the equal-compute literature); simplify toward
  single-agent self-consistency + strong retrieval.
- **Eris claims a closed-book GPQA/MMLU-Pro win over a frontier model** → treat as extraordinary;
  run rephrased-question + decontamination checks before believing it.

## Caveat on the dataset adapters

`datasets.py` mappers follow the dataset cards as of the 2026 brief and use tolerant `.get()`
fallbacks, but HF card schemas drift and some datasets are gated (accept terms on the hub first).
If a live load mis-maps a field, fix the `_*_item` mapper — its logic is unit-tested in
`tests/test_benchmarks.py`, so add a sample row there alongside the fix.
