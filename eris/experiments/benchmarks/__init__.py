"""Two-arm benchmark suite — Eris vs a bare model on identical datasets + scorers, with
equal-token-budget accounting. See docs/benchmarks.md for the full run guide.

The load-bearing logic (BenchItem, the arms, scoring, budget accounting) is dependency-free and
unit-tested offline; the live loaders need `datasets`, the bare arm needs a served model, and the
optional Inspect-AI wrapper needs `inspect_ai`."""
from eris.experiments.benchmarks.core import (
    BenchItem, ArmResult, build_prompt, run_arm, budget_report, accuracy, faithfulness, compare,
)
from eris.experiments.benchmarks.scoring import (
    score_item, score_results, normalize, faithfulness_score)
from eris.experiments.benchmarks.datasets import LOADERS, GROUNDED, CLOSED_BOOK

__all__ = ["BenchItem", "ArmResult", "build_prompt", "run_arm", "budget_report", "accuracy",
           "faithfulness", "compare", "score_item", "score_results", "normalize",
           "faithfulness_score", "LOADERS", "GROUNDED", "CLOSED_BOOK"]
