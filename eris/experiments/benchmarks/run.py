"""CLI runner for the two-arm benchmark.

  # bare model only (works out of the box once Ollama/vLLM is serving):
  python -m eris.experiments.benchmarks.run --benchmark mmlu_pro --arm bare --limit 50

  # both arms head-to-head (wire the Eris arm via a factory, see --eris-factory / docs):
  python -m eris.experiments.benchmarks.run --benchmark frames --arm both --limit 50 \
      --eris-factory my_pkg.bench:make_eris_arm

The factory is `module:function` returning an answer_fn (prompt -> str | (text, tokens)); see
eris/experiments/benchmarks/arms.py::eris_pipeline_arm for the intended wiring."""
import argparse
import importlib
import json
import os
import sys
from typing import Callable, Optional

from eris.experiments.benchmarks.core import run_arm, compare
from eris.experiments.benchmarks.scoring import score_results
from eris.experiments.benchmarks.datasets import LOADERS, GROUNDED


def _load_factory(path: str) -> Callable:
    mod, _, fn = path.partition(":")
    if not mod or not fn:
        raise SystemExit(f"--eris-factory must be 'module:function', got {path!r}")
    return getattr(importlib.import_module(mod), fn)


def main(argv: Optional[list] = None):    # pragma: no cover - CLI orchestration over live arms
    ap = argparse.ArgumentParser(description="Eris vs bare-model two-arm benchmark")
    ap.add_argument("--benchmark", required=True, choices=sorted(LOADERS))
    ap.add_argument("--arm", default="bare", choices=["bare", "eris", "both"])
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--hard-only", action="store_true", help="QuALITY: HARD subset only")
    ap.add_argument("--eris-factory", default="",
                    help="module:function returning the Eris answer_fn (for --arm eris/both)")
    ap.add_argument("--out", default="", help="write the full JSON report here")
    args = ap.parse_args(argv)

    # Loud guard against the #1 silent-truncation footgun: too-small Ollama context window quietly
    # chops long QuALITY/FRAMES passages, so BOTH arms answer from a partial source and the numbers
    # look worse than reality with no signal. Warn before a single token is spent.
    _ctx = os.environ.get("OLLAMA_CONTEXT_LENGTH")
    try:
        _ctx_n = int(_ctx) if _ctx else 0
    except ValueError:
        _ctx_n = 0
    if _ctx_n < 22000:
        print(f"[bench] WARNING: OLLAMA_CONTEXT_LENGTH={_ctx or '(unset → Ollama default ~4096)'} "
              "— long passages will be SILENTLY TRUNCATED by the model, making both arms look worse "
              "than they are. Set OLLAMA_CONTEXT_LENGTH>=22000 before serving for grounded runs.",
              file=sys.stderr)

    loader = LOADERS[args.benchmark]
    kw = {"hard_only": True} if (args.benchmark == "quality" and args.hard_only) else {}
    items = loader(limit=args.limit, **kw)
    print(f"[bench] {args.benchmark}: {len(items)} items "
          f"({'grounded' if args.benchmark in GROUNDED else 'closed-book control'})",
          file=sys.stderr)

    from eris.experiments.benchmarks.arms import default_bare_arm
    report = {"benchmark": args.benchmark, "n": len(items)}
    res_bare = res_eris = None
    if args.arm in ("bare", "both"):
        res_bare = score_results(run_arm(items, default_bare_arm(), "bare"), items)
    if args.arm in ("eris", "both"):
        if not args.eris_factory:
            raise SystemExit("--arm eris/both requires --eris-factory module:function")
        eris_fn = _load_factory(args.eris_factory)()
        res_eris = score_results(run_arm(items, eris_fn, "eris"), items)

    from eris.experiments.benchmarks.core import item_details
    if res_bare and res_eris:
        report["compare"] = compare(res_bare, res_eris)
        arms_for_details = {"bare": res_bare, "eris": res_eris}
    else:
        from eris.experiments.benchmarks.core import accuracy, budget_report, faithfulness
        only = res_bare or res_eris
        label = "bare" if res_bare else "eris"
        report["result"] = {"accuracy": accuracy(only), "budget": budget_report(only),
                            "faithfulness": faithfulness(only)}
        arms_for_details = {label: only}
    # Per-item predictions vs gold — so a 0% score is diagnosable (wrong format? strict scorer?).
    report["items"] = item_details(items, arms_for_details)
    print(json.dumps(report, indent=2, default=str))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":   # pragma: no cover
    main()
