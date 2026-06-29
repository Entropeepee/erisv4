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


# Keys present in BOTH the .env file and the shell, where the shell value DIFFERS — i.e. a stale
# `set`/export that silently overrides the file (the explicit env always wins, by design). Recorded
# so the runner can NAME the override instead of letting it quietly produce a non-attributable run.
_DOTENV_CONFLICTS: list = []


def _load_dotenv(path: str = ".env") -> int:
    """Load KEY=VALUE pairs from a .env file into os.environ — so keys + config live in ONE
    gitignored file, entered once, instead of `set` commands that vanish when the window closes.
    Does NOT override anything already set (an explicit `set`/export still wins). Dependency-free;
    silent on a missing file. Runs BEFORE the eris imports so eris.config picks the values up.

    Records any key that the shell sets to a DIFFERENT value than the file (into _DOTENV_CONFLICTS)
    so the runner can warn that a stale `set` is overriding the .env — the exact footgun that turns
    an intended one-model attributable run into a quietly mismatched one."""
    n = 0
    _DOTENV_CONFLICTS.clear()
    try:
        if not os.path.exists(path):
            return 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if not k:
                    continue
                if k in os.environ:                     # explicit env wins over the file
                    if os.environ[k] != v:              # …but record it so the override is visible
                        _DOTENV_CONFLICTS.append((k, os.environ[k], v))
                    continue
                os.environ[k] = v
                n += 1
    except Exception:
        pass
    return n


def _apply_attributable_preset() -> bool:
    """ERIS_BENCH_ATTRIBUTABLE=1 → the IRON-RULE one-switch: force ONE model into every slot
    (specialists, gap-closing, synthesis, AND the bare arm) so the only variable between the two
    arms is architecture-vs-bare, never the model. This deliberately OVERRIDES any stale shell `set`
    (e.g. a lingering `set ERIS_TIER_SYNTH=…claude…` from a prior speed run) — defeating the exact
    footgun that prints '✗ MISMATCH'. Runs BEFORE the eris imports so eris.config binds the forced
    values (the tier/synth-cloud config is read at import time). Returns True iff it fired."""
    if os.environ.get("ERIS_BENCH_ATTRIBUTABLE", "").strip().lower() not in (
            "1", "on", "true", "yes"):
        return False
    model = os.environ.get("ERIS_BENCH_MODEL", "").strip()
    if not model:
        raise SystemExit("ERIS_BENCH_ATTRIBUTABLE=1 requires ERIS_BENCH_MODEL to name the single "
                         "model to use in every slot (e.g. qwen/qwen-2.5-72b-instruct).")
    for k in ("ERIS_TIER_FREE", "ERIS_TIER_CHEAP", "ERIS_TIER_SYNTH"):
        os.environ[k] = model                    # FORCE (override) — stale values must not survive
    os.environ["ERIS_HIVE_SYNTH_CLOUD"] = "0"    # synthesis REUSES the one model, no cloud escalation
    # route the hive through the same endpoint the bare arm uses, unless a gateway is set explicitly
    if not os.environ.get("ERIS_GATEWAY_BASE_URL") and os.environ.get("ERIS_BENCH_BASE_URL"):
        os.environ["ERIS_GATEWAY_BASE_URL"] = os.environ["ERIS_BENCH_BASE_URL"]
    if not os.environ.get("ERIS_GATEWAY_API_KEY") and os.environ.get("ERIS_BENCH_API_KEY"):
        os.environ["ERIS_GATEWAY_API_KEY"] = os.environ["ERIS_BENCH_API_KEY"]
    return True


_dotenv_loaded = _load_dotenv()        # before any eris import, so config.py reads these values
_attributable = _apply_attributable_preset()   # …and the one-model override lands before config too

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

    if _dotenv_loaded:
        print(f"[bench] loaded {_dotenv_loaded} setting(s) from .env (keys + config — entered once)",
              file=sys.stderr)
    # NO black box for the operational env: name every stale shell var that is overriding the .env.
    # This is the exact failure that prints '✗ MISMATCH — NOT attributable' (a lingering
    # `set ERIS_HIVE_SYNTH_CLOUD=1` from a prior window), so say it in plain terms with the fix.
    if _DOTENV_CONFLICTS:
        print(f"[bench] WARNING: {len(_DOTENV_CONFLICTS)} shell env var(s) OVERRIDE your .env "
              "(an explicit `set`/export always wins over the file):", file=sys.stderr)
        for k, shell_v, file_v in _DOTENV_CONFLICTS:
            print(f"[bench]   {k} = {shell_v!r} (shell)  vs  {file_v!r} (.env)", file=sys.stderr)
        print("[bench]   → stale values like these can make a run NON-attributable. Open a FRESH "
              "terminal (the `set` vars vanish) or clear each (`set VAR=` on Windows / "
              "`unset VAR` on bash), then re-run. Or set ERIS_BENCH_ATTRIBUTABLE=1 to force one "
              "model into every slot regardless.", file=sys.stderr)
    if _attributable:
        print(f"[bench] ATTRIBUTABLE mode: forcing {os.environ.get('ERIS_BENCH_MODEL')!r} into every "
              "slot (specialists / gaps / synthesis / bare), hive-synth-cloud OFF — overriding any "
              "stale shell vars. The only variable left is architecture-vs-bare.", file=sys.stderr)

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
    # Record the run CONFIG + the Eris routing regime IN the saved report — two reports with a
    # different limit / hard_only / model / synth-cloud setting are otherwise indistinguishable, and
    # the routing regime (the single biggest cost + attributability determinant) was previously only
    # on stderr, absent from the --out JSON.
    report = {"benchmark": args.benchmark, "n": len(items),
              "config": {"limit": args.limit, "arm": args.arm, "hard_only": args.hard_only,
                         "eris_factory": args.eris_factory or None,
                         "attributable_mode": _attributable,
                         "dotenv_conflicts": [list(c) for c in _DOTENV_CONFLICTS]}}
    if args.arm in ("eris", "both"):
        report["eris_routing"] = {
            "gateway_base_url": os.environ.get("ERIS_GATEWAY_BASE_URL"),
            "tier_free": os.environ.get("ERIS_TIER_FREE"),
            "tier_synth": os.environ.get("ERIS_TIER_SYNTH"),
            "hive_synth_cloud": os.environ.get("ERIS_HIVE_SYNTH_CLOUD", "0"),
            "bench_model": os.environ.get("ERIS_BENCH_MODEL")}
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
