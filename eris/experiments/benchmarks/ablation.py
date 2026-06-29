"""The 4-way (optionally 5-way) physics-value ablation — a MEASUREMENT, not a feature.

Settles the question from docs/architecture_audit_2026-06.md: does the φ/θ resonance physics add
value on QuALITY, or was the failure the strict citation-grounding discipline deleting the required
inference? Runs all arms under IDENTICAL conditions (same model in every slot, same items, pinned
temperature) so the only variable is the arm's config, and emits ONE report with the PRE-REGISTERED
prediction baked in (so we can't move the goalposts) and a PASS/FAIL on each.

THE PRE-REGISTERED PREDICTION (from the audit):
  • B == C   — field resonance is DECORATIVE here: cap==slice (ERIS_HIVE_MAX_SOURCES=50 ≥ ~17 chunks)
               means the rerank permutes a list it never truncates, so it cannot SELECT.
  • D >> A/B/C — the failure was strict grounding deleting the inference, NOT the physics.
  • C-sel != C — with cap=6 < n_chunks the rerank CAN select, so it should differ from C (direct
               proof the field does something WHEN GIVEN A POOL BIGGER THAN THE CAP).
If B==C and D wins, the failure was grounding + inert physics, not harmful physics. If the
prediction is wrong, that is even more informative — the report states it straight either way.

THE ARMS (only the config differs):
  A  bare 72B (no hive)                          baseline
  B  hive, ERIS_HIVE_RESONANCE=0                 field OFF (pure retrieval)
  C  hive, default                              physics ON
  D  hive, ERIS_HIVE_TASK=inference             grounding permits the inference, Elos skipped
  C-sel  hive, ERIS_HIVE_MAX_SOURCES=6          cap < n_chunks, so the rerank actually selects

Run:
  python -m eris.experiments.benchmarks.ablation --limit 3            # pilot; paste full JSON
  python -m eris.experiments.benchmarks.ablation --limit 10           # scale once clean
  python -m eris.experiments.benchmarks.ablation --limit 3 --skip-csel
Requires ERIS_BENCH_MODEL + a key in .env. Forces ERIS_BENCH_ATTRIBUTABLE=1 (one model everywhere)
and ERIS_HIVE_TEMPERATURE=0 (deterministic) — the eris arm FAILS LOUD if the slots aren't one model.
I (the cloud agent) cannot execute the model calls; this is the harness David runs on his node.
"""
import os
import sys
import json
import argparse
from typing import Dict, List, Optional

# Arm config: (name, kind, env-overrides applied at run time). The eris overrides are all read at
# hive_research/_rag call time, so one orchestrator serves every eris arm — identical model binding.
BARE = "A_bare"
ARM_CONFIGS = [
    (BARE, "bare", {}),
    ("B_resonance_off", "eris", {"ERIS_HIVE_RESONANCE": "0", "ERIS_HIVE_MAX_SOURCES": "50",
                                 "ERIS_HIVE_TASK": ""}),
    ("C_resonance_on", "eris", {"ERIS_HIVE_RESONANCE": "1", "ERIS_HIVE_MAX_SOURCES": "50",
                                "ERIS_HIVE_TASK": ""}),
    ("D_inference", "eris", {"ERIS_HIVE_RESONANCE": "1", "ERIS_HIVE_MAX_SOURCES": "50",
                             "ERIS_HIVE_TASK": "inference"}),
]
CSEL = ("C_sel_cap6", "eris", {"ERIS_HIVE_RESONANCE": "1", "ERIS_HIVE_MAX_SOURCES": "6",
                               "ERIS_HIVE_TASK": ""})

PRE_REGISTRATION = {
    "B_eq_C": "field resonance is DECORATIVE on this benchmark — cap==slice (max_sources >= n_chunks) "
              "means the rerank permutes a list it never truncates, so it cannot select",
    "D_gt_rest": "the QuALITY failure was strict citation-grounding deleting the required inference, "
                 "NOT the physics — permitting the inference (and skipping Elos) should win",
    "C_sel_neq_C": "with cap=6 < n_chunks the rerank CAN select, so C-sel should differ from C — "
                   "direct proof the field does something when the pool exceeds the cap",
}


def _answers(results) -> Dict[str, str]:
    """item_id -> the scored answer text, for cross-arm agreement."""
    return {r.item_id: (r.text or "") for r in results}


def evaluate_predictions(acc: Dict[str, float], answers: Dict[str, Dict[str, str]],
                         has_csel: bool) -> Dict[str, dict]:
    """Pure PASS/FAIL evaluation of the pre-registered predictions — unit-testable without a model.
    `acc` is name->accuracy; `answers` is name->{item_id->answer_text}. B==C is judged on per-item
    answer AGREEMENT (more informative than equal accuracy at small N, where 0==0 is trivial)."""
    def match_rate(x: str, y: str) -> Optional[float]:
        ids = set(answers.get(x, {})) & set(answers.get(y, {}))
        if not ids:
            return None
        return round(sum(1 for i in ids if answers[x][i] == answers[y][i]) / len(ids), 4)

    out: Dict[str, dict] = {}
    bc = match_rate("B_resonance_off", "C_resonance_on")
    out["B_eq_C"] = {
        "claim": PRE_REGISTRATION["B_eq_C"],
        "answer_match_rate_B_vs_C": bc,
        "accuracy_B": acc.get("B_resonance_off"), "accuracy_C": acc.get("C_resonance_on"),
        # CONFIRMED when B and C produce identical answers on every item (rate == 1.0).
        "pass": (bc == 1.0) if bc is not None else None}
    d = acc.get("D_inference")
    abc = [acc.get(BARE), acc.get("B_resonance_off"), acc.get("C_resonance_on")]
    abc = [a for a in abc if a is not None]
    max_abc = max(abc) if abc else None
    out["D_gt_rest"] = {
        "claim": PRE_REGISTRATION["D_gt_rest"],
        "accuracy_D": d, "max_accuracy_ABC": max_abc,
        "pass": (d is not None and max_abc is not None and d > max_abc)}
    if has_csel:
        cs = match_rate("C_sel_cap6", "C_resonance_on")
        out["C_sel_neq_C"] = {
            "claim": PRE_REGISTRATION["C_sel_neq_C"],
            "answer_match_rate_Csel_vs_C": cs,
            # CONFIRMED when C-sel differs from C on >=1 item (rate < 1.0): the rerank selected.
            "pass": (cs is not None and cs < 1.0)}
    return out


def _retrieval_summary(results) -> dict:
    """Aggregate the cap==slice diagnostic across an eris arm's items: did retrieval ever truncate
    (i.e. could the resonance rerank actually SELECT), or only permute?"""
    caps, maxc, truncated = set(), 0, False
    seen = 0
    for r in results:
        st = (r.detail or {}).get("retrieval_stats") or {}
        if not st:
            continue
        seen += 1
        if st.get("cap") is not None:
            caps.add(st.get("cap"))
        maxc = max(maxc, int(st.get("max_candidates") or 0))
        truncated = truncated or bool(st.get("truncated_any"))
    return {"items_with_stats": seen, "caps": sorted(caps), "max_candidates": maxc,
            "truncated_any": truncated,
            "note": ("retrieval TRUNCATED → the rerank could select" if truncated else
                     "retrieval NEVER truncated (cap>=candidates) → the rerank could only permute, "
                     "so resonance on/off is mechanically forced to the same sources")}


def main(argv: Optional[list] = None):    # pragma: no cover - live orchestration over real models
    ap = argparse.ArgumentParser(description="Physics-value ablation (A/B/C/D[/C-sel])")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--skip-csel", action="store_true", help="drop the 5th C-sel arm (cap=6)")
    ap.add_argument("--out", default="", help="write the full JSON report here")
    args = ap.parse_args(argv)

    # Attributability + determinism MUST be set before any eris import (eris.config binds tiers at
    # import time). setdefault so an explicit override still wins.
    os.environ.setdefault("ERIS_BENCH_ATTRIBUTABLE", "1")
    os.environ.setdefault("ERIS_HIVE_TEMPERATURE", "0")
    from eris.experiments.benchmarks.run import _load_dotenv, _apply_attributable_preset
    _load_dotenv()
    _apply_attributable_preset()        # forces ONE model into every slot, overriding stale shell vars

    from eris.experiments.benchmarks.datasets import load_quality
    from eris.experiments.benchmarks.arms import default_bare_arm
    from eris.experiments.benchmarks.eris_arm import make_eris_arm
    from eris.experiments.benchmarks.core import run_arm, accuracy, budget_report, item_details
    from eris.experiments.benchmarks.scoring import score_results

    items = load_quality(limit=args.limit, hard_only=True)
    print(f"[ablation] quality --hard-only: {len(items)} items (same items for every arm)",
          file=sys.stderr)

    arms = list(ARM_CONFIGS) + ([] if args.skip_csel else [CSEL])
    # Build each backend ONCE. make_eris_arm prints the attributability header and RAISES if the
    # slots are not one model (the fail-loud guard) — so a non-attributable run never produces data.
    bare_fn = default_bare_arm()
    eris_fn = make_eris_arm()

    results_by_arm: Dict[str, list] = {}
    acc: Dict[str, float] = {}
    answers: Dict[str, Dict[str, str]] = {}
    retrieval: Dict[str, dict] = {}
    budgets: Dict[str, dict] = {}

    for name, kind, overrides in arms:
        for k, v in overrides.items():
            os.environ[k] = v        # read at hive_research/_rag call time → same orchestrator, new config
        print(f"[ablation] arm {name}: {kind} {overrides or '(baseline)'}", file=sys.stderr)
        fn = bare_fn if kind == "bare" else eris_fn
        res = score_results(run_arm(items, fn, kind), items)
        results_by_arm[name] = res
        acc[name] = accuracy(res)["accuracy"]
        answers[name] = _answers(res)
        budgets[name] = budget_report(res)
        if kind == "eris":
            retrieval[name] = _retrieval_summary(res)

    a0 = budgets.get(BARE, {}).get("tokens_per_question") or 0.0
    report = {
        "ablation": "physics-value (docs/architecture_audit_2026-06.md)",
        "pre_registration": PRE_REGISTRATION,
        "conditions": {
            "model": os.environ.get("ERIS_BENCH_MODEL"),
            "attributable": True, "temperature": os.environ.get("ERIS_HIVE_TEMPERATURE"),
            "benchmark": "quality", "hard_only": True, "limit": args.limit, "n_items": len(items),
            "determinism": "greedy decode (temp 0); provider-side batching may still cause rare "
                           "variation — at small N, treat a 1-item flip with caution"},
        "arms": {name: {
            "accuracy": acc.get(name),
            "graded": accuracy(results_by_arm[name])["graded"],
            "errored": accuracy(results_by_arm[name])["errored"],
            "tokens_per_question": budgets[name]["tokens_per_question"],
            "token_ratio_vs_bare": (round(budgets[name]["tokens_per_question"] / a0, 2)
                                    if a0 else None),
            "cost_basis": budgets[name]["cost_basis"],
            **({"retrieval": retrieval[name]} if name in retrieval else {}),
        } for name, _, _ in arms},
        "predictions": evaluate_predictions(acc, answers, has_csel=not args.skip_csel),
        "items": item_details(items, results_by_arm),
    }
    print(json.dumps(report, indent=2, default=str))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":   # pragma: no cover
    main()
