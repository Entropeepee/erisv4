"""Two-arm benchmark core — the spine of the Eris vs bare-model comparison.

Mirrors Inspect AI's Dataset → Solver → Scorer split (see inspect_glue.py for the optional
Inspect wrapper), but is self-contained and dependency-free so the load-bearing logic — prompt
building, the two arms, scoring, and EQUAL-TOKEN-BUDGET accounting — is unit-testable offline
without inspect_ai, the `datasets` package, or a running model.

The single most-cited credibility failure in "scaffold beats bare model" claims (per the 2024-26
literature in the benchmark brief) is unequal compute. So every arm result carries its token cost
and the runner reports tokens/question for both arms — a win only counts at a matched budget."""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class BenchItem:
    """One benchmark question, normalized across datasets."""
    id: str
    question: str
    context: str = ""                       # provided document(s) for grounded tasks; "" closed-book
    answer: str = ""                        # gold answer (exact-match / span)
    choices: Optional[List[str]] = None     # multiple-choice options (gold = `answer`, a letter/text)
    unanswerable: bool = False              # SQuAD2/abstention: the correct response is to refuse
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArmResult:
    item_id: str
    arm: str                                # "bare" | "eris"
    text: str                               # the model's answer
    tokens: int = 0                         # tokens consumed (equal-budget accounting)
    correct: Optional[bool] = None          # accuracy scorer (None for faithfulness items)
    faithfulness: Optional[float] = None    # hallucination RATE for faithfulness items (lower=better)
    detail: Dict[str, Any] = field(default_factory=dict)   # per-item scorer breakdown


def build_prompt(item: BenchItem) -> str:
    """Compose the user prompt. Grounded tasks lead with the provided document; multiple-choice
    appends lettered options and asks for a single letter.

    Both arms receive the SAME prompt text, but interpret it differently: the bare arm reads the
    source inline in its context window; the Eris arm extracts the source and ingests it for
    RETRIEVAL. Both see the same information, so the comparison is fair — but it is a full-hive vs
    bare-model comparison, not an identical-pipeline one (the retrieval path is architectural)."""
    parts = []
    if item.context:
        parts.append("Read the following source material and answer ONLY from it.\n\n"
                     f"=== SOURCE ===\n{item.context}\n=== END SOURCE ===\n")
    parts.append(f"Question: {item.question}")
    if item.choices:
        opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(item.choices))
        parts.append("Options:\n" + opts +
                     "\n\nAnswer with the single letter of the correct option.")
    elif item.unanswerable is not None and item.meta.get("allow_abstain"):
        parts.append("If the source does not contain the answer, reply exactly: UNANSWERABLE.")
    return "\n\n".join(parts)


def run_arm(items: List[BenchItem], answer_fn: Callable[[str], Any], arm: str) -> List[ArmResult]:
    """Run one arm over the items. `answer_fn(prompt)` returns either a string, or a
    (text, tokens) pair — both arms use the SAME signature so they're interchangeable and the
    runner stays agnostic to whether it's the bare model or the full Eris pipeline."""
    out: List[ArmResult] = []
    for it in items:
        prompt = build_prompt(it)
        try:
            resp = answer_fn(prompt)
        except Exception as e:                       # a failed item scores as wrong, never crashes
            out.append(ArmResult(it.id, arm, f"[error: {e}]", 0, correct=None))
            continue
        detail = {}
        if isinstance(resp, tuple):
            text, tokens = resp[0], int(resp[1] or 0)
        elif isinstance(resp, dict):
            text, tokens = resp.get("text", ""), int(resp.get("tokens", 0) or 0)
            detail = resp.get("detail") or {}        # arm-specific diagnostics (e.g. full synthesis)
        else:
            text, tokens = str(resp), 0
        out.append(ArmResult(it.id, arm, (text or "").strip(), tokens, detail=detail))
    return out


def budget_report(results: List[ArmResult]) -> Dict[str, Any]:
    """Tokens/question for an arm — so a 'win' can be checked at a matched budget. A scaffold that
    spends 5x the tokens hasn't beaten the bare model until the budgets are equalized."""
    toks = [r.tokens for r in results if r.tokens > 0]
    n = len(results)
    total = sum(r.tokens for r in results)
    return {"n": n, "total_tokens": total,
            "tokens_per_question": round(total / n, 1) if n else 0.0,
            "measured": len(toks)}


def accuracy(results: List[ArmResult]) -> Dict[str, Any]:
    graded = [r for r in results if r.correct is not None]
    errored = [r for r in results if (r.text or "").startswith("[error:")]
    n_correct = sum(1 for r in graded if r.correct)
    return {"graded": len(graded), "correct": n_correct,
            "accuracy": round(n_correct / len(graded), 4) if graded else 0.0,
            # errored items are NOT in the accuracy denominator — surface the count so a reliability
            # problem (an arm that times out) can't hide as if it were an accuracy result.
            "errored": len(errored),
            "error_rate": round(len(errored) / len(results), 4) if results else 0.0}


def item_details(items: List[BenchItem],
                 results_by_arm: Dict[str, List[ArmResult]]) -> List[Dict[str, Any]]:
    """Per-item breakdown for diagnosis: the question, the gold answer, and each arm's actual
    prediction + correctness + tokens. Without this a 0% score is a black box — you can't tell a
    wrong-format answer from a genuinely wrong one or a too-strict scorer."""
    by = {label: {r.item_id: r for r in res} for label, res in results_by_arm.items()}
    rows = []
    for it in items:
        # NO truncation — in dev/testing we read every reply in full (a 0% with only the first
        # 300 chars of a 5000-char answer is still a black box). Full question, full answer.
        row = {"id": it.id, "question": it.question,
               "gold": it.answer or ("UNANSWERABLE" if it.unanswerable else "")}
        if it.choices:                             # MC: show what A/B/C/D actually SAY, so a wrong
            row["choices"] = {chr(65 + k): c for k, c in enumerate(it.choices)}  # letter is judgeable
            g = (it.answer or "").strip().upper()
            if len(g) == 1 and g.isalpha():
                row["gold_text"] = row["choices"].get(g, "")
        if (it.meta or {}).get("fetch"):           # FRAMES: how many linked articles actually loaded
            row["source_fetch"] = it.meta["fetch"]
        for label, m in by.items():
            r = m.get(it.id)
            if r is None:
                continue
            cell = {"answer": (r.text or ""), "tokens": r.tokens}
            if it.choices:                         # MC: also show the TEXT of the chosen letter
                a = (r.text or "").strip().upper()
                if len(a) == 1 and a.isalpha():
                    cell["answer_text"] = row["choices"].get(a, "")
            if r.correct is not None:
                cell["correct"] = r.correct
            if r.faithfulness is not None:
                cell["hallucination_rate"] = r.faithfulness
            if r.detail:                           # arm diagnostics: full synthesis, extraction_ok…
                cell.update(r.detail)
            row[label] = cell
        rows.append(row)
    return rows


def faithfulness(results: List[ArmResult]) -> Dict[str, Any]:
    """Aggregate faithfulness for an arm: mean per-item hallucination rate (lower = more faithful).
    This is the comparable per-arm number for RAGTruth — NOT pass/fail accuracy."""
    rated = [r for r in results if r.faithfulness is not None]
    if not rated:
        return {}
    mean = sum(r.faithfulness for r in rated) / len(rated)
    return {"items": len(rated), "mean_hallucination_rate": round(mean, 4)}


def compare(arm_a: List[ArmResult], arm_b: List[ArmResult],
            label_a: str = "bare", label_b: str = "eris") -> Dict[str, Any]:
    """Head-to-head summary: accuracy AND token cost for both arms, with an explicit equal-budget
    flag. Never declares a winner on accuracy alone — the token ratio is reported alongside so an
    unequal-compute 'win' is visible at a glance."""
    acc_a, acc_b = accuracy(arm_a), accuracy(arm_b)
    bud_a, bud_b = budget_report(arm_a), budget_report(arm_b)
    fa_a, fa_b = faithfulness(arm_a), faithfulness(arm_b)
    tpq_a, tpq_b = bud_a["tokens_per_question"], bud_b["tokens_per_question"]
    ratio = round(tpq_b / tpq_a, 2) if tpq_a else None
    out = {
        label_a: {"accuracy": acc_a["accuracy"], "tokens_per_question": tpq_a, **acc_a},
        label_b: {"accuracy": acc_b["accuracy"], "tokens_per_question": tpq_b, **acc_b},
        "delta_accuracy": round(acc_b["accuracy"] - acc_a["accuracy"], 4),
        "token_ratio_b_over_a": ratio,
        "equal_budget": (ratio is not None and 0.8 <= ratio <= 1.25),
        "note": ("equal-budget comparison" if (ratio is not None and 0.8 <= ratio <= 1.25)
                 else "UNEQUAL budget — a higher-accuracy arm that also spends more tokens has "
                      "NOT cleanly won; equalize tokens/question before claiming a scaffold win"),
    }
    if fa_a or fa_b:                       # faithfulness benchmarks (RAGTruth): lower rate = better
        out[label_a]["faithfulness"] = fa_a
        out[label_b]["faithfulness"] = fa_b
        ra = fa_a.get("mean_hallucination_rate") if fa_a else None
        rb = fa_b.get("mean_hallucination_rate") if fa_b else None
        if ra is not None and rb is not None:
            out["delta_hallucination_rate"] = round(rb - ra, 4)   # negative = Eris hallucinates LESS
    return out
