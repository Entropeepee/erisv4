"""Scorers — exact-match, multiple-choice, and abstention. Pure string logic, unit-testable.

Deliberately conservative: normalize lightly (lowercase, strip articles/punctuation) the way
SQuAD/QuALITY scoring does, and for multiple-choice accept either the letter or the option text.
For abstention tasks (SQuAD2 unanswerable, CRAG 'missing'), a correct refusal SCORES, so the
'don't over-claim' axis is rewarded rather than punished."""
import re
import string
from typing import List

from eris.experiments.benchmarks.core import BenchItem, ArmResult

_ARTICLES = {"a", "an", "the"}
_ABSTAIN = re.compile(r"\b(unanswerable|cannot answer|not (?:in|stated|provided|enough)|"
                      r"no answer|insufficient (?:information|context)|the source does not)\b", re.I)


def normalize(text: str) -> str:
    """SQuAD-style normalization: lowercase, drop punctuation + articles, collapse whitespace."""
    t = (text or "").lower()
    t = "".join(ch if ch not in string.punctuation else " " for ch in t)
    t = " ".join(w for w in t.split() if w not in _ARTICLES)
    return t.strip()


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def contains_gold(pred: str, gold: str) -> bool:
    """Looser span match: the normalized gold appears in the normalized prediction (for free-form
    answers where the model wraps the span in a sentence)."""
    g = normalize(gold)
    return bool(g) and g in normalize(pred)


def _letter_for_choice(item: BenchItem) -> str:
    """The gold letter for a multiple-choice item, whether `answer` is a letter or the option text."""
    g = (item.answer or "").strip()
    if len(g) == 1 and g.upper().isalpha():
        return g.upper()
    for i, c in enumerate(item.choices or []):
        if normalize(c) == normalize(g):
            return chr(65 + i)
    return ""


def multiple_choice(pred: str, item: BenchItem) -> bool:
    """Accept the gold letter (e.g. 'B', 'B.', '(B)', 'Answer: B') or the gold option text."""
    gold_letter = _letter_for_choice(item)
    if not gold_letter:
        return False
    m = re.search(r"\b([A-Z])\b", (pred or "").upper())
    if m and m.group(1) == gold_letter:
        return True
    idx = ord(gold_letter) - 65                       # also accept the full option text
    if item.choices and 0 <= idx < len(item.choices):
        return contains_gold(pred, item.choices[idx])
    return False


def abstained(pred: str) -> bool:
    return bool(_ABSTAIN.search(pred or ""))


# ── faithfulness (RAGTruth) — a RATE, not pass/fail accuracy ──────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]


def _content_tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]{4,}", (text or "").lower())}


def _norm_span(s) -> str:
    if isinstance(s, dict):
        return normalize(s.get("text") or s.get("span") or s.get("hallucinated_span") or "")
    return normalize(str(s))


def sentence_supported(sentence: str, context_tokens: set, threshold: float = 0.6) -> bool:
    """A sentence is supported if most of its content words appear in the source context. A
    content-free sentence (e.g. 'In summary,') is never a hallucination."""
    toks = _content_tokens(sentence)
    if not toks:
        return True
    return (len(toks & context_tokens) / len(toks)) >= threshold


def faithfulness_score(output: str, context: str, hallucination_spans=None,
                       threshold: float = 0.6) -> dict:
    """Per-item faithfulness: the fraction of the OUTPUT's sentences whose content is NOT supported
    by the provided context (the hallucination rate; lower = more faithful). If annotated
    hallucination_spans are present (the RAGTruth reference), a sentence overlapping any span counts
    as hallucinated; otherwise fall back to context-entailment by content-word overlap. No new deps;
    see ragchecker_faithfulness() for the optional claim-level upgrade."""
    sents = _sentences(output)
    if not sents:
        return {"hallucination_rate": 0.0, "n_sentences": 0, "supported": 0,
                "unsupported": 0, "hallucinated": []}
    spans = [s for s in (_norm_span(x) for x in (hallucination_spans or [])) if s]
    ctx_tokens = _content_tokens(context)
    hallucinated = []
    for s in sents:
        bad = (any(span in normalize(s) for span in spans) if spans
               else not sentence_supported(s, ctx_tokens, threshold))
        if bad:
            hallucinated.append(s)
    rate = len(hallucinated) / len(sents)
    return {"hallucination_rate": round(rate, 4), "n_sentences": len(sents),
            "supported": len(sents) - len(hallucinated), "unsupported": len(hallucinated),
            "hallucinated": hallucinated[:5]}


def ragchecker_faithfulness(output: str, context: str):     # pragma: no cover - optional dep
    """Optional v2: claim-level precision via RAGChecker/RAGAS if installed, else None. Guarded
    like inspect_glue — importing this module never requires the package."""
    try:
        from ragas import evaluate  # noqa: F401
    except Exception:
        return None
    raise NotImplementedError(
        "RAGAS is installed — wire your judge/LLM config here for claim-level faithfulness. "
        "The built-in faithfulness_score is the no-dep default.")


def score_item(pred: str, item: BenchItem) -> bool:
    """Dispatch by item type. Faithfulness items are a RATE (see score_results), but a bool is
    still defined for direct callers: faithful iff zero unsupported sentences — so a faithfulness
    item is NEVER mis-routed to exact-match against a gold string."""
    if item.meta.get("type") == "faithfulness":
        return faithfulness_score(pred, item.context,
                                  item.meta.get("hallucination_spans"))["hallucination_rate"] == 0.0
    if item.unanswerable:
        return abstained(pred)
    if item.choices:
        return multiple_choice(pred, item)
    return exact_match(pred, item.answer) or contains_gold(pred, item.answer)


def score_results(results: List[ArmResult], items: List[BenchItem]) -> List[ArmResult]:
    """Grade an arm's results in place against the items (matched by id). Faithfulness items get a
    numeric `faithfulness` rate + `detail` (and are left out of accuracy); all others get `correct`."""
    by_id = {it.id: it for it in items}
    for r in results:
        it = by_id.get(r.item_id)
        if it is None or r.text.startswith("[error:"):
            continue
        if it.meta.get("type") == "faithfulness":
            fs = faithfulness_score(r.text, it.context, it.meta.get("hallucination_spans"))
            r.faithfulness = fs["hallucination_rate"]
            r.detail = fs
        else:
            r.correct = score_item(r.text, it)
    return results
