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


def score_item(pred: str, item: BenchItem) -> bool:
    """Dispatch by item type: abstention → refusal is correct; MC → letter/text; else exact/span."""
    if item.unanswerable:
        return abstained(pred)
    if item.choices:
        return multiple_choice(pred, item)
    return exact_match(pred, item.answer) or contains_gold(pred, item.answer)


def score_results(results: List[ArmResult], items: List[BenchItem]) -> List[ArmResult]:
    """Grade an arm's results in place against the items (matched by id)."""
    by_id = {it.id: it for it in items}
    for r in results:
        it = by_id.get(r.item_id)
        if it is not None and not r.text.startswith("[error:"):
            r.correct = score_item(r.text, it)
    return results
