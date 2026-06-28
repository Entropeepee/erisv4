"""Dataset adapters → BenchItem, one per benchmark in the brief.

The live `load_*` functions need the HuggingFace `datasets` package and network; the row→BenchItem
mappers (`_*_item`) are PURE and unit-tested with representative sample rows, so the normalization
logic is verified offline. Schemas follow the dataset cards as of the 2026 brief; mappers use
tolerant .get() with fallbacks so a minor card change degrades to a usable item rather than crash.

Tiers (from the brief):
  • document-grounded (Eris's proving ground): FRAMES, QuALITY-HARD, RAGTruth, MuSiQue
  • closed-book controls (bare-model ceiling):  MMLU-Pro, GPQA-Diamond
"""
from typing import Any, Dict, List, Optional

from eris.experiments.benchmarks.core import BenchItem


def _load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
    except Exception as e:                      # pragma: no cover - environment-dependent
        raise RuntimeError(
            "The `datasets` package is required to load benchmarks: pip install datasets") from e
    return load_dataset(*args, **kwargs)


def _assemble_choices(correct: str, incorrects: List[str], seed: int):
    """Deterministically place the correct answer among the distractors (so it is not always 'A'),
    return (choices, gold_letter). Rotation by `seed` keeps it reproducible without RNG state."""
    opts = [correct] + [c for c in incorrects if c]
    if not opts:
        return [], ""
    pos = seed % len(opts)
    rotated = opts[-pos:] + opts[:-pos] if pos else opts
    gold_idx = rotated.index(correct)
    return rotated, chr(65 + gold_idx)


# ── document-grounded ────────────────────────────────────────────────────────

def _frames_item(row: Dict[str, Any], i: int) -> BenchItem:
    # google/frames-benchmark: a multi-hop question + gold answer; articles referenced by link.
    # Provide article text in `context` if the row carries it (some mirrors inline the wiki text).
    q = row.get("Prompt") or row.get("question") or row.get("prompt") or ""
    ans = row.get("Answer") or row.get("answer") or ""
    ctx = row.get("wiki_text") or row.get("context") or ""
    links = row.get("wikipedia_links") or row.get("wiki_links") or []
    return BenchItem(id=f"frames-{i}", question=q, context=ctx, answer=str(ans),
                     meta={"type": "multi_hop", "links": links,
                           "reasoning": row.get("reasoning_types", "")})


def _quality_questions(article_row: Dict[str, Any], i: int, hard_only: bool) -> List[BenchItem]:
    # nyu-mll/quality: one ~5k-token article with several MC questions; HARD = `difficult`==1.
    article = article_row.get("article") or article_row.get("context") or ""
    out: List[BenchItem] = []
    for j, q in enumerate(article_row.get("questions", []) or []):
        difficult = bool(q.get("difficult", 0))
        if hard_only and not difficult:
            continue
        options = q.get("options") or []
        gold = q.get("gold_label")              # 1-based index per the QuALITY card
        ans_letter = chr(64 + int(gold)) if gold else ""
        out.append(BenchItem(
            id=f"quality-{i}-{j}", question=q.get("question", ""), context=article,
            choices=list(options), answer=ans_letter,
            meta={"type": "long_doc_mc", "difficult": difficult}))
    return out


def _ragtruth_item(row: Dict[str, Any], i: int) -> BenchItem:
    # RAGTruth: a RAG response with span-level hallucination annotations. This is a FAITHFULNESS
    # task, not accuracy — score with span overlap / RAGChecker, not exact-match. We carry the
    # reference + the annotated spans in meta for the faithfulness scorer.
    return BenchItem(
        id=f"ragtruth-{i}",
        question=row.get("prompt") or row.get("question") or "Summarize the source faithfully.",
        context=row.get("reference") or row.get("source_info") or row.get("context") or "",
        answer=row.get("response") or "",
        meta={"type": "faithfulness", "hallucination_spans": row.get("labels")
              or row.get("hallucination_list") or []})


def _musique_item(row: Dict[str, Any], i: int) -> BenchItem:
    # MuSiQue: 2-4 hop; paragraphs provided. Concatenate supporting paragraphs into context.
    paras = row.get("paragraphs") or []
    ctx = "\n\n".join(p.get("paragraph_text", "") if isinstance(p, dict) else str(p)
                      for p in paras)
    return BenchItem(id=f"musique-{i}", question=row.get("question", ""), context=ctx,
                     answer=str(row.get("answer", "")),
                     meta={"type": "multi_hop", "answerable": row.get("answerable", True)})


# ── closed-book controls ─────────────────────────────────────────────────────

def _mmlu_pro_item(row: Dict[str, Any], i: int) -> BenchItem:
    # TIGER-Lab/MMLU-Pro: 10 options; `answer` is the gold letter, `answer_index` 0-based.
    opts = list(row.get("options") or [])
    ans = row.get("answer") or (chr(65 + int(row["answer_index"]))
                                if row.get("answer_index") is not None else "")
    return BenchItem(id=f"mmlupro-{i}", question=row.get("question", ""), context="",
                     choices=opts, answer=str(ans),
                     meta={"type": "closed_book_mc", "category": row.get("category", "")})


def _gpqa_item(row: Dict[str, Any], i: int) -> BenchItem:
    # Idavidrein/gpqa: correct + 3 incorrect answers; assemble + place deterministically.
    correct = row.get("Correct Answer") or row.get("correct_answer") or ""
    incorrects = [row.get("Incorrect Answer 1") or row.get("incorrect_answer_1") or "",
                  row.get("Incorrect Answer 2") or row.get("incorrect_answer_2") or "",
                  row.get("Incorrect Answer 3") or row.get("incorrect_answer_3") or ""]
    choices, gold = _assemble_choices(correct, incorrects, seed=i)
    return BenchItem(id=f"gpqa-{i}", question=row.get("Question") or row.get("question", ""),
                     context="", choices=choices, answer=gold,
                     meta={"type": "closed_book_mc", "domain": row.get("High-level domain", "")})


# ── live loaders (need `datasets` + network) ─────────────────────────────────

def load_frames(limit: Optional[int] = None) -> List[BenchItem]:
    ds = _load_dataset("google/frames-benchmark", split="test")
    rows = ds.select(range(min(limit, len(ds)))) if limit else ds
    return [_frames_item(r, i) for i, r in enumerate(rows)]


def load_quality(limit: Optional[int] = None, hard_only: bool = True) -> List[BenchItem]:
    ds = _load_dataset("emozilla/quality", split="validation")
    items: List[BenchItem] = []
    for i, r in enumerate(ds):
        items.extend(_quality_questions(r, i, hard_only))
        if limit and len(items) >= limit:
            break
    return items[:limit] if limit else items


def load_ragtruth(limit: Optional[int] = None) -> List[BenchItem]:
    ds = _load_dataset("wandb/RAGTruth-processed", split="test")
    rows = ds.select(range(min(limit, len(ds)))) if limit else ds
    return [_ragtruth_item(r, i) for i, r in enumerate(rows)]


def load_musique(limit: Optional[int] = None) -> List[BenchItem]:
    ds = _load_dataset("dgslibisey/MuSiQue", split="validation")
    rows = ds.select(range(min(limit, len(ds)))) if limit else ds
    return [_musique_item(r, i) for i, r in enumerate(rows)]


def load_mmlu_pro(limit: Optional[int] = None) -> List[BenchItem]:
    ds = _load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rows = ds.select(range(min(limit, len(ds)))) if limit else ds
    return [_mmlu_pro_item(r, i) for i, r in enumerate(rows)]


def load_gpqa(limit: Optional[int] = None) -> List[BenchItem]:
    ds = _load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    rows = ds.select(range(min(limit, len(ds)))) if limit else ds
    return [_gpqa_item(r, i) for i, r in enumerate(rows)]


LOADERS = {
    "frames": load_frames, "quality": load_quality, "ragtruth": load_ragtruth,
    "musique": load_musique, "mmlu_pro": load_mmlu_pro, "gpqa": load_gpqa,
}
GROUNDED = {"frames", "quality", "ragtruth", "musique"}     # Eris's proving ground
CLOSED_BOOK = {"mmlu_pro", "gpqa"}                          # bare-model ceiling controls
