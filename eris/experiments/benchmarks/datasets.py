"""Dataset adapters → BenchItem, one per benchmark in the brief.

The live `load_*` functions need the HuggingFace `datasets` package and network; the row→BenchItem
mappers (`_*_item`) are PURE and unit-tested with representative sample rows, so the normalization
logic is verified offline. Schemas follow the dataset cards as of the 2026 brief; mappers use
tolerant .get() with fallbacks so a minor card change degrades to a usable item rather than crash.

Tiers (from the brief):
  • document-grounded (Eris's proving ground): FRAMES, QuALITY-HARD, RAGTruth, MuSiQue
  • closed-book controls (bare-model ceiling):  MMLU-Pro, GPQA-Diamond
"""
import re
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from eris.experiments.benchmarks.core import BenchItem

_WIKI_URL = re.compile(r"https?://[^\s\"'\]\),]*wikipedia\.org/wiki/[^\s\"'\]\),]+", re.I)


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


def _gold_to_letter(gold, options) -> str:
    """Map a QuALITY-style gold answer to an option LETTER, robust to how the HF mirror encodes it:
    a letter ('B'), the option text, a 1-based index (the QuALITY card's gold_label is 1-based), or
    a 0-based index. The 1-based vs 0-based call is the one real ambiguity — QuALITY's native
    `gold_label` is 1-based, so an int is treated 1-based; verify against the printed items block."""
    if gold is None or gold == "":
        return ""
    if isinstance(gold, str):
        g = gold.strip()
        if len(g) == 1 and g.upper().isalpha():
            return g.upper()
        for k, o in enumerate(options or []):           # gold given as the option text
            if str(o).strip() == g:
                return chr(65 + k)
        if g.isdigit():
            gold = int(g)
        else:
            return ""
    if isinstance(gold, int):
        n = len(options or [])
        # 1-based if it fits as 1..n and (n is unknown or gold==n boundary); else 0-based.
        if 1 <= gold <= max(n, 1) and not (n and gold == 0):
            return chr(64 + gold)                       # 1-based (QuALITY native)
        if 0 <= gold < n:
            return chr(65 + gold)                       # 0-based fallback
    return ""


def _quality_item_flat(row: Dict[str, Any], i: int, hard_only: bool):
    """FLAT schema: one MC QUESTION per row (the common HF mirror, e.g. emozilla/quality)."""
    if not row.get("question"):
        return []
    difficult = bool(row.get("hard", row.get("difficult", row.get("is_hard", 0))))
    if hard_only and not difficult:
        return []
    options = row.get("options") or row.get("choices") or []
    article = row.get("article") or row.get("context") or row.get("passage") or ""
    gold = row.get("gold_label", row.get("answer", row.get("label")))
    return [BenchItem(id=f"quality-{i}", question=row["question"], context=article,
                      choices=list(options), answer=_gold_to_letter(gold, options),
                      meta={"type": "long_doc_mc", "difficult": difficult})]


def _quality_questions(article_row: Dict[str, Any], i: int, hard_only: bool) -> List[BenchItem]:
    """NESTED schema (nyu-mll/quality): one ~5k-token article with several MC questions; falls
    through to the FLAT per-question schema when there is no `questions` list."""
    nested = article_row.get("questions")
    if not isinstance(nested, list):
        return _quality_item_flat(article_row, i, hard_only)
    article = article_row.get("article") or article_row.get("context") or ""
    out: List[BenchItem] = []
    for j, q in enumerate(nested or []):
        difficult = bool(q.get("difficult", q.get("hard", 0)))
        if hard_only and not difficult:
            continue
        options = q.get("options") or []
        gold = q.get("gold_label", q.get("answer"))
        out.append(BenchItem(
            id=f"quality-{i}-{j}", question=q.get("question", ""), context=article,
            choices=list(options), answer=_gold_to_letter(gold, options),
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

def _wiki_title_from_url(url: str) -> str:
    """Pull the article title out of a Wikipedia URL: '.../wiki/Ada_Lovelace#Early' -> 'Ada Lovelace'."""
    tail = (url or "").rstrip("/").split("/wiki/", 1)[-1].split("#")[0].split("?")[0]
    return unquote(tail).replace("_", " ").strip()


def _frames_links(row: Dict[str, Any]) -> List[str]:
    """Collect the Wikipedia article URLs from a FRAMES row, wherever they live (a 'wiki_links'
    list, separate link columns, or inline in a text field) — scan all values, dedupe in order."""
    urls: List[str] = []
    for v in (row or {}).values():
        if isinstance(v, str):
            urls += _WIKI_URL.findall(v)
        elif isinstance(v, (list, tuple)):
            for x in v:
                if isinstance(x, str):
                    urls += _WIKI_URL.findall(x)
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out


def _fetch_frames_context(row: Dict[str, Any], max_chars: int = 24000,
                          max_articles: int = 8):
    """FRAMES ships Wikipedia LINKS, not article text — fetch the linked articles and assemble the
    provided 'source' so the grounded benchmark actually has documents. Capped (articles + chars)
    so a 15-article question stays tractable; best-effort per article.

    Returns (context, stats). `stats` records linked / fetched / failed / capped so a PARTIAL fetch
    (e.g. 2 of 6 articles failed) is visible in the report — otherwise an incomplete source makes a
    question unanswerable and Eris looks wrong when really the fetch broke."""
    from eris.knowledge.web_reader import fetch_wikipedia
    links = _frames_links(row)
    parts, total, fetched, failed = [], 0, 0, []
    capped = False
    for url in links[:max_articles]:
        title = _wiki_title_from_url(url)
        if not title:
            continue
        try:
            text = fetch_wikipedia(title)
        except Exception:
            failed.append(title); continue
        if not text:
            failed.append(title); continue
        block = f"== {title} ==\n{text}"
        parts.append(block)
        total += len(block)
        fetched += 1
        if total >= max_chars:
            capped = True
            break
    ctx = ("\n\n".join(parts))[:max_chars]
    stats = {"linked": len(links), "fetched": fetched, "failed": failed,
             "capped_articles": (len(links) > max_articles), "capped_chars": capped,
             "context_chars": len(ctx)}
    return ctx, stats


def load_frames(limit: Optional[int] = None, fetch_context: bool = True,
                max_chars: int = 24000) -> List[BenchItem]:
    """FRAMES — multi-hop over Wikipedia. With fetch_context=True (default) the linked articles are
    fetched into BenchItem.context so the question is answerable from the provided source (the
    paper's 'oracle' setting). fetch_context=False leaves context empty (a no-retrieval baseline)."""
    ds = _load_dataset("google/frames-benchmark", split="test")
    rows = ds.select(range(min(limit, len(ds)))) if limit else ds
    items = []
    for i, r in enumerate(rows):
        it = _frames_item(r, i)
        if fetch_context and not it.context:
            it.context, stats = _fetch_frames_context(r, max_chars=max_chars)
            it.meta["fetch"] = stats             # surfaced per-item in the report (partial fetch visible)
        items.append(it)
    return items


def load_quality(limit: Optional[int] = None, hard_only: bool = True,
                 dataset: str = "emozilla/quality", split: str = "validation") -> List[BenchItem]:
    """QuALITY — long-document MC comprehension. Handles BOTH the nested (one article → questions)
    and flat (one question per row) schemas. Fails LOUDLY if it maps zero items, so a schema
    mismatch (wrong dataset id / field names / a missing `hard` flag) can't masquerade as 'no hard
    questions found'. Override `dataset`/`split` if your mirror differs."""
    ds = _load_dataset(dataset, split=split)
    items: List[BenchItem] = []
    for i, r in enumerate(ds):
        items.extend(_quality_questions(r, i, hard_only))
        if limit and len(items) >= limit:
            break
    if not items:
        keys = list((ds[0] if len(ds) else {}).keys())
        raise RuntimeError(
            f"QuALITY loader mapped 0 items from {dataset}:{split} (hard_only={hard_only}). The row "
            f"schema may differ from what the mapper expects — row keys seen: {keys}. Fix "
            f"_quality_item_flat/_quality_questions to match, or pass hard_only=False to confirm.")
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
