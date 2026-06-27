"""Retrieval eval-set generator (unblocks the #40 resonance-vs-RAG verdict).

Builds the labelled `(question → gold passage)` set the DualPath shadow needs so
`eris/dual/arbiter.gold_passage_at_k` has real data. A question is generated *from*
a library passage, so that passage is its gold — never labelled by what a retriever
returned (RULE A: RAG is a floor, not ground truth). Questions must be conceptual
paraphrases, not lexical echoes (RULE B), or BM25/cosine win by construction and the
verdict is meaningless. For each pair we mine the hardest lexically-similar non-gold
passages (RULE C) — the distractor subset is the row that actually decides the
field's value, reported separately from the full set.

Torch-light and resumable (RULE D): generation goes through the same local-LLM path
the study pipeline uses, injected as a plain `Callable[[str], str]` so tests pass a
deterministic stub; distractor mining reuses the existing BM25 from
`eris/retrieval/hybrid.py`; the run caches by passage id and `--resume` skips done.

Pipeline (one read):
    library passages ─▶ per passage P:
        llm: "N conceptual questions answerable ONLY from P"   (RULE A: P is gold)
          ├─ FILTER answerable  (P's terms suffice)
          └─ FILTER paraphrastic (RULE B: reject high trigram overlap vs P)
      ─▶ (question, gold=record_id(P)) ─▶ DISTRACTOR MINING (RULE C):
            BM25 top-m non-gold → has_distractor + distractor_ids
      ─▶ eval_set.jsonl (+ held-out split)  ─▶  gold_passage_at_k  ─▶  #40 report
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import hashlib
import json
import os
import re

from eris.dual.types import record_id

# The library-origin source prefixes (mirror eris/orchestrator.py _LIBRARY_PREFIXES):
# read the quality-gated library, never the conversation/thought-stream.
LIBRARY_PREFIXES = ("reading:", "study:", "exploration:", "ponder:", "deepread",
                    "research", "expert:")

LLM = Callable[[str], str]                    # prompt -> text (injected; stub in tests)

_WORD = re.compile(r"[a-z0-9]{3,}")
_STOP = {"the", "and", "for", "with", "that", "this", "what", "how", "does", "are",
         "was", "were", "from", "into", "about", "which", "who", "why", "you",
         "your", "its", "his", "her", "their", "they", "can", "could", "would",
         "should", "will", "have", "has", "had", "but", "not", "any", "all", "one"}


# ── config ──────────────────────────────────────────────────────────────────
@dataclass
class EvalSetConfig:
    n_questions: int = 3              # questions requested per passage
    paraphrastic_threshold: float = 0.5   # RULE B: reject trigram overlap above this
    ngram_n: int = 3
    min_answer_terms: int = 1        # answerable if ≥ this many shared content terms
    max_per_passage: int = 3         # cap kept questions so no source dominates
    heldout_frac: float = 0.2        # stratified held-out split
    distractor_ratio: float = 0.8    # RULE C: top non-gold BM25 ≥ ratio·gold ⇒ hard
    distractor_top_m: int = 5        # store up to this many distractor ids
    seed: int = 0
    llm_answerable: bool = False     # optional LLM answerability check (flagged)


# ── §1 generator: prompt + parsing ──────────────────────────────────────────
_QGEN_SYS = (
    "You write evaluation questions for a retrieval system. Given a passage, you "
    "produce conceptual questions that test whether a reader UNDERSTOOD the passage "
    "— not whether they can keyword-match it."
)


def make_qgen_prompt(passage: str, n: int) -> str:
    """Demand conceptual questions and forbid copying the passage's wording (RULE B
    is enforced by the filter regardless, but we ask for it too)."""
    return (
        f"Read this passage:\n\n\"\"\"\n{passage.strip()}\n\"\"\"\n\n"
        f"Write {n} questions that can be answered ONLY from this passage. Rules:\n"
        f"- Ask about the MEANING/concepts, not surface keywords.\n"
        f"- Do NOT copy phrases or sentences from the passage — paraphrase fully.\n"
        f"- Each question on its own line, no numbering, no preamble."
    )


def parse_questions(text: str) -> List[str]:
    """Pull questions out of an LLM reply: one per line, strip numbering/bullets,
    drop blanks/headers/too-short lines."""
    out: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        # strip leading "1.", "1)", "- ", "* ", "Q: " enumerations
        line = re.sub(r"^\s*(?:\d+[.)]\s*|[-*\u2022]\s*|[Qq][:.]\s*)", "", line)
        line = line.strip().strip('"')
        if len(line.split()) < 3:           # not a real question
            continue
        out.append(line)
    return out


# ── §2 filters (RULE A/B) ───────────────────────────────────────────────────
def _terms(text: str) -> set:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP}


def _trigrams(text: str, n: int = 3) -> set:
    toks = _WORD.findall((text or "").lower())
    if len(toks) < n:
        return set()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def ngram_overlap(question: str, passage: str, n: int = 3) -> float:
    """Fraction of the QUESTION's n-grams that also appear in the passage. A
    verbatim copy ⇒ ~1.0; a true paraphrase ⇒ low (RULE B's lever).

    A question shorter than n tokens has no n-grams; rather than pass it for free
    (which would let a 1–2 word verbatim echo slip RULE B), fall back to a contiguous
    token-subsequence check — a short echo IS a contiguous run of the passage ⇒ 1.0."""
    q = _trigrams(question, n)
    if not q:
        qt = _WORD.findall((question or "").lower())
        if not qt:
            return 0.0
        pt = _WORD.findall((passage or "").lower())
        for i in range(len(pt) - len(qt) + 1):
            if pt[i:i + len(qt)] == qt:
                return 1.0           # short verbatim echo
        return 0.0
    p = _trigrams(passage, n)
    return len(q & p) / len(q)


def is_paraphrastic(question: str, passage: str, cfg: EvalSetConfig) -> bool:
    """RULE B: True (keep) when the question is NOT a lexical echo of the passage."""
    return ngram_overlap(question, passage, cfg.ngram_n) <= cfg.paraphrastic_threshold


def is_answerable(question: str, passage: str, cfg: EvalSetConfig,
                  llm: Optional[LLM] = None) -> bool:
    """Cheap heuristic (RULE A sanity): the passage must share enough salient terms
    with the question to plausibly answer it (so the question is on-topic, not a
    hallucination). Optional LLM confirmation behind cfg.llm_answerable.

    min_answer_terms stays low (1) ON PURPOSE: this is in TENSION with RULE B — a good
    conceptual paraphrase deliberately shares FEW literal terms with its passage (e.g.
    "what molecule carries the energy" vs a passage naming only "ATP, the energy
    currency"), so demanding many shared terms here would reject exactly the questions
    RULE B wants. Use cfg.llm_answerable for a stronger, non-lexical check instead."""
    q = _terms(question)
    if not q:
        return False
    shared = len(q & _terms(passage))
    if shared < cfg.min_answer_terms:
        return False
    if cfg.llm_answerable and llm is not None:
        ans = llm(f"Can this question be answered using ONLY the passage? "
                  f"Answer yes or no.\n\nPassage:\n{passage}\n\nQuestion: {question}")
        if "yes" not in (ans or "").strip().lower()[:8]:
            return False
    return True


def _dedup(questions: List[str]) -> List[str]:
    """Drop near-duplicate questions (normalized token-set Jaccard ≥ 0.9)."""
    kept: List[str] = []
    seen_termsets: List[set] = []
    for qn in questions:
        ts = _terms(qn)
        norm = " ".join(sorted(ts))
        dup = False
        for prev in seen_termsets:
            inter = len(ts & prev)
            union = len(ts | prev) or 1
            if norm and inter / union >= 0.9:
                dup = True
                break
        if not dup:
            kept.append(qn)
            seen_termsets.append(ts)
    return kept


# ── §3 distractor mining (RULE C) ───────────────────────────────────────────
def _build_bm25(texts: List[str]):
    """Reuse the existing BM25 (eris/retrieval/hybrid.py) — do not write a second."""
    from eris.retrieval.hybrid import BM25
    return BM25(texts)


def mine_distractor(question: str, gold_id: str, ids: List[str], bm25,
                    cfg: EvalSetConfig) -> Tuple[bool, List[str]]:
    """Score the question against the whole library with BM25; if the top non-gold
    passage scores within `distractor_ratio` of the gold's score, this is a hard
    negative where cosine is most likely to be fooled (RULE C). Returns
    (has_distractor, distractor_ids).

    Gold is excluded by ID, not index: with duplicate passages (same content ⇒ same
    record_id ⇒ same id as gold) an index-only exclusion would let the gold's own twin
    masquerade as a distractor. Excluding every index whose id == gold_id prevents that."""
    import numpy as np
    scores = np.asarray(bm25.scores(question), dtype=float)
    gold_idxs = [i for i, rid in enumerate(ids) if rid == gold_id]
    if not gold_idxs:
        return False, []
    gold_score = float(np.max(scores[gold_idxs]))
    gold_set = set(gold_idxs)
    order = [i for i in np.argsort(-scores) if i not in gold_set]
    top = [i for i in order if scores[i] > 0.0][:cfg.distractor_top_m]
    distractor_ids = [ids[i] for i in top]
    has = bool(top) and gold_score > 0.0 and \
        float(scores[top[0]]) >= cfg.distractor_ratio * gold_score
    return has, distractor_ids


# ── split assignment (deterministic, by passage to avoid leakage) ───────────
def _hash01(s: str) -> float:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()
    return int(h, 16) / float(1 << 64)


def assign_split(gold: str, cfg: EvalSetConfig) -> str:
    """Deterministic held-out split keyed by GOLD (the passage id), so EVERY question
    from a passage shares one split — a passage never straddles train/heldout, which
    would leak the gold across the boundary. Hashing the gold spreads passages
    uniformly, so ~heldout_frac of passages (not of questions) land held-out."""
    return "heldout" if _hash01(f"{gold}|{cfg.seed}") < cfg.heldout_frac else "train"


def _qid(gold: str, question: str) -> str:
    return "q:" + hashlib.blake2b((gold + "\n" + question).encode("utf-8"),
                                  digest_size=8).hexdigest()


def _source_meta(passage) -> dict:
    meta = getattr(passage, "metadata", None) or {}
    return {"source": getattr(passage, "source", "") or "",
            "title": meta.get("title"), "sha256": meta.get("sha256")}


# ── core generation (in-memory, testable) ───────────────────────────────────
def generate_rows(passages: List[Any], llm: LLM, cfg: Optional[EvalSetConfig] = None,
                  *, corpus_bm25=None, corpus_ids=None) -> Tuple[List[dict], dict]:
    """Generate, filter, dedup, mine distractors, split — all in memory. Returns
    (rows, stats). `passages` are MemoryRecord-like (`.text`, `.source`, `.metadata`).

    Each passage's gold is its OWN record_id. Distractor mining (RULE C) ranks against
    the whole library, so the caller can pass `corpus_bm25`/`corpus_ids` over the full
    passage set (run() does this when processing one passage at a time) — distinct from
    the passages being generated from. When omitted, the corpus IS `passages`."""
    cfg = cfg or EvalSetConfig()
    passages = list(passages)
    own_ids = [record_id(p) for p in passages]               # this passage's gold
    corpus_ids = corpus_ids if corpus_ids is not None else own_ids
    bm25 = corpus_bm25 if corpus_bm25 is not None else \
        _build_bm25([getattr(p, "text", "") or "" for p in passages])

    rows: List[dict] = []
    stats = {"passages": len(passages), "raw_questions": 0, "rejected_unanswerable": 0,
             "rejected_paraphrastic": 0, "kept": 0, "has_distractor": 0,
             "llm_errors": 0, "capped": 0}

    for p, gold in zip(passages, own_ids):
        ptext = getattr(p, "text", "") or ""
        if not ptext.strip():
            continue
        try:
            reply = llm(make_qgen_prompt(ptext, cfg.n_questions))
        except Exception:
            reply = ""
            stats["llm_errors"] += 1          # surfaced, never silent (RULE D)
        cands = _dedup(parse_questions(reply))
        stats["raw_questions"] += len(cands)
        kept = 0
        for qn in cands:
            if kept >= cfg.max_per_passage:
                stats["capped"] += 1          # dropped by the per-passage cap — logged
                continue
            if not is_answerable(qn, ptext, cfg, llm):
                stats["rejected_unanswerable"] += 1
                continue
            if not is_paraphrastic(qn, ptext, cfg):       # RULE B
                stats["rejected_paraphrastic"] += 1
                continue
            qid = _qid(gold, qn)
            has, dids = mine_distractor(qn, gold, corpus_ids, bm25, cfg)
            rows.append({
                "qid": qid, "question": qn, "gold": gold,
                "source_meta": _source_meta(p),
                "has_distractor": has, "distractor_ids": dids,
                "split": assign_split(gold, cfg),
            })
            kept += 1
            stats["kept"] += 1
            if has:
                stats["has_distractor"] += 1
    return rows, stats


# ── resumable run + IO (mirror trace_gen) ───────────────────────────────────
def _done_golds(out_path: str) -> set:
    """Passage ids already represented in the output (resume skips them)."""
    done = set()
    for sidecar in (out_path + ".done", out_path):
        try:
            with open(sidecar, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if sidecar.endswith(".done"):
                        done.add(line)
                    else:
                        try:
                            done.add(json.loads(line).get("gold", ""))
                        except (json.JSONDecodeError, ValueError):
                            continue
        except FileNotFoundError:
            continue
    done.discard("")
    return done


def run(passages: Iterable[Any], llm: LLM, out_path: str,
        cfg: Optional[EvalSetConfig] = None, *, resume: bool = True) -> dict:
    """Generate the eval set over `passages`, writing rows to `out_path` (JSONL)
    incrementally and recording processed passage ids in `out_path + '.done'` so an
    interrupted run resumes. Returns aggregate stats. No time caps (checkpoint rule)."""
    cfg = cfg or EvalSetConfig()
    passages = list(passages)
    texts = [getattr(p, "text", "") or "" for p in passages]
    ids = [record_id(p) for p in passages]
    bm25 = _build_bm25(texts)            # full corpus once → distractor mining
    done = _done_golds(out_path) if resume else set()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    agg = {"passages": len(passages), "kept": 0, "has_distractor": 0,
           "rejected_paraphrastic": 0, "rejected_unanswerable": 0, "skipped": 0,
           "llm_errors": 0, "capped": 0, "errors": 0}
    with open(out_path, "a", encoding="utf-8") as f, \
            open(out_path + ".done", "a", encoding="utf-8") as df:
        for p, gold in zip(passages, ids):
            if gold in done:
                agg["skipped"] += 1
                continue
            # One bad passage must never abort a long overnight run. Mark it done so a
            # deterministically-failing passage isn't retried forever; count it.
            try:
                rows, st = generate_rows([p], llm, cfg, corpus_bm25=bm25, corpus_ids=ids)
            except Exception:
                agg["errors"] += 1
                df.write(gold + "\n"); df.flush()
                done.add(gold)
                continue
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            df.write(gold + "\n"); df.flush()
            done.add(gold)
            for kk in ("kept", "has_distractor", "rejected_paraphrastic",
                       "rejected_unanswerable", "llm_errors", "capped"):
                agg[kk] += st.get(kk, 0)
    agg["reject_rate"] = round(
        agg["rejected_paraphrastic"] / max(1, agg["rejected_paraphrastic"] + agg["kept"]), 4)
    return agg


def load_eval_set(path: str) -> List[dict]:
    rows: List[dict] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except (json.JSONDecodeError, ValueError):
                        continue
    except FileNotFoundError:
        pass
    return rows


# ── adapters: real library passages + real LLM ──────────────────────────────
def iter_library(memory, *, prefixes: Tuple[str, ...] = LIBRARY_PREFIXES,
                 limit: Optional[int] = None) -> List[Any]:
    """Library-origin passages only (quality-gated store, not the thought-stream):
    `memory.all_records()` filtered by source prefix."""
    recs = memory.all_records(limit=limit) if hasattr(memory, "all_records") else []
    return [r for r in recs
            if any((getattr(r, "source", "") or "").startswith(pre) for pre in prefixes)]


def mediator_llm(mediator) -> LLM:
    """Wrap an LLMMediator as a sync `Callable[[str], str]` (the study pipeline's
    bridge): run the async generate to completion and return its text."""
    def _gen(prompt: str) -> str:
        try:
            from eris.interface.mediator import run_blocking
            resp = run_blocking(mediator.generate(prompt=prompt, system=_QGEN_SYS))
            return getattr(resp, "text", "") or ""
        except Exception:
            return ""
    return _gen


# ── §4 evaluation: hit@k on full set AND distractor subset, BOTH paths ───────
def evaluate(rows: List[dict], retrievers: Dict[str, Callable[[str], Any]],
             k: int = 8) -> dict:
    """Run each retriever (question -> RetrievalResult) over the eval set and report
    hit@1 / hit@k on the FULL set and on the has_distractor SUBSET, for every path.
    The distractor row is the verdict: does the field see coupling cosine misses?"""
    from eris.dual.arbiter import gold_passage_at_k
    names = list(retrievers)
    full = {n: {"hit1": 0, "hitk": 0} for n in names}
    dist = {n: {"hit1": 0, "hitk": 0} for n in names}
    n_full, n_dist = 0, 0
    for row in rows:
        q, gold, is_d = row["question"], row["gold"], bool(row.get("has_distractor"))
        n_full += 1
        if is_d:
            n_dist += 1
        for n in names:
            res = retrievers[n](q)
            h1, _ = gold_passage_at_k(res, gold, 1)
            hk, _ = gold_passage_at_k(res, gold, k)
            full[n]["hit1"] += (h1 or 0.0)
            full[n]["hitk"] += (hk or 0.0)
            if is_d:
                dist[n]["hit1"] += (h1 or 0.0)
                dist[n]["hitk"] += (hk or 0.0)

    def _norm(d, denom):
        return {n: {"hit@1": round(d[n]["hit1"] / denom, 4) if denom else None,
                    f"hit@{k}": round(d[n]["hitk"] / denom, 4) if denom else None}
                for n in names}

    return {"k": k, "n_full": n_full, "n_distractor": n_dist,
            "full": _norm(full, n_full), "distractor": _norm(dist, n_dist)}


# ── CLI ([machine] — real library + local LLM; report over both #40 paths) ──
def main(argv=None):   # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="Retrieval eval-set generator (#40 verdict)")
    ap.add_argument("command", choices=["generate", "report"])
    ap.add_argument("--out", default="eris_data/dual/eval_set.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="cap library passages")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    from eris.orchestrator import ErisOrchestrator
    orch = ErisOrchestrator()
    if args.command == "generate":
        passages = iter_library(orch.memory, limit=args.limit)
        llm = mediator_llm(orch.mediator)
        print(f"library passages: {len(passages)}")
        agg = run(passages, llm, args.out, resume=not args.no_resume)
        print("eval-set:", agg)
        return
    # report: evaluate the produced set with BOTH #40 retrievers
    from eris.dual.retrieval import traditional_retriever, novel_retriever
    from eris.dual.report import print_eval_report
    rows = load_eval_set(args.out)
    retrievers = {"trad": traditional_retriever(orch.memory, top_k=args.k),
                  "novel": novel_retriever(orch.memory, top_k=args.k)}
    print_eval_report(evaluate(rows, retrievers, k=args.k))


if __name__ == "__main__":   # pragma: no cover
    main()
