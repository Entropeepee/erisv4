"""§5: retrieval eval-set generator — offline, deterministic, stub LLM.

Proves the three load-bearing rules:
  RULE A — gold is the SOURCE passage's id, never a retriever's pick.
  RULE B — a verbatim-copy question is rejected by the paraphrastic filter; a
           conceptual one passes.
  RULE C — a planted lexically-similar non-gold passage is found and flagged.
And that the produced set drives the #40 arbiter's gold_passage_at_k correctly.
"""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import tempfile
import unittest

from eris.dual.types import RetrievalResult, record_id
from eris.dual.eval_set import (
    EvalSetConfig, generate_rows, run, load_eval_set, evaluate,
    ngram_overlap, is_paraphrastic, is_answerable, mine_distractor,
    parse_questions, _build_bm25,
)


class _Rec:
    """Minimal MemoryRecord-like passage."""
    def __init__(self, text, source="reading:doc", sha=None):
        self.text = text
        self.source = source
        self.metadata = {"title": source.split(":")[-1], "sha256": sha} if sha else {"title": source}


# A deterministic "LLM": maps each passage to canned questions keyed by a marker
# word in the passage, so generation is reproducible without a model.
def _stub_llm(text):
    if "photosynthesis" in text:
        # one conceptual paraphrase + one verbatim echo (the echo must be rejected)
        return ("By what process do green plants turn light into stored chemical energy?\n"
                "Plants convert sunlight into chemical energy stored as glucose.")
    if "mitochondria" in text:
        return ("Which organelle is chiefly responsible for producing a cell's usable energy?\n"
                "What molecule carries the energy the organelle releases?")
    return "What is the central idea conveyed by this material here?"


class TestEvalSet(unittest.TestCase):
    def setUp(self):
        self.passages = [
            _Rec("Plants convert sunlight into chemical energy stored as glucose. "
                 "This photosynthesis process happens in the chloroplast.", "reading:bio1"),
            _Rec("The mitochondria generate ATP, the energy currency of the cell, "
                 "through respiration.", "study:bio2"),
        ]
        self.cfg = EvalSetConfig(n_questions=2, seed=0)

    # ── RULE A ────────────────────────────────────────────────────────────
    def test_gold_is_source_hash_not_retriever(self):
        rows, stats = generate_rows(self.passages, _stub_llm, self.cfg)
        self.assertTrue(rows)
        golds = {r["gold"] for r in rows}
        # every gold equals the record_id of one of the SOURCE passages
        source_ids = {record_id(p) for p in self.passages}
        self.assertTrue(golds <= source_ids)
        for r in rows:
            self.assertIn("qid", r)
            self.assertIn("question", r)
            self.assertIn(r["split"], ("train", "heldout"))

    # ── RULE B ────────────────────────────────────────────────────────────
    def test_paraphrastic_filter_rejects_verbatim_keeps_conceptual(self):
        passage = self.passages[0].text
        echo = "Plants convert sunlight into chemical energy stored as glucose."
        concept = "By what process do green plants turn light into stored chemical energy?"
        self.assertGreater(ngram_overlap(echo, passage), 0.5)
        self.assertLessEqual(ngram_overlap(concept, passage), 0.5)
        self.assertFalse(is_paraphrastic(echo, passage, self.cfg))    # rejected
        self.assertTrue(is_paraphrastic(concept, passage, self.cfg))  # kept

    def test_generation_drops_the_echo_row(self):
        rows, stats = generate_rows(self.passages[:1], _stub_llm, self.cfg)
        # the stub emitted one conceptual + one verbatim echo; only the concept survives
        self.assertEqual(stats["rejected_paraphrastic"], 1)
        self.assertEqual(len(rows), 1)
        self.assertIn("process", rows[0]["question"].lower())

    def test_answerable_requires_on_topic(self):
        passage = self.passages[0].text
        self.assertTrue(is_answerable("How do plants store energy from sunlight?",
                                      passage, self.cfg))
        self.assertFalse(is_answerable("Who won the football match yesterday?",
                                       passage, self.cfg))

    # ── RULE C ────────────────────────────────────────────────────────────
    def test_distractor_mining_flags_planted_hard_negative(self):
        gold = _Rec("Mitochondria produce ATP and release carbon dioxide during "
                    "cellular respiration in the cell.", "study:gold")
        # a lexically near-symmetric NON-gold passage (BM25 can't separate them)
        distractor = _Rec("Mitochondria also release carbon dioxide and generate heat "
                          "during respiration in the cell.", "study:distract")
        far = _Rec("Rainbows form when sunlight refracts through water droplets.",
                   "reading:far")
        passages = [gold, distractor, far]
        ids = [record_id(p) for p in passages]
        bm25 = _build_bm25([p.text for p in passages])
        question = "What do mitochondria release during respiration in the cell?"
        has, dids = mine_distractor(question, 0, ids, bm25, self.cfg)
        self.assertTrue(has)
        self.assertIn(record_id(distractor), dids)
        self.assertNotIn(record_id(gold), dids)        # gold is never its own distractor

    def test_no_distractor_when_corpus_is_unrelated(self):
        gold = _Rec("Quaternions extend complex numbers to four dimensions.", "reading:math")
        far1 = _Rec("Bananas are rich in potassium.", "reading:food")
        far2 = _Rec("The Eiffel Tower is in Paris.", "reading:travel")
        passages = [gold, far1, far2]
        ids = [record_id(p) for p in passages]
        bm25 = _build_bm25([p.text for p in passages])
        has, dids = mine_distractor("What do quaternions extend to four dimensions?",
                                    0, ids, bm25, self.cfg)
        self.assertFalse(has)

    # ── feeds gold_passage_at_k (the #40 arbiter) ─────────────────────────
    def test_set_drives_gold_passage_at_k(self):
        rows, _ = generate_rows(self.passages, _stub_llm, self.cfg)
        qgold = {r["question"]: r["gold"] for r in rows}

        def perfect(q):     # ranks each query's true gold first
            g = qgold[q]
            recs = sorted(self.passages, key=lambda p: record_id(p) != g)
            return RetrievalResult(records=recs)

        def useless(q):     # never returns the gold
            g = qgold[q]
            return RetrievalResult(records=[p for p in self.passages
                                            if record_id(p) != g])

        res = evaluate(rows, {"good": perfect, "bad": useless}, k=2)
        self.assertEqual(res["full"]["good"]["hit@1"], 1.0)
        self.assertEqual(res["full"]["bad"]["hit@1"], 0.0)
        self.assertEqual(res["n_full"], len(rows))

    def test_report_formatter_prints_both_subsets(self):
        from eris.dual.report import print_eval_report
        result = {"k": 8, "n_full": 10, "n_distractor": 4,
                  "full": {"trad": {"hit@1": 0.7, "hit@8": 0.9},
                           "novel": {"hit@1": 0.6, "hit@8": 0.9}},
                  "distractor": {"trad": {"hit@1": 0.25, "hit@8": 0.5},
                                 "novel": {"hit@1": 0.5, "hit@8": 0.75}}}
        out = print_eval_report(result)        # must not raise; echoes the dict back
        self.assertEqual(out["n_distractor"], 4)

    # ── resumable run + IO ────────────────────────────────────────────────
    def test_run_writes_and_resumes(self):
        out = os.path.join(tempfile.mkdtemp(), "eval_set.jsonl")
        agg1 = run(self.passages, _stub_llm, out, self.cfg)
        self.assertGreater(agg1["kept"], 0)
        self.assertEqual(agg1["skipped"], 0)
        rows = load_eval_set(out)
        self.assertEqual(len(rows), agg1["kept"])
        # second run skips every already-processed passage
        agg2 = run(self.passages, _stub_llm, out, self.cfg)
        self.assertEqual(agg2["kept"], 0)
        self.assertEqual(agg2["skipped"], len(self.passages))
        self.assertEqual(len(load_eval_set(out)), len(rows))   # no duplicates


if __name__ == "__main__":
    unittest.main()
