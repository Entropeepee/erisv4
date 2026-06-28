"""Two-arm benchmark harness — the load-bearing logic (prompt building, the runner, scoring,
equal-token-budget accounting, dataset row→BenchItem mapping) is dependency-free and tested
offline. The live loaders (need `datasets`), the bare arm (needs a served model), and the Inspect
wrapper (needs `inspect_ai`) are NOT exercised here — only the pure logic that decides correctness
and fairness."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest

from eris.experiments.benchmarks.core import (
    BenchItem, build_prompt, run_arm, budget_report, accuracy, compare)
from eris.experiments.benchmarks.scoring import (
    normalize, exact_match, multiple_choice, abstained, score_item, score_results)
from eris.experiments.benchmarks.arms import callable_arm, _split_source
from eris.experiments.benchmarks import datasets as D


class TestPromptAndRunner(unittest.TestCase):
    def test_grounded_prompt_includes_source_and_question(self):
        it = BenchItem(id="x", question="Who?", context="Ada wrote the first algorithm.")
        p = build_prompt(it)
        self.assertIn("=== SOURCE ===", p)
        self.assertIn("Ada wrote", p)
        self.assertIn("Question: Who?", p)

    def test_multiple_choice_prompt_lists_lettered_options(self):
        it = BenchItem(id="x", question="2+2?", choices=["3", "4", "5"], answer="B")
        p = build_prompt(it)
        self.assertIn("A. 3", p)
        self.assertIn("B. 4", p)
        self.assertIn("single letter", p)

    def test_run_arm_accepts_text_or_text_token_pairs(self):
        items = [BenchItem(id="a", question="q1"), BenchItem(id="b", question="q2")]
        res = run_arm(items, callable_arm(lambda p: ("answer", 42)), "bare")
        self.assertEqual([r.item_id for r in res], ["a", "b"])
        self.assertEqual(res[0].tokens, 42)

    def test_run_arm_isolates_a_failing_item(self):
        def boom(_): raise ValueError("model down")
        res = run_arm([BenchItem(id="a", question="q")], boom, "bare")
        self.assertTrue(res[0].text.startswith("[error:"))
        self.assertIsNone(res[0].correct)


class TestScoring(unittest.TestCase):
    def test_normalize_and_exact_match(self):
        self.assertEqual(normalize("The Answer, Is:  42."), "answer is 42")
        self.assertTrue(exact_match("the answer is 42", "Answer is 42!"))
        self.assertFalse(exact_match("43", "42"))

    def test_multiple_choice_accepts_letter_or_text(self):
        it = BenchItem(id="x", question="?", choices=["apple", "banana", "cherry"], answer="B")
        self.assertTrue(multiple_choice("The answer is B.", it))
        self.assertTrue(multiple_choice("banana", it))
        self.assertFalse(multiple_choice("A", it))

    def test_multiple_choice_gold_as_text_resolves_to_letter(self):
        it = BenchItem(id="x", question="?", choices=["apple", "banana"], answer="banana")
        self.assertTrue(multiple_choice("(B)", it))

    def test_abstention_scores_a_refusal_correct(self):
        it = BenchItem(id="x", question="?", unanswerable=True)
        self.assertTrue(score_item("UNANSWERABLE", it))
        self.assertTrue(abstained("The source does not contain this information."))
        self.assertFalse(score_item("It is 42.", it))

    def test_score_results_matches_by_id_and_skips_errors(self):
        items = [BenchItem(id="a", question="?", answer="42")]
        res = run_arm(items, callable_arm(lambda p: "42"), "bare")
        score_results(res, items)
        self.assertTrue(res[0].correct)


class TestBudgetAndCompare(unittest.TestCase):
    def test_equal_budget_flagged_true_when_token_ratio_near_one(self):
        a = run_arm([BenchItem(id="1", question="q", answer="x")],
                    callable_arm(lambda p: ("x", 100)), "bare")
        b = run_arm([BenchItem(id="1", question="q", answer="x")],
                    callable_arm(lambda p: ("x", 110)), "eris")
        items = [BenchItem(id="1", question="q", answer="x")]
        score_results(a, items); score_results(b, items)
        c = compare(a, b)
        self.assertTrue(c["equal_budget"])
        self.assertEqual(c["token_ratio_b_over_a"], 1.1)

    def test_unequal_budget_flagged_false_and_warned(self):
        a = run_arm([BenchItem(id="1", question="q", answer="x")],
                    callable_arm(lambda p: ("x", 100)), "bare")
        b = run_arm([BenchItem(id="1", question="q", answer="x")],
                    callable_arm(lambda p: ("x", 500)), "eris")   # 5x the tokens
        items = [BenchItem(id="1", question="q", answer="x")]
        score_results(a, items); score_results(b, items)
        c = compare(a, b)
        self.assertFalse(c["equal_budget"])
        self.assertIn("UNEQUAL", c["note"])

    def test_budget_and_accuracy_helpers(self):
        items = [BenchItem(id=str(i), question="q", answer="x") for i in range(2)]
        res = run_arm(items, callable_arm(lambda p: ("x", 50)), "bare")
        score_results(res, items)
        self.assertEqual(budget_report(res)["tokens_per_question"], 50.0)
        self.assertEqual(accuracy(res)["accuracy"], 1.0)


class TestDatasetMappers(unittest.TestCase):
    def test_frames_item(self):
        it = D._frames_item({"Prompt": "Who founded X?", "Answer": "Ada",
                             "wikipedia_links": ["http://x"]}, 0)
        self.assertEqual(it.question, "Who founded X?")
        self.assertEqual(it.answer, "Ada")
        self.assertEqual(it.meta["type"], "multi_hop")

    def test_quality_flattens_questions_and_filters_hard(self):
        row = {"article": "A long passage.", "questions": [
            {"question": "easy?", "options": ["a", "b"], "gold_label": 1, "difficult": 0},
            {"question": "hard?", "options": ["a", "b", "c"], "gold_label": 3, "difficult": 1}]}
        hard = D._quality_questions(row, 0, hard_only=True)
        self.assertEqual(len(hard), 1)
        self.assertEqual(hard[0].question, "hard?")
        self.assertEqual(hard[0].answer, "C")               # gold_label 3 → letter C
        self.assertEqual(hard[0].context, "A long passage.")
        allq = D._quality_questions(row, 0, hard_only=False)
        self.assertEqual(len(allq), 2)

    def test_mmlu_pro_item_uses_letter_or_index(self):
        it = D._mmlu_pro_item({"question": "?", "options": ["w", "x", "y"],
                               "answer": "C", "category": "math"}, 0)
        self.assertEqual(it.answer, "C")
        self.assertEqual(it.context, "")                    # closed-book
        self.assertEqual(len(it.choices), 3)

    def test_gpqa_places_correct_deterministically_not_always_A(self):
        row = {"Question": "?", "Correct Answer": "right",
               "Incorrect Answer 1": "w1", "Incorrect Answer 2": "w2", "Incorrect Answer 3": "w3"}
        # the gold letter must point at "right" for several indices, and not be constant
        letters = []
        for i in range(4):
            it = D._gpqa_item(row, i)
            self.assertEqual(it.choices[ord(it.answer) - 65], "right")   # gold letter → correct opt
            letters.append(it.answer)
        self.assertGreater(len(set(letters)), 1)            # position varies, not always 'A'

    def test_ragtruth_is_faithfulness_typed(self):
        it = D._ragtruth_item({"prompt": "Summarize.", "reference": "src text",
                               "response": "a summary", "labels": [{"span": "x"}]}, 0)
        self.assertEqual(it.meta["type"], "faithfulness")
        self.assertEqual(it.context, "src text")
        self.assertEqual(it.meta["hallucination_spans"], [{"span": "x"}])


class TestArmsHelpers(unittest.TestCase):
    def test_split_source_recovers_context_and_question(self):
        it = BenchItem(id="x", question="Who wrote it?", context="Ada did.")
        ctx, q = _split_source(build_prompt(it))
        self.assertEqual(ctx, "Ada did.")
        self.assertEqual(q, "Who wrote it?")

    def test_split_source_closed_book_has_no_context(self):
        ctx, q = _split_source("Question: 2+2?")
        self.assertEqual(ctx, "")
        self.assertIn("2+2", q)


if __name__ == "__main__":
    unittest.main()
