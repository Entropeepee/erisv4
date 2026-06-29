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
    BenchItem, build_prompt, run_arm, budget_report, accuracy, faithfulness, compare, item_details)
from eris.experiments.benchmarks.scoring import (
    normalize, exact_match, multiple_choice, abstained, score_item, score_results,
    faithfulness_score, sentence_supported)
from eris.experiments.benchmarks.arms import callable_arm, _split_source, _answer_text_from_message
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

    def test_normalize_preserves_decimals(self):
        # the swarm's decimal false-negative: 3.14 must not shatter into '3 14'
        from eris.experiments.benchmarks.scoring import contains_gold
        self.assertEqual(normalize("3.14"), "3.14")
        self.assertTrue(exact_match("3.14", "3.14"))                 # pure numeric answer matches
        self.assertTrue(contains_gold("The value is 3.14", "3.14"))  # free-form span survives
        self.assertFalse(exact_match("3.15", "3.14"))                # different numbers still differ

    def test_multiple_choice_accepts_letter_or_text(self):
        it = BenchItem(id="x", question="?", choices=["apple", "banana", "cherry"], answer="B")
        self.assertTrue(multiple_choice("The answer is B.", it))
        self.assertTrue(multiple_choice("banana", it))
        self.assertFalse(multiple_choice("A", it))

    def test_multiple_choice_gold_as_text_resolves_to_letter(self):
        it = BenchItem(id="x", question="?", choices=["apple", "banana"], answer="banana")
        self.assertTrue(multiple_choice("(B)", it))

    def test_multiple_choice_ignores_leading_article_capital(self):
        # the swarm's false-negative: a stray leading 'A'/'I' must not be read as the choice
        it = BenchItem(id="x", question="?", choices=["red", "blue", "yellow"], answer="C")
        self.assertTrue(multiple_choice("A reasonable answer is C.", it))   # not 'A'
        self.assertTrue(multiple_choice("I would pick C", it))              # not 'I'
        self.assertTrue(multiple_choice("Answer: **C**", it))              # markdown marker
        self.assertFalse(multiple_choice("A reasonable answer is B.", it))  # genuinely wrong

    def test_multiple_choice_restricts_to_valid_option_letters(self):
        # 'E' is past the 3 options → never a valid pick
        it = BenchItem(id="x", question="?", choices=["red", "blue", "yellow"], answer="C")
        self.assertFalse(multiple_choice("The grade is E", it))

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

    def test_item_details_pairs_predictions_with_gold(self):
        items = [BenchItem(id="q1", question="Who wrote it?", answer="Ada")]
        bare = run_arm(items, callable_arm(lambda p: ("Babbage", 50)), "bare")
        eris = run_arm(items, callable_arm(lambda p: ("Ada", 900)), "eris")
        score_results(bare, items); score_results(eris, items)
        rows = item_details(items, {"bare": bare, "eris": eris})
        self.assertEqual(rows[0]["gold"], "Ada")
        self.assertEqual(rows[0]["bare"]["answer"], "Babbage")
        self.assertFalse(rows[0]["bare"]["correct"])
        self.assertEqual(rows[0]["eris"]["answer"], "Ada")
        self.assertTrue(rows[0]["eris"]["correct"])
        self.assertEqual(rows[0]["eris"]["tokens"], 900)

    def test_item_details_shows_mc_option_texts(self):
        # so a wrong letter is judgeable: we must SEE what A/B/C/D say + the chosen option's text
        it = BenchItem(id="q", question="?", choices=["she felt controlled", "she was bored",
                                                      "she was tired"], answer="A")
        res = run_arm([it], callable_arm(lambda p: ("C", 5)), "eris")
        score_results(res, [it])
        rows = item_details([it], {"eris": res})
        self.assertEqual(rows[0]["choices"]["A"], "she felt controlled")
        self.assertEqual(rows[0]["gold_text"], "she felt controlled")
        self.assertEqual(rows[0]["eris"]["answer_text"], "she was tired")   # what 'C' actually said
        self.assertFalse(rows[0]["eris"]["correct"])

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

    def test_frames_wiki_title_from_url(self):
        self.assertEqual(D._wiki_title_from_url(
            "https://en.wikipedia.org/wiki/Ada_Lovelace#Early_life"), "Ada Lovelace")
        self.assertEqual(D._wiki_title_from_url(
            "https://en.wikipedia.org/wiki/Charles_Babbage?foo=bar"), "Charles Babbage")
        self.assertEqual(D._wiki_title_from_url(
            "https://en.wikipedia.org/wiki/Caf%C3%A9"), "Café")          # url-decoded

    def test_frames_links_collected_from_any_field(self):
        # FRAMES ships links, not text — they may be a list OR inline in a string field
        row = {"Prompt": "see https://en.wikipedia.org/wiki/Foo for context",
               "wiki_links": ["https://en.wikipedia.org/wiki/Bar",
                              "https://en.wikipedia.org/wiki/Bar"],   # dup
               "Answer": "x"}
        links = D._frames_links(row)
        self.assertIn("https://en.wikipedia.org/wiki/Foo", links)
        self.assertIn("https://en.wikipedia.org/wiki/Bar", links)
        self.assertEqual(len(links), 2)                                # deduped, order-preserved

    def test_frames_item_context_empty_without_fetch(self):
        # the bug the smoke test caught: a bare FRAMES row carries NO article text → context ""
        # (which is why load_frames(fetch_context=True) must assemble it)
        it = D._frames_item({"Prompt": "Q?", "Answer": "A",
                             "wiki_links": ["https://en.wikipedia.org/wiki/Foo"]}, 0)
        self.assertEqual(it.context, "")

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

    def test_gold_to_letter_handles_every_encoding(self):
        opts = ["apple", "banana", "cherry", "date"]
        self.assertEqual(D._gold_to_letter("B", opts), "B")            # letter
        self.assertEqual(D._gold_to_letter("cherry", opts), "C")       # option text
        self.assertEqual(D._gold_to_letter(2, opts), "B")             # 1-based (QuALITY native)
        self.assertEqual(D._gold_to_letter("4", opts), "D")           # numeric string, 1-based
        self.assertEqual(D._gold_to_letter(None, opts), "")

    def test_quality_flat_schema_one_question_per_row(self):
        # the emozilla/quality mirror is flat (one MC question per row), not nested
        row = {"question": "Why?", "options": ["a", "b", "c"], "answer": 2,
               "hard": True, "article": "A long passage about the topic."}
        items = D._quality_questions(row, 0, hard_only=True)           # routes to the flat mapper
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].question, "Why?")
        self.assertEqual(items[0].answer, "B")                        # 1-based 2 → B
        self.assertEqual(items[0].context, "A long passage about the topic.")
        self.assertEqual(items[0].meta["type"], "faithfulness" if False else "long_doc_mc")

    def test_quality_flat_hard_only_filters_easy(self):
        easy = {"question": "Q?", "options": ["a", "b"], "answer": 1, "hard": False}
        self.assertEqual(D._quality_questions(easy, 0, hard_only=True), [])
        self.assertEqual(len(D._quality_questions(easy, 0, hard_only=False)), 1)

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


class TestFaithfulnessScorer(unittest.TestCase):
    CTX = ("The SGT method derives its gating threshold from the measurement statistics, "
           "T equals k times sigma. The novelty is the integration into a dual-path "
           "architecture with a shared accumulator, not the well-known scaling.")

    def test_supported_sentence_passes_unsupported_flagged(self):
        ctx_tokens = {w for w in __import__("re").findall(r"[a-z0-9]{4,}", self.CTX.lower())}
        self.assertTrue(sentence_supported(
            "The gating threshold is derived from the measurement statistics.", ctx_tokens))
        self.assertFalse(sentence_supported(
            "The device cured cancer in nineteen clinical trials worldwide.", ctx_tokens))

    def test_faithfulness_rate_zero_when_fully_grounded(self):
        out = ("The gating threshold is derived from the measurement statistics. "
               "The novelty is the integration into a dual-path architecture.")
        fs = faithfulness_score(out, self.CTX)
        self.assertEqual(fs["hallucination_rate"], 0.0)
        self.assertEqual(fs["unsupported"], 0)

    def test_faithfulness_flags_unsupported_sentences(self):
        out = ("The gating threshold is derived from the measurement statistics. "
               "It also reduces power consumption by exactly forty-two percent in every device.")
        fs = faithfulness_score(out, self.CTX)
        self.assertGreater(fs["hallucination_rate"], 0.0)   # the fabricated stat is caught
        self.assertEqual(fs["unsupported"], 1)

    def test_annotated_spans_take_precedence(self):
        out = "Claim one is here. The fabricated wear claim sits in sentence two."
        fs = faithfulness_score(out, "anything", hallucination_spans=[{"text": "fabricated wear claim"}])
        self.assertEqual(fs["unsupported"], 1)
        self.assertIn("fabricated wear", fs["hallucinated"][0])

    def test_annotated_span_does_not_match_inside_a_longer_word(self):
        # the swarm's substring collision: span 'cat' must NOT fire inside 'category'
        out = "The category of students was assigned by faculty."
        fs = faithfulness_score(out, "ctx", hallucination_spans=[{"span": "cat"}])
        self.assertEqual(fs["unsupported"], 0)
        self.assertEqual(fs["hallucination_rate"], 0.0)

    def test_score_item_does_not_misroute_faithfulness_to_exact_match(self):
        it = BenchItem(id="rt", question="Summarize.", context=self.CTX,
                       answer="(a reference response that exact-match would compare against)",
                       meta={"type": "faithfulness", "hallucination_spans": []})
        grounded = "The novelty is the integration into a dual-path architecture."
        self.assertTrue(score_item(grounded, it))           # faithful → True, NOT a gold-string match
        self.assertFalse(score_item("It tripled battery life across all tested hardware.", it))

    def test_score_results_attaches_rate_and_leaves_accuracy_alone(self):
        items = [BenchItem(id="rt", question="?", context=self.CTX,
                           meta={"type": "faithfulness"})]
        res = run_arm(items, callable_arm(
            lambda p: "The novelty is the integration into a dual-path architecture."), "eris")
        score_results(res, items)
        self.assertEqual(res[0].faithfulness, 0.0)
        self.assertIsNone(res[0].correct)                   # not folded into accuracy
        self.assertEqual(accuracy(res)["graded"], 0)

    def test_compare_reports_hallucination_delta_lower_is_better(self):
        items = [BenchItem(id="rt", question="?", context=self.CTX,
                           meta={"type": "faithfulness"})]
        bare = run_arm(items, callable_arm(
            lambda p: ("It cut power use by ninety percent on every device.", 100)), "bare")
        eris = run_arm(items, callable_arm(
            lambda p: ("The novelty is the integration into a dual-path architecture.", 110)), "eris")
        score_results(bare, items); score_results(eris, items)
        c = compare(bare, eris)
        self.assertIn("faithfulness", c["bare"])
        self.assertLess(c["delta_hallucination_rate"], 0)   # Eris hallucinates LESS than bare
        self.assertEqual(c["eris"]["faithfulness"]["mean_hallucination_rate"], 0.0)


class TestObservabilityNoSilentFailures(unittest.TestCase):
    def test_run_arm_captures_detail_from_dict_return(self):
        # the Eris arm returns a dict with full diagnostics; run_arm must preserve detail
        items = [BenchItem(id="a", question="q", answer="x")]
        res = run_arm(items, lambda p: {"text": "x", "tokens": 5,
                                        "detail": {"synthesis": "FULL HIVE TEXT",
                                                   "extraction_ok": True}}, "eris")
        self.assertEqual(res[0].tokens, 5)
        self.assertEqual(res[0].detail["synthesis"], "FULL HIVE TEXT")

    def test_accuracy_reports_error_rate(self):
        # an arm that errors on some items must not hide the reliability hit in the denominator
        items = [BenchItem(id=str(i), question="q", answer="x") for i in range(4)]
        calls = {"n": 0}
        def fn(p):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("boom")
            return ("x", 10)
        res = score_results(run_arm(items, fn, "bare"), items)
        a = accuracy(res)
        self.assertEqual(a["errored"], 1)
        self.assertEqual(a["error_rate"], 0.25)
        self.assertEqual(a["graded"], 3)              # errored item excluded from accuracy denom

    def test_item_details_surfaces_full_synthesis_and_fetch_stats(self):
        it = BenchItem(id="f1", question="q", answer="x",
                       meta={"fetch": {"linked": 3, "fetched": 1, "failed": ["A", "B"]}})
        res = run_arm([it], lambda p: {"text": "x", "tokens": 9,
                                       "detail": {"synthesis": "FULL HIVE TEXT",
                                                  "extraction_ok": False}}, "eris")
        score_results(res, [it])
        rows = item_details([it], {"eris": res})
        self.assertEqual(rows[0]["source_fetch"]["fetched"], 1)        # partial fetch is visible
        self.assertEqual(rows[0]["source_fetch"]["failed"], ["A", "B"])
        self.assertEqual(rows[0]["eris"]["synthesis"], "FULL HIVE TEXT")  # full, untruncated
        self.assertFalse(rows[0]["eris"]["extraction_ok"])             # extraction failure flagged

    def test_item_details_does_not_truncate_long_answers(self):
        long = "word " * 500                          # ~2500 chars
        it = BenchItem(id="a", question="q" * 400, answer="x")
        res = run_arm([it], callable_arm(lambda p: (long, 1)), "bare")
        rows = item_details([it], {"bare": res})
        self.assertEqual(rows[0]["bare"]["answer"], long.strip())     # whole answer
        self.assertEqual(len(rows[0]["question"]), 400)               # whole question

    def test_frames_fetch_records_partial_failures(self):
        import eris.knowledge.web_reader as wr
        def fake_fetch(title, lang="en"):
            if title == "Good":
                return "the article text body"
            raise RuntimeError("404")
        orig = wr.fetch_wikipedia
        wr.fetch_wikipedia = fake_fetch
        try:
            row = {"wiki_links": ["https://en.wikipedia.org/wiki/Good",
                                  "https://en.wikipedia.org/wiki/Bad"]}
            ctx, stats = D._fetch_frames_context(row)
        finally:
            wr.fetch_wikipedia = orig
        self.assertIn("article text", ctx)
        self.assertEqual(stats["linked"], 2)
        self.assertEqual(stats["fetched"], 1)
        self.assertIn("Bad", stats["failed"])                        # the failed article is recorded


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

    def test_bare_arm_reads_reasoning_models(self):
        # the live bug: qwen3 put its answer in reasoning_content with empty content → scored 0
        self.assertEqual(_answer_text_from_message({"content": "42"}), "42")
        self.assertEqual(_answer_text_from_message(
            {"content": "", "reasoning_content": "let me think... the answer is 42"}),
            "let me think... the answer is 42")
        self.assertEqual(_answer_text_from_message(
            {"content": "<think>2+2 is 4, not 5</think>4"}), "4")          # strip the think block
        self.assertEqual(_answer_text_from_message(
            {"content": "", "reasoning": "<think>hmm</think>Paris"}), "Paris")


class TestBlackboxAuditFixes(unittest.TestCase):
    """The 2026-06 blackbox audit found 39 places the report computed a value then dropped it.
    These lock the high-severity surfacings so a regression can't quietly re-hide them."""

    def test_faithfulness_surfaces_mode_n_spans_and_full_list(self):
        from eris.experiments.benchmarks.scoring import faithfulness_score
        # overlap-proxy mode (no annotated spans) — must be labelled as the CRUDE fallback
        out = "Alpha beta gamma delta. " * 8                        # 8 unsupported sentences
        fs = faithfulness_score(out, "totally unrelated source text")
        self.assertEqual(fs["mode"], "overlap_proxy")
        self.assertEqual(fs["n_spans"], 0)
        self.assertEqual(fs["threshold"], 0.6)
        self.assertEqual(len(fs["hallucinated"]), 5)               # display cap unchanged
        self.assertEqual(len(fs["hallucinated_all"]), fs["unsupported"])   # FULL list, uncapped
        self.assertGreater(fs["unsupported"], 5)
        # spans mode — the trustworthy RAGTruth path must be labelled 'spans'
        fs2 = faithfulness_score("A fabricated claim here.", "ctx",
                                 hallucination_spans=[{"text": "fabricated claim"}])
        self.assertEqual(fs2["mode"], "spans")
        self.assertEqual(fs2["n_spans"], 1)

    def test_score_results_merges_detail_not_clobber(self):
        # a faithfulness item must KEEP the arm's hive diagnostics, not have them overwritten
        it = BenchItem(id="r1", question="q", context="the cat sat on the mat",
                       meta={"type": "faithfulness"})
        res = run_arm([it], lambda p: {"text": "the cat sat on the mat.", "tokens": 3,
                                       "detail": {"synthesis": "FULL HIVE TEXT",
                                                  "specialist_divergence": 0.42}}, "eris")
        score_results(res, [it])
        d = res[0].detail
        self.assertEqual(d["synthesis"], "FULL HIVE TEXT")          # survived
        self.assertEqual(d["specialist_divergence"], 0.42)         # survived
        self.assertIn("hallucination_rate", d)                      # faithfulness merged in
        self.assertEqual(d["mode"], "overlap_proxy")

    def test_multiple_choice_detail_records_letter_text_and_path(self):
        from eris.experiments.benchmarks.scoring import multiple_choice_detail
        it = BenchItem(id="m", question="q", choices=["apple", "banana", "cherry"], answer="B")
        d = multiple_choice_detail("After weighing it, the answer is B because…", it)
        self.assertTrue(d["correct"])
        self.assertEqual(d["extracted_letter"], "B")
        self.assertEqual(d["extracted_text"], "banana")
        self.assertEqual(d["mc_match"], "letter")
        # option-text path: a verbose answer that names the option but no letter
        d2 = multiple_choice_detail("I'd say it is the banana.", it)
        self.assertTrue(d2["correct"])
        self.assertEqual(d2["mc_match"], "option_text")
        # genuine miss
        d3 = multiple_choice_detail("Definitely A.", it)
        self.assertFalse(d3["correct"])
        self.assertEqual(d3["mc_match"], "none")
        self.assertEqual(d3["extracted_letter"], "A")              # what it parsed is still shown

    def test_score_results_surfaces_mc_decision_in_detail(self):
        it = BenchItem(id="m2", question="q", choices=["x", "y"], answer="A")
        res = run_arm([it], callable_arm(lambda p: ("The answer is A.", 4)), "bare")
        score_results(res, [it])
        d = res[0].detail
        self.assertEqual(d["extracted_letter"], "A")
        self.assertEqual(d["mc_match"], "letter")
        self.assertEqual(d["gold_letter"], "A")

    def test_budget_report_flags_proxy_and_distribution(self):
        from dataclasses import dataclass
        items = [BenchItem(id=str(i), question="q", answer="x") for i in range(3)]
        res = run_arm(items, lambda p: {"text": "x", "tokens": 10,
                                        "detail": {"tokens_are_proxy": True}}, "eris")
        b = budget_report(res)
        self.assertEqual(b["proxy_items"], 3)
        self.assertEqual(b["cost_basis"], "call_count_proxy")
        self.assertEqual(b["max_tokens"], 10)
        self.assertEqual(b["median_tokens"], 10.0)
        self.assertEqual(b["measured"], 3)
        # real-token arm
        res2 = run_arm(items, callable_arm(lambda p: ("x", 20)), "bare")
        self.assertEqual(budget_report(res2)["cost_basis"], "real_tokens")

    def test_compare_surfaces_ungraded_and_cost_caveat(self):
        items = [BenchItem(id=str(i), question="q", answer="x") for i in range(2)]
        bare = run_arm(items, callable_arm(lambda p: ("x", 10)), "bare")
        eris = run_arm(items, lambda p: {"text": "x", "tokens": 11,
                                         "detail": {"tokens_are_proxy": True}}, "eris")
        score_results(bare, items); score_results(eris, items)
        c = compare(bare, eris)
        self.assertIn("ungraded", c["bare"])
        self.assertEqual(c["cost_basis"]["eris"], "call_count_proxy")
        self.assertIn("equal_budget_caveat", c)                     # proxy → verdict qualified

    def test_item_details_shows_context_chars_for_grounded_items(self):
        it = BenchItem(id="g", question="q", context="x" * 1234, answer="a")
        res = run_arm([it], callable_arm(lambda p: ("a", 1)), "bare")
        rows = item_details([it], {"bare": res})
        self.assertEqual(rows[0]["context_chars"], 1234)
        # closed-book items carry no context_chars (nothing to ground against)
        it2 = BenchItem(id="cb", question="q", answer="a")
        rows2 = item_details([it2], {"bare": run_arm([it2], callable_arm(lambda p: ("a", 1)), "bare")})
        self.assertNotIn("context_chars", rows2[0])

    def test_openai_chat_arm_surfaces_finish_reason_and_truncation(self):
        import sys, types
        fake = types.SimpleNamespace()

        class _Resp:
            def __init__(self, data): self._d = data
            def raise_for_status(self): pass
            def json(self): return self._d

        def post(url, json=None, headers=None, timeout=None):
            return _Resp({"choices": [{"message": {"content": "C"}, "finish_reason": "length"}],
                          "usage": {"total_tokens": 1500, "prompt_tokens": 1400,
                                    "completion_tokens": 100}})
        fake.post = post
        orig = sys.modules.get("requests")
        sys.modules["requests"] = fake
        try:
            from eris.experiments.benchmarks.arms import openai_chat_arm
            out = openai_chat_arm(base_url="http://x/v1", model="m", api_key="k")("Question: ?")
        finally:
            if orig is not None:
                sys.modules["requests"] = orig
            else:
                sys.modules.pop("requests", None)
        self.assertEqual(out["text"], "C")
        self.assertEqual(out["tokens"], 1500)
        self.assertTrue(out["detail"]["truncated"])               # clipped ≠ wrong answer
        self.assertEqual(out["detail"]["finish_reason"], "length")
        self.assertEqual(out["detail"]["completion_tokens"], 100)


class TestDotenvAndAttributable(unittest.TestCase):
    def test_dotenv_records_shell_override_conflict(self):
        import tempfile
        from eris.experiments.benchmarks.run import _load_dotenv, _DOTENV_CONFLICTS
        d = tempfile.mkdtemp()
        p = os.path.join(d, ".env")
        with open(p, "w") as f:
            f.write("ERIS_HIVE_SYNTH_CLOUD=0\nERIS_TIER_FREE=qwen/qwen-2.5-72b\n")
        os.environ["ERIS_HIVE_SYNTH_CLOUD"] = "1"                  # stale shell value overrides file
        os.environ.pop("ERIS_TIER_FREE", None)
        try:
            _load_dotenv(p)
            self.assertEqual(os.environ["ERIS_HIVE_SYNTH_CLOUD"], "1")     # shell still wins
            self.assertEqual(os.environ["ERIS_TIER_FREE"], "qwen/qwen-2.5-72b")  # file loaded clean
            conflicts = {c[0]: (c[1], c[2]) for c in _DOTENV_CONFLICTS}
            self.assertEqual(conflicts.get("ERIS_HIVE_SYNTH_CLOUD"), ("1", "0"))  # named!
            self.assertNotIn("ERIS_TIER_FREE", conflicts)          # loaded cleanly, no conflict
        finally:
            os.environ.pop("ERIS_HIVE_SYNTH_CLOUD", None)
            os.environ.pop("ERIS_TIER_FREE", None)

    def test_attributable_preset_forces_one_model_over_stale_vars(self):
        from eris.experiments.benchmarks.run import _apply_attributable_preset
        keys = ("ERIS_BENCH_ATTRIBUTABLE", "ERIS_BENCH_MODEL", "ERIS_TIER_FREE", "ERIS_TIER_SYNTH",
                "ERIS_TIER_CHEAP", "ERIS_HIVE_SYNTH_CLOUD")
        saved = {k: os.environ.get(k) for k in keys}
        try:
            os.environ["ERIS_BENCH_ATTRIBUTABLE"] = "1"
            os.environ["ERIS_BENCH_MODEL"] = "qwen/qwen-2.5-72b-instruct"
            os.environ["ERIS_TIER_SYNTH"] = "anthropic/claude-3.5-sonnet"   # stale → must override
            os.environ["ERIS_HIVE_SYNTH_CLOUD"] = "1"                       # stale → must flip off
            self.assertTrue(_apply_attributable_preset())
            for k in ("ERIS_TIER_FREE", "ERIS_TIER_CHEAP", "ERIS_TIER_SYNTH"):
                self.assertEqual(os.environ[k], "qwen/qwen-2.5-72b-instruct")
            self.assertEqual(os.environ["ERIS_HIVE_SYNTH_CLOUD"], "0")
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    def test_attributable_preset_requires_bench_model(self):
        from eris.experiments.benchmarks.run import _apply_attributable_preset
        saved = {k: os.environ.get(k) for k in ("ERIS_BENCH_ATTRIBUTABLE", "ERIS_BENCH_MODEL")}
        try:
            os.environ["ERIS_BENCH_ATTRIBUTABLE"] = "1"
            os.environ.pop("ERIS_BENCH_MODEL", None)
            with self.assertRaises(SystemExit):
                _apply_attributable_preset()
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    def test_attributable_preset_noop_when_unset(self):
        from eris.experiments.benchmarks.run import _apply_attributable_preset
        saved = os.environ.get("ERIS_BENCH_ATTRIBUTABLE")
        try:
            os.environ.pop("ERIS_BENCH_ATTRIBUTABLE", None)
            self.assertFalse(_apply_attributable_preset())
        finally:
            if saved is not None:
                os.environ["ERIS_BENCH_ATTRIBUTABLE"] = saved


if __name__ == "__main__":
    unittest.main()
