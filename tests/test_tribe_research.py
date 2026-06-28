"""§A2 + §5.4: the two-cycle research engine reasons across specialists, runs a second
gap-driven cycle, and canonizes a citation-grounded thought — stripping any claim that
cites a source that does not resolve. Offline, stub retriever + model, deterministic."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import tempfile
import unittest

from eris.computation.activations import BVec
from eris.tribe.specialists import TRIBE
from eris.tribe.research import run_two_cycle_research, _ground_citations
from eris.memory.thought_stream import ThoughtStream

GOAL = BVec(B=0.4, F=0.5, E=0.4, C=0.4, D=0.2, S=0.3)


def _retriever(query):
    return [f"Source about {query[:30]} — fact A.", "Second source — fact B."]


class TestTribeResearch(unittest.TestCase):
    def test_two_cycle_runs_and_multiple_specialists_contribute(self):
        def model(prompt):
            # domain reasoning + a named gap so cycle 2 fires; cite a real source
            if "Kairos" in prompt:
                return "Integrated view grounded in [s:0].\n- gap: long-term dynamics unclear"
            return "A specific domain analysis grounded in [s:0], adding non-obvious structure."
        active = TRIBE[:4]
        res = run_two_cycle_research("phase transitions", retriever=_retriever, model=model,
                                     specialists=active, goal_bvec=GOAL)
        self.assertEqual(res.n_active, 4)
        self.assertGreaterEqual(res.n_contributors, 2)     # the hive, not one voice
        self.assertEqual(res.cycles, 2)                    # a gap drove a second cycle
        self.assertTrue(res.synthesis)

    def test_unciteable_claim_is_stripped_at_canonize(self):
        # §5.4: a claim citing a source that does not exist must be stripped, not shipped.
        def model(prompt):
            if "Final synthesis" in prompt:
                return ("Supported claim grounded in [s:0]. Fabricated claim grounded in [s:99].")
            if "Kairos" in prompt:
                return "Synthesis [s:0]."
            return "Domain analysis [s:0]."
        res = run_two_cycle_research("topic", retriever=_retriever, model=model,
                                     specialists=TRIBE[:3], goal_bvec=GOAL)
        self.assertGreaterEqual(res.stripped_claims, 1)
        self.assertNotIn("[s:99]", res.synthesis)
        self.assertIn("[s:0]", res.synthesis)

    def test_ground_citations_strips_only_unresolved(self):
        text = "Good claim [s:0]. Good [s:1]. Bad claim [s:42]."
        out, n = _ground_citations(text, n_sources=2)
        self.assertEqual(n, 1)
        self.assertIn("[s:0]", out); self.assertIn("[s:1]", out)
        self.assertNotIn("[s:42]", out)

    def test_ground_citations_keeps_valid_claim_in_mixed_sentence(self):
        # mixed valid+fabricated cite in one sentence → strip the bad token, KEEP the claim
        out, n = _ground_citations("Source shows X [s:0] but also [s:99] elsewhere.", n_sources=1)
        self.assertIn("[s:0]", out)
        self.assertNotIn("[s:99]", out)
        self.assertIn("Source shows X", out)        # the valid claim survived

    def test_ground_citations_recognizes_citation_forms(self):
        # models cite in many forms: [s:0], (s:1), bare s:2, and ranges (s:1-3)
        from eris.tribe.research import _cited_ids
        self.assertEqual(_cited_ids("a [s:0] b (s:1) c s:2"), {0, 1, 2})
        self.assertEqual(_cited_ids("see (s:1-3) and [s:5]"), {1, 2, 3, 5})

    def test_uncited_honest_reasoning_is_kept_not_nuked(self):
        # an absence/negative finding ("the sources don't discuss X") is honest, not a
        # hallucination — it must survive grounding (the bug that emptied a whole synthesis)
        text = "The sources describe a LaTeX template and never mention resonance at all."
        out, n = _ground_citations(text, n_sources=2)
        self.assertEqual(n, 0)
        self.assertIn("resonance", out)             # kept

    def test_ungrounded_synthesis_is_visible_but_not_canonized(self):
        path = os.path.join(tempfile.mkdtemp(), "t.jsonl")
        ts = ThoughtStream(path=path)
        # model never cites a real source → nothing resolves → must NOT pollute memory,
        # but the synthesis text must still be returned so it's inspectable
        res = run_two_cycle_research("topic", retriever=_retriever,
                                     model=lambda p: "The sources do not cover this topic.",
                                     specialists=TRIBE[:2], goal_bvec=GOAL, thought_stream=ts)
        self.assertTrue(res.synthesis)              # visible, not empty
        self.assertIsNone(res.thought_id)           # NOT canonized (no grounded support)
        self.assertEqual(ts.size(), 0)

    def test_canonized_thought_has_embedding_when_embed_fn_given(self):
        import numpy as np
        path = os.path.join(tempfile.mkdtemp(), "t.jsonl")
        ts = ThoughtStream(path=path)
        res = run_two_cycle_research("emergence", retriever=_retriever,
                                     model=lambda p: "Grounded [s:0].", specialists=TRIBE[:2],
                                     goal_bvec=GOAL, thought_stream=ts,
                                     embed_fn=lambda t: np.array([0.1, 0.2, 0.3], dtype=np.float32))
        stored = ts.get(res.thought_id)
        self.assertIsNotNone(stored.embedding)       # retrievable, not invisible

    def test_canonizes_into_thought_stream(self):
        path = os.path.join(tempfile.mkdtemp(), "thoughts.jsonl")
        ts = ThoughtStream(path=path)
        res = run_two_cycle_research("emergence", retriever=_retriever,
                                     model=lambda p: "Grounded synthesis [s:0].",
                                     specialists=TRIBE[:2], goal_bvec=GOAL, thought_stream=ts)
        self.assertIsNotNone(res.thought_id)
        self.assertEqual(ts.get(res.thought_id).topic, "emergence")

    def test_gaps_parsing_skips_section_headers(self):
        from eris.tribe.research import _gaps_from
        text = ("**Open GAPS**\n- empirical validation is missing\n"
                "- the matrix sign convention is unspecified\nGaps\n- another real gap here")
        gaps = _gaps_from(text)
        self.assertNotIn("Open GAPS", " | ".join(gaps))   # header not treated as a gap
        self.assertTrue(any("empirical validation" in g for g in gaps))
        self.assertTrue(any("sign convention" in g for g in gaps))

    def test_parallel_map_fn_matches_sequential_and_preserves_order(self):
        # specialist reasoning via a concurrent map_fn must produce the same findings as the
        # sequential default (order preserved, all contributors present, hub populated)
        import concurrent.futures as cf
        def model(prompt):
            return "Domain analysis grounded in [s:0]."
        def par_map(fn, items):
            items = list(items)
            with cf.ThreadPoolExecutor(max_workers=4) as ex:
                return list(ex.map(fn, items))
        seq = run_two_cycle_research("topic", retriever=_retriever, model=model,
                                     specialists=TRIBE[:4], goal_bvec=GOAL)
        par = run_two_cycle_research("topic", retriever=_retriever, model=model,
                                     specialists=TRIBE[:4], goal_bvec=GOAL, map_fn=par_map)
        self.assertEqual(par.n_active, seq.n_active)
        self.assertEqual(par.n_contributors, seq.n_contributors)
        self.assertGreaterEqual(par.n_contributors, 2)

    def test_empty_sources_short_circuits_hive_no_scaffolding(self):
        # 0 sources → ONE honest refusal, NOT 2000 chars of [s:nil] scaffolding from 5
        # specialists. The model must never even be called (nothing to ground in).
        calls = {"n": 0}
        def model(prompt):
            calls["n"] += 1
            return "fabricated structure [s:nil]"
        res = run_two_cycle_research("obscure topic", retriever=lambda q: [], model=model,
                                     specialists=TRIBE[:5], goal_bvec=GOAL)
        self.assertEqual(res.n_sources, 0)
        self.assertEqual(res.cycles, 0)
        self.assertIsNone(res.thought_id)              # never canonized
        self.assertNotIn("[s:nil]", res.synthesis)     # no fabricated citation token
        self.assertLess(len(res.synthesis), 400)       # a sentence, not a scaffold
        self.assertEqual(calls["n"], 0)                # didn't spin up the specialists

    def test_empty_sources_short_circuits_single_pass(self):
        res = run_two_cycle_research("obscure topic", retriever=lambda q: [],
                                     model=lambda p: "should not be called",
                                     specialists=TRIBE[:2], goal_bvec=GOAL, single_pass=True)
        self.assertEqual(res.n_sources, 0)
        self.assertLess(len(res.synthesis), 400)

    def test_single_pass_control_is_one_call_no_specialists(self):
        # the A/B control: single-pass RAG summary — no specialists, no second cycle, but the
        # same retrieval + citation grounding so the delta isolates what the hive adds
        calls = {"n": 0}
        def model(prompt):
            calls["n"] += 1
            return "Single-pass summary grounded in [s:0]."
        res = run_two_cycle_research("topic", retriever=_retriever, model=model,
                                     specialists=TRIBE[:4], goal_bvec=GOAL, single_pass=True)
        self.assertEqual(calls["n"], 1)                 # exactly one model call
        self.assertEqual(res.n_active, 0)               # no specialists ran
        self.assertEqual(res.cycles, 0)
        self.assertGreater(res.n_sources, 0)            # retrieval still happened
        self.assertIn("[s:0]", res.synthesis)

    def test_gaps_parsing_ignores_body_bullets_takes_only_gap_section(self):
        # the bug from David's BLECD run: body findings (bulleted) were captured as "gaps",
        # bloating cycle-2 retrieval with whole paragraphs. Only the post-header section counts.
        from eris.tribe.research import _gaps_from
        text = ("**What BLECD says**\n"
                "- Criticality is one of six domains in the interaction matrix.\n"
                "- The off-diagonal coefficient is the tuning lever.\n"
                "**Open Gaps**\n"
                "- no empirical mapping from symbols to control knobs\n"
                "- sign convention is unspecified")
        gaps = _gaps_from(text)
        self.assertTrue(any("empirical mapping" in g for g in gaps))
        self.assertTrue(any("sign convention" in g for g in gaps))
        self.assertFalse(any("six domains" in g for g in gaps))      # body finding, NOT a gap
        self.assertFalse(any("tuning lever" in g for g in gaps))

    def test_gaps_parsing_unwraps_markdown_table_rows(self):
        # the BLECD run leaked raw '| col | col | gap |' rows; keep the gap cell, not the row
        from eris.tribe.research import _gaps_from
        text = ("**Open Gaps**\n"
                "| Lens | Claim | Uncertainty |\n"
                "| --- | --- | --- |\n"
                "| Anthropos | smooth ramp | whether it protects against failure is unverified |")
        gaps = _gaps_from(text)
        self.assertTrue(any("unverified" in g for g in gaps))
        self.assertFalse(any(g.startswith("|") for g in gaps))     # no raw table rows
        self.assertFalse(any("---" in g for g in gaps))            # no separator rows

    def test_gaps_parsing_falls_back_to_gap_language_without_header(self):
        from eris.tribe.research import _gaps_from
        text = ("Integrated view grounded in [s:0].\n"
                "- gap: long-term dynamics remain unclear")
        gaps = _gaps_from(text)
        self.assertTrue(any("long-term dynamics" in g for g in gaps))

    def test_elos_falsifies_when_active(self):
        seen = {"elos": False}
        def model(prompt):
            if "Elos" in prompt and "FALSIFY" in prompt:
                seen["elos"] = True
                return "Weakest claim: the causal direction is unsupported."
            return "Analysis [s:0]."
        res = run_two_cycle_research("topic", retriever=_retriever, model=model,
                                     specialists=[TRIBE[3]], goal_bvec=GOAL)  # Elos
        self.assertTrue(seen["elos"])
        self.assertTrue(res.elos_critique)


if __name__ == "__main__":
    unittest.main()
