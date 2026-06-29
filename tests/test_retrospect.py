"""Retrospective metacognition + the two upstream guards (grounding check,
deep-reasoning input-gate). The load-bearing property: no fabricated past
thoughts — every [t:id] must resolve to a real reviewed thought-stream entry."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import json
import tempfile
import unittest

from eris.memory.thought_stream import ThoughtStream, link_and_store
from eris.metacognition.retrospect import (
    gather_retrospective_material, build_grounded_context, unresolved_ids,
    run_retrospective,
)
from eris.reasoning.calibration import (
    verify_grounding, is_synthesis_task, _looks_conversational,
)


def _stream():
    return ThoughtStream(path=os.path.join(tempfile.mkdtemp(), "t.jsonl"))


def _embed(_text):
    return [0.1, 0.2, 0.3]


class TestScopingAndContext(unittest.TestCase):
    def test_must_be_scoped(self):
        s = _stream()
        with self.assertRaises(ValueError):
            gather_retrospective_material(s)              # no topic, no since

    def test_topic_scoped_uses_active_trajectory(self):
        s = _stream()
        link_and_store(s, "conformity", "plastic", "first take")
        link_and_store(s, "conformity", "elastic", "second take")
        items = gather_retrospective_material(s, topic="conformity")
        self.assertEqual([t.text for t in items], ["first take", "second take"])

    def test_context_carries_real_ids(self):
        s = _stream()
        a = link_and_store(s, "x", "plastic", "thought a")
        ctx = build_grounded_context(s.by_topic("x"))
        self.assertEqual(ctx[0]["id"], a.id)


class TestUnresolvedIds(unittest.TestCase):
    def test_detects_fabricated_citation(self):
        bad = unresolved_ids("I built on [t:abc123] and also [t:deadbeef].", ["abc123"])
        self.assertEqual(bad, {"deadbeef"})

    def test_all_valid(self):
        self.assertEqual(unresolved_ids("only [t:abc123] here", ["abc123", "x"]), set())


class TestRunRetrospective(unittest.TestCase):
    def _seed(self):
        s = _stream()
        a = link_and_store(s, "conformity", "plastic", "Norms feel like a current.")
        b = link_and_store(s, "conformity", "elastic", "Breaking a norm is self-assertion.")
        return s, a, b

    def test_needs_at_least_two(self):
        s = _stream()
        link_and_store(s, "conformity", "plastic", "only one")
        self.assertIsNone(run_retrospective(s, "conformity", lambda p: "{}", _embed))

    def test_happy_path_stores_linked_synthesis(self):
        s, a, b = self._seed()
        before = s.size()
        payload = json.dumps({
            "movement": f"My view moved from [t:{a.id}] to [t:{b.id}].",
            "now_grounded": [{"text": "norms shape behavior", "tier": "fact",
                              "source_id": a.id}],
            "still_open": [{"text": "is non-conformity always growth?", "tier": "bridge"}],
            "mind_changes": [],
        })
        retro = run_retrospective(s, "conformity", lambda p: payload, _embed,
                                  regime="elastic")
        self.assertIsNotNone(retro)
        self.assertEqual(set(retro.reviewed_ids), {a.id, b.id})
        # A new synthesis thought was stored, linked to exactly the reviewed ids.
        self.assertEqual(s.size(), before + 1)
        synth = s.all()[-1]
        self.assertEqual(set(synth.prior), {a.id, b.id})

    def test_grounding_check_demotes_unsupported_fact(self):
        s, a, b = self._seed()
        payload = json.dumps({
            "movement": f"See [t:{a.id}].",
            "now_grounded": [{"text": "the patent proves it", "tier": "fact",
                              "source_id": "ghost99"}],   # id NOT in reviewed set
            "still_open": [], "mind_changes": [],
        })
        retro = run_retrospective(s, "conformity", lambda p: payload, _embed)
        c = retro.now_grounded[0]
        self.assertEqual(c["tier"], "speculation")        # demoted
        self.assertIn("not found", c["note"])

    def test_fabricated_id_regenerates_then_strips(self):
        s, a, b = self._seed()
        calls = []
        ghost = "deadbeef0000"                            # hex, but not a real id
        def gen(prompt):
            calls.append(prompt)
            # Always cites a ghost id; first sentence ties to it, second is clean.
            return json.dumps({
                "movement": f"Recall [t:{ghost}] my old note. But [t:{a.id}] holds.",
                "now_grounded": [], "still_open": [], "mind_changes": [],
            })
        retro = run_retrospective(s, "conformity", gen, _embed)
        self.assertEqual(len(calls), 2)                   # regenerated once
        self.assertNotIn(ghost, retro.movement)           # unsupported sentence stripped
        self.assertIn(a.id, retro.movement)               # the real one survives

    def test_mind_change_supersedes_reviewed_id(self):
        s, a, b = self._seed()
        payload = json.dumps({
            "movement": f"Revised [t:{a.id}].",
            "now_grounded": [], "still_open": [],
            "mind_changes": [{"from_id": a.id,
                              "to_claim": "Norms are negotiable, not fixed.",
                              "why": "later reading"}],
        })
        run_retrospective(s, "conformity", lambda p: payload, _embed)
        active = [t.id for t in s.active_by_topic("conformity", limit=50)]
        self.assertNotIn(a.id, active)                    # superseded -> out of active
        self.assertIn(a.id, [t.id for t in s.by_topic("conformity", limit=50)])  # still history


class TestGroundingGuard(unittest.TestCase):
    # The cited source TEXT for the substance check. The claim under test is judged against this.
    SRC = {"a": "All twelve shards reported success during the migration."}

    @staticmethod
    def _judge(label, quote):
        """Stub local judge returning a fixed LABEL + QUOTE in the scorer's wire format."""
        return lambda prompt: f'LABEL: {label}\nQUOTE: "{quote}"\nREASON: stub\n'

    def test_fact_without_source_demoted(self):
        # (1) resolution: cited id doesn't exist in the reviewed set → demoted, no judge needed.
        c = verify_grounding({"text": "x", "tier": "fact", "source_id": "nope"}, {"a"})
        self.assertEqual(c["tier"], "speculation")

    def test_fact_with_real_support_kept(self):
        # (2) substance: id resolves AND the source SUPPORTS the claim (quote is verbatim) → fact.
        c = verify_grounding(
            {"text": "Every shard succeeded.", "tier": "fact", "source_id": "a"}, {"a", "b"},
            source_texts=self.SRC,
            model=self._judge("SUPPORTED", "All twelve shards reported success during the migration."))
        self.assertEqual(c["tier"], "fact")

    def test_fact_with_resolving_id_but_no_support_demoted(self):
        # THE false-confidence fix: the id resolves, but the source does NOT back the claim (the
        # judge's quote is fabricated → forced UNSUPPORTED) → demoted to speculation, never 'fact'.
        c = verify_grounding(
            {"text": "The migration cut costs by 40%.", "tier": "fact", "source_id": "a"}, {"a"},
            source_texts=self.SRC,
            model=self._judge("SUPPORTED", "Costs were cut by forty percent."))  # not in source
        self.assertEqual(c["tier"], "speculation")
        self.assertIn("does not support", c["note"])

    def test_inferred_kept_as_inference_tier_with_provenance(self):
        # source IMPLIES (not states) the claim → its own 'inference' tier, carrying provenance.
        c = verify_grounding(
            {"text": "The migration finished.", "tier": "fact", "source_id": "a"}, {"a"},
            source_texts=self.SRC,
            model=self._judge("INFERRED", "All twelve shards reported success during the migration."))
        self.assertEqual(c["tier"], "inference")
        self.assertTrue(c["provenance"]["spans"])

    def test_bridge_untouched(self):
        c = verify_grounding({"text": "x", "tier": "bridge", "source_id": "nope"}, {"a"})
        self.assertEqual(c["tier"], "bridge")


class TestInputGate(unittest.TestCase):
    def test_conversational_is_not_synthesis(self):
        self.assertFalse(is_synthesis_task("want me to write the spec?"))
        self.assertFalse(is_synthesis_task("should I add a test?"))
        self.assertTrue(_looks_conversational("can you do this?"))

    def test_real_synthesis_still_detected(self):
        # Must NOT regress the existing behavior.
        self.assertTrue(is_synthesis_task("how does the LNCS paper inform my patent?"))
        self.assertTrue(is_synthesis_task("compare X and Y"))
        self.assertTrue(is_synthesis_task("a simple question", named_sources=2))
        self.assertFalse(is_synthesis_task("what time is it?"))


if __name__ == "__main__":
    unittest.main()
