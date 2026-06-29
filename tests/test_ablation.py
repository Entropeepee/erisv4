"""The physics-value ablation harness: the inference-grounding mode (arm D) and the pre-registered
prediction evaluator. Offline — stub retriever + model, deterministic, no network or real LLM."""
import os
os.environ.setdefault("ERIS_GPU", "0")
os.environ.setdefault("ERIS_EMBEDDINGS", "off")

import unittest

from eris.computation.activations import BVec
from eris.tribe.specialists import TRIBE
from eris.tribe.research import run_two_cycle_research
from eris.experiments.benchmarks.ablation import evaluate_predictions

GOAL = BVec(B=0.4, F=0.5, E=0.4, C=0.4, D=0.2, S=0.3)


def _retriever(query):
    return [f"Source about {query[:30]} — fact A.", "Second source — fact B."]


class TestInferenceMode(unittest.TestCase):
    """ERIS_HIVE_TASK=inference (arm D): the canon/synth prompts permit the supported inference and
    the Elos strike-the-weakest-claim pass is SKIPPED. Strict grounding stays the default."""

    def _run(self, inference_mode):
        prompts = []

        def model(prompt):
            prompts.append(prompt)
            if "Kairos" in prompt:
                return "Integrated view grounded in [s:0].\n- gap: the motive is unclear"
            if "As Elos" in prompt:
                return "Weakest claim: X. Ruling: strike."
            if "Final synthesis" in prompt:
                return "Final answer grounded in [s:0]."
            return "Domain analysis grounded in [s:0]."

        elos = [s for s in TRIBE if s.id == "elos"]
        self.assertTrue(elos, "tribe must contain an 'elos' specialist for this test")
        active = elos + [s for s in TRIBE if s.id != "elos"][:3]
        run_two_cycle_research("why does X feel Y", retriever=_retriever, model=model,
                               specialists=active, goal_bvec=GOAL, inference_mode=inference_mode)
        return prompts

    def test_strict_mode_runs_elos_and_strict_grounding(self):
        joined = "\n".join(self._run(inference_mode=False))
        self.assertIn("As Elos (adversarial)", joined)                       # Elos ran
        self.assertIn("assert nothing the sources don't support", joined)    # strict canon prompt
        self.assertIn("mark what the sources SUPPORT vs what is inference", joined)  # strict synth

    def test_inference_mode_skips_elos_and_permits_inference(self):
        joined = "\n".join(self._run(inference_mode=True))
        self.assertNotIn("As Elos (adversarial)", joined)                    # Elos SKIPPED
        self.assertIn("best-supported INFERENCE the sources IMPLY", joined)  # inference synth
        self.assertIn("do NOT omit a well-supported inference", joined)      # inference canon
        self.assertNotIn("assert nothing the sources don't support", joined)  # strict text gone

    def test_default_is_strict(self):
        # the parameter defaults to False — nothing changes unless explicitly opted in
        joined = "\n".join(self._run(inference_mode=False))
        self.assertNotIn("best-supported INFERENCE the sources IMPLY", joined)


class TestPredictionEvaluator(unittest.TestCase):
    def test_b_eq_c_pass_and_csel_differs(self):
        acc = {"A_bare": 0.0, "B_resonance_off": 0.0, "C_resonance_on": 0.0,
               "D_inference": 0.67, "C_sel_cap6": 0.0}
        answers = {
            "A_bare": {"q1": "C", "q2": "C"},
            "B_resonance_off": {"q1": "C", "q2": "B"},
            "C_resonance_on": {"q1": "C", "q2": "B"},     # identical to B → B==C confirmed
            "D_inference": {"q1": "A", "q2": "B"},
            "C_sel_cap6": {"q1": "A", "q2": "B"},          # differs from C on q1 → rerank selected
        }
        out = evaluate_predictions(acc, answers, has_csel=True)
        self.assertEqual(out["B_eq_C"]["answer_match_rate_B_vs_C"], 1.0)
        self.assertTrue(out["B_eq_C"]["pass"])
        self.assertTrue(out["D_gt_rest"]["pass"])          # 0.67 > max(0,0,0)
        self.assertTrue(out["C_sel_neq_C"]["pass"])        # C-sel != C on q1

    def test_b_neq_c_and_d_not_winning_fail(self):
        acc = {"A_bare": 0.0, "B_resonance_off": 0.33, "C_resonance_on": 0.0, "D_inference": 0.0}
        answers = {"A_bare": {"q1": "D"}, "B_resonance_off": {"q1": "A"},
                   "C_resonance_on": {"q1": "B"}, "D_inference": {"q1": "C"}}
        out = evaluate_predictions(acc, answers, has_csel=False)
        self.assertEqual(out["B_eq_C"]["answer_match_rate_B_vs_C"], 0.0)
        self.assertFalse(out["B_eq_C"]["pass"])
        self.assertFalse(out["D_gt_rest"]["pass"])         # 0.0 is not > 0.33
        self.assertNotIn("C_sel_neq_C", out)               # skipped when no C-sel

    def test_missing_arm_is_none_not_crash(self):
        out = evaluate_predictions({}, {}, has_csel=False)
        self.assertIsNone(out["B_eq_C"]["pass"])           # no data → None, not a false pass


if __name__ == "__main__":
    unittest.main()
