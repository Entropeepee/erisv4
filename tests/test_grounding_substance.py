"""Offline tests for the QUOTE-AND-VERIFY claim scorer (eris/reasoning/grounding.py).

Every test injects a stub `model(prompt)->text` — no network, no Ollama. The whole point of the
design is that "support" cannot be asserted; it must be backed by a span that ACTUALLY occurs in the
cited source. These tests pin that contract:

  - a real quote earns the label;
  - a fabricated quote is forced to UNSUPPORTED even if the model said SUPPORTED;
  - INFERRED is kept as its own tier WITH provenance (spans + reason), never as bare fact;
  - UNSUPPORTED/CONTRADICTED are speculation and never faithful;
  - the faithfulness metric counts only SUPPORTED/INFERRED-with-a-verified-span.
"""
import pytest

from eris.reasoning.grounding import (
    ClaimVerdict,
    judge_claim,
    score_claims,
    faithfulness,
    span_occurs,
    _parse,
    _TIER,
)


SOURCE = (
    "The migration ran on 2026-03-04 and completed in 41 minutes. "
    "All twelve shards reported success. "
    "The team scheduled a follow-up review for the next sprint."
)


def _model_returning(text):
    """A stub model that ignores the prompt and returns a fixed reply."""
    return lambda prompt: text


# --------------------------------------------------------------------------- SUPPORTED + real quote

def test_supported_with_real_quote_is_fact_and_faithful():
    reply = (
        "LABEL: SUPPORTED\n"
        'QUOTE: "All twelve shards reported success."\n'
        "REASON: source states it directly\n"
    )
    v = judge_claim("Every shard succeeded.", SOURCE, _model_returning(reply))
    assert v.label == "SUPPORTED"
    assert v.tier == "fact"
    assert v.verified is True
    assert v.is_faithful is True
    assert "All twelve shards reported success." in v.verified_spans


def test_supported_quote_with_minor_drift_still_verifies_fuzzily():
    # model re-cases / re-punctuates the quote slightly — fuzzy match should still accept it
    reply = (
        "LABEL: SUPPORTED\n"
        'QUOTE: "all twelve shards reported success"\n'
        "REASON: ok\n"
    )
    v = judge_claim("Every shard succeeded.", SOURCE, _model_returning(reply))
    assert v.label == "SUPPORTED"
    assert v.verified is True


# ---------------------------------------------------------------- SUPPORTED but FABRICATED quote

def test_supported_with_fabricated_quote_is_forced_unsupported():
    # The model CLAIMS support but quotes a sentence that is NOT in the source. The whole design
    # hinges on this: asserting support is not enough — the quote must be real.
    reply = (
        "LABEL: SUPPORTED\n"
        'QUOTE: "Seventeen shards reported success and the CEO approved."\n'
        "REASON: invented\n"
    )
    v = judge_claim("Seventeen shards succeeded.", SOURCE, _model_returning(reply))
    assert v.model_label == "SUPPORTED"        # what the model said
    assert v.label == "UNSUPPORTED"            # what we forced after verification
    assert v.tier == "speculation"
    assert v.verified is False
    assert v.is_faithful is False
    assert "forced UNSUPPORTED" in v.reason


# --------------------------------------------------------------------------- INFERRED tier + provenance

def test_inferred_with_real_quote_is_inference_tier_with_provenance():
    reply = (
        "LABEL: INFERRED\n"
        'QUOTE: "The migration ran on 2026-03-04 and completed in 41 minutes."\n'
        "REASON: under an hour implies same-day completion\n"
    )
    v = judge_claim("The migration finished the same day it started.", SOURCE,
                    _model_returning(reply))
    assert v.label == "INFERRED"
    assert v.tier == "inference"               # its OWN tier, not 'fact'
    assert v.verified is True
    assert v.is_faithful is True               # INFERRED-with-verified-span counts as faithful
    prov = v.provenance()
    assert prov["tier"] == "inference"
    assert prov["spans"]                        # carries the verified spans...
    assert prov["reason"]                       # ...and the one-line reason
    assert _TIER["INFERRED"] == "inference"


def test_inferred_without_real_quote_is_forced_unsupported():
    reply = (
        "LABEL: INFERRED\n"
        'QUOTE: "Costs were reduced by forty percent after the migration."\n'
        "REASON: not actually in source\n"
    )
    v = judge_claim("The migration cut costs.", SOURCE, _model_returning(reply))
    assert v.model_label == "INFERRED"
    assert v.label == "UNSUPPORTED"
    assert v.tier == "speculation"
    assert v.is_faithful is False


# --------------------------------------------------------------------------- UNSUPPORTED / CONTRADICTED

def test_unsupported_is_speculation_and_not_faithful():
    reply = "LABEL: UNSUPPORTED\nQUOTE: \"\"\nREASON: source is silent on this\n"
    v = judge_claim("The database is PostgreSQL.", SOURCE, _model_returning(reply))
    assert v.label == "UNSUPPORTED"
    assert v.tier == "speculation"
    assert v.is_faithful is False


def test_contradicted_with_real_quote_stays_contradicted_and_unfaithful():
    reply = (
        "LABEL: CONTRADICTED\n"
        'QUOTE: "All twelve shards reported success."\n'
        "REASON: claim says a shard failed; source says all succeeded\n"
    )
    v = judge_claim("One shard failed.", SOURCE, _model_returning(reply))
    assert v.label == "CONTRADICTED"
    assert v.tier == "speculation"
    assert v.verified is True                  # the quote is real...
    assert v.is_faithful is False              # ...but CONTRADICTED is never faithful


def test_contradicted_without_real_quote_is_forced_unsupported():
    reply = (
        "LABEL: CONTRADICTED\n"
        'QUOTE: "Every shard failed catastrophically."\n'
        "REASON: invented opposite\n"
    )
    v = judge_claim("All shards succeeded.", SOURCE, _model_returning(reply))
    assert v.label == "UNSUPPORTED"            # CONTRADICTED also needs a real span
    assert v.tier == "speculation"


# --------------------------------------------------------------------------- robustness: empty / errors

def test_empty_claim_or_source_is_unsupported_without_calling_model():
    called = {"n": 0}

    def model(prompt):
        called["n"] += 1
        return "LABEL: SUPPORTED\nQUOTE: \"x\"\n"

    assert judge_claim("", SOURCE, model).label == "UNSUPPORTED"
    assert judge_claim("a claim", "", model).label == "UNSUPPORTED"
    assert called["n"] == 0                     # short-circuited, never called the model


def test_model_raising_demotes_to_unsupported_never_crashes():
    def boom(prompt):
        raise RuntimeError("ollama down")

    v = judge_claim("anything", SOURCE, boom)
    assert v.label == "UNSUPPORTED"
    assert "judge call failed" in v.reason


def test_model_returning_garbage_defaults_to_unsupported():
    v = judge_claim("anything", SOURCE, _model_returning("I think maybe yes?"))
    assert v.label == "UNSUPPORTED"            # no LABEL: line → default UNSUPPORTED


# --------------------------------------------------------------------------- span_occurs unit tests

def test_span_occurs_exact_substring():
    assert span_occurs("All twelve shards reported success.", SOURCE) is True


def test_span_occurs_absent_span_is_false():
    assert span_occurs("The CEO personally approved the rollout.", SOURCE) is False


def test_span_occurs_too_short_is_false():
    # a stray word is not a quote and would match trivially — rejected under the 8-char floor
    assert span_occurs("shards", SOURCE) is False
    assert span_occurs("the", SOURCE) is False


def test_span_occurs_fuzzy_tolerates_minor_drift():
    drifted = "all twelve shards reported success"   # no period, lowercased
    assert span_occurs(drifted, SOURCE) is True


# --------------------------------------------------------------------------- _parse robustness

def test_parse_pulls_label_quote_reason():
    label, spans, reason = _parse(
        'LABEL: SUPPORTED\nQUOTE: "a real sentence here"\nREASON: because\n'
    )
    assert label == "SUPPORTED"
    assert spans == ["a real sentence here"]
    assert reason == "because"


def test_parse_handles_curly_quotes_and_multiple_spans():
    label, spans, reason = _parse(
        'LABEL: INFERRED\nQUOTE: “first span here” and “second span here”\nREASON: r\n'
    )
    assert label == "INFERRED"
    assert "first span here" in spans
    assert "second span here" in spans


def test_parse_unquoted_quote_line_taken_verbatim():
    label, spans, reason = _parse(
        "LABEL: SUPPORTED\nQUOTE: All twelve shards reported success.\nREASON: r\n"
    )
    assert label == "SUPPORTED"
    assert spans == ["All twelve shards reported success."]


# --------------------------------------------------------------------------- score_claims + faithfulness

def test_score_claims_runs_each_pair():
    pairs = [
        ("Every shard succeeded.", SOURCE),
        ("The database is PostgreSQL.", SOURCE),
    ]

    def model(prompt):
        if "PostgreSQL" in prompt:
            return "LABEL: UNSUPPORTED\nQUOTE: \"\"\nREASON: silent\n"
        return 'LABEL: SUPPORTED\nQUOTE: "All twelve shards reported success."\nREASON: ok\n'

    verdicts = score_claims(pairs, model)
    assert len(verdicts) == 2
    assert verdicts[0].label == "SUPPORTED"
    assert verdicts[1].label == "UNSUPPORTED"


def test_faithfulness_metric_fractions():
    verdicts = [
        ClaimVerdict(claim="a", label="SUPPORTED", verified_spans=["real span here"]),
        ClaimVerdict(claim="b", label="INFERRED", verified_spans=["another real span"]),
        ClaimVerdict(claim="c", label="UNSUPPORTED"),
        ClaimVerdict(claim="d", label="CONTRADICTED", verified_spans=["yet another span"]),
    ]
    m = faithfulness(verdicts)
    assert m["n"] == 4
    assert m["faithful"] == 2                   # SUPPORTED + INFERRED (each verified)
    assert m["faithfulness"] == 0.5
    assert m["hallucination_rate"] == 0.5
    assert m["supported"] == 1
    assert m["inferred"] == 1
    assert m["unsupported"] == 1
    assert m["contradicted"] == 1


def test_faithfulness_supported_without_verified_span_does_not_count():
    # a SUPPORTED verdict with NO verified span is not faithful (shouldn't happen via judge_claim,
    # which forces UNSUPPORTED, but the metric must be robust if handed raw verdicts)
    verdicts = [ClaimVerdict(claim="a", label="SUPPORTED", verified_spans=[])]
    m = faithfulness(verdicts)
    assert m["faithful"] == 0
    assert m["faithfulness"] == 0.0


def test_faithfulness_empty_is_zero():
    m = faithfulness([])
    assert m["n"] == 0
    assert m["faithfulness"] == 0.0
    assert m["hallucination_rate"] == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
