"""
Remediation v4.1 regression tests (Tiers 0–4).
Each test pins one behavior the ERIS_V4 remediation introduced.
"""
import asyncio
import os
import tempfile

import numpy as np
import pytest


# ── Tier 1 ───────────────────────────────────────────────────────────────
def test_regime_is_self_calibrating_and_reachable():
    from eris.field.pde import FractalField
    f = FractalField(size=16)
    seen = set()
    for i in range(30):
        f.seed_from_text(f"input number {i} about phase transitions")
        f.run(4)
        seen.add(f.detect_regime())
    # Regime must no longer be pinned to a single value (was always 'elastic').
    assert seen - {"warmup"}, seen
    assert seen <= {"warmup", "elastic", "plastic", "transfixed"}


def test_empty_confidence_rename_with_alias():
    from eris.executive.workspace import TransfixionDetector
    td = TransfixionDetector()
    assert hasattr(td, "check_empty_confidence_signature")
    # deprecated name still callable (back-compat alias)
    assert hasattr(td, "check_hallucination_signature")


def test_clone_no_longer_crashes():
    from eris.field.pde import FractalField
    f = FractalField(size=16)
    f.seed_from_text("hello"); f.run(3)
    c = f.clone()  # used to raise AttributeError on _C_hist
    assert c.size == f.size


# ── Tier 2 ───────────────────────────────────────────────────────────────
def test_stm_novelty_is_direction_aware():
    from eris.memory.tiers import ShortTermMemory, MemoryRecord
    from eris.computation.activations import BVec
    stm = ShortTermMemory()
    stm.store(MemoryRecord(text="a", bvec=BVec(B=1, F=0, E=0, C=0, D=0, S=0)))
    # same total activation, different direction -> should still be novel
    nov = stm.novelty(BVec(B=0, F=0, E=0, C=0, D=0, S=1))
    assert nov > 0.9


def test_consolidate_prunes_and_reports():
    from eris.memory.tiers import MemorySystem
    from eris.computation.activations import BVec
    d = tempfile.mkdtemp()
    m = MemorySystem(data_dir=d)
    for i in range(12):
        m.store_turn(text=f"t{i}", bvec=BVec(B=0.2, F=0.3, E=0.4, C=0.1, D=0.0, S=0.1))
    out = m.consolidate()
    assert "mtm_pruned" in out


# ── Tier 3 ───────────────────────────────────────────────────────────────
def test_specialist_findings_are_field_bids_not_echoes():
    from eris.tribe.specialists import make_field_finding, TRIBE
    from eris.computation.activations import BVec
    field_bvec = BVec(B=0.5, F=0.4, E=0.3, C=0.6, D=0.1, S=0.2)
    f0 = make_field_finding(TRIBE[0], field_bvec)
    f3 = make_field_finding(TRIBE[3], field_bvec)
    assert "Analysis of" not in f0.content and "Focused on" not in f0.content
    assert f0.content != f3.content                 # bids differ across specialists
    assert f0.bvec.as_array().tolist() != field_bvec.as_array().tolist()  # projected


# ── Tier 4 ───────────────────────────────────────────────────────────────
def test_embeddings_are_deterministic_and_discriminating():
    from eris.knowledge.embeddings import get_embedding
    a = get_embedding("Kuramoto critical coupling")
    b = get_embedding("Kuramoto critical coupling")
    c = get_embedding("a recipe for sourdough bread")
    assert np.allclose(a, b)                          # deterministic
    assert float(a @ b) > float(a @ c) + 0.5          # discriminating


def test_ask_expert_is_dormant_without_key():
    import eris.knowledge.ask_expert as ae
    old = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        assert ae.is_available() is False
        assert ae.ask("anything") is None             # returns instantly, no hang
    finally:
        if old is not None:
            os.environ["ANTHROPIC_API_KEY"] = old


def test_field_resonance_self_match():
    from eris.retrieval.field_interference import field_resonance
    phi = np.random.rand(16, 16); theta = np.random.rand(16, 16) * 6.28
    r_self = field_resonance(phi, theta, phi, theta)
    other = np.random.rand(16, 16)
    r_other = field_resonance(phi, theta, other, theta * 0 + 3.14)
    assert r_self > r_other


def test_research_bundle_shape():
    from eris.knowledge.research import ResearchBundle
    b = ResearchBundle(query="x")
    assert b.grounding == "" and b.sources == [] and b.used_expert is False


# ── Tier 0 + integration ─────────────────────────────────────────────────
def test_full_turn_runs_without_llm_backend():
    from eris.orchestrator import ErisOrchestrator
    d = tempfile.mkdtemp()
    orch = ErisOrchestrator(field_size=16, data_dir=d)
    res = asyncio.run(orch.process("Tell me about emergence"))
    v = orch.get_vitals()
    assert "dissonance" in v and "dCdX" in v          # Tier 1.4 split
    assert res.specialist_source                       # a field bid won
    assert hasattr(orch, "_router_gate")               # Tier 0 SGT router gate


# ── Tier 6: sine-aware resonant retrieval ────────────────────────────────
def test_resonant_retrieval_returns_cosine_and_sine_sets():
    """cosine (elastic) returns the near-duplicate; sine (plastic) returns the
    coupled-but-unresolved 'teacher' — not the near-duplicate, not noise."""
    import tempfile
    from eris.memory.tiers import MemorySystem, MemoryRecord
    from eris.computation.activations import BVec
    m = MemorySystem(data_dir=tempfile.mkdtemp())
    q = BVec(B=0.2, F=0.2, E=0.7, C=0.7, D=0.1, S=0.1)
    m.ltm.store(MemoryRecord(text="ALIGNED", bvec=BVec(B=0.2, F=0.2, E=0.7, C=0.7, D=0.1, S=0.1)))
    m.ltm.store(MemoryRecord(text="TENSION", bvec=BVec(B=0.8, F=0.2, E=0.5, C=0.2, D=0.8, S=0.3)))
    m.ltm.store(MemoryRecord(text="UNRELATED", bvec=BVec(B=0.0, F=0.0, E=0.0, C=0.0, D=0.0, S=0.9)))
    aligned, tension = m.retrieve_resonant(q, top_k=1, tension_k=1)
    assert aligned and aligned[0].text == "ALIGNED"
    assert tension and tension[0].text == "TENSION"   # sine surfaces the teacher
