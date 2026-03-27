"""
Phase 2 Tests — Memory + Updated BLC Compiler
================================================
Run: cd eris_echo_v4 && python -m pytest tests/test_memory.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import tempfile
import pytest


class TestFieldCompiler:
    """Tests for the rewritten BLC that generates φ-θ seed geometries."""

    def test_no_contradiction_no_seeds(self):
        from eris.computation.activations import BVec
        from eris.field.compiler import compile_contradiction
        a = BVec(B=0.5, F=0.5, E=0.5, C=0.5, D=0.5, S=0.5)
        result = compile_contradiction(a, a, field_size=32)
        assert result.n_seeds == 0

    def test_contradiction_produces_field_seeds(self):
        from eris.computation.activations import BVec
        from eris.field.compiler import compile_contradiction
        a = BVec(B=0.9, F=0.1, E=0.1, C=0.1, D=0.1, S=0.1)
        b = BVec(B=0.1, F=0.9, E=0.9, C=0.9, D=0.9, S=0.9)
        result = compile_contradiction(a, b, field_size=64,
                                       sgt_threshold=1.0, sgt_mean=0.0, sgt_var=0.01)
        assert result.n_seeds > 0, "Strong contradiction should produce seeds"
        seed = result.seeds[0]
        assert seed.phi_patch.shape == (64, 64)
        assert seed.theta_patch.shape == (64, 64)
        assert seed.strength > 0

    def test_seeds_have_correct_gate_types(self):
        """C+D contradiction should produce XOR gate geometry."""
        from eris.computation.activations import BVec
        from eris.field.compiler import compile_contradiction, FieldGateType
        # Contradiction primarily in C and D
        a = BVec(B=0.5, F=0.5, E=0.5, C=0.1, D=0.1, S=0.5)
        b = BVec(B=0.5, F=0.5, E=0.5, C=0.9, D=0.9, S=0.5)
        result = compile_contradiction(a, b, field_size=32,
                                       sgt_threshold=0.5, sgt_mean=0.0, sgt_var=0.01)
        if result.n_seeds > 0:
            assert result.seeds[0].gate_type == FieldGateType.XOR

    def test_inject_seeds_modifies_field(self):
        """Injecting seeds should actually change the PDE field."""
        from eris.computation.activations import BVec
        from eris.field.pde import FractalField
        from eris.field.compiler import compile_contradiction, inject_seeds

        field = FractalField(size=32)
        phi_before = np.asarray(field.phi).copy()

        a = BVec(B=0.9, F=0.1, E=0.1, C=0.1, D=0.1, S=0.1)
        b = BVec(B=0.1, F=0.9, E=0.9, C=0.9, D=0.9, S=0.9)
        result = compile_contradiction(a, b, field_size=32,
                                       sgt_threshold=1.0, sgt_mean=0.0, sgt_var=0.01)
        inject_seeds(field, result)

        phi_after = np.asarray(field.phi)
        diff = np.sum(np.abs(phi_after - phi_before))
        assert diff > 0.01, "Injecting seeds should modify the field"

    def test_xor_geometry_has_dual_lobes(self):
        """XOR gate should create two distinct φ regions with phase opposition."""
        from eris.field.compiler import generate_xor_gate
        phi, theta = generate_xor_gate(64, 32, 32, 10, 0.5)
        # Should have nonzero phi in two regions
        assert phi.max() > 0.3
        # Left and right halves should both have phi
        left_energy = phi[:, :32].sum()
        right_energy = phi[:, 32:].sum()
        assert left_energy > 0 and right_energy > 0, "XOR should have dual lobes"

    def test_diode_geometry_is_asymmetric(self):
        """DIODE gate should have more φ on the forward side."""
        from eris.field.compiler import generate_diode_gate
        phi, theta = generate_diode_gate(64, 32, 32, 10, 0.5)
        # Right side (forward) should have more energy than left
        left = phi[:, :32].sum()
        right = phi[:, 32:].sum()
        assert right > left * 1.2, f"DIODE should be asymmetric: L={left:.2f}, R={right:.2f}"


class TestMemoryTiers:
    """Tests for the three-tier memory system."""

    def test_stm_stores_and_retrieves(self):
        from eris.memory.tiers import ShortTermMemory, MemoryRecord
        from eris.computation.activations import BVec
        stm = ShortTermMemory(capacity=5)
        stm.store(MemoryRecord(text="hello", bvec=BVec(E=0.5)))
        stm.store(MemoryRecord(text="world", bvec=BVec(C=0.7)))
        assert stm.size == 2
        recent = stm.get_recent(1)
        assert recent[0].text == "world"

    def test_stm_drops_oldest(self):
        from eris.memory.tiers import ShortTermMemory, MemoryRecord
        from eris.computation.activations import BVec
        stm = ShortTermMemory(capacity=3)
        for i in range(5):
            stm.store(MemoryRecord(text=f"msg_{i}", bvec=BVec()))
        assert stm.size == 3
        texts = [r.text for r in stm.get_all()]
        assert "msg_0" not in texts
        assert "msg_4" in texts

    def test_mtm_persists_to_disk(self):
        from eris.memory.tiers import MediumTermMemory, MemoryRecord
        from eris.computation.activations import BVec
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            mtm = MediumTermMemory(storage_path=path)
            mtm.store(MemoryRecord(text="persistent", bvec=BVec(F=0.8)))
            # Reload from disk
            mtm2 = MediumTermMemory(storage_path=path)
            assert mtm2.size == 1
            assert mtm2._records[0].text == "persistent"
        finally:
            os.unlink(path)

    def test_ebbinghaus_freshness_decays(self):
        from eris.memory.tiers import MemoryRecord
        from eris.computation.activations import BVec
        import time as _time
        rec = MemoryRecord(text="old", bvec=BVec(),
                           timestamp=_time.time() - 7 * 24 * 3600)  # 1 week ago
        f = rec.freshness(half_life_hours=168.0)  # 1 week half-life
        assert 0.4 < f < 0.6, f"1 week old with 1 week half-life ≈ 0.5, got {f}"

    def test_memory_system_retrieve(self):
        from eris.memory.tiers import MemorySystem
        from eris.computation.activations import BVec
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = MemorySystem(data_dir=tmpdir)
            mem.store_turn("about emergence", BVec(E=0.9, C=0.3))
            mem.store_turn("about boundary", BVec(B=0.9, S=0.2))
            mem.store_turn("about feedback", BVec(F=0.8, B=0.3))

            # Retrieve with emergence-heavy query
            results = mem.retrieve(query_bvec=BVec(E=0.8, C=0.4), top_k=2)
            assert len(results) >= 1
            # The emergence record should rank highest
            assert "emergence" in results[0].text

    def test_consolidation_sgt_gated(self):
        from eris.memory.tiers import MemorySystem
        from eris.computation.activations import BVec
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = MemorySystem(data_dir=tmpdir)
            # Store several turns with varying novelty
            for i in range(15):
                bvec = BVec(E=0.1 * (i % 3), C=0.05 * i)
                mem.store_turn(f"turn_{i}", bvec)

            counts = mem.consolidate()
            # Some should promote, some shouldn't (SGT gating)
            assert isinstance(counts, dict)
            assert "stm_to_mtm" in counts


class TestInterference:
    """Tests for CSBA-enhanced multi-memory interference."""

    def test_self_interference_is_positive(self):
        from eris.memory.interference import compute_interference
        from eris.memory.tiers import MemoryRecord
        from eris.computation.activations import BVec
        phi = np.random.rand(32, 32).astype(np.float32) * 0.5
        theta = np.random.rand(32, 32).astype(np.float32) * np.pi
        rec = MemoryRecord(text="self", bvec=BVec(E=0.5),
                           phi_snapshot=phi, theta_snapshot=theta)
        result = compute_interference(rec, rec)
        assert result.total > 0.9, f"Self-interference should be ~1.0, got {result.total}"
        assert result.used_field_integral is True

    def test_opposing_phases_conflict(self):
        from eris.memory.interference import compute_interference
        from eris.memory.tiers import MemoryRecord
        from eris.computation.activations import BVec
        phi = np.ones((16, 16), dtype=np.float32) * 0.5
        theta_a = np.zeros((16, 16), dtype=np.float32)
        theta_b = np.full((16, 16), np.pi, dtype=np.float32)
        rec_a = MemoryRecord(text="a", bvec=BVec(), phi_snapshot=phi, theta_snapshot=theta_a)
        rec_b = MemoryRecord(text="b", bvec=BVec(), phi_snapshot=phi, theta_snapshot=theta_b)
        result = compute_interference(rec_a, rec_b)
        assert result.total < -0.5, f"Opposing phases should conflict: R={result.total}"
        assert result.regime == "conflicting"

    def test_csba_coupling_no_field_snapshots(self):
        """Without field snapshots, uses CSBA coupling geometry (NOT single cosine)."""
        from eris.memory.interference import compute_interference
        from eris.memory.tiers import MemoryRecord
        from eris.computation.activations import BVec
        # No phi/theta snapshots — forces CSBA path
        rec_a = MemoryRecord(text="a", bvec=BVec(E=0.9, C=0.1))
        rec_b = MemoryRecord(text="b", bvec=BVec(E=0.1, C=0.9))
        result = compute_interference(rec_a, rec_b)
        assert result.used_field_integral is False
        assert "E" in result.per_domain and "C" in result.per_domain
        # These should be in different domains → low total
        assert result.total < 0.5

    def test_csba_similar_memories_resonate(self):
        """Similar BFECDS profiles should show resonance via CSBA."""
        from eris.memory.interference import compute_interference
        from eris.memory.tiers import MemoryRecord
        from eris.computation.activations import BVec
        rec_a = MemoryRecord(text="a", bvec=BVec(E=0.8, F=0.7, C=0.2))
        rec_b = MemoryRecord(text="b", bvec=BVec(E=0.75, F=0.65, C=0.25))
        result = compute_interference(rec_a, rec_b)
        assert result.total > 0.5, f"Similar BVecs should resonate: R={result.total}"
        assert result.regime == "resonant"

    def test_find_conflicts(self):
        from eris.memory.interference import find_conflicts
        from eris.memory.tiers import MemoryRecord
        from eris.computation.activations import BVec
        phi = np.ones((8, 8), dtype=np.float32) * 0.5
        mems = [
            MemoryRecord(text="a", bvec=BVec(E=0.9),
                         phi_snapshot=phi, theta_snapshot=np.zeros((8, 8), dtype=np.float32)),
            MemoryRecord(text="b", bvec=BVec(C=0.9),
                         phi_snapshot=phi, theta_snapshot=np.full((8, 8), np.pi, dtype=np.float32)),
        ]
        conflicts = find_conflicts(mems, threshold=-0.1)
        assert len(conflicts) >= 1
        assert conflicts[0][2].regime == "conflicting"

    def test_find_resonances(self):
        from eris.memory.interference import find_resonances
        from eris.memory.tiers import MemoryRecord
        from eris.computation.activations import BVec
        phi = np.ones((8, 8), dtype=np.float32) * 0.5
        theta = np.zeros((8, 8), dtype=np.float32)
        mems = [
            MemoryRecord(text="a", bvec=BVec(E=0.9), phi_snapshot=phi, theta_snapshot=theta),
            MemoryRecord(text="b", bvec=BVec(E=0.8), phi_snapshot=phi, theta_snapshot=theta + 0.1),
        ]
        resonances = find_resonances(mems, threshold=0.5)
        assert len(resonances) >= 1


class TestGPW:
    """Tests for the wave-interference GPW."""

    def test_moe_gate_interference_scoring(self):
        """MoEGate should score via field interference, not single cosine."""
        from eris.executive.workspace import MoEGate
        from eris.tribe.specialists import SpecialistFinding
        from eris.computation.activations import BVec

        gate = MoEGate()
        gate.set_goal(BVec(E=0.9, C=0.7))

        # Aligned bid should score higher than misaligned
        good = SpecialistFinding("kairos", "emergence", BVec(E=0.8, C=0.6))
        bad = SpecialistFinding("ploutos", "fiscal", BVec(B=0.9, S=0.8))

        score_good = gate.score_bid(good)
        score_bad = gate.score_bid(bad)
        assert score_good > score_bad, (
            f"Aligned bid should score higher: {score_good:.3f} vs {score_bad:.3f}"
        )

    def test_transfixion_reads_dCdX(self):
        """TransfixionDetector should detect stagnation via dC/dX ≈ 0."""
        from eris.executive.workspace import TransfixionDetector
        from eris.computation.activations import BVec

        det = TransfixionDetector(history_length=6, dCdX_stagnation_threshold=0.01)

        # Feed stagnant signals: high C but zero dC/dX
        for _ in range(8):
            det.record(BVec(F=0.5, B=0.5), dCdX=0.001, coherence=0.8)

        assert det.is_transfixed(), "Should detect stagnation from dC/dX ≈ 0 + high C"

    def test_transfixion_not_triggered_by_active_processing(self):
        """Active processing (nonzero dC/dX) should NOT trigger transfixion."""
        from eris.executive.workspace import TransfixionDetector
        from eris.computation.activations import BVec

        det = TransfixionDetector(history_length=6)

        # Feed active processing signals: varying dC/dX
        for i in range(8):
            det.record(BVec(E=0.3 * (i % 3), C=0.2 * i),
                       dCdX=0.1 * (i - 4), coherence=0.5)

        assert not det.is_transfixed(), "Active processing should not trigger transfixion"

    def test_hallucination_signature_uses_dCdX(self):
        from eris.executive.workspace import TransfixionDetector
        from eris.computation.activations import BVec

        det = TransfixionDetector()
        # Conservation law version: dC/dX ≈ 0 + high C
        assert det.check_hallucination_signature(
            BVec(E=0.5), coherence=0.85, tau_rms=0.3, dCdX=0.001
        ), "Zero dC/dX + high coherence = hallucination"

        # Normal processing: nonzero dC/dX
        assert not det.check_hallucination_signature(
            BVec(E=0.5), coherence=0.6, tau_rms=0.3, dCdX=0.15
        )

    def test_transfixion_override_picks_novel(self):
        from eris.executive.workspace import MoEGate
        from eris.tribe.specialists import SpecialistFinding
        from eris.computation.activations import BVec

        gate = MoEGate()
        gate.set_goal(BVec(F=0.5))

        # Force transfixion by filling history with stagnant signals
        for _ in range(15):
            gate.transfixion_detector.record(BVec(F=0.5, B=0.5), dCdX=0.001, coherence=0.8)

        findings = [
            SpecialistFinding("logos", "same old", BVec(F=0.8, E=0.1)),
            SpecialistFinding("aesthetes", "novel idea", BVec(E=0.9, C=0.3)),
        ]
        winner = gate.select_winner(findings, dCdX=0.001, coherence=0.8)
        assert winner.specialist_id == "aesthetes"

    def test_tribe_activation_is_selective(self):
        from eris.tribe.specialists import get_active_specialists
        from eris.computation.activations import BVec
        active = get_active_specialists(BVec(E=0.9, C=0.7))
        assert len(active) < 11


class TestOrchestrator:
    """Tests for the full cognitive pipeline orchestrator."""

    def test_orchestrator_initializes(self):
        from eris.orchestrator import ErisOrchestrator
        with tempfile.TemporaryDirectory() as tmpdir:
            eris = ErisOrchestrator(field_size=16, data_dir=tmpdir)
            assert eris.turn_count == 0
            assert eris.field.size == 16

    def test_vitals_endpoint(self):
        from eris.orchestrator import ErisOrchestrator
        with tempfile.TemporaryDirectory() as tmpdir:
            eris = ErisOrchestrator(field_size=16, data_dir=tmpdir)
            vitals = eris.get_vitals()
            assert "coherence" in vitals
            assert "regime" in vitals
            assert "transfixed" in vitals
            assert "llm_backends" in vitals

    def test_process_without_llm(self):
        """Should work even with no LLM backend — uses specialist finding."""
        import asyncio
        from eris.orchestrator import ErisOrchestrator

        async def _run():
            with tempfile.TemporaryDirectory() as tmpdir:
                eris = ErisOrchestrator(field_size=16, data_dir=tmpdir)
                result = await eris.process("Tell me about emergence")
                return result

        result = asyncio.run(_run())
        assert result.response_text != ""
        assert result.input_bvec is not None
        assert result.regime in ("warmup", "elastic", "plastic", "transfixed")
        assert result.latency_ms > 0

    def test_process_tracks_autobiography(self):
        """Each process call should log to autobiography."""
        import asyncio
        from eris.orchestrator import ErisOrchestrator

        async def _run():
            with tempfile.TemporaryDirectory() as tmpdir:
                eris = ErisOrchestrator(field_size=16, data_dir=tmpdir)
                await eris.process("First message")
                await eris.process("Second message")
                return eris

        eris = asyncio.run(_run())
        entries = eris.autobiography.get_today()
        assert len(entries) == 2
        assert eris.turn_count == 2

    def test_frt_mode_works(self):
        """FRT seeding mode should produce valid results."""
        import asyncio
        from eris.orchestrator import ErisOrchestrator

        async def _run():
            with tempfile.TemporaryDirectory() as tmpdir:
                eris = ErisOrchestrator(
                    field_size=16, data_dir=tmpdir, use_frt_seeding=True
                )
                result = await eris.process("Testing FRT reflexive pathway")
                return result

        result = asyncio.run(_run())
        assert result.response_text != ""
        assert result.input_bvec is not None


class TestDreamingLoop:
    """Tests for the metacognitive dreaming cycle."""

    def test_dream_cycle_processes_tensions(self):
        from eris.metacognition.dreaming import DreamingLoop
        from eris.memory.autobiography import Autobiography
        from eris.memory.tiers import MemorySystem
        from eris.computation.activations import BVec

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            auto_path = f.name
        with tempfile.TemporaryDirectory() as mem_dir:
            try:
                auto = Autobiography(path=auto_path)
                mem = MemorySystem(data_dir=mem_dir)

                # Log some interactions with varying dissonance
                auto.log_interaction("low tension", "agreement",
                                     BVec(E=0.5), BVec(E=0.5))
                auto.log_interaction("high tension", "total disagreement",
                                     BVec(E=0.1, C=0.1), BVec(C=0.9, D=0.8))

                loop = DreamingLoop(auto, mem, field_size=16,
                                     torsion_threshold=0.1)
                report = loop.run_cycle()

                assert report.tensions_scanned >= 1
                assert report.tensions_processed + report.tensions_gated_out == report.tensions_scanned
            finally:
                os.unlink(auto_path)

    def test_research_trigger_conditions(self):
        from eris.tribe.specialists import should_trigger_research
        from eris.computation.activations import BVec

        # C > 0.4 AND E > 0.2 → trigger
        assert should_trigger_research(BVec(C=0.5, E=0.3))
        # C too low → don't trigger
        assert not should_trigger_research(BVec(C=0.2, E=0.5))
        # E too low → don't trigger
        assert not should_trigger_research(BVec(C=0.6, E=0.1))


class TestAutobiography:
    """Tests for the autobiography logger."""

    def test_log_and_retrieve(self):
        from eris.memory.autobiography import Autobiography
        from eris.computation.activations import BVec
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            auto = Autobiography(path=path)
            auto.log_interaction(
                input_text="hello",
                response_text="world",
                input_bvec=BVec(E=0.5),
                response_bvec=BVec(F=0.3, C=0.7),
                coherence=0.8, exchange=0.3, dCdX=0.4,
                regime="elastic",
            )
            today = auto.get_today()
            assert len(today) == 1
            assert today[0].dominant_domain == "C"  # C=0.7 is highest
            assert today[0].archetype != ""
        finally:
            os.unlink(path)

    def test_high_torsion_filter(self):
        from eris.memory.autobiography import Autobiography
        from eris.computation.activations import BVec
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            auto = Autobiography(path=path)
            # Low dissonance entry
            auto.log_interaction("a", "b", BVec(E=0.5), BVec(E=0.5))
            # High dissonance entry
            auto.log_interaction("c", "d", BVec(E=0.1), BVec(C=0.9, D=0.8))

            high_torsion = auto.get_high_torsion(threshold=0.3)
            assert len(high_torsion) >= 1, "Should find the high-dissonance entry"
        finally:
            os.unlink(path)


class TestFullPipeline:
    """Integration test: text → PDE → BVec → BLC → inject → memory."""

    def test_text_to_memory_with_contradiction(self):
        from eris.field.pde import FractalField
        from eris.field.compiler import compile_contradiction, inject_seeds
        from eris.memory.autobiography import Autobiography
        from eris.computation.activations import BVec

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            auto_path = f.name

        try:
            auto = Autobiography(path=auto_path)

            # Process two texts through the PDE
            f1 = FractalField(size=32)
            f1.seed_from_text("Emergence creates new structure from chaos")
            f1.run(30)
            bvec1 = f1.compute_bvec()

            f2 = FractalField(size=32)
            f2.seed_from_text("Decay dissolves all structure into entropy")
            f2.run(30)
            bvec2 = f2.compute_bvec()

            # Compile the contradiction
            result = compile_contradiction(bvec1, bvec2, field_size=32,
                                           sgt_threshold=0.5, sgt_mean=0.0, sgt_var=0.01)

            # Inject into field and let it resolve
            inject_seeds(f1, result)
            f1.run(20)
            resolved_bvec = f1.compute_bvec()

            # Log the interaction
            auto.log_interaction(
                input_text="Emergence creates new structure from chaos",
                response_text="Decay dissolves all structure into entropy",
                input_bvec=bvec1,
                response_bvec=bvec2,
                coherence=f1.coherence,
                exchange=f1.exchange,
                dCdX=f1.dCdX,
                regime=f1.detect_regime(),
            )

            entry = auto.get_today()[0]
            print(f"\n  Pipeline results:")
            print(f"  Input archetype:    {bvec1.archetype()}")
            print(f"  Response archetype: {bvec2.archetype()}")
            print(f"  Contradiction:      {result.total_contradiction:.3f}")
            print(f"  Seeds generated:    {result.n_seeds}")
            print(f"  Resolved archetype: {resolved_bvec.archetype()}")
            print(f"  Regime:             {entry.regime}")
            print(f"  dC/dX:              {entry.dCdX:.4f}")
            print(f"  Dissonance:         {entry.dissonance:.3f}")

            assert entry.dissonance > 0
            assert entry.regime in ("warmup", "elastic", "plastic", "transfixed")
        finally:
            os.unlink(auto_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
