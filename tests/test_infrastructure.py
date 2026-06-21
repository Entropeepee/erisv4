"""
Tests for Phase 5-7: Sandbox, Retrieval, Knowledge, Server
============================================================
Run: cd eris_echo_v4 && python -m pytest tests/test_infrastructure.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from eris.config import to_numpy, xp
import tempfile
import pytest


# ─── Sandbox Tests ────────────────────────────────────────────────────────

class TestValidator:
    def test_safe_code_passes(self):
        from eris.sandbox.validator import validate_code
        ok, msg = validate_code("import numpy as np\nfrom eris.config import to_numpy, xp\nprint(np.pi)")
        assert ok, msg

    def test_blocked_import_os(self):
        from eris.sandbox.validator import validate_code
        ok, msg = validate_code("import os\nos.system('rm -rf /')")
        assert not ok
        assert "Blocked" in msg  # Could be import or pattern

    def test_blocked_subprocess(self):
        from eris.sandbox.validator import validate_code
        ok, msg = validate_code("import subprocess\nsubprocess.run(['ls'])")
        assert not ok

    def test_blocked_eval(self):
        from eris.sandbox.validator import validate_code
        ok, msg = validate_code("eval('1+1')")
        assert not ok

    def test_syntax_error_caught(self):
        from eris.sandbox.validator import validate_code
        ok, msg = validate_code("def f(\n  broken")
        assert not ok
        assert "Syntax error" in msg

    def test_eris_import_allowed(self):
        from eris.sandbox.validator import validate_code
        ok, msg = validate_code("from eris.computation.activations import BVec")
        assert ok, msg


class TestSandboxExecutor:
    def test_basic_execution(self):
        from eris.sandbox.executor import SandboxExecutor, ExecutionStatus
        with tempfile.TemporaryDirectory() as tmpdir:
            sb = SandboxExecutor(workspace_dir=tmpdir)
            result = sb.execute("print(42 + 58)")
            assert result.status == ExecutionStatus.COMPLETED
            assert "100" in result.stdout

    def test_blocked_code_doesnt_run(self):
        from eris.sandbox.executor import SandboxExecutor, ExecutionStatus
        with tempfile.TemporaryDirectory() as tmpdir:
            sb = SandboxExecutor(workspace_dir=tmpdir)
            result = sb.execute("import os; os.system('echo hacked')")
            assert result.status == ExecutionStatus.BLOCKED

    def test_timeout_enforced(self):
        from eris.sandbox.executor import SandboxExecutor, ExecutionStatus
        with tempfile.TemporaryDirectory() as tmpdir:
            sb = SandboxExecutor(workspace_dir=tmpdir)
            result = sb.execute("import time; time.sleep(30)", timeout=2)
            assert result.status == ExecutionStatus.TIMEOUT

    def test_error_captured(self):
        from eris.sandbox.executor import SandboxExecutor, ExecutionStatus
        with tempfile.TemporaryDirectory() as tmpdir:
            sb = SandboxExecutor(workspace_dir=tmpdir)
            result = sb.execute("raise ValueError('test error')")
            assert result.status == ExecutionStatus.ERROR
            assert "ValueError" in result.stderr

    def test_stats_tracked(self):
        from eris.sandbox.executor import SandboxExecutor
        with tempfile.TemporaryDirectory() as tmpdir:
            sb = SandboxExecutor(workspace_dir=tmpdir)
            sb.execute("print(1)")
            sb.execute("import os")  # blocked
            assert sb.stats["total"] == 2
            assert sb.stats["successful"] == 1
            assert sb.stats["blocked"] == 1


# ─── GLNCS Filter Tests ──────────────────────────────────────────────────

class TestGLNCSFilter:
    def test_calibrate_and_apply(self):
        from eris.retrieval.glncs_filter import GLNCSFilter
        glncs = GLNCSFilter(input_dim=128)

        # Create noise vectors (systematic bias)
        rng = np.random.default_rng(42)
        noise = rng.standard_normal((50, 128)).astype(np.float32)
        glncs.calibrate(noise, bias_fraction=0.1)

        assert glncs.is_calibrated
        assert glncs.n_bias_dims > 0

        # Apply to a vector
        vec = rng.standard_normal(128).astype(np.float32)
        clean = glncs.apply(vec)
        assert clean.shape == (128,)
        # Clean vector should have reduced projection onto bias subspace
        assert not np.allclose(vec, clean)

    def test_uncalibrated_passthrough(self):
        from eris.retrieval.glncs_filter import GLNCSFilter
        glncs = GLNCSFilter(input_dim=64)
        vec = np.ones(64, dtype=np.float32)
        result = glncs.apply(vec)
        np.testing.assert_array_equal(vec, result)

    def test_compress(self):
        from eris.retrieval.glncs_filter import GLNCSFilter
        glncs = GLNCSFilter(input_dim=128)
        rng = np.random.default_rng(42)
        noise = rng.standard_normal((50, 128)).astype(np.float32)
        glncs.calibrate(noise)
        corpus = rng.standard_normal((100, 128)).astype(np.float32)
        glncs.calibrate_compression(corpus, target_dim=32)

        vec = rng.standard_normal(128).astype(np.float32)
        compressed = glncs.compress(vec, target_dim=32)
        assert compressed.shape == (32,)


# ─── Vector Index Tests ───────────────────────────────────────────────────

class TestVectorIndex:
    def test_add_and_search(self):
        from eris.retrieval.vector_index import VectorIndex
        idx = VectorIndex(dim=16)
        rng = np.random.default_rng(42)

        # Add some vectors
        for i in range(10):
            v = rng.standard_normal(16).astype(np.float32)
            idx.add(f"doc_{i}", v, metadata={"i": i})

        # Search
        query = rng.standard_normal(16).astype(np.float32)
        results = idx.search(query, top_k=3)
        assert len(results) == 3
        assert all(r.score <= 1.0 for r in results)

    def test_tier_promotion(self):
        from eris.retrieval.vector_index import VectorIndex
        idx = VectorIndex(dim=8)
        v = np.ones(8, dtype=np.float32)
        idx.add("doc_1", v, tier="warm")
        assert idx.tier_sizes["warm"] == 1
        idx.promote("doc_1")
        assert idx.tier_sizes["hot"] == 1
        assert idx.tier_sizes["warm"] == 0

    def test_total_size(self):
        from eris.retrieval.vector_index import VectorIndex
        idx = VectorIndex(dim=4)
        for i in range(5):
            idx.add(f"d{i}", np.random.randn(4).astype(np.float32))
        assert idx.total_size == 5


# ─── Retrieval Swarm Tests ───────────────────────────────────────────────

class TestRetrievalSwarm:
    def test_swarm_returns_results(self):
        from eris.retrieval.swarm import RetrievalSwarm
        from eris.memory.tiers import MemorySystem, MemoryRecord
        from eris.computation.activations import BVec

        with tempfile.TemporaryDirectory() as tmpdir:
            mem = MemorySystem(data_dir=tmpdir, stm_capacity=5, mtm_capacity=10)
            # Add directly to MTM (bypass SGT consolidation gate)
            mem.mtm.store(MemoryRecord(text="about emergence", bvec=BVec(E=0.9, C=0.3)))
            mem.mtm.store(MemoryRecord(text="about boundary", bvec=BVec(B=0.9, S=0.2)))

            swarm = RetrievalSwarm(mem)
            results = swarm.search(query_bvec=BVec(E=0.8), top_k=2)
            assert len(results) >= 1


class TestKnowledgeExtractor:
    def test_extract_short_text(self):
        from eris.knowledge.extractor import KnowledgeExtractor
        with tempfile.TemporaryDirectory() as tmpdir:
            ex = KnowledgeExtractor(
                output_dir=tmpdir, field_size=8, pde_steps=5,
            )
            descs = ex.extract_text("Short test text", title="short")
            assert len(descs) == 1
            eris_files = [f for f in os.listdir(tmpdir) if f.endswith(".eris")]
            assert len(eris_files) == 1


class TestCorpusProcessor:
    def test_process_text_directory(self):
        from eris.knowledge.corpus import CorpusProcessor
        with tempfile.TemporaryDirectory() as src_dir:
            with tempfile.TemporaryDirectory() as out_dir:
                for i in range(2):
                    with open(os.path.join(src_dir, f"doc_{i}.txt"), "w") as f:
                        f.write(f"Test document {i}.")

                proc = CorpusProcessor(output_dir=out_dir, field_size=8, pde_steps=5)
                stats = proc.process_text_directory(src_dir)
                assert stats["files"] == 2


# ─── Server Tests (no FastAPI dependency needed) ──────────────────────────

class TestServerModule:
    def test_server_module_importable(self):
        """Server module should import without FastAPI installed."""
        from eris.server import app as server_module
        assert hasattr(server_module, 'create_app')

    def test_minimal_ui_string_exists(self):
        from eris.server.app import _MINIMAL_UI
        assert "<html>" in _MINIMAL_UI
        assert "Eris Echo" in _MINIMAL_UI


# ─── Full Integration Test ────────────────────────────────────────────────

class TestFullSystemIntegration:
    def test_orchestrator_with_knowledge(self):
        """Pipeline: process message → create .eris descriptor."""
        import asyncio
        from eris.orchestrator import ErisOrchestrator
        from eris.knowledge.descriptor import ErisDescriptor

        async def _run():
            with tempfile.TemporaryDirectory() as tmpdir:
                eris = ErisOrchestrator(field_size=8, data_dir=tmpdir)
                r1 = await eris.process("Emergence creates new patterns")
                assert r1.input_bvec is not None

                desc = ErisDescriptor.from_text(
                    "Emergence creates new patterns",
                    title="test", field_size=8, pde_steps=5,
                )
                assert desc.bvec is not None
                assert desc.verify_integrity()
                return True

        assert asyncio.run(_run())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
