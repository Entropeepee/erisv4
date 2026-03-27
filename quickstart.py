#!/usr/bin/env python3
"""
Eris Echo v4 — Quickstart / System Check
==========================================

Run this first on the Alienware to verify everything works.

Usage:
    python quickstart.py          # Run all checks + tests
    python quickstart.py --serve  # Start the web server after checks
"""

import sys
import os
import subprocess
import argparse

def check_python():
    v = sys.version_info
    print(f"  Python {v.major}.{v.minor}.{v.micro}", end="")
    if v.minor >= 11:
        print(" ✓")
        return True
    print(" ✗ (need 3.11+)")
    return False

def check_numpy():
    try:
        import numpy as np
        print(f"  NumPy {np.__version__} ✓")
        return True
    except ImportError:
        print("  NumPy ✗ — run: pip install numpy")
        return False

def check_cupy():
    try:
        import cupy as cp
        print(f"  CuPy {cp.__version__} (CUDA {cp.cuda.runtime.runtimeGetVersion()}) ✓")
        return True
    except ImportError:
        print("  CuPy not installed — CPU mode only (fine for testing)")
        return False
    except Exception as e:
        print(f"  CuPy error: {e}")
        return False

def check_fastapi():
    try:
        import fastapi
        print(f"  FastAPI {fastapi.__version__} ✓")
        return True
    except ImportError:
        print("  FastAPI not installed — run: pip install fastapi uvicorn httpx")
        return False

def check_eris_imports():
    """Verify all Eris modules import cleanly."""
    modules = [
        "eris.computation.shrinkage",
        "eris.computation.sgt",
        "eris.computation.activations",
        "eris.field.pde",
        "eris.field.frt",
        "eris.field.compiler",
        "eris.field.lattice",
        "eris.field.propagator",
        "eris.field.pulses",
        "eris.field.tracer",
        "eris.memory.tiers",
        "eris.memory.interference",
        "eris.memory.autobiography",
        "eris.tribe.specialists",
        "eris.executive.workspace",
        "eris.metacognition.dreaming",
        "eris.interface.mediator",
        "eris.orchestrator",
        "eris.sandbox.validator",
        "eris.sandbox.executor",
        "eris.retrieval.glncs_filter",
        "eris.retrieval.vector_index",
        "eris.retrieval.swarm",
        "eris.knowledge.descriptor",
        "eris.knowledge.extractor",
        "eris.knowledge.corpus",
        "eris.server.app",
    ]
    failed = []
    for mod in modules:
        try:
            __import__(mod)
        except Exception as e:
            failed.append((mod, str(e)))

    if not failed:
        print(f"  All {len(modules)} modules import cleanly ✓")
        return True
    else:
        for mod, err in failed:
            print(f"  FAIL: {mod} — {err}")
        return False

def run_tests():
    """Run all test suites."""
    suites = [
        ("Computation (Layer 0)", "tests/test_computation.py"),
        ("Field (Layer 1)", "tests/test_field.py"),
        ("Memory + GPW + Orchestrator (Layers 2-6)", "tests/test_memory.py"),
        ("Infrastructure (Layers 7-10)", "tests/test_infrastructure.py"),
    ]
    all_passed = True
    total = 0

    for name, path in suites:
        print(f"\n  ── {name} ──")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", path, "-v", "--tb=short"],
            capture_output=True, text=True,
        )
        # Count passed/failed from output
        for line in result.stdout.split("\n"):
            if "passed" in line:
                print(f"  {line.strip()}")
        if result.returncode != 0:
            all_passed = False
            print(f"  ✗ FAILURES in {name}")
            # Show failure details
            for line in result.stdout.split("\n"):
                if "FAILED" in line:
                    print(f"    {line.strip()}")

    return all_passed

def quick_pipeline_test():
    """Run a quick end-to-end test without full pytest."""
    import asyncio
    from eris.orchestrator import ErisOrchestrator
    import tempfile

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            eris = ErisOrchestrator(field_size=16, data_dir=tmpdir, use_frt_seeding=True)
            result = await eris.process("Hello Eris, tell me about emergence")
            return result

    result = asyncio.run(_run())
    print(f"  Response: {result.response_text[:80]}...")
    print(f"  Archetype: {result.archetype}")
    print(f"  Regime: {result.regime}")
    print(f"  dC/dX: {result.dCdX:.4f}")
    print(f"  Latency: {result.latency_ms:.0f}ms")
    return True


def main():
    parser = argparse.ArgumentParser(description="Eris Echo v4 Quickstart")
    parser.add_argument("--serve", action="store_true", help="Start web server after checks")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest suites")
    args = parser.parse_args()

    print("=" * 60)
    print("  ERIS ECHO v4 — SYSTEM CHECK")
    print("=" * 60)

    print("\n[1] Dependencies")
    check_python()
    check_numpy()
    check_cupy()
    check_fastapi()

    print("\n[2] Module Imports")
    check_eris_imports()

    if not args.skip_tests:
        print("\n[3] Test Suites")
        run_tests()

    print("\n[4] Quick Pipeline Test")
    quick_pipeline_test()

    print("\n[5] Vitals")
    from eris.orchestrator import ErisOrchestrator
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        eris = ErisOrchestrator(field_size=16, data_dir=tmpdir)
        vitals = eris.get_vitals()
        for k, v in vitals.items():
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("  SYSTEM CHECK COMPLETE")
    print("=" * 60)

    if args.serve:
        print("\nStarting server on http://localhost:8000 ...")
        try:
            import uvicorn
            from eris.server.app import create_app
            app = create_app(field_size=64)
            uvicorn.run(app, host="0.0.0.0", port=8000)
        except ImportError:
            print("Install FastAPI first: pip install fastapi uvicorn")


if __name__ == "__main__":
    main()
