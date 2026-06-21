#!/usr/bin/env python3
"""
bench_orchestration.py — the A/B "ruler" for the CIP cross-stage orchestration
=============================================================================

Tier 0 of ERIS_ORCHESTRATION_REMEDIATION.md. This ships the MEASUREMENT layer
only — there are no gates yet. Its job is to prove, offline and deterministically,
that:

  1. the benchmark runs with no network, no cloud key, no model download;
  2. the "baseline" arm (orchestration disabled) reproduces current behavior —
     the per-turn counters equal the un-gated pipeline; and
  3. the "orchestrated" arm, with no gates wired, is byte-identical to baseline
     (max counter Δ = 0, max answer Δ = 0.000). The ruler is calibrated.

Later tiers add gates behind flags; this same harness then reports, per message
and per class, the resources saved AND whether the answer is preserved within
tolerance. Value is declared ONLY when resources drop and answer-Δ stays under
`orch_answer_tol`.

What it measures (per turn, via eris.computation.orch_counters):
  - PDE steps executed (main field)        - response-field steps executed
  - field rebuilds (the cold 2nd field)    - would-be cloud-expert calls

Isolation (so it measures orchestration, not LLM variance):
  - The LLM is a deterministic FakeBackend (prompt-hashed text, zero latency).
    Identical prompts → identical text, so any text change a gate causes is a
    REAL behavioral effect, which we surface.
  - Deterministic embeddings (ERIS_EMBEDDINGS=off) and CPU field math
    (ERIS_GPU=0) are forced for reproducibility and to keep it offline.
  - Fixed field seed per message in both arms; M-seed repeats report mean ± std.

Read the numbers correctly: with the LLM stubbed, WALL-CLOCK UNDERSTATES real
value — the router gate's payoff is skipped CLOUD CALLS, shown by that counter,
not by wall-clock. The table calls this out.

Usage:
    python bench_orchestration.py                 # defaults: seeds=5, field=64
    python bench_orchestration.py --seeds 3 --field-size 32   # quicker
"""
from __future__ import annotations

# ── Force offline + deterministic BEFORE importing eris ───────────────────
import os
os.environ.setdefault("ERIS_EMBEDDINGS", "off")   # deterministic hash embeddings
os.environ.setdefault("ERIS_GPU", "0")            # CPU field math, reproducible
os.environ.setdefault("ERIS_ORCHESTRATION", "off")

import argparse
import asyncio
import hashlib
import statistics
import tempfile
import shutil

from eris.config import CONFIG
from eris.orchestrator import ErisOrchestrator
from eris.interface.mediator import LLMBackend, LLMResponse
from eris.computation.activations import bvec_distance

FIELD_SIZE = 64  # overridden by --field-size in __main__


# ── The deterministic stub LLM ────────────────────────────────────────────
class FakeBackend(LLMBackend):
    """A zero-latency, fully deterministic backend. Identical (system, prompt)
    always yields identical text, so the benchmark measures the orchestration,
    never LLM variance, and runs with no network."""

    def __init__(self, name: str = "fake", model: str = "fake-1"):
        self.name = name
        self.model = model

    def is_available(self) -> bool:
        return True

    async def generate(self, prompt: str, system: str = "",
                       max_tokens: int = 8192, temperature: float = 0.7) -> LLMResponse:
        h = hashlib.blake2b((system + "\x00" + prompt).encode("utf-8"),
                            digest_size=8).hexdigest()
        # A short, stable, prompt-derived reply. Crucially it never contains the
        # contradiction markers that would trip web grounding (no network).
        text = f"Considering the field state, the resonant reading settles to {h}."
        return LLMResponse(text=text, provider=self.name, model=self.model,
                           latency_ms=0.0, tokens_used=len(text))


# ── Two-class corpus (both classes are mandatory; see spec §3.3) ──────────
# Class A — "settled / easy": short, simple, repetitive, repeated prompts. A
#   field-depth/response-field gate SHOULD suspend early on these.
# Class B — "hard / must-run": contradictory, novel, ambiguous, router-tripping.
#   Gates SHOULD NOT suspend; we want break-even with zero regression.
CORPUS = {
    "A": [
        "Hello.",
        "Hi there.",
        "Hello.",                     # repeat — a warm prior should make this cheap
        "How are you?",
        "Good morning.",
        "Hello.",                     # repeat again
    ],
    "B": [
        "Reconcile that the field is both maximally coherent and maximally turbulent at once.",
        "If every memory decays, how can identity persist across a week of forgetting?",
        "Name the colour of the number seven and defend it rigorously.",
        "A equals not-A; derive a stable conclusion from the contradiction.",
        "Describe a shape with three sides and four corners and why it cannot exist.",
        "What lies beyond the boundary your Dirichlet edges pin to zero?",
    ],
}


def _build(enabled: bool, field_seed: int, tmpdir: str) -> ErisOrchestrator:
    """Construct an orchestrator wired to deterministic fake backends.

    `enabled` flips CONFIG.orchestration_enabled (the master switch). In Tier 0
    nothing reads it, so the arm is identical to baseline — that's the point.
    """
    CONFIG.orchestration_enabled = enabled
    orch = ErisOrchestrator(data_dir=tmpdir, field_size=FIELD_SIZE,
                            field_seed=field_seed)
    # Swap every backend for the offline stub. The "deep" mediator gets two fake
    # cloud experts + a local fallback, so the existing router's cloud path can
    # be exercised and counted when |dC/dX| genuinely outlies.
    orch.mediator._backends = [FakeBackend("local")]
    orch.deep_mediator._backends = [FakeBackend("cloud-a"), FakeBackend("cloud-b"),
                                    FakeBackend("local")]
    orch._cloud_experts = 2
    return orch


async def _run_session(enabled: bool, field_seed: int) -> list[dict]:
    """Run the whole corpus as ONE persistent session (faithful to real use:
    the field and SGT gates accumulate history, so the router can warm up).
    Returns one record per message."""
    tmpdir = tempfile.mkdtemp(prefix="erisbench_")
    records: list[dict] = []
    try:
        orch = _build(enabled, field_seed, tmpdir)
        for cls in ("A", "B"):
            for idx, msg in enumerate(CORPUS[cls]):
                res = await orch.process(msg)
                c = orch.counters
                records.append({
                    "class": cls, "idx": idx,
                    "pde_steps": c.pde_steps,
                    "resp_field_steps": c.resp_field_steps,
                    "field_rebuilds": c.field_rebuilds,
                    "cloud_calls": c.cloud_calls,
                    "wall_ms": res.latency_ms,
                    "bvec": res.response_bvec,
                    "coherence": res.coherence,
                    "dCdX": res.dCdX,
                })
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return records


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    return statistics.fmean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def _pct_saved(base: float, orch: float) -> float:
    if base == 0:
        return 0.0
    return 100.0 * (base - orch) / base


async def main(seeds: int) -> int:
    print("=" * 78)
    print("ERIS ORCHESTRATION BENCHMARK — Tier 0 (the ruler; no gates wired)")
    print(f"  field_size={FIELD_SIZE}  seeds={seeds}  "
          f"corpus: A(easy)={len(CORPUS['A'])}  B(hard)={len(CORPUS['B'])}")
    print(f"  offline: ERIS_GPU={os.environ['ERIS_GPU']}  "
          f"ERIS_EMBEDDINGS={os.environ['ERIS_EMBEDDINGS']}  LLM=FakeBackend")
    print("=" * 78)

    # Collect baseline and orchestrated records across M seeds, keyed by message.
    base_runs: list[list[dict]] = []
    orch_runs: list[list[dict]] = []
    for s in range(seeds):
        seed = 42 + s
        base_runs.append(await _run_session(enabled=False, field_seed=seed))
        orch_runs.append(await _run_session(enabled=True, field_seed=seed))

    metrics = ("pde_steps", "resp_field_steps", "field_rebuilds", "cloud_calls", "wall_ms")
    worst_counter_delta = 0
    worst_answer_delta = 0.0

    for cls in ("A", "B"):
        # Gather per-(seed, message) values for this class.
        agg = {m: {"base": [], "orch": []} for m in metrics}
        answer_deltas: list[float] = []
        coh_deltas: list[float] = []
        n_msgs = len(CORPUS[cls])
        for run_b, run_o in zip(base_runs, orch_runs):
            rb = [r for r in run_b if r["class"] == cls]
            ro = [r for r in run_o if r["class"] == cls]
            for b, o in zip(rb, ro):
                for m in metrics:
                    agg[m]["base"].append(float(b[m]))
                    agg[m]["orch"].append(float(o[m]))
                d = bvec_distance(o["bvec"], b["bvec"]) if (o["bvec"] and b["bvec"]) else 0.0
                answer_deltas.append(d)
                coh_deltas.append(abs(o["coherence"] - b["coherence"]))
                worst_counter_delta = max(
                    worst_counter_delta,
                    *(abs(int(o[m]) - int(b[m])) for m in
                      ("pde_steps", "resp_field_steps", "field_rebuilds", "cloud_calls"))
                )
        worst_answer_delta = max(worst_answer_delta, max(answer_deltas, default=0.0))

        # Baseline absolute means (the ruler's reading of current behavior).
        bmean = {m: _mean_std(agg[m]["base"])[0] for m in metrics}
        # Savings, orchestrated vs baseline.
        sav = {m: _pct_saved(*_mean_std_pair(agg[m])) for m in metrics}
        maxd = max(answer_deltas, default=0.0)
        verdict = "PASS" if maxd < CONFIG.orch_answer_tol else "FAIL"

        label = "easy" if cls == "A" else "hard"
        print(f"\nCLASS {cls} ({label}, n={n_msgs}, seeds={seeds})")
        print(f"  baseline/turn : PDE {bmean['pde_steps']:.0f} | resp-field "
              f"{bmean['resp_field_steps']:.0f} | rebuilds {bmean['field_rebuilds']:.0f} "
              f"| cloud {bmean['cloud_calls']:.1f} | wall {bmean['wall_ms']:.1f} ms")
        print(f"  orch savings  : PDE {sav['pde_steps']:+.0f}% | resp-field "
              f"{sav['resp_field_steps']:+.0f}% | rebuilds {sav['field_rebuilds']:+.0f}% "
              f"| cloud {sav['cloud_calls']:+.0f}% | wall {sav['wall_ms']:+.0f}%")
        print(f"  fidelity      : max answer Δ {maxd:.3f} (tol {CONFIG.orch_answer_tol}) "
              f"| max coherence Δ {max(coh_deltas, default=0.0):.3f}  -> {verdict}")

    print("\n" + "-" * 78)
    print("NOTE: with the LLM stubbed, wall-clock UNDERSTATES real value — the "
          "router's\n      payoff is skipped CLOUD CALLS (see the cloud counter), "
          "not wall-clock.")
    print("-" * 78)
    if worst_counter_delta == 0 and worst_answer_delta < 1e-9:
        print("TIER 0 VERDICT: ruler calibrated. No gates wired; the orchestrated "
              "arm\n  reproduces baseline EXACTLY (max counter Δ = 0, max answer "
              "Δ = 0.000).\n  Baseline reproduces current behavior. Add gates in Tier 2+.")
        return 0
    print(f"TIER 0 WARNING: orchestrated arm diverged from baseline with no gates "
          f"wired\n  (max counter Δ = {worst_counter_delta}, max answer Δ = "
          f"{worst_answer_delta:.4f}). The ruler must read zero here — investigate "
          f"before adding gates.")
    return 1


def _mean_std_pair(d: dict) -> tuple[float, float]:
    """(baseline_mean, orchestrated_mean) for _pct_saved."""
    return _mean_std(d["base"])[0], _mean_std(d["orch"])[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Eris orchestration A/B ruler (Tier 0)")
    ap.add_argument("--seeds", type=int, default=5, help="repeat over M field seeds")
    ap.add_argument("--field-size", type=int, default=64, help="PDE grid size")
    args = ap.parse_args()
    FIELD_SIZE = args.field_size  # noqa: F811 — intentional global override
    raise SystemExit(asyncio.run(main(args.seeds)))
