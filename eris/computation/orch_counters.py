"""Cross-stage orchestration counters (Tier 0 — the benchmark's ruler).

Cheap integer counters incremented at each expensive pipeline boundary so the
A/B harness (`bench_orchestration.py`) can measure resources spent per turn:
main-field PDE steps, response-field steps, cold field rebuilds, and would-be
cloud-expert calls.

They are pure instrumentation — present in BOTH the baseline and the
orchestrated arm so their (tiny) integer-add overhead cancels in the
comparison. Counting never changes a decision: with orchestration disabled the
numbers simply equal the un-gated pipeline, which is exactly the Tier 0
acceptance check (baseline reproduces current behavior).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class OrchestrationCounters:
    """Per-turn resource tally. Reset at the top of each `process()`."""

    pde_steps: int = 0          # main-field Kuramoto steps executed this turn
    resp_field_steps: int = 0   # response-field steps executed this turn
    field_rebuilds: int = 0     # cold FractalField builds for the response bvec
    cloud_calls: int = 0        # would-be cloud-expert calls (router escalations)

    def reset(self) -> None:
        self.pde_steps = 0
        self.resp_field_steps = 0
        self.field_rebuilds = 0
        self.cloud_calls = 0

    def as_dict(self) -> dict:
        return {
            "pde_steps": self.pde_steps,
            "resp_field_steps": self.resp_field_steps,
            "field_rebuilds": self.field_rebuilds,
            "cloud_calls": self.cloud_calls,
        }
