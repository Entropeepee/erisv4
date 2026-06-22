# Orchestration gates — per-tier verdicts (the §13 hand-back)

Every gate is **default OFF** in `eris/config.py`. This file records, per the
benchmark + fidelity probes, whether each gate earned its place. The rule
(spec §3.1): a gate may merge on by default **only** when resources drop **and**
answer-Δ stays under `orch_answer_tol` (0.05). Otherwise it stays behind its
flag — available for future tuning, off in production.

Run the ruler yourself: `python bench_orchestration.py` (offline, deterministic).

| Tier | Gate | Flag | Verdict | Status |
|---|---|---|---|---|
| 2 | field-evolution depth | `gate_field_depth` | **flagged OFF** — no safe savings on this engine | implemented, measured |
| 3 | response-field warm-start | `gate_response_field` | (pending) | |
| 4 | formalized router | `gate_router` | (pending) | |
| 5 | failure reports → dreams | `gate_failure_reports` | (pending) | |
| 6 | β-star bridge | `use_beta_star` | (pending) | |

---

## Tier 2 — field-evolution depth gate → **leave OFF**

**What it does.** `FractalField.run_gated()` evolves up to `pde_steps_per_input`
but suspends early once the windowed change in global coherence drops to a low
outlier below its own noise floor (the shared `CriticalityMonitor`, "settle"
mode), with a hard `min_steps` floor.

**Why it earns nothing here (measured, not guessed).** Two independent probes:

1. *The settle signal never fires.* After the one-time seeding transient, the
   coherence delta **plateaus at a roughly constant** ~2.4e-3 (easy) / ~0.5e-3
   (hard) and stays there — the field reaches a steady *drift*, not a fixed
   point. There is no "change falls below floor" event to detect, so the gate
   correctly never suspends (benchmark: PDE 50→50, +0%, fidelity Δ 0.000).

2. *Forcing an early stop would regress the answer badly.* Stopping at n steps
   vs the full 50 gives response-bvec Δ of **0.24–0.27 at n=30**, **0.36–0.65 at
   n=12–20** — all far above the 0.05 tolerance. The bvec is still converging at
   step 50; the φ/θ keep moving even after coherence plateaus.

**Conclusion.** The gate is implemented and its mechanism is unit-tested (a
settled trajectory suspends, a turbulent one runs full, `min_steps` is honored),
but on this Kuramoto engine early field termination is a fidelity regression, not
a saving. It stays **OFF by default**. The real, safe wins live downstream (the
router's skipped cloud calls — Tier 4).
