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
| 3 | response-field warm-start | `gate_response_field` | **flagged OFF** — regresses dissonance, no savings | implemented, measured |
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

## Tier 3 — response-field warm-start → **leave OFF** (isolated, as flagged)

**What it does.** Reuses a persistent response field across turns (no cold
rebuild), blends the new response text into its warm φ/θ prior
(`warm_reseed`, `orch_resp_blend=0.7`), and suspends once the response bvec
stabilizes (`run_gated_response`, settle mode).

**Why it doesn't earn its place (measured).**
- *No step savings.* The response bvec plateaus just like the main field, so the
  settle signal never fires: resp-field steps 25→25 (+0%).
- *It regresses the answer.* The warm prior (30% carryover) plus the persistent
  field's advanced RNG stream shift the response bvec by **0.15** on both classes
  — and since this bvec **is** the dissonance input, the dissonance delta is also
  **~0.15**, 3× over the 0.05 tolerance. FAIL on A and B.

So the warm-start is strictly worse here: it costs fidelity and saves nothing.
This is exactly the drift the spec anticipated in marking Tier 3 isolated and
fidelity-gated. It stays **OFF by default**. (A fidelity-safe variant would need
to reset the RNG per turn and use blend≈1.0 — at which point it's just instance
reuse saving a cheap allocation, not a real amortization win.)
