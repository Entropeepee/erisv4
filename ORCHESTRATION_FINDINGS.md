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
| 4 | formalized router | `gate_router` | **fidelity-safe** — enabled by `ERIS_ORCHESTRATION=on` | implemented, measured |
| 5 | failure reports → dreams | `gate_failure_reports` | **safe** — metacognition feature, no perf cost | implemented, tested |
| 6 | β-star bridge | `use_beta_star` | **flagged OFF** — neutral, but consumer is dormant | implemented, tested |

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

## Tier 4 — formalized router → **fidelity-safe; the one to enable**

**What it does.** Replaces the binary local-vs-ensemble `_router_gate` with the
shared `CriticalityMonitor` on the |dC/dX| anomaly, widened to four decisions:
CONTINUE → one local call (default); SWITCH → a **single** cloud expert (cheaper);
ESCALATE → the full cloud ensemble; SUSPEND → specialist finding, no LLM. The
transfixion override in `workspace.py` is unified under the same SWITCH
vocabulary (kept on its richer reactivity probe; not rewired, to preserve winner
selection). Behind `gate_router`; baseline path unchanged when off.

**Measured (router-only A/B, `--gates router`).**
- *Easy (A): provably untouched* — cloud 0→0, **answer Δ 0.000**. Easy turns
  stay local, byte-for-byte the baseline answer.
- *Hard (B): engages cloud within tolerance* — it takes a cloud path on genuine
  outliers (cloud 0→0.2 mean) with **answer Δ 0.028 < 0.05** and dissonance Δ
  0.027. No regression.
- *The saving is structural, not corpus-visible.* The local-first baseline
  escalates ~never on this offline corpus, so there's no "both escalate" turn to
  show SWITCH (1 call) beating the old full ensemble (N calls). That saving is
  proven by unit tests (`test_escalate_counts_full_ensemble` = N,
  `test_switch_counts_single_expert` = 1) and pays off in production where real
  cloud experts and real outlier patterns exist.

**Conclusion.** The only gate that is **fidelity-safe** (Δ ≤ 0.028 on hard, 0 on
easy) and structurally better than the old binary router. It is what
`ERIS_ORCHESTRATION=on` enables. Emits `FailureModeReport`s on SWITCH/ESCALATE
for Tier 5.

## Tier 5 — failure reports → dream queue → **safe (metacognition, not perf)**

**What it does.** When a gate makes a mechanism-changing decision (router
SWITCH/ESCALATE), the orchestrator turns its `FailureModeReport` into a question
in `dreaming_loop.pending_questions` (CIP §0111 — never silently proceed; reflect
on it). The orchestrator mediates, so gates stay decoupled from the dream loop.
Behind `gate_failure_reports`.

**Measured.** Pure plumbing — no resource or fidelity effect. Unit-tested both
ways: a forced ESCALATE adds exactly one dream question when the flag is on, and
none when off. Safe to enable alongside the router; it makes the router's
escalations *observable* in Eris's metacognition rather than invisible.

## Tier 6 — β-star bridge → **neutral, but inert; leave OFF**

**What it does.** Self-tunes the Davidian β threshold from the sample-size ratio
ψ via `beta_star(ψ)` instead of the hand-set ψ baseline, in `params_from_bvec`.
Behind `use_beta_star`.

**Measured.**
- *Neutral on winner selection.* Toggling β-star changes β (0.62 → 0.75 on the
  test bvec) but the dominant eigenvalue/winner is unchanged; only near-mean
  components (shrunk to ~equal, irrelevant to selection) reshuffle. Unit-tested.
- *Inert at runtime.* The feared "highest blast radius" doesn't materialize here:
  `params_from_bvec`/`shrink_eigenvalues` are **not called anywhere in the live
  pipeline** (only `davidian_weight` is consumed). So the bridge has zero runtime
  effect today — the benchmark with `--gates beta_star` reads Δ 0.000.

**Conclusion.** Neutral by the spec's bar, but it delivers no benefit because its
consumer is dormant. Per "merge only if neutral-or-**better**," it stays **OFF**.
Wired and verified, ready if a future caller routes `params_from_bvec` into
MoEGate/CSBA scoring.

---

# Summary — what to enable

The Tier 0 discipline ("don't trust guesses; measure") delivered a clear,
honest result on this engine + offline corpus:

- **Enable (fidelity-safe, structurally better):** the **router** (Tier 4) +
  **failure-reports → dreams** (Tier 5). This is exactly what
  `ERIS_ORCHESTRATION=on` turns on. Easy turns are provably untouched (Δ 0);
  hard turns engage cloud within tolerance; moderate outliers take the cheaper
  single-expert SWITCH instead of the full ensemble (a real production saving).
- **Leave OFF (don't earn their place here):** the **field-depth** (Tier 2) and
  **response-field** (Tier 3) gates regress the answer because the field bvec is
  still converging at the step budget — early termination costs 0.2–0.65 / 0.15
  in answer Δ for no real savings. The **β-star** bridge (Tier 6) is neutral but
  inert (dormant consumer).

Net: the orchestration machinery (shared noise floor, criticality monitors, the
four-decision interface, the A/B ruler) is built, tested (148 green), and
measured. The one change that helps in production is on; everything that would
trade the answer for speed is off — by measurement, not by guess.
