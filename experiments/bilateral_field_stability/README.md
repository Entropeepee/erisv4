# Bilateral Coherence-Field Stability Test

A standalone, pure-numpy probe asking one question decisively:

> **Does splitting a single near-critical coherence field into two coupled mirror
> lobes (a corpus-callosum membrane of permeability μ) make it resist collapse?**

This is a *probe to inform Eris*, not a change to Eris. Nothing here imports from
or wires into the live architecture — the field dynamics are a faithful **lift**
of `eris/field/pde.py`, re-implemented self-contained so the result transfers.

---

## Background

Eris evolves a 2D coherence field — amplitude `phi(x,y)` and phase `theta(x,y)`
on a 64×64 torus — under PDE dynamics; the *shape* of that evolving field steers
retrieval/routing/generation. Earlier versions suffered **standing-wave collapse**
("transfixion"): the field locked into a static pattern and stopped changing — a
dead cognition.

The bilateral hypothesis (from the FRACTAL–BLECD "Why Brains Are Bilateral"
derivation): a *single* field near criticality is collapse-prone; **two coupled
mirror lobes** can perturb each other out of standing-wave locks — but **only at
partial coupling**. μ=0 is two independent fields (no shared benefit);
μ→large synchronises them into effectively one field (no benefit); an
**intermediate μ band** is the predicted win.

---

## What's faithful, what's stripped

Lifted **verbatim** (numpy form) from `eris/field/pde.py` into `field_core.py`:

| Repo (`pde.py`)        | Here (`field_core.py`)        |
|------------------------|-------------------------------|
| `FractalField.step`    | `SingleField.step_with_coupling` |
| `_base_ops` (8 BLECD domains S,D,F,TH,B,E,IBT,ZBT) | `SingleField._base_ops` (identical) |
| `_vorticity` (τ = ∇ρ×∇θ, wrap-safe) | `vorticity` |
| `_phase_step` (Kuramoto) | `SingleField._phase_step` |
| soft ceiling (`B` op) + hard `clip(.,0,B_max)` | same |
| `W_ELASTIC` / `W_PLASTIC` regime weights | same |
| `hill_power`, `sigmoid_gate`, `colored_noise`, `local_coherence`, `wrap_diff` | same |

Deliberate, documented deviations:

* **Torus kept perfect** — all stencils periodic (`np.roll`); the repo's
  `_enforce_dirichlet` edge-zeroing is dropped (per the brief: keep the torus).
  Dirichlet edges would inject boundary artefacts that confound a collapse metric.
* **numpy only** — the repo's `xp`/`to_gpu` CuPy indirection is removed.
* **No LLM / embeddings / orchestrator** — seeding is a colored-noise field.

The **bilateral** variant (`BilateralField`) adds a second mirror lobe and a
Robin/Newton-cooling membrane; nothing else in the dynamics changes:

```
dphi_L += mu * (phi_R - phi_L)              # amplitude exchange
dtheta_L += mu * wrap_diff(theta_R, theta_L)  # wrap-safe phase exchange
```

Mirror init: `phi_R = phi_L`, `theta_R = -theta_L`, `omega_R = -omega_L`
(=> `∇²theta_L = -∇²theta_R`, the bilateral derivation's ansatz).

---

## The fair-fight design (why lobe-L is the headline metric)

Lobe **L** is constructed with the **same seed** as the single-field baseline.
At **μ=0** the membrane is off, so **lobe L evolves bit-for-bit identically to the
single field** — an exact control. Any change at μ>0 is purely the membrane acting
on the *same* seed. So the headline is **lobe-L collapse fraction vs μ**, with the
single-field fraction as a horizontal baseline. (The run verifies the μ=0 identity:
lobe-L collapse == single collapse.)

The **combined / averaged** readout is reported as a secondary curve, with a
caveat: averaging two mirror lobes *cancels* their anti-phase temporal motion, so
the averaged descriptor looks more locked than either lobe. A concatenated readout
would be the right architectural choice; averaging is shown only for contrast.

---

## Collapse metrics (operational, raw numbers logged — `metrics.py`)

* **LOCK** (standing wave): rolling per-cell temporal variance of `phi` over a
  window → below `eps_lock` for `n_consec` steps while spatial structure persists.
* **DEATH**: spatial variance of `phi` → 0 (FLAT) **or** mean local Kuramoto
  coherence → 0 (NOISE), sustained `n_consec` steps.
* **DIVERGE**: NaN/Inf or `|phi|` blows past a ceiling.
* **ALIVE**: none of the above — sustained temporal variation **and** persistent
  spatial structure **and** bounded.
* **COLLAPSE = LOCK ∪ DEATH ∪ DIVERGE** within `T` steps.

---

## Collapse-prone regime (drives transfixion on purpose)

`collapse_params()`: strong saturation (`r_sat=0.85`), near-ceiling start
(`phi_init=0.85`), minimal novelty (`sigma_noise=sigma_phase=0.004`), tight
intrinsic frequencies (`omega_spread=0.25`, → easy global phase sync, which *is*
the standing-wave lock). A regime map (`regime_map.py`) confirmed collapse is
confined to low `omega_spread` + low noise; this point sits in it with the single
field locking in the large majority of seeds.

---

## Run it

```bash
cd experiments/bilateral_field_stability
python3 run_experiment.py --quick           # 16 seeds, sanity (~10 min)
python3 run_experiment.py --seeds 60 --T 800 --tag full   # full sweep
python3 plot.py full                         # collapse-vs-mu plot
```

Checkpoints land in `results/sweep_<tag>.json` after every μ (resumable-ish; a
long sweep can be inspected mid-flight). Raw per-run series are decimated into the
JSON for audit.

---

## Results & verdict

**Decisive answer: YES** — partial coupling (μ ≈ 0.1) cuts collapse from **70% → 30%**
(n=60, p≈1.2×10⁻⁵), while μ=0 is inert and μ≥0.3 is *worse* than a single field.
See **`VERDICT.md`** for the full raw table and interpretation.

Figures: `results/collapse_vs_mu_cool.png` (the metastable, fair-fight regime —
**the headline**) and `results/collapse_vs_mu_full.png` (the maximally-hot σ=0.004
regime, single=100%, where the lock is too deterministic for the membrane to help).

Raw data: `results/sweep_cool.json`, `results/sweep_full.json`,
`results/scan_cool.log` (the σ-criticality scan).
