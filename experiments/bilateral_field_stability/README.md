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

### Adversarial controls (T1–T4) — **`CONTROLS_VERDICT.md`**

The 70%→30% result was then put through an adversarial control battery
(`controls.py` + `analyze.py` + `verify_extra.py`, reusing the same dynamics). Two
conclusions:

1. **Real and mirror-specific.** Matched noise of the membrane's measured power
   (white 65% / colored 68%) can't reproduce it; *only* mirror coupling helps —
   mutual-non-mirror coupling does nothing (82% ≈ single 70%, p=0.136) and
   one-way/frozen partners *hurt* (98–100%). All shams are significantly worse than
   mirror (p<1e-7). A threshold-free Mann-Whitney rank test confirms mirror is the
   most temporally-active arm (p≤1e-5 vs single, ≤2e-7 vs shams).
2. **Delay, not prevention.** The absolute effect size is threshold-inflated (collapse
   is a hard cut through a **unimodal** cluster — Hartigan dip p=0.70, GMM BIC favors
   1 component), and a long-T check shows the σ=0.007 regime is **monostable**: mirror
   coupling only *postpones* lock, and by T≈1200 **every** architecture is 100% locked.
   The "30%" is a T=800 snapshot of a slow relaxation, not a stable alive state.

Figures: `results/verify_delay_curve.png` (the decisive delay-not-prevention curve),
`results/controls_bars.png`, `results/controls_unimodality.png`. Raw data:
`results/controls/`, `results/verify_extra.json`. Full write-up: **`CONTROLS_VERDICT.md`**.

---

## Experiment 3 — Sustained two-ness via E-gated coupling — **`BIFURCATE_VERDICT.md`**

The mirror work showed diffusive coupling only *fuses-then-locks* (one attractor =
sameness). This experiment asks whether **E-gated** coupling (the coupling law
`E(Δ)=cos²Δ·sin²Δ`, with `sin²` zero at sameness) between two *distinct, detuned*
agents can sustain a **stable related-but-distinct** coupling angle θ\* — genuine
two-ness — that cosine-only coupling cannot.

**Verdict: CONFIRMED — `sin²` is the cause.** In a sustained-alive regime (σ≥0.013,
T=2500), egate holds a continuous **family of stable interior attractors** (θ\*≈18–58°
across μ): perturbing θ_LR toward sameness *or* orthogonality, it returns to the same
interior value from both sides, with both domains alive and exchange live. The
cosine-only control (`cos`) has **no** interior attractor at *any* μ — it collapses to
fusion; plain `diff` fuses; `iso` segregates. Faithful caveat: this requires the
coupling law to gate the **phase** membrane (θ_LR is phase-dominated; the amplitude-only
contrast is a null) — the handoff's sanctioned `egate_phase`. The §7 dispersion probe
finds **no** Turing band, so spontaneous *spatial* division is not native to this PDE;
the relational route is the route.

Engine: `bifurcate.py` (+ `field_core.py` gated coupling, backward-compatible).
Regime: `stage_a_regime.py`. Map: `run_bifurcate_sweep.sh` → `analyze_bifurcate.py`.
Attractor test: `run_attractor.sh`. Dispersion: `dispersion.py`. Figures:
`results/bifurcate_map.png`, `bifurcate_curves.png`, `attractor_test.png`,
`attractor_vs_mu.png` (decisive control), `dispersion.png`.
