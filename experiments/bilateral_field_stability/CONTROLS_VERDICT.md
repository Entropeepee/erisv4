# Adversarial Controls — Verdict

**Question carried in (from the handoff):** the headline "single 70% → bilateral 30%
collapse" looked suspicious — possibly a **detector artifact** (a hard `eps_lock`
threshold slicing one **unimodal** variance cluster), and it was **open** whether the
membrane does anything **bilateral/mirror-specific** or whether *any* co-evolving
partner near a knife-edge would do the same. The discriminator was **T3 (mirror vs
sham)**, never run.

**Verdict in one line:** The mirror membrane does something **real and
mirror-specific** — but what it does is **DELAY collapse, not prevent it**. At the
T=800 snapshot the whole experiment used, mirror coupling is genuinely and
specifically better than single / matched-noise / every non-mirror partner; but a
long-T check shows this is a **timing shift in a monostable slow relaxation** — by
T≈1200 *every* architecture is fully locked. So the handoff's deepest suspicion holds
(the "30%" is a non-equilibrium snapshot, not a stable alive basin), while the
mechanism result survives (the delay is mirror-specific and not reproducible by noise).

> ⚠️ **The single most important number in this document:** at σ=0.007, collapse
> fraction by step T is — single: 2%(600) **75%(800)** 95%(1000) 100%(1200); bilateral
> lobe-L: 2%(600) **30%(800)** 100%(1000) 100%(1200). **Both reach 100%.** The 45pp gap
> exists only around T=800. See `results/verify_delay_curve.png`.

> All numbers: `controls.py` (dynamics reused verbatim from `field_core`/`metrics`),
> N=60 seeds (T2/T3), N=40 (long-T), regime σ=0.007, μ=0.1. Raw data in
> `results/controls/` and `results/verify_extra.json`.

---

## What the handoff got RIGHT (its suspicion confirmed)

**T1 — the single-vs-bilateral collapse gap is threshold-inflated.** Sweeping
`eps_lock`, the gap peaks exactly at the canonical 3e-5 and vanishes on both sides:

| eps_lock | single | bilateral | gap |
|---|---|---|---|
| 2.6e-5 | 10% | 0% | 10pp |
| 2.8e-5 | 38% | 12% | 27pp |
| **3.0e-5** | 80% | 52% | **28pp** |
| 3.2e-5 | 92% | 78% | 13pp |
| 3.4e-5 | 100% | 97% | 3pp |
| 3.6e-5 | 100% | 98% | 2pp |

**T4 — the variance distribution is UNIMODAL.** Hartigan dip p = 0.99 / 0.70 / 0.38
(σ = 0.006 / 0.007 / 0.008); GMM BIC favors **1 component** at all three. So
"collapse fraction" near σ=0.007 is a threshold cut through **one cluster** that
*shifts* with σ — not two basins. (See `results/controls_unimodality.png`.)

→ **The absolute "30%" is partly a threshold/timing readout. Do not headline it.**

**Nuance from T4 hysteresis:** LOCK is nonetheless an **absorbing** state. Kick a
naturally-alive field by starving novelty for 50 steps and **83%** stay locked after
novelty is restored; kick a locked field with a 5× noise burst and only **2%** escape
and stay escaped. So "staying out of lock" is a *real* dynamical achievement — but the
*unforced* ensemble at σ=0.007 is a slow relaxation (unimodal), not clean bistability.

---

## What the controls NEWLY ESTABLISH (mechanism is real and mirror-specific)

**T2 — it is NOT reproducible by equivalent noise.** Inject the membrane's *measured*
power (amp = μ·RMS|φ_L−φ_R| = 0.1·0.0214 = 0.00214) as amplitude noise into a single
field: **white → 65%, colored → 68%**, versus single 70% and bilateral 30%. Matched
noise barely moves the needle. (A *global* σ-rise to ~0.0078 would reach 30%, but
that's a far larger total perturbation than the membrane injects.)

**T3 — the decisive discriminator: mirror beats every sham.** Collapse %, with a
two-proportion z-test of each arm **against true bilateral (30%)**:

| arm | collapse | z vs bilateral | p |
|---|---|---|---|
| **true bilateral (mirror, mutual)** | **30%** | — | — |
| single baseline | 70% | 4.38 | 1.2e-5 |
| matched colored noise | 68% | 4.20 | 2.7e-5 |
| sham: **mutual, NON-mirror** | 82% | 5.70 | 1.2e-8 |
| sham: independent free partner | 98% | 7.81 | 6e-15 |
| sham: frozen partner | 100% | 8.04 | 9e-16 |

Every non-mirror alternative is **significantly worse than true bilateral** (the
load-bearing claim). Stated precisely against each baseline:

* vs **bilateral (30%)**: frozen, independent, **and** mutual-non-mirror are all
  significantly worse (z = 5.7–8.0, p < 1e-7). Mirror is required.
* vs **single (70%)**: frozen (100%, z=8.0) and independent (98%, z=7.8) are
  significantly worse — a *one-way / frozen* partner actively **promotes** lock.
  But **mutual-non-mirror (82%) is *not* significantly different from single**
  (z=1.49, **p=0.136**) — a mutual non-mirror partner gives **no benefit**, like
  injected noise, neither helping nor clearly hurting.

So the clean reading is: **mutual coupling alone does nothing** (sham_mutual ≈
single); **one-way coupling hurts**; **only the mirror (θ_R = −θ_L, anti-phase)
coupling helps**. The `sham_mutual` arm is the key isolator — mutual **and** coupled
exactly like `BilateralField` but **not** mirror-initialized — and it forfeits the
entire benefit (30% → 82%). **Mirror init is the load-bearing ingredient.**
(See `results/controls_bars.png`.)

> **Why the headline (70/30) differs from T1's canonical row (80/52):** they are two
> *classifiers on the same data*. The headline `.collapsed` uses the real detector
> (tvar < eps_lock AND svar > eps_flat **sustained 30 consecutive steps**, plus death
> modes). The T1 eps_lock sweep uses a cheaper **final-state proxy** (the condition at
> the last step only), which over-counts locks (single 80% vs 70%, bilateral 52% vs
> 30%). T1 is a *sensitivity* tool, not the headline number. No undisclosed parameter
> change.

**Threshold-FREE confirmation (Mann-Whitney U on raw `temporal_var_final`; higher =
more alive).** Bilateral's temporal activity is stochastically **greater** than every
arm — *no threshold involved*:

| vs | p (bilateral more active) | median tvar (×1e-5) |
|---|---|---|
| sham frozen | 2.1e-13 | 2.72 |
| sham mutual (non-mirror) | 1.0e-7 | 2.81 |
| single | 8.5e-6 | 2.83 |
| sham independent | 2.0e-8 | 2.84 |
| white matched | 1.7e-4 | 2.88 |
| colored matched | 3.9e-4 | 2.87 |
| **bilateral (mirror)** | — | **2.99 (highest)** |

Bilateral is the **most temporally-active** arm; non-mirror partners are **more
locked than even a single field** (note: mutual-non-mirror's *median tvar* is below
single, though its collapse *fraction* is not significantly different from single —
the distribution shifts down without crossing the threshold for most seeds). This
ordering is a distributional fact independent of where any threshold sits.

---

## The decisive caveat: DELAY, not prevention (long-T check, N=40)

Everything above is measured at T=800. Running to T=2000 reframes it. Cumulative
collapse fraction vs evolution steps:

| T | single | bilateral lobe-L | gap |
|---|---|---|---|
| 600 | 2% | 2% | 0 |
| **800** | **75%** | **30%** | **45pp** |
| 1000 | 95% | 100% | −5pp |
| 1200 | 100% | 100% | 0 |
| 1500 | 100% | 100% | 0 |
| 2000 | 100% | 100% | 0 |

At T=2000 **all 40/40 seeds are locked in every arm** (single, lobe-L, lobe-R). The
mirror membrane **delays** entry into the absorbing lock state by a few hundred steps;
it does **not** create a sustained alive state. The σ=0.007 regime has **one
attractor (lock)**, reached slowly — consistent with T4 (unimodal + absorbing). So
"collapse fraction within T" is a **clock reading**, and the headline 30%-vs-70% is
the value of that clock at T=800. (See `results/verify_delay_curve.png`.)

This also retires the **lobe-asymmetry** worry (lobe-L vs lobe-R reporting): at long T
both lobes are 100% locked; at T=800 lobe-L≈30% and lobe-R is somewhat higher — both
far below single, but both are snapshots on the same delayed clock.

The delay is still **real and mirror-specific** (at any fixed T, mirror locks later
than single / matched-noise / every non-mirror partner — the shams sit at 82–100% at
T=800 vs mirror's 30%). For a system like Eris that **re-seeds every turn**, delaying
lock *within the active window* may be exactly the useful property. But as a claim
about asymptotic stability, **bifurcation does not prevent collapse in this regime.**

---

## Mechanism (why mirror specifically)

The membrane couples θ_L toward θ_R via `μ·wrap_diff(θ_R, θ_L)`. With the **mirror**
init θ_R = −θ_L this injects an **anti-phase** perturbation that disrupts the
standing-wave phase lock and so **postpones** it. A **non-mirror** partner instead
pulls θ_L toward a *shared* phase — it **synchronizes/biases** the lobe, giving no
delay (mutual-non-mirror ≈ single) or *accelerating* lock (one-way/frozen, which lock
sooner than single). So the result is sharper than "two lobes are better than one":
**only mirror-symmetric anti-phase coupling postpones collapse; generic coupling does
nothing or hurts.** That is a more specific — and more falsifiable — version of the
"why brains are bilateral" claim than the original sweep showed, even though (per the
long-T check) the postponement does not become permanent in this regime.

---

## Honest bottom line

- **Is the membrane doing something real?** Yes. Not reproducible by matched noise
  (T2); bilateral is the most temporally-active arm by a threshold-free rank test
  (p ≤ 1e-5 vs single, ≤ 2e-7 vs shams).
- **Is it bilateral/mirror-specific?** Yes (T3). Mirror coupling is the *only*
  intervention that postpones lock; mutual-non-mirror coupling does nothing
  (≈ single, p=0.136), one-way/frozen coupling *hurts*. All shams are significantly
  worse than mirror (p < 1e-7).
- **Does it PREVENT collapse?** **No.** Long-T (T4): the σ=0.007 regime is monostable
  with an absorbing lock; mirror coupling only *delays* it, and by T≈1200 every arm is
  100% locked. It buys time, not stability.
- **Is the headline 70%→30% effect size trustworthy as a number?** No — it is a
  **clock reading at T=800**, threshold-inflated and read from a unimodal cluster
  (T1 + T4). **State the result as: mirror coupling specifically delays lock —
  ordering + distribution shift + mechanism, not the percentage, and not "prevention."**
- **Net:** a real, mirror-specific *postponement* of collapse; **not** the sustained
  anti-collapse the original sweep implied. To claim true collapse-resistance you must
  first find a regime with a *sustained* alive attractor at long T (none here), then
  test mirror coupling there.

## Adversarial panel (cross-check)

A 4-reviewer adversarial panel (prosecution / defense / independent statistics /
independent mechanism) was run over these raw numbers; their convergent findings,
all reflected above:

* **Defense** (real-and-mirror-specific): matched-noise falsification + sham ladder
  + threshold-free MWU + the non-monotonic μ-band. Own weakest link: the honest
  mechanism is *"mirror delays/slows entry into the absorbing lock state,"* not
  *"creates a distinct alive basin"*; the absolute 30% is a non-equilibrium,
  T-dependent snapshot.
* **Statistics** (mixed): MWU is valid and **survives Bonferroni** (m=6; weakest,
  colored, adj-p=2.3e-3); the collapse-% headline is partly a threshold readout and
  should *not* be the primary evidence. Flagged the 70/30-vs-80/52 discrepancy →
  resolved above (proxy vs sustained detector).
* **Prosecution** (mixed, retreating): strongest artifact case = threshold straddle
  of one unimodal cluster (a ~5.7% median-tvar shift dramatized into a 40pp swing) +
  tuned-σ + favorable-lobe reporting. **Conceded** its core attack fails: it verified
  the matched-noise injection is fairly calibrated (per-step increment dt·amp =
  dt·μ·RMS = 1.06e-4) yet matched noise gives 65–68% while mirror gives 30%; "it's
  just extra noise" is empirically refuted. Retreats to *"real but small, inflated by
  presentation."*
* **Mechanism** (partially decisive): the discriminator cleanly kills "any
  co-evolving partner" (sham_mutual measured RMS phase-drive 1.81 ≈ mirror's, yet
  82% vs 30%), but does **not** isolate mirror-init to a *single* knob — it's a
  conjunction of anti-phase θ, anti-phase ω, and amplitude identity. And
  sham_mutual is not significantly worse than single (confirmed: p=0.136).

**Panel consensus:** the membrane effect is **real and mirror-specific**, the
**absolute effect size is threshold-inflated**, and the cleanest defensible claim is
the *direction + mechanism*, carried by the threshold-free rank test and the sham
ladder — not the headline percentage.

## Recommended next (the long-T result makes the regime question primary)

The mirror mechanism is real, but "collapse fraction within T" is a clock reading in a
monostable regime. Before any further bilateral claims:

1. **Find a regime with a *sustained* alive attractor at long T** — not just slow
   relaxation. Sweep r_sat × σ × omega_spread and, at each point, run to large T and
   require a *stable* alive fraction (and ideally verified bimodality via dip/BIC),
   *not* a T=800 collapse fraction. The σ=0.007 "metastable" point is monostable; it
   is the wrong operating point for an asymptotic-stability claim.
2. In such a regime (if one exists), re-run **mirror vs sham_mutual** at long T: a real
   anti-collapse effect must keep a lobe alive *permanently*, beating the sham where
   "alive" is a true basin — not merely lock a few hundred steps later.
3. Only then the stretch: **functional κ/λ lateralization** vs mirror symmetry — do
   *differentiated* lobes beat *mirror* lobes?
4. **Relevance to Eris as-is:** Eris re-seeds the field every turn, so a *delay* of
   lock within the per-turn active window (a few hundred steps) could already be the
   useful property — the asymptotic-lock result may not matter operationally. Worth
   measuring lock-onset-step vs the actual per-turn step budget before deciding.
5. None of this is wired into Eris; it remains a standalone probe.
