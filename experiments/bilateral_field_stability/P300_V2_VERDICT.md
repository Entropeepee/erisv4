# P300 v2 — The Prediction-Error Wave — VERDICT

**Scope note (read first).** This is a *standalone probe* built beside the coherence-field
experiments to inform Eris, not to be wired into it. Nothing here is merged to `main`, nothing is
wired into the Eris architecture, and no semantic/meaning layer was started. The branch decision
(whether to pursue a Level-2 semantic wave) remains open and is explicitly **not** taken here.

---

## PROOF STATUS

| Prediction | Base field | Excitable field | Verdict |
|---|---|---|---|
| **P1** double dissociation (wave⊥exchange) | **NULL** (p=0.20) | **CONFIRMED** (A=0.993, p=1.3e-20) | emergent-with-excitability |
| **P2** ignition / sigmoid | graded/linear | **CONFIRMED** sigmoid (thr≈0.6) | excitable only |
| **P3** precision → bigger wave | null (ρ=+0.08) | **CONFIRMED** (ρ=+0.66) | excitable only |
| **P4** propagation + self-extinction | no pulse | **CONFIRMED** for the *activator* (spread 0.13→0, a→0); coherence **steps** (does not return) | partial — see caveat |
| **P5** flow control (predicted@max-exchange → no wave) | weak | **CONFIRMED** (A=0.993, p=4.7e-14) | excitable only |

**One-line bottom line.** The base coherence field reproduces the Level-1 *null*: a
prediction-error drive sits below intrinsic fluctuations and produces no measurable wave. Adding a
**minimal, generic FitzHugh-Nagumo excitable layer** (not tuned to the headline — verified by a
25-variant robustness sweep) turns the same architecture into a clean, thresholded, precision-gated,
propagating, self-extinguishing, prediction-locked transient that dissociates from the exchange
rate. The dissociation's *orthogonality to Δ* is architectural (by construction); its *existence as
a measurable effect*, its *threshold*, its *precision-gating*, and its *adaptation* are emergent
dynamical properties of the excitable regime, absent in the base field.

All numbers below are raw, from 20 seeds/cell, threshold-free statistics (Mann-Whitney U with
tie-corrected normal approximation; Spearman ρ; common-language effect size A = P(X>Y)). Data:
`results/p300v2/*.json`; code: `p300_v2_prederror.py`; analysis: `analyze_p300v2.py`.

---

## 1. What was built

Two **separately measured** channels, reusing `field_core.py` / `bifurcate.py` verbatim behind flags:

- **EXCHANGE** — E(Δ)-gated transport, a pure function of the coupling angle Δ:
  `exchange = mean(coupling_gate(Δ, "egate"))` = ¼·sin²(2Δ). Reported as a magnitude. By design it
  is 0 at Δ=0°/90° and maximal (0.25) at Δ=45°.
- **WAVE** — the receiver's global phase-coherence transient triggered by the **precision-weighted
  prediction-error residual**, injected **ungated by Δ**:
  `residual = wrap_diff(stimulus_texture, κ_angle)`, `drive = w_gain · precision · |residual|`.

**κ_pred** is a running prediction of the next stimulus texture: a circular EMA (`circ_ema`, α=0.25)
of the stimulus history. **Precision** = the EMA resultant length |κ| ∈ [0,1] (sharpness of the
prior; the Fisher knob for P3). κ_pred is updated **only after** the response is measured — it never
peeks at the current stimulus (audit (i) below).

**Excitability (flag).** The base field is relaxational — a residual kick simply re-settles, giving
no ignition. When `--excitable` is set, the residual drive seeds a **FitzHugh-Nagumo activator**
layer `a` (threshold cubic + Laplacian diffusion + slow refractory recovery `w`) which supports a
thresholded, propagating, self-extinguishing pulse; the activator then forces the receiver phase.
The excitable layer is reported strictly **pre/post** so its contribution is visible, per the
handoff's instruction not to hand-tune to manufacture the result.

---

## 2. The five predictions (raw numbers)

### P1 — Double dissociation (the headline)

`crossed` design: Factor A = coupling angle Δ ∈ {0°,45°,90°}, Factor B = prediction match
{predicted, violating}. n=20 seeds/cell.

```
                        Δ=0°            Δ=45°           Δ=90°
BASE   predicted    0.0078±0.0011   0.0078±0.0011   0.0078±0.0011
       violating    0.0084±0.0010   0.0084±0.0010   0.0084±0.0010     <- NULL
EXC    predicted    0.0078±0.0011   0.0078±0.0011   0.0078±0.0011
       violating    0.0509±0.0053   0.0509±0.0053   0.0509±0.0053     <- CONFIRMED
exchange (both)     0.0000          0.2500          0.0000
```

- **Base:** violating (0.0084) vs predicted (0.0078), Mann-Whitney **p=0.20, A=0.568** — no
  dissociation. The prediction-error drive is below the intrinsic coherence fluctuation floor
  (~0.0078). This is the Level-1 null, reproduced.
- **Excitable:** violating (0.0509) vs predicted (0.0078), **U=27, z=-9.31, p=1.3e-20, A=0.993**.
- **Dissociation geometry (excitable):** wave relative-range across Δ = **0.00** (flat — wave
  ignores Δ); exchange relative-range across Δ = **3.00** (wave tracks *error*, exchange tracks
  *angle*). This is the double dissociation ii≈iii ≫ i≈iv.

### P2 — Ignition / sigmoid

`p2` sweep: violation size ∈ {0, 0.2, 0.4, 0.6, 0.9, 1.2, 1.6, 2.2} at Δ=45°.

```
viol   0.00   0.20   0.40   0.60   0.90   1.20   1.60   2.20
BASE  .0078  .0079  .0078  .0079  .0082  .0084  .0090  .0096   sigmoid RMSE 0.0013 vs linear 0.0002 -> LINEAR
EXC   .0078  .0077  .0078  .0173  .0373  .0509  .0597  .0618   sigmoid RMSE 0.0027 vs linear 0.0070 -> SIGMOID
a_pk  .0000  .0017  .0120  .0269  .0509  .0719  .0912  .1073   (excitable activator peak)
```

Excitable: below viol≈0.4 the wave stays at floor; between 0.6 and 1.2 it ignites; then saturates.
A genuine threshold, not graded gain. Base: monotone-but-tiny, best fit linear.

### P3 — Precision → bigger wave

`p3` sweep varies the prelude spread (input inconsistency), which sets prior precision |κ|.

```
prelude_spread   0.05    0.40    0.90    1.60
precision |κ|    0.999   0.934   0.711   0.418
BASE wave       .0084   .0083   .0081   .0080     Spearman(prec,wave) = +0.078  (no effect)
EXC  wave       .0508   .0477   .0344   .0105     Spearman(prec,wave) = +0.657  (strong)
```

Higher precision → bigger wave, but only in the excitable regime. This is the Fisher
precision-weighting realized dynamically: low precision keeps the drive sub-threshold.

### P4 — Propagation + self-extinction

`crossed`/`trace`, violating cells, excitable:

- **Propagation:** activated fraction `spread_peak = 0.129±0.001` of the 64×64 field (the pulse
  spreads via diffusion), from a localized seed.
- **Activator self-extinction:** `a_peak 0.072 → a_final 0.000`, `spread_final 0.000` — the pulse
  extinguishes by refractory recovery. Trace confirms a rise+fall shape (`a: 0 → 0.091 @ t12 → 0`).
- **CAVEAT (honest, and important):** the *activator* is a genuine transient, but the *global
  coherence order parameter* does **not** return to baseline — `coh_return = 0.26` (0 = permanent
  step, 1 = full return). The receiver **integrates** the prediction-error input and settles to a
  new coherence level (a registered change). We therefore claim self-extinction **only of the
  activator pulse**, and report `coh_return` transparently. Calling the coherence excursion itself a
  "self-extinguishing transient" would be an overstatement (see audit (ii)).

### P5 — Flow control (the second crux)

`p5`: a sustained stream of **predicted** inputs at Δ=45° (exchange = 0.25 = maximal), then one
violation at the end.

```
                          wave           exchange
EXC predicted stream   0.0047±0.0001     0.250 (max)     <- flat, NO waves despite max flow
EXC end violation      0.0642±0.0082                     <- spikes
   violation > stream: A=0.993, p=4.7e-14
BASE predicted stream  0.0048   /  end violation 0.0069  (A=0.704, p=1.9e-3; weak)
```

Sustained maximal exchange with predicted input produces **no wave**; a single violation ignites
one. Exchange and wave are orthogonal, exactly as predicted.

---

## 3. Controls

- **No-prediction (frozen κ).** With κ frozen the drive is removed and the wave collapses to the
  0.0078 floor even for violating input (excitable: A(viol>frozen)=0.993, p=1.3e-20). The wave
  *requires* the running prediction. (This control zeroes the error channel by fiat; the stronger
  positive evidence that κ is genuinely predictive is the **adaptation** test below.)
- **Adaptation (κ genuinely learns).** After a step-change in the input mean, the wave spikes then
  decays back to floor as κ_pred tracks the new mean:

  ```
  after-shift k:     0      1      2      3      ...   13
  EXC  wave       0.0556 0.0228 0.0171 0.0072  ...  0.0045   (decay ratio 0.08, Spearman(k,wave)=-0.523)
  error (both)    1.2181 1.0179 0.7643 0.4264  ...  0.0222   (55× decay as κ learns)
  ```

  A fixed (non-learning) reference could not show this. The independent audit (i) additionally ran a
  frozen-κ control and confirmed the decay is caused by κ learning, not field habituation.
- **Base-field / flow / equiprobable controls** are covered by the base-vs-excitable contrast and P5
  above.

---

## 4. Independent adversarial audit

Four independent skeptical auditors read the actual source and tried to **refute** one claim each.
(The multi-agent Workflow permission stream failed, as it has intermittently this session; the audit
was run as four parallel `Explore` subagents instead — same adversarial protocol.)

**(i) κ_pred is genuinely predictive and never peeks — HOLDS.** The auditor verified the causal
order (measure with the old κ, *then* update; `update_pred` is never called inside `probe`), that
the wave is driven by the prediction *error* (`wrap_diff(texture, κ_angle)`) not the raw stimulus,
and independently ran a frozen-κ control to falsify the field-habituation hypothesis (error/wave
stay flat when κ can't learn). `make_texture("predicted")` building from κ is legitimate (it
generates the stimuli the prior should predict; the residual still exposes any mismatch).

**(ii) The observable is a transient, not a step — PARTIAL → framing corrected.** The auditor's
sharp and correct catch: the *activator* is genuinely transient (a_final=0), but the *coherence*
order parameter steps to a new plateau and does **not** return to baseline, because the activator
accumulates a phase shift. The original draft's "self-extinguishing wave" language conflated the
two. **Fixed:** the code now records `coh_return` (=0.26, making the step explicit), and the verdict
claims self-extinction **only of the activator pulse**. (An attempted leaky-integrator variant that
would have made the coherence itself return destroyed the signal amplitude; forcing it would have
been exactly the hand-tuning the handoff warns against, so we instead report the step honestly.)

**(iii) Excitability is minimal / not knife-edge — REFUTED → addressed.** The auditor correctly
found the original robustness sweep (a) left load-bearing constants un-swept (`seed_thr`, `rec_w`,
the `dt=0.5` step, the `[0,1.5]` clip) and (b) only tested "violating > predicted" (which the
ungated channel guarantees) rather than the ignition **threshold**. **Fixed:** the sweep now
perturbs **all 9** FHN constants ±40% one-at-a-time **plus 6 joint perturbations** (every constant
jittered together), and tests the threshold structure directly — a sub-threshold violation (0.3,
must stay at floor) vs a supra-threshold one (1.4, must ignite). Result: **the ignition threshold
survives all 25 variants** (sub stays 0.0075–0.0079 at floor; supra ignites 0.044–0.061;
A=0.93–0.998, all p<4e-6). The *qualitative* signature (floor-below, ignite-above, self-extinguish)
is generic to excitable media; the *quantitative* threshold location (viol≈0.6) is a free parameter
and is not itself claimed as a prediction.

**(iv) By-construction critique — PARTIAL, and it reshapes the headline.** The harshest auditor is
right that the wave's *orthogonality to Δ* is tautological: the wave is wired ungated by Δ and the
exchange is a pure function of Δ, so they *must* dissociate. The redeeming fact — which becomes the
honest headline — is that the **base field shows no measurable dissociation (p=0.20) under the exact
same wiring**. So the dissociation-*as-a-phenomenon* is not guaranteed by the architecture; it
requires the excitable dynamics. The genuinely non-trivial, non-tautological findings are: the
**base-null vs excitable-effect contrast**, **precision-gating** (ρ=+0.66, absent in base),
**adaptation** (learning-driven decay), and the **sigmoid ignition threshold** — none of which are
"wired in."

---

## 5. Honest ledger — by-construction vs emergent

**By construction (not a discovery):**
- The wave is Δ-independent and the exchange is Δ-tuned. (Two separate channels, wired that way.)
- That violating > predicted *in sign* (the drive ∝ |residual|).

**Emergent / non-trivial (the actual science):**
- The dissociation only becomes a **measurable effect** in the excitable regime (base p=0.20 →
  excitable p=1.3e-20 under identical wiring).
- **Sigmoid ignition threshold** (P2) — a bifurcation-like all-or-none, absent in the base field.
- **Precision-gating** (P3, ρ=+0.66 vs +0.08) — the wave scales with prior sharpness because low
  precision keeps the drive sub-threshold.
- **Adaptation** (κ learns; wave self-extinguishes across presentations as error decays).
- **Robustness** of the ignition threshold across 25 FHN perturbations.

**Contingent on an added ingredient (disclosed):**
- All of P2/P4 and the *magnitude* of P1/P3/P5 depend on adding the FHN excitable layer. The base
  coherence field is **not** intrinsically excitable. The correct claim is "*if* the field carries a
  generic excitable layer, *then* prediction error produces a P300-like ignition wave orthogonal to
  the exchange rate," not "the coherence field is intrinsically a P300 generator."

---

## 6. Limitations

- The coherence excursion is a **step**, not a return-to-baseline transient (§2 P4 caveat). A true
  return-to-baseline P300 analogue would need a coherence-restoring mechanism we did not add (and
  did not tune in).
- Excitability is an **external** ingredient, not derived from the coherence PDE. Whether the Eris
  field naturally supports such excitability is untested and is the obvious next question.
- The specific threshold location and wave amplitudes are parameter-dependent; only their
  *qualitative existence and robustness* are claimed.
- P1's Δ-orthogonality is architectural; the load-bearing science is the base-vs-excitable contrast,
  P3, and adaptation.

---

## 7. Verdict

**P300 v2 is a sound, honestly-scoped result with one corrected overclaim.** On the base coherence
field it is a **null** (the Level-1 result reproduced). With a **minimal, generic, robustness-tested
FHN excitable layer**, the prediction-error wave shows a clean double dissociation from the exchange
rate (P1, P5), sigmoid ignition (P2), precision-gating (P3), and a propagating, self-extinguishing
*activator* pulse (P4). κ_pred is a genuine, non-peeking, adapting predictor (audit i, confirmed).
The excitability is not knife-edge tuning (audit iii, 25-variant threshold sweep). The one audit
correction: the *global coherence* integrates the input to a new level rather than returning to
baseline, so self-extinction is claimed only of the activator (audit ii) — reported transparently
via `coh_return`.

The decisive, non-tautological claim: **the same two-channel architecture is inert on the base field
(p=0.20) and produces a robust, precision-gated, adapting P300-like ignition wave once the field is
excitable — a wave that tracks prediction error orthogonally to the coupling angle.**

**Not merged to `main`. Not wired into Eris. No semantic/meaning layer started.** Branch decision
(Level-2 semantic wave) deferred.

---

### Reproduce

```
cd experiments/bilateral_field_stability
python3 p300_v2_prederror.py crossed --seeds 20 [--excitable]   # P1 + no-prediction control
python3 p300_v2_prederror.py p2      --seeds 20 [--excitable]   # P2 ignition sweep
python3 p300_v2_prederror.py p3      --seeds 20 [--excitable]   # P3 precision sweep
python3 p300_v2_prederror.py p5      --seeds 20 [--excitable]   # P5 flow control
python3 p300_v2_prederror.py adapt   --seeds 20 [--excitable]   # audit (i): κ adaptation
python3 p300_v2_prederror.py trace   --seeds 20 [--excitable]   # audit (ii): transient shape
python3 p300_v2_prederror.py robust  --seeds 20 --excitable     # audit (iii): 25-variant threshold sweep
python3 analyze_p300v2.py                                       # threshold-free stats for all of the above
```
