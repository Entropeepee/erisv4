# CIP v9 patent ↔ Eris orchestration (Tiers 0–6): does it land in the right places?

Review of `CIP_v9_full_USPTO_double_spaced.docx` ("Cross-Stage Computational
Orchestration with Shared-Sigma Quiet Zones", Willow IP Group LLC, inv. D. Pope)
against the Tier 0–6 implementation on branch `orchestration-cip`.

**Bottom line.** The patent is sound and the *discipline* was transplanted
faithfully. But the patent gates **one specific structure** — a multi-stage
computation where a cheap Stage 1 produces an answer with a **numerical residual
that converges monotonically to a user tolerance**, and an expensive Stage 2 only
refines it. That structure exists in Eris at **exactly one boundary** (local→cloud
LLM = the patent's own speculative-decoding embodiment), which Tier 4 nailed.
It does **not** exist at the field-evolution boundaries (Tiers 2–3), and applying
it there was a category error — now proven empirically, not just "no savings."
**Salvageable: yes, narrowly. Scrap the field gates; keep the router.**

---

## 1. What the patent actually requires (the precondition everything rests on)

The invention (claims 1, 16; §§[0004]–[0014]) instruments a **stage boundary**
of a **multi-stage numerical workload** with a criticality monitor that issues
continue/suspend/switch/escalate. Its validity rests on four preconditions:

1. **A defined answer with a user tolerance** (`10^-6` relative residual in the
   validation, §[0095]). "Suspend" means *continuing would not change the answer
   within tolerance.*
2. **A convergent residual** — a signal that marches **monotonically toward a
   known target** (zero / tolerance). The schedule-convolution predictor
   (§[0057]) forecasts the residual's **log-linear decay**; the gate fires when
   the predicted residual is already inside the band, or **stalls** vs that decay
   (→ switch/escalate, §[0098]).
3. **A genuinely skippable expensive stage** — Stage 2 (Conjugate Gradient)
   *refines* Stage 1's answer; skipping it when Stage 1 already hit tolerance
   loses nothing. That's the 1.5×–44× win (§[0096], FIG 4).
4. **Shared σ across boundaries of the same workload** (claims 2, 12) — DST→CG
   and CG-mid-iteration gates share one variance estimate.

Everything else (six BFECDS Stage-1 families, seven resource axes, hardware-event
dispatch, beta-star Hill-Power, failure reports) hangs off this skeleton.

## 2. The mapping, tier by tier

| Tier | Patent basis | Right boundary? | Correct assumption? | Verdict |
|---|---|---|---|---|
| 0 ruler | §[0014], FIG 4 (A/B value structure) | n/a (measurement) | yes — measures answer-Δ vs tolerance | **correct** |
| 1 shared σ + monitor | claims 1–2, §§[0058]–[0067] | n/a (substrate) | **improved** — see §3 | **correct (with fix)** |
| 2 field-depth | claim 1 + §[0069F] early-termination | **NO** | **NO** — field has no convergent residual | **mis-placed → shelve** |
| 3 response-field | §[0053] warm-start/amortization | **NO** | **NO** — same; bvec never converges | **mis-placed → shelve** |
| 4 router | **§[0101] / claim 20 — the patent's OWN speculative-decoding embodiment** | **YES** | mostly — see §4 | **correctly placed** |
| 5 failure reports | claim 10, §§[0109]–[0111] | YES | yes | **correct** |
| 6 beta-star | claims 4, 17; §[0068] | the consumer is dormant | yes (neutral) | **inert** |

## 3. Where Tier 1 was *more* correct than the patent's literal text

The patent says "a **single sigma** value used by a plurality of monitors"
(claims 2, 12). Taken literally that breaks Eris, because its gated signals live
at wildly different scales (dissonance ~1.0, dC/dX ~1e-3). The implementation
correctly read §[0044] (the σ is a *robust estimator family*, MAD×1.4826 /
Hill-Power) and built **per-signal local scale + one shared global multiplier**.
That is the right reading of the patent's intent and a genuine improvement over
its literal wording. ✔

## 4. Where Tiers 2–3 went wrong (the category error), proven

The remediation spec treated the Kuramoto **field evolution** as "an iterative
solver" (§5 of the spec, citing §[0069F]). It is not. Three measurements on the
real engine:

- **The settling signal never fires.** After the seeding transient the coherence
  delta **plateaus** (~2.4e-3 easy / ~0.5e-3 hard) — a steady *drift*, not decay
  to a fixed point. There is no residual converging to a tolerance.
- **The answer is still moving at the step budget.** Response-bvec distance for an
  early stop vs the full 50 steps: **0.24–0.27 at n=30, 0.36–0.65 at n=12–20**
  (tol 0.05). "Continuing would not change the answer" is simply false here.
- **Even the discrete decision hasn't converged.** The archetype the field feeds
  downstream is *still flipping between steps 30 and 50* (Emergence Catalyst →
  Organic Flow Field), at different steps for different prompts. So there is **no
  convergent signal anywhere** — not the bvec, not the categorical output.

The patent's gate **requires** precondition #2 (a convergent residual). The field
has none — by design; Eris is a *living dynamical system*, not a solver seeking a
fixed point. So the field-evolution boundary is the wrong place for this gate, and
the benchmark's "no savings / fidelity regression" results were the symptom, not
the disease. **This is the correction:** Tiers 2–3 are not "untuned," they are
mis-placed.

### The BFECDS trap (why the mis-placement was tempting)
The patent organizes its six Stage-1 compression families by **BFECDS** (Boundary,
Feedback, Emergence, Criticality, Decay, Saturation) and maps each to a numerical
mechanism (FFT, RNLA, Strassen, multigrid, constraint-projection, Lie-reduction,
§[0046]–[0051]). Eris *also* uses BFECDS — as **cognitive field activation
channels** (the BVec). **Same vocabulary, different mathematics.** The patent's
BFECDS are operator-compression families with a convergent residual; Eris's are
dynamical activation coordinates with no convergence target. The shared name makes
the orchestration look like it should "obviously" apply to the field. It doesn't.
That nominal-not-structural overlap is, I believe, the origin of the mis-mapping.

## 5. Where Tier 4 is exactly right

The router is **the patent's own alternative embodiment**, almost verbatim:
§[0101] / claim 20 — "the boundary between a **fast model proposing tokens** and a
**larger verifier model** … the criticality gate replaces the heuristic
confidence threshold … with a noise-floor-relative gate." Eris's local Ollama =
draft/Stage 1; cloud experts = verifier/Stage 2. The four decisions map cleanly
(CONTINUE=local, SWITCH=one expert, ESCALATE=full ensemble, SUSPEND=specialist).
Measured fidelity-safe (easy Δ 0.000, hard Δ 0.028). This boundary has a real
"is the cheap stage good enough, or escalate to the expensive one?" decision —
the patent's structure, genuinely present.

**One honest gap (a refinement, not a correction):** the patent gates on a
*residual vs tolerance* with a *trajectory predictor + CUSUM stall detection*
(§[0067], claim 5). The router gates on a single-sample |dC/dX| **anomaly**
z-score. That's a defensible proxy (there's no numerical "is the local answer
correct" residual without running the cloud), but it is a heuristic, not the
patent's predicted-residual mechanism. A more faithful router would track the
dC/dX trajectory and escalate on a change-point. Worth doing only if the router
ever proves too eager/lazy in production.

## 6. Is the concept salvageable?

**Yes — but only the part that was already kept.** The salvage is a *scoping*
conclusion, not new code:

- **Keep & enable (correctly placed):** Tier 4 router + Tier 5 failure-reports.
  This is the patent applied where its preconditions actually hold. It's
  fidelity-safe and is what `ERIS_ORCHESTRATION=on` turns on.
- **Shelve as negative results (mis-placed):** Tiers 2–3. Leave the code (it's
  correct, unit-tested machinery) flagged OFF and documented as "the field is not
  a convergent solver." Do **not** invest further in tuning them — the boundary,
  not the tuning, is wrong.
- **Leave inert:** Tier 6 beta-star — wired, neutral, but its consumer
  (`params_from_bvec`) isn't on any hot path.

**The honest scope of the patent in Eris:** the patent is a *numerical-computing*
orchestrator. Eris's only hot, gateable, multi-stage boundary with the patent's
structure is the **local→cloud LLM** decision. The patent's bulk — six operator
families, seven resource axes, hardware dispatch, convergent-residual gating — has
no counterpart in a resonant cognitive field and should not be forced onto it.

### If you want more value from the patent later (candidate boundaries, unbuilt)
Only boundaries with a *cheap-stage → expensive-external-stage* decision qualify:
- **Research/grounding escalation** (`_ground_if_contradicting`, dreaming research
  cascade): Stage 1 = local memory retrieval; Stage 2 = expensive web/expert call.
  Gate: skip the web pass when local retrieval coverage is within a confidence
  band. *Currently this path is rare and already cheaply gated by contradiction
  markers, so the payoff is small — measure before building.*
- **Deep retrieval / FAISS depth**: gate how many tiers/chunks to search when the
  top-k similarity already saturates. Small, safe, but small payoff.

Both are the *same* structure as the router. Neither touches the field. That is
the rule the patent gives you: **gate resource-escalation decisions, never the
cognition itself.**

## 7. Recommendation

1. **Do not scrap the work** — the ruler, shared estimator, monitor, four-decision
   interface, router, and failure-reports are correct and the router earns its
   place. Merge the branch; run with `ERIS_ORCHESTRATION=on`.
2. **Formally shelve Tiers 2–3** as documented negative results (done in
   `ORCHESTRATION_FINDINGS.md`). They proved a real fact: the field has no
   convergent residual to gate.
3. **Don't chase the rest of the patent into Eris.** It's a numerical-solver
   invention; its home is numerical solvers, not a cognitive field. The one place
   it belongs in Eris is already wired.
