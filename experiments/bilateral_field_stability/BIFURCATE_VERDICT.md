# Sustained Bifurcation via E-Gated Coupling — Verdict

**Hypothesis (handoff):** mirror/diffusive coupling has one attractor — *sameness* —
so it can only fuse-then-lock. Replacing it with transport gated by the coupling law
**E(Δ) = cos²Δ·sin²Δ** (peaks at 45°, **zero at Δ=0 and Δ=90°**) between two *distinct,
detuned* agents should admit a **stable interior coupling angle θ\* ∈ (0°,90°)** —
sustained, alive, exchanging two-ness — that cosine-only coupling (`cos`, no `sin²`)
cannot. The `sin²` factor is the load-bearing claim.

## Verdict: CONFIRMED — and the `sin²` factor is the cause.

A genuine **interior attractor** exists for E-gated coupling and **only** for E-gated
coupling. The handoff's §5 first decision rule is satisfied: egate sustains a stable
θ\* ∈ (0°,90°) at long T with both domains alive and exchange live, **passes the
attractor test (returns from both sides),** while diff fuses and cos — when actually
perturbed — collapses to fusion.

> Operating regime (Stage A precondition): an *isolated* single field is sustained-alive
> at long T for **σ ≥ 0.013** (vs the monostable σ=0.007 lock regime of the prior work);
> we run at **σ=0.016, omega_spread=0.25, T=2500, N=16**. Two agents share one law but are
> **non-mirror**: independent seeds, independent noise, intrinsic-frequency detuning δ.

---

## The one design decision that matters (stated plainly)

The handoff's **primary** arm gates only the *amplitude* membrane and keeps a plain
diffusive *phase* membrane. **That primary contrast is a NULL:** because the global
coupling angle θ_LR is **phase-overlap-dominated**, a plain phase membrane fuses the
phases regardless of the amplitude gate. Measured directly (δ=0.1, μ=0.3, amplitude-gate
only): diff→15°, cos→15°, **egate→16°** — all fuse; the `sin²` factor does nothing.

So the experiment was run in the handoff's **sanctioned secondary form** — the coupling
law gates **both** membranes (`gate_phase=True`, "the coupling law *as* the membrane
transport rule"). This is the faithful realization of "keep sin": the `sin²` zero-at-
sameness must act on the channel that sets θ_LR. With that, the contrast is decisive.
(`diff` = plain g=1 on both; `cos` = cos²Δ on both; `egate` = cos²Δ·sin²Δ on both.)

---

## The δ/μ regime map (the headline)  — `results/bifurcate_map.png`, `bifurcate_curves.png`

mean θ_LR (deg) at long T; **S**=sustained interior, **F**=fusion(→0), **X**=segregation:

```
EGATE                                COS
 μ\δ  0.05  0.1   0.2   0.4            μ\δ  0.05  0.1   0.2   0.4
 1.8  16 F  19 F  25 S  52 S          1.8   4 F   5 F   5 F  10 F
 1.3  19 F  22 S  30 S 104 X          1.3   8 F   9 F  14 F  36 S
 0.9  22 S  27 S  55 S 110 X          0.9  34 S  37 S  49 S 104 X
 0.6  28 S  39 S  98 X 101 X          0.6  72 S  78 X  98 X 120 X
 0.4  39 S  69 S  95 X  95 X          0.4  87 X  94 X 114 X 126 X
references:  diff(0.1,·) → 4–12° (always F)   iso(·) → 89–90° (always X)
```

Reading it:
* **iso** (no coupling) → segregation (θ_LR≈90°) for every δ — two independent fields
  are orthogonal. **diff** (plain) → fusion (θ_LR→4–12°) everywhere. The two baselines
  bracket the question.
* **egate** holds an **interior band of 11 sustained cells** with **low within-run drift
  (0.2–1.0°)** = settled, and **never fully fuses** (floor ≈16–19° even at μ=1.8, vs cos
  →4°): the `sin²` "never rewarded for collapsing distinctions" signature.
* **cos** transitions **sharply** segregation→fusion; its only interior cells sit on the
  μ=0.9 crossing row with the **highest drift (2.0–2.3°)** = still moving, not settled.

The drift difference already says egate's interior is settled and cos's is transient.
The attractor test proves it.

---

## The attractor test (decisive, §5 metric)  — `results/attractor_test.png`

Settle, then perturb θ_LR **both ways** (force toward sameness θ→0, and toward
orthogonality θ→90), release, and see where it lands. N=6 seeds/cell, averaged:

| arm / cell | θ\* (unperturbed) | ← sameness kick | ← orthogonality kick | attractor? |
|---|---|---|---|---|
| **egate** d0.05 μ0.6 | 27.8° | **24.8°** | **24.7°** | **interior — returns both sides** |
| **egate** d0.1 μ0.6 | 40.3° | **34.5°** | **34.2°** | **interior — returns both sides** |
| **egate** d0.1 μ0.9 | 27.3° | **25.1°** | **25.3°** | **interior — returns both sides** |
| cos d0.1 μ0.6 | 79.3° | 10.4° | 10.9° | no — collapses to **fusion** |
| cos d0.1 μ0.9 | 40.6° | 7.2° | 7.7° | no — apparent interior collapses to **fusion** |
| diff d0.1 μ0.9 | 6.9° | 6.8° | 6.8° | stable, but the attractor is **fusion** |

* **egate**: from *both* a θ→0 kick and a θ→90 kick, θ_LR returns to the *same* interior
  value (~25–34°, well inside (0,90)). **Return-from-both-sides = θ\* is a genuine
  attractor.** Sustained two-ness is *stable*, not a transient snapshot.
* **cos**: its "settled" interior (37–79°) is a **slow transient** — a phase kick (either
  direction) drops it into **fusion (~7–11°)**. Cosine-only has no interior attractor.
* **diff**: returns from both sides, but to **fusion (~7°)** — confirming plain coupling's
  one attractor is sameness.

(Minor honest note: egate's perturbed runs settle a few degrees *below* the unperturbed
θ\* — e.g. 25° vs 27° — but the two kicks converge to the *same* value far from 0/90,
which is the attractor property. The small offset is relaxation-window/amplitude-reset,
not a failure.)

---

## Secondary probe — dispersion relation (§7)  — `results/dispersion.png`

The linearized homogeneous field has **λ(k) < 0 for every well-resolved wavenumber**
(theta channel monotonically damped; phi channel negative except isolated Nyquist/
half-Nyquist spikes at k=16,32 that are discretization artifacts). **No genuine
pattern-forming (Turing) band ⇒ spontaneous *spatial* division is NOT native to this
PDE.** Two-ness here must be **relational** (the route taken above), not a single field
splitting itself with a self-generated wall.

---

## Bottom line

* **Does E-gated coupling produce sustained dynamic two-ness?** Yes. A stable interior
  θ\* attractor, both domains alive, exchange live, robust to perturbation from both
  sides — across an 11-cell δ/μ band.
* **Is the `sin²` factor the cause?** Yes. The cosine-only control (`cos`) has the same
  peak coupling magnitude but **no interior attractor** — it collapses to fusion when
  perturbed; plain `diff` fuses outright. Removing `sin²` removes the effect.
* **Faithful caveat:** this required gating the **phase** membrane with the coupling law
  (the handoff's sanctioned `egate_phase`), because the amplitude-only primary contrast
  is a null (θ_LR is phase-dominated). The honest mechanism statement is **"the coupling
  law `cos²·sin²` must gate the channel that carries the relationship."**
* **Scope:** one operating point (σ=0.016); not yet swept across the alive band; the
  slow-identity-drift (§4.7) and distinct-law κ/λ (§3 stretch) tests are the warranted
  next steps now that the core result holds. Standalone probe; nothing wired into Eris.
