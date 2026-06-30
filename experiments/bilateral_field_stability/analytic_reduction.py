#!/usr/bin/env python3
"""
analytic_reduction.py -- TASK T-A
=================================================================
Reduce the two-lobe membrane phase dynamics (field_core.py:_phase_step,
gate_phase=True) to a SINGLE ODE for the relative phase Psi = theta_R - theta_L,
find the fixed points and their stability for each coupling arm, and show the
mechanism: the egate gate cg = cos^2 sin^2 vanishes QUADRATICALLY at Psi->0, so
the restoring product cg(Psi)*Psi ~ Psi^3 and the fixed point sits at a
cube-root FLOOR Psi* ~ (delta_eff/2mu)^(1/3) -- a "no-pull-at-sameness" interior
attractor that diff (linear, Psi*~delta/2mu ~ 0, fusion) and cos (O(1) at 0,
fusion branch) do not have.

Exact per-lobe phase update (field_core.py:_phase_step):
    dtheta = omega0 + K_phase * g(phi) * coupling_kuramoto
           + mu * cg(Delta) * Delta          # membrane, gate_phase=True
    theta <- theta + dt * dtheta + noise
with Delta = wrap_diff(theta_other, theta_self) in (-pi,pi], and
    cg(Delta) = coupling_gate(Delta, kind):
        diff  -> 1
        cos   -> cos(Delta)^2
        egate -> cos(Delta)^2 * sin(Delta)^2 = (1/4) sin(2 Delta)^2

Detuning enters via omega0 (bifurcate.py): omega_L += delta/2, omega_R -= delta/2,
so the bare inter-lobe detuning is delta. The membrane is INTEGRATED with dt=0.05
and the Kuramoto neighbour term renormalises the drift, so the EFFECTIVE detuning
that the membrane must balance is delta_eff = s * delta for a single positive scale
factor s (calibrated below against the numerical egate column).

Outputs:
  - symbolic derivation/printout (sympy)
  - results/analytic_band.json : predicted class + Psi* per (kind,delta,mu) cell
"""
from __future__ import annotations
import os, json
import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

PI = np.pi

# --------------------------------------------------------------------------- #
# 0. the gates, in numeric form (must match field_core.coupling_gate exactly)
# --------------------------------------------------------------------------- #
def cg(Psi, kind):
    if kind == "diff":
        return np.ones_like(Psi) if hasattr(Psi, "shape") else 1.0
    if kind == "cos":
        return np.cos(Psi) ** 2
    if kind == "egate":
        return (np.cos(Psi) ** 2) * (np.sin(Psi) ** 2)
    raise ValueError(kind)

def F(Psi, kind):
    """Restoring product cg(Psi)*Psi. dPsi/dt = delta_eff - 2*mu*F(Psi)."""
    return cg(Psi, kind) * Psi

def dF(Psi, kind):
    """d/dPsi [cg*Psi]; stable interior fixed point iff this is > 0 there."""
    h = 1e-6
    return (F(Psi + h, kind) - F(Psi - h, kind)) / (2 * h)


# =========================================================================== #
# 1 + 2.  SYMBOLIC derivation and fixed-point algebra (sympy)
# =========================================================================== #
def symbolic_derivation():
    print("=" * 74)
    print("STEP 1.  Reduction of the two-lobe membrane to one ODE for Psi")
    print("=" * 74)
    Psi, mu, deff = sp.symbols("Psi mu delta_eff", real=True)
    thL, thR, oL, oR = sp.symbols("theta_L theta_R omega_L omega_R", real=True)

    # cg is EVEN in its argument; verify symbolically for each arm.
    d = sp.symbols("d", real=True)
    cg_sym = {
        "diff": sp.Integer(1),
        "cos": sp.cos(d) ** 2,
        "egate": sp.cos(d) ** 2 * sp.sin(d) ** 2,
    }
    print("\n  cg(Delta) for each arm (even-function check cg(-d)-cg(d)=0):")
    for k, e in cg_sym.items():
        even = sp.simplify(e.subs(d, -d) - e)
        print(f"    {k:6s}: cg = {e}   even? {even == 0}")
        assert even == 0

    print("\n  Per-lobe membrane drift (gate_phase=True), Delta_self = wrap(other-self):")
    print("    lobe L: dtheta_L gets  +mu*cg(+Psi)*(+Psi)   [Delta_L = theta_R-theta_L = +Psi]")
    print("    lobe R: dtheta_R gets  +mu*cg(-Psi)*(-Psi)   [Delta_R = theta_L-theta_R = -Psi]")
    # Psi = theta_R - theta_L ; subtract the two lobe drifts.
    cgP = sp.Function("cg")
    # symbolic, using evenness cg(-Psi)=cg(Psi):
    dthL_mem = mu * cgP(Psi) * Psi
    dthR_mem = mu * cgP(Psi) * (-Psi)            # cg(-Psi) -> cg(Psi) by evenness
    dPsi_mem = sp.simplify(dthR_mem - dthL_mem)
    dPsi = (oR - oL) + dPsi_mem
    print("\n  dPsi/dt = (omega_R - omega_L) + [dtheta_R - dtheta_L]_membrane")
    print(f"          = (omega_R - omega_L) + ({sp.simplify(dthR_mem)}) - ({sp.simplify(dthL_mem)})")
    print(f"          = delta_eff - 2*mu*cg(Psi)*Psi     [omega_R-omega_L -> delta_eff]")
    print("\n  ==> REDUCED EQUATION:   dPsi/dt = delta_eff - 2*mu*cg(Psi)*Psi")
    print("      (membranes antisymmetric in the +/-Psi exchange; cg even => the two")
    print("       membrane terms add to -2*mu*cg(Psi)*Psi, an ODD restoring force.)")

    print("\n" + "=" * 74)
    print("STEP 2.  Fixed points  cg(Psi*)*Psi* = delta_eff/(2 mu)  and stability")
    print("=" * 74)
    print("  Fixed point: dPsi/dt = 0  <=>  cg(Psi*)*Psi* = delta_eff/(2 mu).")
    print("  Stability  : linearise dPsi/dt = -2 mu * d/dPsi[cg*Psi].")
    print("               stable  iff  d/dPsi[cg(Psi)*Psi] > 0  at Psi*.")

    print("\n  Small-Psi behaviour of the restoring product P(Psi)=cg(Psi)*Psi:")
    for k, e in cg_sym.items():
        P = sp.simplify(e.subs(d, Psi) * Psi)
        ser = sp.series(P, Psi, 0, 6).removeO()
        slope0 = sp.diff(P, Psi).subs(Psi, 0)
        print(f"    {k:6s}: P(Psi) = {sp.simplify(P)}")
        print(f"            series @0: {ser}")
        print(f"            P'(0) = {slope0}")

    print("\n  KEY CONTRAST at Psi -> 0:")
    print("    diff : cg=1            => P=Psi, P'(0)=1  (linear restoring)")
    print("           Psi* = delta_eff/(2 mu)  -> small  => FUSION branch.")
    print("    cos  : cg=cos^2        => P~Psi-Psi^3..., P'(0)=1 (still O(1) pull at 0)")
    print("           small-Psi root Psi*~delta_eff/(2 mu) => FUSION branch")
    print("           (plus folds: cos^2*Psi is non-monotone, max near Psi~0.65 rad).")
    print("    egate: cg=cos^2 sin^2  => P = (1/4) sin^2(2Psi) * Psi ~ Psi^3 near 0,")
    print("           P'(0)=0  => the restoring force VANISHES QUADRATICALLY at sameness.")
    print("           cg(Psi*)Psi* = delta_eff/(2mu): near 0, Psi*^3 ~ delta_eff/(2mu)")
    print("           => Psi* ~ (delta_eff/(2 mu))^(1/3)  -- a CUBE-ROOT FLOOR away from 0.")

    # confirm the cube-root scaling for egate symbolically near 0
    Peg = sp.Rational(1, 4) * sp.sin(2 * Psi) ** 2 * Psi
    approx = sp.series(Peg, Psi, 0, 6).removeO()
    print(f"\n    egate P(Psi) small-Psi:  {approx}   (leading term Psi^3).")
    print("    Set Psi^3 = delta_eff/(2 mu)  =>  Psi* = (delta_eff/(2 mu))^(1/3).")
    return cg_sym


# =========================================================================== #
# 3.  Numeric fixed-point solve + classification over the (delta,mu) grid
# =========================================================================== #
# domain of Psi for a relative phase: (0, pi/2] is the physically meaningful
# half (theta_LR = |Psi| folded into [0,90] deg by the arccos overlap readout).
# We search interior roots on (0, pi/2).  A "fold to segregation" = the lock is
# lost (no stable interior root below pi/2 -> Psi runs up toward pi/2 or pi).

FUSE_DEG, SEG_DEG = 20.0, 75.0   # same thresholds as bifurcate.classify

def all_roots(kind, mu, deff, n=20000):
    """All Psi in (eps, pi/2) with cg(Psi)*Psi = deff/(2 mu), with stability flag."""
    target = deff / (2.0 * mu)
    grid = np.linspace(1e-4, PI / 2 - 1e-4, n)
    g = F(grid, kind) - target
    roots = []
    sign = np.sign(g)
    idx = np.where(np.diff(sign) != 0)[0]
    for i in idx:
        a, b = grid[i], grid[i + 1]
        # bisection refine
        for _ in range(60):
            m = 0.5 * (a + b)
            if (F(a, kind) - target) * (F(m, kind) - target) <= 0:
                b = m
            else:
                a = m
        r = 0.5 * (a + b)
        roots.append((r, dF(r, kind) > 0))   # (Psi*, stable?)
    return roots, target

def classify_cell(kind, mu, deff):
    """Return (cls, Psi*_deg). cls in {fusion, sustained, segregation, runaway}."""
    roots, target = all_roots(kind, mu, deff)
    stable = [r for (r, s) in roots if s]
    if not stable:
        # no stable interior root on (0,pi/2): does the trajectory settle at the
        # max-coupling edge (segregation) or run away? For these even gates with
        # target>max(F) the flow pushes Psi up toward / past pi/2 -> segregation.
        maxF = np.max(F(np.linspace(1e-4, PI / 2 - 1e-4, 4000), kind))
        if target > maxF:
            return "segregation", 90.0      # detuning exceeds max lock-in -> drifts apart
        return "runaway", float("nan")
    # pick the stable root the basin actually selects: the SMALLEST stable Psi*
    # reachable from below (the system relaxes from small offsets up to it).
    Psi_star = min(stable)
    deg = np.degrees(Psi_star)
    if deg < FUSE_DEG:
        cls = "fusion"
    elif deg > SEG_DEG:
        cls = "segregation"
    else:
        cls = "sustained"
    return cls, float(deg)


# =========================================================================== #
# Calibration of delta_eff = s * delta against the numerical egate column
# =========================================================================== #
# numerical mean theta_LR (deg), gate_phase=True, sigma=0.016, delta=0.1:
NUM_EGATE_DELTA01 = {0.4: 69.0, 0.6: 39.0, 0.9: 27.0, 1.3: 22.0, 1.8: 19.0}
NUM_COS_DELTA01   = {0.4: 94.0, 0.6: 78.0, 0.9: 37.0, 1.3: 9.0, 1.8: 5.0}

def calibrate_scale():
    """Fit single positive s so egate Psi*(deff=s*0.1) matches NUM_EGATE_DELTA01.
    Pure cube-root floor model: theta* = degrees( (s*0.1/(2 mu))^(1/3) )."""
    mus = np.array(sorted(NUM_EGATE_DELTA01))
    num = np.array([NUM_EGATE_DELTA01[m] for m in mus])
    def pred(s):
        out = []
        for m in mus:
            _, deg = classify_cell("egate", m, s * 0.1)
            out.append(deg)
        return np.array(out)
    # 1-D search on s (log-spaced); minimise relative SSE
    cand = np.geomspace(1e-3, 5.0, 4000)
    best_s, best_e = None, np.inf
    for s in cand:
        p = pred(s)
        e = np.sum(((p - num) / num) ** 2)
        if e < best_e:
            best_e, best_s = e, s
    return float(best_s), pred(best_s), num, mus


def main():
    cg_sym = symbolic_derivation()

    print("\n" + "=" * 74)
    print("STEP 3.  Calibrate delta_eff = s*delta on the egate column (delta=0.1)")
    print("=" * 74)
    s, pred_eg, num_eg, mus = calibrate_scale()
    print(f"  Best-fit positive scale factor  s = {s:.4f}")
    print(f"  (delta_eff = {s:.4f} * delta;  dt=0.05 sets the order; Kuramoto renorm")
    print( "   absorbs the rest.  Only the SHAPE, not s, carries the claim.)")
    print("\n   mu     numeric  predicted(egate cube-root floor)")
    for m, n, p in zip(mus, num_eg, pred_eg):
        print(f"   {m:.1f}    {n:6.1f}    {p:8.1f}")

    # ---- build the band over the full grid -------------------------------- #
    deltas = [0.05, 0.1, 0.2, 0.4]
    mus_all = [0.4, 0.6, 0.9, 1.3, 1.8]
    kinds = ["diff", "cos", "egate"]
    band = {"scale_factor_s": s, "delta_eff_def": "delta_eff = s * delta",
            "reduced_ode": "dPsi/dt = delta_eff - 2*mu*cg(Psi)*Psi",
            "fixed_point": "cg(Psi*)*Psi* = delta_eff/(2*mu)",
            "stability": "stable iff d/dPsi[cg(Psi)*Psi] > 0",
            "cells": {}}
    for kind in kinds:
        for dl in deltas:
            for mu in mus_all:
                cls, deg = classify_cell(kind, mu, s * dl)
                band["cells"][f"{kind}|delta={dl}|mu={mu}"] = {
                    "kind": kind, "delta": dl, "mu": mu,
                    "delta_eff": s * dl,
                    "class": cls,
                    "Psi_star_deg": None if (deg != deg) else round(deg, 2),
                }

    outp = os.path.join(RESULTS, "analytic_band.json")
    with open(outp, "w") as f:
        json.dump(band, f, indent=2)
    print(f"\n  wrote {outp}  ({len(band['cells'])} cells)")

    # ---- printed cell-by-cell comparison at delta=0.1 --------------------- #
    print("\n" + "=" * 74)
    print("STEP 4.  Predicted vs numerical class (delta=0.1)")
    print("=" * 74)
    def num_cls(v):
        if v < FUSE_DEG: return "fusion"
        if v > SEG_DEG: return "segregation"
        return "sustained"
    print("\n  EGATE:")
    print("   mu   pred_deg pred_class | num_deg num_class  match")
    for mu in mus_all:
        cls, deg = classify_cell("egate", mu, s * 0.1)
        nv = NUM_EGATE_DELTA01[mu]; nc = num_cls(nv)
        print(f"   {mu:.1f}   {deg:6.1f}  {cls:11s}|  {nv:5.1f}  {nc:11s} {'OK' if cls==nc else 'x'}")
    print("\n  COS:")
    print("   mu   pred_deg pred_class | num_deg num_class  match")
    for mu in mus_all:
        cls, deg = classify_cell("cos", mu, s * 0.1)
        nv = NUM_COS_DELTA01[mu]; nc = num_cls(nv)
        dd = "nan" if deg != deg else f"{deg:6.1f}"
        print(f"   {mu:.1f}   {dd:>6s}  {cls:11s}|  {nv:5.1f}  {nc:11s} {'OK' if cls==nc else 'x'}")

    print("\n  DIFF (all mu, delta=0.1): expect fusion")
    for mu in mus_all:
        cls, deg = classify_cell("diff", mu, s * 0.1)
        print(f"   mu={mu:.1f}: Psi*={deg:.2f} deg -> {cls}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
