"""Two reviewer challenges, verified directly:
 (2) lobe asymmetry: report BOTH lobe-L and lobe-R collapse + the joint outcome table
     (is L=30% the cherry-picked favorable lobe?).
 (4) delay vs prevention: run to long T and see whether the single-vs-bilateral gap
     PERSISTS or closes (collapse fraction by time, from collapse_step)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import field_core as cc
import metrics as ms

N, T = 40, 2000
PHI_INIT, JITTER = 0.85, 0.04
SIGMA, MU = 0.007, 0.1


def regime():
    p = cc.PDEParams()
    p.r_sat = 0.85; p.d_decay = 0.20
    p.sigma_noise = SIGMA; p.sigma_phase = SIGMA; p.omega_spread = 0.25
    return p


def run_single(seed, p):
    mon = ms.CollapseMonitor(ms.CollapseThresholds(), log_every=50)
    f = cc.SingleField(64, p, seed=seed, phi_init=PHI_INIT, phi_jitter=JITTER)
    for t in range(T):
        f.step(); mon.observe(t, f.phi, f.theta)
    r = mon.finalize()
    return r.collapsed, r.collapse_step


def run_bilat(seed, p):
    f = cc.BilateralField(64, p, seed=seed, mu=MU, phi_init=PHI_INIT, phi_jitter=JITTER)
    mL = ms.CollapseMonitor(ms.CollapseThresholds(), log_every=50)
    mR = ms.CollapseMonitor(ms.CollapseThresholds(), log_every=50)
    for t in range(T):
        f.step()
        mL.observe(t, f.L.phi, f.L.theta)
        mR.observe(t, f.R.phi, f.R.theta)
    rl, rr = mL.finalize(), mR.finalize()
    return (rl.collapsed, rl.collapse_step), (rr.collapsed, rr.collapse_step)


def surv_curve(steps, label):
    """collapse fraction vs time t (fraction with collapse_step <= t)."""
    pts = []
    for t in [600, 800, 1000, 1200, 1500, 2000]:
        frac = np.mean([(s is not None and s <= t) for s in steps])
        pts.append((t, frac))
    return pts


def main():
    p = regime()
    print(f"verify N={N} T={T} sigma={SIGMA} mu={MU}", flush=True)
    s_col, s_step = [], []
    L_col, L_step, R_col, R_step = [], [], [], []
    for seed in range(N):
        c, st = run_single(seed, p); s_col.append(c); s_step.append(st)
        (lc, lst), (rc, rst) = run_bilat(seed, p)
        L_col.append(lc); L_step.append(lst); R_col.append(rc); R_step.append(rst)
        if (seed + 1) % 5 == 0:
            print(f"  {seed+1}/{N}", flush=True)

    s_col = np.array(s_col); L_col = np.array(L_col); R_col = np.array(R_col)
    print("\n=== (2) lobe asymmetry + joint outcomes (alive = not collapsed) ===")
    print(f"single   collapse = {s_col.mean():.0%}")
    print(f"lobe-L   collapse = {L_col.mean():.0%}")
    print(f"lobe-R   collapse = {R_col.mean():.0%}")
    La, Ra = ~L_col, ~R_col
    print(f"joint: both-alive={np.sum(La&Ra)}  L-only={np.sum(La&~Ra)}  "
          f"R-only={np.sum(~La&Ra)}  both-collapsed={np.sum(~La&~Ra)}  (n={N})")
    either_alive = np.mean(La | Ra)
    print(f"either-lobe-alive = {either_alive:.0%}  (architecture survives if EITHER lobe lives)")

    print("\n=== (4) delay vs prevention: collapse fraction by time T ===")
    print(f"{'T':>6} | {'single':>7} | {'lobe-L':>7} | {'gap':>5}")
    sc = surv_curve(s_step, "single"); lc = surv_curve(L_step, "L")
    for (t, sf), (_, lf) in zip(sc, lc):
        print(f"{t:>6} | {sf*100:6.0f}% | {lf*100:6.0f}% | {(sf-lf)*100:4.0f}")

    out = {"N": N, "T": T, "single": float(s_col.mean()),
           "lobeL": float(L_col.mean()), "lobeR": float(R_col.mean()),
           "either_alive": float(either_alive),
           "joint": {"both_alive": int(np.sum(La & Ra)), "L_only": int(np.sum(La & ~Ra)),
                     "R_only": int(np.sum(~La & Ra)), "both_collapsed": int(np.sum(~La & ~Ra))},
           "surv_single": sc, "surv_L": lc}
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "results", "verify_extra.json"), "w"), indent=2)
    print("\nwrote results/verify_extra.json")


if __name__ == "__main__":
    main()
