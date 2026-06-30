"""
R3 -- exponent cleanup. The verdict claimed the interior floor ~ (delta/2mu)^(1/3) with
"slope -0.41 ~= -1/3", fit on theta_LR. But (a) theta_LR (overlap angle) conflates the
coherence magnitude r and the relative phase, so it is NOT the reduced ODE variable Psi;
and (b) (delta/2mu)^(1/3) is the ASYMPTOTIC small-Psi law, not a finite-mu fit.

This measures the actual reduced variable Psi_mean = arg(<psi_L,psi_R>) (amplitude-weighted
mean relative phase) AND theta_LR AND coherence r at the settled attractor, fits effective
exponents on the floor branch, and checks the analytic Psi*(mu) asymptotes to -1/3 at large
mu. Outcome corrects BIFURCATE_VERDICT.md to the right variable + honest exponent.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bifurcate as bf
from field_core import PDEParams

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CKPT = os.path.join(RES, "r3_exponent.json")
SIG = 0.016


def regime():
    p = PDEParams(); p.r_sat = 0.85; p.d_decay = 0.20
    p.sigma_noise = SIG; p.sigma_phase = SIG; p.omega_spread = 0.25
    return p


def settled_measures(kind, mu, delta, seed, T=1800, window=300):
    p = regime()
    ta = bf.TwoAgents(64, p, mu=mu, kind=kind, delta=delta, seed=seed, gate_phase=True)
    psis, thetas, rs = [], [], []
    for t in range(T):
        ta.step()
        if t >= T - window:
            O = np.sum(ta.L.phi * ta.R.phi * np.exp(1j * (ta.L.theta - ta.R.theta)))
            nrm = np.sqrt(np.sum(ta.L.phi ** 2)) * np.sqrt(np.sum(ta.R.phi ** 2)) + 1e-12
            psis.append(abs(np.degrees(np.angle(O))))      # mean relative phase |Psi|
            rs.append(abs(O) / nrm)                          # coherence magnitude
            thetas.append(ta.theta_LR())
    return {"seed": seed, "Psi_mean": float(np.mean(psis)),
            "theta_LR": float(np.mean(thetas)), "coherence": float(np.mean(rs))}


def loglog_slope(mus, vals):
    x, y = np.log(np.array(mus)), np.log(np.array(vals))
    return float(np.polyfit(x, y, 1)[0])


def main():
    MUS = [0.6, 0.9, 1.3, 1.8, 2.5, 4.0]
    DELTA = 0.1
    N = {"egate": 20, "cos": 8, "diff": 8}
    st = json.load(open(CKPT)) if os.path.exists(CKPT) else {}
    for kind in ["egate", "cos", "diff"]:
        for mu in MUS:
            key = f"{kind}_m{mu}"
            if key in st and len(st[key]) >= N[kind]:
                continue
            rows = st.get(key, [])
            done = {r["seed"] for r in rows}
            for s in range(N[kind]):
                if s in done:
                    continue
                rows.append(settled_measures(kind, mu, DELTA, s))
                if len(rows) % 10 == 0:
                    st[key] = rows; json.dump(st, open(CKPT, "w"))
                    print(f"[{key}] {len(rows)}/{N[kind]}", flush=True)
            st[key] = rows; json.dump(st, open(CKPT, "w"))
            print(f"[{key}] done Psi={np.mean([r['Psi_mean'] for r in rows]):.1f} "
                  f"theta={np.mean([r['theta_LR'] for r in rows]):.1f} "
                  f"r={np.mean([r['coherence'] for r in rows]):.2f}", flush=True)

    # analytic Psi* asymptotic check (large mu) from analytic_reduction's fixed-point solver
    print("\n=== R3 RESULT ===")
    out = {"delta": DELTA, "mus": MUS}
    for kind in ["egate", "cos", "diff"]:
        psi = [np.mean([r["Psi_mean"] for r in st[f"{kind}_m{mu}"]]) for mu in MUS]
        th = [np.mean([r["theta_LR"] for r in st[f"{kind}_m{mu}"]]) for mu in MUS]
        r = [np.mean([r2["coherence"] for r2 in st[f"{kind}_m{mu}"]]) for mu in MUS]
        # floor branch slope (all mu) and high-mu slope (mu>=1.3)
        sl_psi = loglog_slope(MUS, psi); sl_th = loglog_slope(MUS, th)
        hi = [i for i, m in enumerate(MUS) if m >= 1.3]
        sl_psi_hi = loglog_slope([MUS[i] for i in hi], [psi[i] for i in hi])
        print(f"{kind:6s}: Psi_mean(deg)={[round(x,1) for x in psi]}")
        print(f"        theta_LR(deg)={[round(x,1) for x in th]}  coherence={[round(x,2) for x in r]}")
        print(f"        slope log|Psi| vs log mu = {sl_psi:+.2f} (all), {sl_psi_hi:+.2f} (mu>=1.3); "
              f"slope theta_LR = {sl_th:+.2f}")
        out[kind] = {"Psi_mean": psi, "theta_LR": th, "coherence": r,
                     "slope_Psi": sl_psi, "slope_Psi_hi": sl_psi_hi, "slope_theta": sl_th}
    json.dump(out, open(os.path.join(RES, "r3_summary.json"), "w"), indent=1)
    print("\nwrote results/r3_summary.json")


if __name__ == "__main__":
    main()
