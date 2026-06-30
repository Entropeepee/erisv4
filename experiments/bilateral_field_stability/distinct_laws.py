"""
distinct_laws.py -- T-F: distinct laws (kappa/lambda lateralization) + realignment (CLS).

Two parts, both reusing bifurcate.TwoAgents (params_R) + the FIXED snapshot/restore:
 1) distinct-law attractor test: give L and R DIFFERENT laws (r_sat/d_decay/K_phase),
    NO frequency detuning (delta=0) -- the law asymmetry is the only difference. Does the
    egate interior attractor survive when the agents differ in *law*, not just frequency?
 2) realignment / slow drift (CLS): slowly modulate the two lobes' laws in anti-phase so
    their characters MIGRATE/SWAP over a slow timescale, under egate coupling, and check
    whether theta_LR stays interior (away from 0 and 90) throughout = fast metastable
    coordination + slow identity drift.
"""
from __future__ import annotations
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bifurcate as bf
from field_core import PDEParams

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SIGMA = 0.016


def law(r_sat, d_decay, K_phase, sigma=SIGMA, omega_spread=0.25):
    p = PDEParams()
    p.r_sat = r_sat; p.d_decay = d_decay; p.K_phase = K_phase
    p.sigma_noise = sigma; p.sigma_phase = sigma; p.omega_spread = omega_spread
    return p


# kappa-dominant (more saturation/structure, stronger phase coupling) vs lambda-dominant
# (more decay, weaker phase coupling). Both in the sustained-alive regime.
KAPPA = dict(r_sat=0.95, d_decay=0.15, K_phase=2.0)
LAMBDA = dict(r_sat=0.72, d_decay=0.28, K_phase=1.1)


def part1_distinct_attractor(N=12, mu=0.9):
    """Attractor test with DISTINCT LAWS (delta=0). egate should keep an interior attractor;
    cos should not."""
    pL = law(**KAPPA); pR = law(**LAMBDA)
    out = {"KAPPA": KAPPA, "LAMBDA": LAMBDA, "mu": mu, "delta": 0.0, "N": N, "arms": {}}
    print(f"=== T-F part 1: distinct-law attractor test (kappa-L vs lambda-R, delta=0, mu={mu}) ===")
    for kind in ["egate", "cos", "diff"]:
        rows = []
        for s in range(N):
            r = bf.attractor_test(kind, 0.0, mu, SIGMA, s, gate_phase=True,
                                  params=pL, params_R=pR)
            rows.append(r)
        ts = np.mean([r["theta_star"] for r in rows])
        sa = np.mean([r["after_same_kick"] for r in rows])
        so = np.mean([r["after_orth_kick"] for r in rows])
        conv = abs(sa - so)
        interior = 15 < (sa + so) / 2 < 80
        verdict = "INTERIOR ATTRACTOR" if (conv < 12 and interior) else (
            "fusion" if (sa + so) / 2 <= 15 else "segregation/none")
        print(f"  {kind:6s}: theta*={ts:5.1f} <-same={sa:5.1f} <-orth={so:5.1f} "
              f"conv={conv:4.1f} -> {verdict}")
        out["arms"][kind] = {"theta_star": float(ts), "after_same": float(sa),
                             "after_orth": float(so), "verdict": verdict}
    json.dump(out, open(os.path.join(RES, "tf_distinct_attractor.json"), "w"), indent=1)
    print("wrote results/tf_distinct_attractor.json")
    return out


def part2_realignment(kind="egate", mu=0.9, T=6000, seed=0):
    """Slowly modulate the two lobes' laws in ANTI-PHASE (r_sat and K_phase swap kappa<->lambda
    over the run) under coupling; track theta_LR and the two identities. Remaining interior
    while identities migrate = CLS slow/fast structure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pL = law(**KAPPA); pR = law(**LAMBDA)
    ta = bf.TwoAgents(64, pL, mu=mu, kind=kind, delta=0.0, seed=seed, gate_phase=True, params_R=pR)
    # slow schedule s(t) in [0,1], one full swap over the run (kappa<->lambda) and back
    rs_k, dd_k, kp_k = KAPPA["r_sat"], KAPPA["d_decay"], KAPPA["K_phase"]
    rs_l, dd_l, kp_l = LAMBDA["r_sat"], LAMBDA["d_decay"], LAMBDA["K_phase"]
    th, rsL, rsR, kpL, kpR = [], [], [], [], []
    settle = 800
    for t in range(T):
        if t >= settle:
            s = 0.5 * (1 - np.cos(2 * np.pi * (t - settle) / (T - settle)))  # 0->1->0 smooth swap
        else:
            s = 0.0
        # L morphs kappa->lambda by s, R morphs lambda->kappa by s (anti-phase identity swap)
        ta.L.p.r_sat = (1 - s) * rs_k + s * rs_l
        ta.L.p.d_decay = (1 - s) * dd_k + s * dd_l
        ta.L.p.K_phase = (1 - s) * kp_k + s * kp_l
        ta.R.p.r_sat = (1 - s) * rs_l + s * rs_k
        ta.R.p.d_decay = (1 - s) * dd_l + s * dd_k
        ta.R.p.K_phase = (1 - s) * kp_l + s * kp_k
        ta.step()
        if t % 20 == 0:
            th.append(ta.theta_LR()); rsL.append(ta.L.p.r_sat); rsR.append(ta.R.p.r_sat)
            kpL.append(ta.L.p.K_phase); kpR.append(ta.R.p.K_phase)
    th = np.array(th)
    post = th[settle // 20:]
    interior_frac = float(np.mean((post > 10) & (post < 80)))
    print(f"=== T-F part 2: realignment (kind={kind}, mu={mu}) ===")
    print(f"  theta_LR during identity swap: min={post.min():.0f} mean={post.mean():.0f} "
          f"max={post.max():.0f} | fraction interior (10-80deg) = {interior_frac:.0%}")
    res = {"kind": kind, "mu": mu, "interior_frac": interior_frac,
           "theta_min": float(post.min()), "theta_mean": float(post.mean()),
           "theta_max": float(post.max())}
    json.dump(res, open(os.path.join(RES, "tf_realignment.json"), "w"), indent=1)

    tt = np.arange(len(th)) * 20
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(tt, rsL, color="navy", label="L r_sat (identity)")
    ax[0].plot(tt, rsR, color="crimson", label="R r_sat (identity)")
    ax[0].axvline(settle, color="gray", ls=":")
    ax[0].set_ylabel("r_sat (lobe identity)"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"T-F realignment: identities migrate/swap (top) while theta_LR stays interior (bottom) [{kind}]")
    ax[1].plot(tt, th, color="seagreen"); ax[1].axhspan(10, 80, color="lightgreen", alpha=0.2)
    ax[1].axvline(settle, color="gray", ls=":")
    ax[1].set_ylabel("theta_LR (deg)"); ax[1].set_xlabel("step"); ax[1].set_ylim(0, 95)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "tf_realignment.png"), dpi=130)
    print("wrote results/tf_realignment.png, tf_realignment.json")
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "both"
    if cmd in ("attractor", "both"):
        N = int(sys.argv[sys.argv.index("--N") + 1]) if "--N" in sys.argv else 12
        part1_distinct_attractor(N=N)
    if cmd in ("realign", "both"):
        part2_realignment()
