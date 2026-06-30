"""
twoness_tasks.py -- T-B (aliveness axis), T-C (robustness), T-E (exchange benchmark).

Reuses bifurcate.TwoAgents and field_core VERBATIM (no new dynamics). Adds runners with
full regime control (sigma, r_sat) so we can drive toward the LOCK EDGE -- the regime
the sigma=0.016 run never stressed -- and ask the decisive question: at an operating
point where plain diffusive coupling drives mutual LOCK, does E-gated coupling keep BOTH
lobes ALIVE while diff/cos do not?

Per-cell JSON (parallel-safe), --resume, raw per-seed records.
"""
from __future__ import annotations
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bifurcate as bf
from field_core import PDEParams
from metrics import CollapseMonitor, CollapseThresholds

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "results", "bifurcate")
os.makedirs(OUTDIR, exist_ok=True)
CKPT_EVERY = 10


def regime(sigma, r_sat=0.85, omega_spread=0.25, d_decay=0.20):
    p = PDEParams()
    p.r_sat = r_sat; p.d_decay = d_decay
    p.sigma_noise = sigma; p.sigma_phase = sigma; p.omega_spread = omega_spread
    return p


def run_seed(kind, delta, mu, p, seed, T, gate_phase=True):
    ta = bf.TwoAgents(64, p, mu=mu, kind=kind, delta=delta, seed=seed, gate_phase=gate_phase)
    mL = CollapseMonitor(CollapseThresholds(), log_every=1000)
    mR = CollapseMonitor(CollapseThresholds(), log_every=1000)
    th, tr = [], []
    for t in range(T):
        ta.step()
        mL.observe(t, ta.L.phi, ta.L.theta)
        mR.observe(t, ta.R.phi, ta.R.theta)
        if t % 10 == 0:
            th.append(ta.theta_LR()); tr.append(ta._transport)
    rL, rR = mL.finalize(), mR.finalize()
    post_th = th[len(th) // 2:]
    post_tr = tr[len(tr) // 2:]
    return {
        "seed": seed, "kind": kind,
        "L_collapsed": bool(rL.collapsed), "R_collapsed": bool(rR.collapsed),
        "both_alive": bool((not rL.collapsed) and (not rR.collapsed)),
        "either_alive": bool((not rL.collapsed) or (not rR.collapsed)),
        "L_outcome": rL.outcome, "R_outcome": rR.outcome,
        "L_collapse_step": rL.collapse_step, "R_collapse_step": rR.collapse_step,
        "mean_theta_LR": float(np.mean(post_th)),
        "transport": float(np.mean(post_tr)),
        "L_tvar": float(rL.temporal_var_final), "R_tvar": float(rR.temporal_var_final),
    }


def run_cell(kind, delta, mu, sigma, r_sat, N, T, gate_phase=True, tag=""):
    p = regime(sigma, r_sat=r_sat)
    key = f"alive{tag}_{kind}_d{delta}_m{mu}_s{sigma}_r{r_sat}"
    path = os.path.join(OUTDIR, key + ".json")
    st = json.load(open(path)) if os.path.exists(path) else {"key": key, "records": []}
    recs = st["records"]; done = {r["seed"] for r in recs}
    for s in range(N):
        if s in done:
            continue
        recs.append(run_seed(kind, delta, mu, p, s, T, gate_phase=gate_phase))
        if (s + 1) % CKPT_EVERY == 0:
            st["records"] = recs
            st["meta"] = {"N": N, "T": T, "sigma": sigma, "r_sat": r_sat, "delta": delta, "mu": mu}
            json.dump(st, open(path, "w"))
            print(f"[{key}] {len(recs)}/{N}", flush=True)
    st["records"] = recs
    st["meta"] = {"N": N, "T": T, "sigma": sigma, "r_sat": r_sat, "delta": delta, "mu": mu, "done": True}
    json.dump(st, open(path, "w"))
    ba = np.mean([r["both_alive"] for r in recs])
    Lc = np.mean([r["L_collapsed"] for r in recs]); Rc = np.mean([r["R_collapsed"] for r in recs])
    mt = np.mean([r["mean_theta_LR"] for r in recs])
    print(f"[{key}] DONE both_alive={ba:.0%} L_lock={Lc:.0%} R_lock={Rc:.0%} "
          f"theta={mt:.0f} n={len(recs)}", flush=True)


def main():
    a = sys.argv
    def opt(name, d, cast=float):
        return cast(a[a.index(name) + 1]) if name in a else d
    if len(a) < 2:
        raise SystemExit(__doc__)
    cmd = a[1]
    N = int(opt("--N", 20, int)); T = int(opt("--T", 2500, int))
    sigma = opt("--sigma", 0.010); r_sat = opt("--r_sat", 0.85)
    gate_phase = "--no_gate_phase" not in a
    tag = a[a.index("--tag") + 1] if "--tag" in a else ""
    if cmd == "cell":
        kind, delta, mu = a[2], float(a[3]), float(a[4])
        run_cell(kind, delta, mu, sigma, r_sat, N, T, gate_phase=gate_phase, tag=tag)
    else:
        raise SystemExit(f"unknown cmd {cmd}")


if __name__ == "__main__":
    main()
