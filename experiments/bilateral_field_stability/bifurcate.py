"""
bifurcate.py -- Sustained dynamic two-ness via E-gated coupling
===============================================================

Tests whether E-gated membrane transport (the coupling law E(Delta)=cos^2 Delta *
sin^2 Delta, peaks at 45 deg, ZERO at 0 and 90 deg) between two DISTINCT, detuned
field-agents produces a STABLE related-but-distinct coupling angle theta* in (0,90) --
sustained two-ness -- where plain diffusive coupling only fuses-then-locks.

Dynamics reuse field_core.SingleField verbatim (coupling_kind/gate_phase added there,
backward-compatible). Two agents share one law/params but are NON-mirror: independent
seeds, independent noise, intrinsic-frequency detuning delta (omega_L mean +delta/2,
omega_R mean -delta/2). Coupling arms (amplitude transport gate):
  iso   : mu=0                      (independent)
  diff  : g=1                       (fusion / mirror-class)
  cos   : g=cos^2 Delta             (fusion; NEGATIVE CONTROL)
  egate : g=cos^2 Delta sin^2 Delta (sustained two-ness in the delta/mu band)
The decisive contrast is egate vs cos -- isolates the sin^2 factor.

Metrics are threshold-free and attractor-based at long T (>=2500): the global coupling
angle theta_LR trajectory, per-domain liveness (BOTH lobes), live exchange (transport
magnitude), classification (fusion/sustained/segregation), and an attractor test
(perturb theta_LR both ways, does it return to theta*?).
"""
from __future__ import annotations
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import field_core as cc
from metrics import CollapseMonitor, CollapseThresholds

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "results", "bifurcate")
os.makedirs(OUTDIR, exist_ok=True)

PHI_INIT, JITTER = 0.85, 0.04
SEED_OFFSET = 20000
CKPT_EVERY = 10


def regime(sigma, omega_spread=0.25, **ov):
    p = cc.PDEParams()
    p.r_sat = 0.85; p.d_decay = 0.20
    p.sigma_noise = sigma; p.sigma_phase = sigma; p.omega_spread = omega_spread
    for k, v in ov.items():
        setattr(p, k, v)
    return p


class TwoAgents:
    def __init__(self, size, params, mu, kind, delta, seed, gate_phase=False):
        self.L = cc.SingleField(size, params, seed=seed, phi_init=PHI_INIT, phi_jitter=JITTER)
        self.R = cc.SingleField(size, params, seed=seed + SEED_OFFSET,
                                phi_init=PHI_INIT, phi_jitter=JITTER)
        # intrinsic-frequency detuning between two DISTINCT fields (not mirror sign-flip)
        self.L.omega0 = self.L.omega0 + delta / 2.0
        self.R.omega0 = self.R.omega0 - delta / 2.0
        self.mu, self.kind, self.delta, self.gate_phase = mu, kind, delta, gate_phase
        self._transport = 0.0

    def step(self):
        if self.kind == "iso" or self.mu == 0.0:
            self.L.step(); self.R.step()
            self._transport = 0.0
            return
        lp, lt = self.L.phi.copy(), self.L.theta.copy()
        rp, rt = self.R.phi.copy(), self.R.theta.copy()
        self.L.step_with_coupling(rp, rt, self.mu, coupling_kind=self.kind, gate_phase=self.gate_phase)
        self.R.step_with_coupling(lp, lt, self.mu, coupling_kind=self.kind, gate_phase=self.gate_phase)
        self._transport = 0.5 * (self.L._last_transport + self.R._last_transport)

    def theta_LR(self):
        """Global coupling angle in degrees [0,180] from the normalized complex-field
        overlap cos = Re<psiL,psiR>/(|psiL||psiR|), psi = rho e^{i theta}."""
        psiL = self.L.phi * np.exp(1j * self.L.theta)
        psiR = self.R.phi * np.exp(1j * self.R.theta)
        ov = np.sum(psiL * np.conj(psiR))
        denom = np.sqrt(np.sum(self.L.phi ** 2)) * np.sqrt(np.sum(self.R.phi ** 2)) + 1e-12
        return float(np.degrees(np.arccos(np.clip(np.real(ov) / denom, -1.0, 1.0))))

    def snapshot(self):
        return (self.L.phi.copy(), self.L.theta.copy(), self.R.phi.copy(), self.R.theta.copy())

    def restore(self, snap):
        self.L.phi, self.L.theta, self.R.phi, self.R.theta = (s.copy() for s in snap)


# classification thresholds (degrees)
FUSE_DEG, SEG_DEG = 20.0, 75.0


def classify(mean_theta, bothalive):
    if mean_theta < FUSE_DEG:
        return "fusion"
    if mean_theta > SEG_DEG:
        return "segregation"
    return "sustained" if bothalive else "segregation"


def run_cell_seed(kind, delta, mu, seed, T, gate_phase=False, rec_every=10):
    p = regime(run_cell_seed.sigma)
    ta = TwoAgents(64, p, mu=mu, kind=kind, delta=delta, seed=seed, gate_phase=gate_phase)
    monL = CollapseMonitor(CollapseThresholds(), log_every=500)
    monR = CollapseMonitor(CollapseThresholds(), log_every=500)
    th, tr = [], []
    for t in range(T):
        ta.step()
        monL.observe(t, ta.L.phi, ta.L.theta)
        monR.observe(t, ta.R.phi, ta.R.theta)
        if t % rec_every == 0:
            th.append(ta.theta_LR()); tr.append(ta._transport)
    rL, rR = monL.finalize(), monR.finalize()
    th = np.array(th); tr = np.array(tr)
    post = th[len(th) // 2:]            # post-transient half
    post_tr = tr[len(tr) // 2:]
    bothalive = (not rL.collapsed) and (not rR.collapsed)
    mean_theta = float(np.mean(post))
    rec = {
        "seed": seed, "kind": kind, "delta": delta, "mu": mu,
        "mean_theta_LR": mean_theta, "std_theta_LR": float(np.std(post)),
        "theta_LR_final": float(th[-1]),
        "L_collapsed": bool(rL.collapsed), "R_collapsed": bool(rR.collapsed),
        "L_tvar": float(rL.temporal_var_final), "R_tvar": float(rR.temporal_var_final),
        "both_alive": bool(bothalive),
        "mean_transport_post": float(np.mean(post_tr)),
        "class": classify(mean_theta, bothalive),
        # decimated trajectory for plotting (further thinned)
        "theta_traj": [round(x, 2) for x in th[::5].tolist()],
    }
    return rec


# --------------------------------------------------------------------------- #
def _path(key):
    return os.path.join(OUTDIR, key + ".json")


def run_cell(kind, delta, mu, N, T, sigma, gate_phase=False):
    run_cell_seed.sigma = sigma
    gp = "_gp" if gate_phase else ""
    key = f"{kind}{gp}_d{delta}_m{mu}_s{sigma}"
    st = json.load(open(_path(key))) if os.path.exists(_path(key)) else {"key": key, "records": []}
    recs = st["records"]
    done = {r["seed"] for r in recs}
    for s in range(N):
        if s in done:
            continue
        recs.append(run_cell_seed(kind, delta, mu, s, T, gate_phase=gate_phase))
        if (s + 1) % CKPT_EVERY == 0:
            st["records"] = recs; st["meta"] = {"N": N, "T": T, "sigma": sigma}
            json.dump(st, open(_path(key), "w"))
            print(f"[{key}] {len(recs)}/{N}", flush=True)
    st["records"] = recs; st["meta"] = {"N": N, "T": T, "sigma": sigma, "done": True}
    json.dump(st, open(_path(key), "w"))
    from collections import Counter
    cl = Counter(r["class"] for r in recs)
    mt = float(np.mean([r["mean_theta_LR"] for r in recs]))
    ba = float(np.mean([r["both_alive"] for r in recs]))
    tr = float(np.mean([r["mean_transport_post"] for r in recs]))
    print(f"[{key}] DONE  mean_theta={mt:.1f}deg both_alive={ba:.0%} transport={tr:.2e}  {dict(cl)}",
          flush=True)


def attractor_test(kind, delta, mu, sigma, seed, T_settle=1500, T_relax=1000, gate_phase=False):
    """Settle, record theta*, then perturb theta_LR BOTH ways and watch for return."""
    run_cell_seed.sigma = sigma
    p = regime(sigma)
    ta = TwoAgents(64, p, mu=mu, kind=kind, delta=delta, seed=seed, gate_phase=gate_phase)
    for t in range(T_settle):
        ta.step()
    base = np.mean([ta.theta_LR() for _ in _peek(ta, 50)])
    snap = ta.snapshot()

    def relax(perturb):
        ta.restore(snap)
        perturb(ta)
        traj = []
        for t in range(T_relax):
            ta.step()
            if t % 10 == 0:
                traj.append(ta.theta_LR())
        return traj

    # toward sameness: copy L's phase into R (theta_LR -> ~0)
    traj_same = relax(lambda a: setattr(a.R, "theta", a.L.theta.copy()))
    # toward orthogonality: rotate R's phase by +90 deg (theta_LR -> larger)
    traj_orth = relax(lambda a: setattr(a.R, "theta", (a.L.theta + np.pi / 2) % (2 * np.pi)))
    return {"theta_star": float(base),
            "after_same_kick": float(np.mean(traj_same[-10:])),
            "after_orth_kick": float(np.mean(traj_orth[-10:])),
            "traj_same": [round(x, 2) for x in traj_same],
            "traj_orth": [round(x, 2) for x in traj_orth]}


def _peek(ta, n):
    # advance a few steps to average theta* over a small window
    for _ in range(n):
        ta.step()
        yield None


def main():
    a = sys.argv
    if len(a) < 2:
        raise SystemExit(__doc__)
    cmd = a[1]
    def opt(name, default, cast=float):
        return cast(a[a.index(name) + 1]) if name in a else default
    N = int(opt("--N", 16, int)); T = int(opt("--T", 2500, int)); sigma = opt("--sigma", 0.016)
    gate_phase = "--gate_phase" in a
    if cmd == "cell":
        kind, delta, mu = a[2], float(a[3]), float(a[4])
        run_cell(kind, delta, mu, N, T, sigma, gate_phase=gate_phase)
    elif cmd == "attractor":
        kind, delta, mu = a[2], float(a[3]), float(a[4])
        seed = int(opt("--seed", 0, int))
        r = attractor_test(kind, delta, mu, sigma, seed, gate_phase=gate_phase)
        key = f"attractor_{kind}_d{delta}_m{mu}_s{sigma}_seed{seed}"
        json.dump(r, open(_path(key), "w"), indent=1)
        print(f"[{key}] theta*={r['theta_star']:.1f}  after_same={r['after_same_kick']:.1f}  "
              f"after_orth={r['after_orth_kick']:.1f}")
    else:
        raise SystemExit(f"unknown cmd {cmd}")


if __name__ == "__main__":
    main()
