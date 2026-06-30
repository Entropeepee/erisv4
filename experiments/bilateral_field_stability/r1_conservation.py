"""
R1 (numerical) -- conservation-budget test.

Q1: in the conservative limit (forcing/decay/noise OFF) does the E-gated two-lobe membrane
    conserve a total budget Q = Q_L + Q_R while only re-splitting it between lobes?
Q2: is the reallocation RATE shaped like E(theta) = (1/4) sin^2(2 theta) -- peaking at the
    related-but-distinct angle (45deg) and vanishing at BOTH sameness (0) and orthogonality
    (90) -- and is THIS the egate-specific part (vs sum-conservation being generic to any
    symmetric pore: diff/cos conserve the sum too)?
Q3: under FULL dynamics (forcing/noise/decay on) is Q approximately (adiabatically) conserved
    or destroyed?

Reuses field_core + bifurcate.TwoAgents verbatim.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bifurcate as bf
from field_core import PDEParams, coupling_gate, wrap_diff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DOMAINS = ["S", "D", "F", "TH", "B", "E", "IBT", "ZBT"]


def conservative_params(sigma=0.0):
    p = PDEParams()
    p.activations = {k: 0.0 for k in DOMAINS}   # forcing OFF (non-empty -> no decay branch)
    p.memory_coupling = 0.0
    p.sigma_noise = sigma; p.sigma_phase = sigma
    p.r_sat = 0.85; p.d_decay = 0.0; p.omega_spread = 0.25
    return p


def Q_amp(ta):   # total amplitude budget
    return float(np.sum(ta.L.phi) + np.sum(ta.R.phi))

def Q_action(ta):  # total "action"/energy
    return float(np.sum(ta.L.phi ** 2) + np.sum(ta.R.phi ** 2))


def test_conservation(kind="egate", mu=0.9, delta=0.1, T=2000, seed=0):
    p = conservative_params(sigma=0.0)
    ta = bf.TwoAgents(64, p, mu=mu, kind=kind, delta=delta, seed=seed, gate_phase=True)
    Q0, A0 = Q_amp(ta), Q_action(ta)
    split, Qrel, Arel, th = [], [], [], []
    for t in range(T):
        ta.step()
        split.append(float(np.sum(ta.L.phi) - np.sum(ta.R.phi)))  # the inter-lobe split
        Qrel.append((Q_amp(ta) - Q0) / Q0)
        Arel.append((Q_action(ta) - A0) / A0)
        th.append(ta.theta_LR())
    return {"Q0": Q0, "A0": A0, "Q_drift_rel": Qrel, "action_drift_rel": Arel,
            "split": split, "theta": th}


def reallocation_rate_vs_theta(mu=0.9, seed=1):
    """Impose a uniform relative-phase offset c on two amplitude-distinct lobes; measure the
    membrane reallocation coefficient (mean gate) per arm vs c, overlay E(c)=1/4 sin^2(2c)."""
    rng = np.random.default_rng(seed)
    # two distinct amplitude fields (so phi_R - phi_L != 0 -> there is something to reallocate)
    from field_core import colored_noise
    phiL = np.clip(0.6 + 0.15 * colored_noise((64, 64), rng, 3), 0.05, 0.95)
    phiR = np.clip(0.6 + 0.15 * colored_noise((64, 64), np.random.default_rng(seed + 7), 3), 0.05, 0.95)
    cs = np.linspace(0, np.pi / 2, 46)
    out = {"theta_deg": np.degrees(cs).tolist()}
    for kind in ["diff", "cos", "egate"]:
        rates = []
        for c in cs:
            # uniform offset => per-cell Delta = c everywhere
            g = coupling_gate(np.full((64, 64), c), kind)
            flux = g * (phiR - phiL)                 # membrane reallocation (per unit mu*dt)
            rates.append(float(np.mean(np.abs(flux))))
        out[kind] = rates
    out["E_curve"] = (0.25 * np.sin(2 * cs) ** 2).tolist()   # E(theta), scaled
    return out


def main():
    print("=== R1.Q1/Q3: conservation of Q=sum(phi_L+phi_R) ===")
    res = {}
    for label, sigma_full in [("conservative", False), ("full_dynamics", True)]:
        # conservative limit always uses conservative_params; full re-enables forcing+noise
        if label == "conservative":
            r = test_conservation("egate", mu=0.9, T=2000)
        else:
            p = PDEParams(); p.r_sat = 0.85; p.d_decay = 0.20
            p.sigma_noise = 0.016; p.sigma_phase = 0.016; p.omega_spread = 0.25
            ta = bf.TwoAgents(64, p, mu=0.9, kind="egate", delta=0.1, seed=0, gate_phase=True)
            Q0 = Q_amp(ta); drift = []
            for t in range(2000):
                ta.step(); drift.append((Q_amp(ta) - Q0) / Q0)
            r = {"Q_drift_rel": drift}
        qd = np.array(r["Q_drift_rel"])
        print(f"  {label:14s}: |Q drift|/Q  max={np.max(np.abs(qd)):.2e}  final={qd[-1]:+.2e}")
        res[label] = {"max_abs_drift": float(np.max(np.abs(qd))), "final_drift": float(qd[-1])}
        if label == "conservative":
            sp = np.array(r["split"])
            print(f"                  split (Q_L-Q_R): start={sp[0]:.2f} -> end={sp[-1]:.2f} "
                  f"(range {sp.max()-sp.min():.2f}); action drift max="
                  f"{np.max(np.abs(r['action_drift_rel'])):.2e}")
            res["conservative"]["split_start"] = float(sp[0]); res["conservative"]["split_end"] = float(sp[-1])
            res["conservative"]["action_max_drift"] = float(np.max(np.abs(r["action_drift_rel"])))
            cons = r

    print("\n=== R1.Q2: reallocation-rate shape vs theta (egate ~ E, vanishes at 0 AND 90) ===")
    rr = reallocation_rate_vs_theta()
    th = np.array(rr["theta_deg"])
    for kind in ["diff", "cos", "egate"]:
        a = np.array(rr[kind]); peak = th[np.argmax(a)]
        print(f"  {kind:6s}: rate(0deg)={a[0]:.3f} rate(45deg)={a[len(a)//2]:.3f} "
              f"rate(90deg)={a[-1]:.3f}  peak at {peak:.0f}deg")
    res["reallocation"] = {k: rr[k] for k in ["diff", "cos", "egate", "theta_deg", "E_curve"]}
    json.dump(res, open(os.path.join(RES, "r1_conservation.json"), "w"))

    # plots
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    qd = np.array(cons["Q_drift_rel"]); sp = np.array(cons["split"])
    ax[0].plot(qd, color="navy", label="total Q=Σ(φ_L+φ_R) drift / Q0")
    ax[0].plot(sp / abs(sp[0] if sp[0] != 0 else 1), color="crimson", alpha=0.6,
               label="split (Q_L−Q_R), normalized")
    ax[0].set_title(f"Conservative limit: Q conserved (drift~{np.max(np.abs(qd)):.0e}) "
                    f"while split relaxes")
    ax[0].set_xlabel("step"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    for kind, col in [("diff", "gray"), ("cos", "crimson"), ("egate", "navy")]:
        a = np.array(rr[kind]); ax[1].plot(th, a / a.max(), color=col, lw=2, label=kind)
    ax[1].plot(th, np.array(rr["E_curve"]) / max(rr["E_curve"]), "k--", lw=1.5, label="E(θ)=¼sin²2θ")
    ax[1].axvline(45, color="green", ls=":", alpha=0.5)
    ax[1].set_title("Reallocation-rate shape vs relatedness angle (normalized)")
    ax[1].set_xlabel("relative-phase angle θ (deg)"); ax[1].set_ylabel("rate (norm)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "r1_conservation.png"), dpi=130)
    print("\nwrote results/r1_conservation.json, results/r1_conservation.png")


if __name__ == "__main__":
    main()
