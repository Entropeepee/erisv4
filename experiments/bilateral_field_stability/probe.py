"""Quick exploratory probe: find a collapse-prone regime + see metric scales.
Not part of the deliverable; used to calibrate thresholds."""
import numpy as np
from field_core import PDEParams, SingleField, BilateralField
from metrics import CollapseMonitor, CollapseThresholds


def collapse_params(**over):
    """Collapse-prone (transfixion) regime: high saturation, near-ceiling, low novelty."""
    p = PDEParams()
    p.phi_yield_hi = 0.62
    p.r_sat = 0.85          # strong saturation drive
    p.d_decay = 0.20        # weak decay (less churn)
    p.sigma_noise = 0.004   # minimal amplitude novelty
    p.sigma_phase = 0.004   # minimal phase novelty
    p.omega_spread = 0.25   # tighter frequencies -> easier phase sync
    p.noise_structured = True
    for k, v in over.items():
        setattr(p, k, v)
    return p


def probe_run(kind, seed, T=900, mu=0.1, phi_init=0.85, jitter=0.04):
    p = collapse_params()
    thr = CollapseThresholds()
    mon = CollapseMonitor(thr, log_every=10)
    if kind == "single":
        f = SingleField(64, p, seed=seed, phi_init=phi_init, phi_jitter=jitter)
    else:
        f = BilateralField(64, p, seed=seed, mu=mu, phi_init=phi_init, phi_jitter=jitter)
    for t in range(T):
        f.step()
        mon.observe(t, f.phi, f.theta)
    r = mon.finalize()
    print(f"[{kind:9s} seed={seed} mu={mu if kind!='single' else '-'}] "
          f"outcome={r.outcome:11s} collapse@{r.collapse_step} "
          f"tvar={r.temporal_var_final:.2e} svar={r.spatial_var_final:.2e} "
          f"kur={r.kuramoto_final:.3f} tau={r.tau_rms_final:.3f} phimax={r.phi_max_final:.3f}")
    return r


if __name__ == "__main__":
    print("=== single field, collapse regime ===")
    singles = [probe_run("single", s) for s in range(8)]
    print("\n=== bilateral mu=0.1 ===")
    bil = [probe_run("bilateral", s, mu=0.1) for s in range(8)]

    # print temporal-var trajectory of one single run to set eps_lock
    print("\n=== sample tvar trajectory (single seed0) ===")
    p = collapse_params(); mon = CollapseMonitor(CollapseThresholds(), log_every=25)
    f = SingleField(64, p, seed=0, phi_init=0.85, phi_jitter=0.04)
    for t in range(900):
        f.step(); mon.observe(t, f.phi, f.theta)
    r = mon.finalize()
    for i in range(len(r.t)):
        print(f"  t={r.t[i]:4d} tvar={r.temporal_var[i]:.3e} svar={r.spatial_var[i]:.3e} "
              f"kur={r.kuramoto[i]:.3f} tau={r.tau_rms[i]:.3f}")
