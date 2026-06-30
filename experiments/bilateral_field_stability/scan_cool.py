"""Fast single-only scan at T=800 to locate a METASTABLE regime
(single-field collapse ~70-90%), giving the membrane headroom to help."""
import numpy as np
from collections import Counter
from field_core import PDEParams, SingleField
from metrics import CollapseMonitor, CollapseThresholds


def params(omega_spread, sigma):
    p = PDEParams()
    p.r_sat = 0.85; p.d_decay = 0.20
    p.sigma_noise = sigma; p.sigma_phase = sigma
    p.omega_spread = omega_spread
    return p


def run(seed, p, T, thr, phi_init):
    mon = CollapseMonitor(thr, log_every=100)
    f = SingleField(64, p, seed=seed, phi_init=phi_init, phi_jitter=0.04)
    for t in range(T):
        f.step(); mon.observe(t, f.phi, f.theta)
    return mon.finalize()


if __name__ == "__main__":
    thr = CollapseThresholds(); T = 800; n = 16
    print(f"single-only metastability scan  T={T} n={n}")
    for sp in [0.25, 0.30, 0.35, 0.40]:
        for sig in [0.004, 0.008, 0.012]:
            p = params(sp, sig)
            outs = [run(s, p, T, thr, 0.85) for s in range(n)]
            frac = np.mean([o.collapsed for o in outs])
            tv = np.mean([o.temporal_var_final for o in outs])
            print(f"  omega_spread={sp:.2f} sigma={sig:.3f} -> collapse={frac:4.0%} "
                  f"tvar={tv:.2e}  {dict(Counter(o.outcome for o in outs))}", flush=True)
