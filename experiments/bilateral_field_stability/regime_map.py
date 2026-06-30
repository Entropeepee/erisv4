"""Map single-field collapse fraction vs the criticality knob (omega_spread)
to locate a metastable regime (single collapses SOMETIMES, not always)."""
import numpy as np
from field_core import PDEParams, SingleField
from metrics import CollapseMonitor, CollapseThresholds


def params(omega_spread, sigma_noise, sigma_phase, r_sat=0.85, d_decay=0.20):
    p = PDEParams()
    p.r_sat = r_sat; p.d_decay = d_decay
    p.sigma_noise = sigma_noise; p.sigma_phase = sigma_phase
    p.omega_spread = omega_spread
    return p


def run_single(seed, p, T, thr, phi_init=0.85, jitter=0.04):
    mon = CollapseMonitor(thr, log_every=50)
    f = SingleField(64, p, seed=seed, phi_init=phi_init, phi_jitter=jitter)
    for t in range(T):
        f.step(); mon.observe(t, f.phi, f.theta)
    return mon.finalize()


if __name__ == "__main__":
    thr = CollapseThresholds()
    T = 600
    nseed = 12
    print(f"T={T} nseed={nseed}  (LOCK if tvar<{thr.eps_lock:.0e} for {thr.n_consec} steps)")
    for sp in [0.25, 0.5, 0.8, 1.2, 1.8, 2.5]:
        for sn, spn in [(0.004, 0.004), (0.02, 0.02), (0.05, 0.05)]:
            p = params(sp, sn, spn)
            outs = [run_single(s, p, T, thr) for s in range(nseed)]
            frac = np.mean([o.collapsed for o in outs])
            from collections import Counter
            c = Counter(o.outcome for o in outs)
            tv = np.mean([o.temporal_var_final for o in outs])
            kur = np.mean([o.kuramoto_final for o in outs])
            print(f"  omega_spread={sp:4.2f} sig_n={sn:.3f} sig_p={spn:.3f} -> "
                  f"collapse={frac:4.0%}  tvar={tv:.2e} kur={kur:.3f}  {dict(c)}")
