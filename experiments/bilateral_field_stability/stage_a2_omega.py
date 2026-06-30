"""Stage A (cont.): if sigma alone won't sustain aliveness at long T, sweep the real
criticality knob -- omega_spread (intrinsic-frequency heterogeneity). Higher spread =>
frustrated phase sync => no global standing-wave lock => sustained dynamics. Find a
(omega_spread, sigma) where an isolated single field is robustly ALIVE at T=2500."""
import numpy as np
from collections import Counter
from field_core import PDEParams, SingleField
from metrics import CollapseMonitor, CollapseThresholds

T, N = 2500, 16
PHI_INIT, JITTER = 0.85, 0.04


def params(omega_spread, sigma):
    p = PDEParams()
    p.r_sat = 0.85; p.d_decay = 0.20
    p.sigma_noise = sigma; p.sigma_phase = sigma; p.omega_spread = omega_spread
    return p


def run(seed, p):
    mon = CollapseMonitor(CollapseThresholds(), log_every=500)
    f = SingleField(64, p, seed=seed, phi_init=PHI_INIT, phi_jitter=JITTER)
    for t in range(T):
        f.step(); mon.observe(t, f.phi, f.theta)
    return mon.finalize()


if __name__ == "__main__":
    print(f"Stage A2: omega_spread x sigma sustained-alive scan  T={T} N={N}")
    print(f"{'om_spr':>7} {'sigma':>7} | {'collapse%':>9} | {'kur':>6} | outcomes")
    for osp in [0.5, 0.8, 1.2]:
        for sig in [0.006, 0.010]:
            p = params(osp, sig)
            outs = [run(s, p) for s in range(N)]
            frac = np.mean([o.collapsed for o in outs])
            kur = np.mean([o.kuramoto_final for o in outs])
            print(f"{osp:7.2f} {sig:7.3f} | {frac*100:8.0f}% | {kur:6.3f} | "
                  f"{dict(Counter(o.outcome for o in outs))}", flush=True)
