"""Stage A precondition: find a SUSTAINED-ALIVE operating regime for an isolated
single field at long T (T=2500). The monostable sigma=0.007 lock regime is wrong for
this experiment -- we need 'lock' to be a real event, not the only attractor. Sweep
sigma up; want collapse~0 AND coherence maintained (not DEATH_NOISE)."""
import numpy as np
from collections import Counter
from field_core import PDEParams, SingleField
from metrics import CollapseMonitor, CollapseThresholds

T, N = 2500, 16
PHI_INIT, JITTER = 0.85, 0.04


def params(sigma, omega_spread=0.25):
    p = PDEParams()
    p.r_sat = 0.85; p.d_decay = 0.20
    p.sigma_noise = sigma; p.sigma_phase = sigma; p.omega_spread = omega_spread
    return p


def run(seed, p):
    mon = CollapseMonitor(CollapseThresholds(), log_every=200)
    f = SingleField(64, p, seed=seed, phi_init=PHI_INIT, phi_jitter=JITTER)
    for t in range(T):
        f.step(); mon.observe(t, f.phi, f.theta)
    return mon.finalize()


if __name__ == "__main__":
    print(f"Stage A: single-field sustained-alive scan  T={T} N={N}")
    print(f"{'sigma':>7} | {'collapse%':>9} | {'mean kur':>8} | outcomes")
    for sigma in [0.010, 0.013, 0.016, 0.020, 0.025, 0.030]:
        p = params(sigma)
        outs = [run(s, p) for s in range(N)]
        frac = np.mean([o.collapsed for o in outs])
        kur = np.mean([o.kuramoto_final for o in outs])
        print(f"{sigma:7.3f} | {frac*100:8.0f}% | {kur:8.3f} | "
              f"{dict(Counter(o.outcome for o in outs))}", flush=True)
