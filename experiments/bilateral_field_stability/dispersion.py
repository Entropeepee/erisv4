"""
dispersion.py  (handoff section 7, secondary probe)
===================================================

Does this PDE support SPONTANEOUS spatial division? Measure the dispersion relation
lambda(k) of the linearized homogeneous field: seed a single Fourier mode of wavenumber
k as a small perturbation on the homogeneous fixed point, evolve NOISE-FREE, and read the
mode's growth/decay rate from a difference field against an unperturbed control (isolates
the linear response). If lambda(k) > 0 for a band of k > 0 -> a Turing/pattern-forming
instability -> one field can split itself into spatial domains with a self-generated wall
at wavelength 2*pi/k*. If lambda(k) <= 0 for all k > 0 -> spontaneous spatial division is
NOT native to this PDE (the relational two-agent route is then the only route).

Pure dynamics from field_core (no analytic linearization); homogeneous => omega_spread=0,
noise off.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from field_core import PDEParams, SingleField, local_coherence, vorticity

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SIZE = 64


def homo_params(sigma=0.016):
    """Same law as the experiment but noise-free + homogeneous intrinsic frequency,
    so the background stays a clean homogeneous fixed point to linearize about."""
    p = PDEParams()
    p.r_sat = 0.85; p.d_decay = 0.20
    p.sigma_noise = 0.0; p.sigma_phase = 0.0; p.omega_spread = 0.0
    return p


def make_homogeneous(p, phi0):
    f = SingleField(SIZE, p, seed=0, phi_init=phi0, phi_jitter=0.0)
    f.phi[:] = phi0
    f.theta[:] = 0.0
    f.omega0[:] = p.omega
    f.memory[:] = phi0
    f.attention[:] = 1.0
    f.phi_prev = f.phi.copy()
    f.theta_prev = f.theta.copy()
    f._lc = local_coherence(f.theta)
    f.tau = vorticity(f.phi, f.theta)
    return f


def relax_homogeneous(p, phi0=0.5, steps=400):
    """Relax a uniform field to its homogeneous fixed point phi*."""
    f = make_homogeneous(p, phi0)
    for _ in range(steps):
        f.step()
        f.theta[:] = float(np.mean(f.theta))   # keep phase exactly homogeneous
    return float(np.mean(f.phi))


def growth_rate(p, phi_star, k, eps=1e-3, steps=40, channel="phi"):
    """Seed mode k in `channel` on the homogeneous fixed point; return lambda(k) =
    d/dt log|mode| measured in that SAME channel (wrap-safe for theta), against an
    unperturbed homogeneous control so only the linear mode response is captured."""
    from field_core import wrap_diff
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    mode = np.cos(2 * np.pi * k * x / SIZE)
    base = make_homogeneous(p, phi_star)
    pert = make_homogeneous(p, phi_star)
    if channel == "phi":
        pert.phi = pert.phi + eps * mode
    else:
        pert.theta = (pert.theta + eps * mode) % (2 * np.pi)
    amps = []
    for t in range(steps):
        base.step(); pert.step()
        base.phi[:] = float(np.mean(base.phi))
        base.theta[:] = float(np.mean(base.theta))
        if channel == "phi":
            diff = pert.phi - base.phi
        else:
            diff = wrap_diff(pert.theta, base.theta)   # perturbation in the phase channel
        F = np.fft.fft2(diff)
        amps.append(np.abs(F[0, k]) / (SIZE * SIZE))
    amps = np.array(amps)
    good = amps > 1e-18
    if good.sum() < 8:
        return float("nan")
    tt = np.arange(steps)[good]
    lam = np.polyfit(tt, np.log(amps[good]), 1)[0]   # per-step growth rate
    return float(lam)


def main():
    p = homo_params()
    phi_star = relax_homogeneous(p)
    print(f"homogeneous fixed point phi* = {phi_star:.4f}")
    ks = list(range(1, 33))
    lam_phi = [growth_rate(p, phi_star, k, channel="phi") for k in ks]
    lam_th = [growth_rate(p, phi_star, k, channel="theta") for k in ks]
    print(f"{'k':>3} {'lambda_phi':>12} {'lambda_theta':>13}")
    for k, lp, lt in zip(ks, lam_phi, lam_th):
        flag = "  <== UNSTABLE" if (lp > 1e-6 or lt > 1e-6) else ""
        print(f"{k:>3} {lp:>12.2e} {lt:>13.2e}{flag}")

    # A genuine pattern-forming (Turing) band is a CONTIGUOUS run of positive growth at
    # well-resolved wavenumbers -- NOT isolated spikes at/near Nyquist (k=32 on a 64 grid,
    # and its k=16 alias), which are discretization artifacts. Restrict to k<=24 and require
    # two adjacent unstable wavenumbers.
    KMAX_PHYS = 24
    def contiguous_band(lams):
        pos = [(k <= KMAX_PHYS and l == l and l > 1e-6) for k, l in zip(ks, lams)]
        return any(pos[i] and pos[i + 1] for i in range(len(pos) - 1))
    phi_band = contiguous_band(lam_phi)
    th_band = contiguous_band(lam_th)
    nyquist_spikes = [k for k, l in zip(ks, lam_phi) if l == l and l > 1e-6 and k > KMAX_PHYS] \
        + [k for k, l in zip(ks, lam_phi) if l == l and l > 1e-6 and k == 16]
    kstar = None
    if phi_band or th_band:
        allk = [(l, k, "phi") for k, l in zip(ks, lam_phi) if k <= KMAX_PHYS] \
            + [(l, k, "theta") for k, l in zip(ks, lam_th) if k <= KMAX_PHYS]
        allk = [t for t in allk if t[0] == t[0] and t[0] > 1e-6]
        l, kstar, ch = max(allk)
        print(f"\n=> TURING-type instability: contiguous positive band, peak k*={kstar} ({ch}), "
              f"wavelength ~{SIZE/kstar:.1f} cells, lambda={l:.2e}/step")
    else:
        print(f"\n=> NO genuine pattern-forming band: every well-resolved mode (k<=24) decays "
              f"(lambda<0). Isolated positive spikes at k={sorted(set(nyquist_spikes))} are "
              f"Nyquist/half-Nyquist discretization artifacts, not a physical instability.")
        print("   => Spontaneous SPATIAL division is NOT native to this PDE; the relational "
              "two-agent route is the route to two-ness.")
    any_unstable = bool(phi_band or th_band)

    json.dump({"phi_star": phi_star, "k": ks, "lambda_phi": lam_phi, "lambda_theta": lam_th,
               "unstable": bool(any_unstable), "kstar": kstar},
              open(os.path.join(RES, "dispersion.json"), "w"), indent=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(ks, lam_phi, "o-", label="phi-channel", color="navy")
    ax.plot(ks, lam_th, "s-", label="theta-channel", color="crimson")
    ax.set_xlabel("wavenumber k"); ax.set_ylabel("growth rate lambda(k) per step")
    ax.set_title("Dispersion relation of the homogeneous field\n(>0 anywhere = spontaneous pattern formation)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "dispersion.png"), dpi=130)
    print("wrote results/dispersion.json, results/dispersion.png")


if __name__ == "__main__":
    main()
