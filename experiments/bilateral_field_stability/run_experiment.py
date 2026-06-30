"""
run_experiment.py -- Bilateral coherence-field stability test
=============================================================

Decisive minimal experiment: does splitting a single near-critical coherence
field into TWO coupled mirror lobes (a corpus-callosum membrane of permeability
mu) reduce its tendency to COLLAPSE (lock into a standing wave, or decohere)?

Design (fair fight):
  * Collapse-prone regime: high saturation, near-ceiling amplitude, minimal
    novelty (low noise) -> the single field locks ~>90% of seeds. This is the
    condition that froze "Eris 2.0" (standing-wave transfixion).
  * SINGLE baseline: one field per seed.
  * BILATERAL: two lobes. Lobe L is constructed with the SAME seed as the single
    baseline; lobe R is its mirror (theta_R=-theta_L, omega_R=-omega_L, an
    independent noise stream). At mu=0 lobe L evolves IDENTICALLY to the single
    field -> exact control. Any change at mu>0 is purely the membrane acting on
    the same seed. We report lobe L's collapse fraction as the headline (apples-
    to-apples with single), plus the combined readout and lobe R as secondary.
  * Sweep mu in {0, 0.01, 0.05, 0.1, 0.3, 0.6}; same seed set at every mu.

Outputs raw collapse fractions per mu, a plot, a wall-time ratio, and a
controllability check. Checkpoints after every mu so a long sweep can resume.
"""
from __future__ import annotations
import sys, os, json, time, argparse
from collections import Counter
import numpy as np

from field_core import PDEParams, SingleField, BilateralField
from metrics import CollapseMonitor, CollapseThresholds

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

MU_SWEEP = [0.0, 0.01, 0.05, 0.1, 0.3, 0.6]


def collapse_params(**over) -> PDEParams:
    """Collapse-prone (transfixion) regime, calibrated so the single field locks
    in the large majority of seeds: strong saturation drive, near-ceiling start,
    minimal amplitude/phase novelty, tight intrinsic frequencies (easy phase
    sync == the standing-wave lock)."""
    p = PDEParams()
    p.r_sat = 0.85
    p.d_decay = 0.20
    p.sigma_noise = 0.004
    p.sigma_phase = 0.004
    p.omega_spread = 0.25
    p.noise_structured = True
    for k, v in over.items():
        setattr(p, k, v)
    return p


# defaults; overridable on the CLI to probe cooler/hotter regimes
PHI_INIT, JITTER = 0.85, 0.04


def run_single(seed, p, T, thr):
    mon = CollapseMonitor(thr, log_every=25)
    f = SingleField(64, p, seed=seed, phi_init=PHI_INIT, phi_jitter=JITTER)
    for t in range(T):
        f.step(); mon.observe(t, f.phi, f.theta)
    return mon.finalize(), f


def run_bilateral(seed, mu, p, T, thr):
    f = BilateralField(64, p, seed=seed, mu=mu, phi_init=PHI_INIT, phi_jitter=JITTER)
    mon_L = CollapseMonitor(thr, log_every=25)
    mon_R = CollapseMonitor(thr, log_every=25)
    mon_C = CollapseMonitor(thr, log_every=25)
    for t in range(T):
        f.step()
        mon_L.observe(t, f.L.phi, f.L.theta)
        mon_R.observe(t, f.R.phi, f.R.theta)
        mon_C.observe(t, f.phi, f.theta)        # combined (averaged) readout
    return mon_L.finalize(), mon_R.finalize(), mon_C.finalize(), f


def summarize(results):
    frac = float(np.mean([r.collapsed for r in results]))
    cnt = dict(Counter(r.outcome for r in results))
    csteps = [r.collapse_step for r in results if r.collapse_step is not None]
    return {
        "collapse_fraction": frac,
        "outcomes": cnt,
        "median_collapse_step": float(np.median(csteps)) if csteps else None,
        "mean_temporal_var_final": float(np.mean([r.temporal_var_final for r in results])),
        "mean_kuramoto_final": float(np.mean([r.kuramoto_final for r in results])),
        "mean_spatial_var_final": float(np.mean([r.spatial_var_final for r in results])),
    }


def controllability_check(p, T, thr):
    """Is the combined bilateral readout stable & seed-dependent (not chaotic mush)?
       - DETERMINISM: same seed twice -> identical final descriptor (bitwise-ish).
       - SEED-DEPENDENCE: different seeds -> distinguishable descriptors.
       - BOUNDEDNESS: descriptor finite, within ceiling.
       Descriptor = concatenation of (mean phi, spatial var phi, mean tau, kuramoto)
       over the combined field, plus a coarse 8x8 down-pooled phi map."""
    def descriptor(f):
        from field_core import vorticity, local_coherence
        phi, th = f.phi, f.theta
        pooled = phi.reshape(8, 8, 8, 8).mean(axis=(1, 3)).ravel()
        scal = np.array([phi.mean(), phi.var(),
                         np.sqrt(np.mean(vorticity(phi, th) ** 2)),
                         local_coherence(th).mean()])
        return np.concatenate([scal, pooled])

    mu = 0.1
    _, _, _, fa = run_bilateral(123, mu, p, T, thr)
    _, _, _, fb = run_bilateral(123, mu, p, T, thr)     # repeat same seed
    _, _, _, fc = run_bilateral(999, mu, p, T, thr)     # different seed
    da, db, dc = descriptor(fa), descriptor(fb), descriptor(fc)
    return {
        "determinism_L2_same_seed": float(np.linalg.norm(da - db)),
        "seed_separation_L2_diff_seed": float(np.linalg.norm(da - dc)),
        "descriptor_finite": bool(np.all(np.isfinite(da))),
        "descriptor_max_abs": float(np.max(np.abs(da))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--T", type=int, default=800)
    ap.add_argument("--quick", action="store_true", help="16 seeds, T=700, subset of mu")
    ap.add_argument("--tag", default="full")
    # regime overrides (defaults = the collapse-prone regime in collapse_params)
    ap.add_argument("--omega_spread", type=float, default=None)
    ap.add_argument("--sigma", type=float, default=None, help="sets sigma_noise=sigma_phase")
    ap.add_argument("--phi_init", type=float, default=None)
    args = ap.parse_args()

    global PHI_INIT
    if args.phi_init is not None:
        PHI_INIT = args.phi_init

    if args.quick:
        nseed, T = 16, 700
        mus = [0.0, 0.05, 0.1, 0.3]
    else:
        nseed, T = args.seeds, args.T
        mus = MU_SWEEP

    thr = CollapseThresholds()
    over = {}
    if args.omega_spread is not None:
        over["omega_spread"] = args.omega_spread
    if args.sigma is not None:
        over["sigma_noise"] = args.sigma; over["sigma_phase"] = args.sigma
    p = collapse_params(**over)
    seeds = list(range(nseed))
    ckpt = os.path.join(OUT, f"sweep_{args.tag}.json")

    report = {
        "config": {
            "nseed": nseed, "T": T, "mu_sweep": mus, "grid": 64,
            "phi_init": PHI_INIT, "jitter": JITTER,
            "thresholds": vars(thr),
            "regime": {k: getattr(p, k) for k in
                       ["r_sat", "d_decay", "sigma_noise", "sigma_phase", "omega_spread"]},
        },
        "single": None, "bilateral": {}, "timing": {}, "controllability": None,
    }

    # ---- SINGLE baseline ----
    print(f"[single] {nseed} seeds, T={T} ...", flush=True)
    t0 = time.time()
    singles = [run_single(s, p, T, thr)[0] for s in seeds]
    t_single = time.time() - t0
    report["single"] = summarize(singles)
    report["timing"]["single_total_s"] = t_single
    report["timing"]["single_per_run_s"] = t_single / nseed
    print(f"  single collapse = {report['single']['collapse_fraction']:.0%} "
          f"({report['single']['outcomes']})  [{t_single:.1f}s]", flush=True)

    # ---- BILATERAL mu sweep ----
    bil_times = []
    for mu in mus:
        print(f"[bilateral mu={mu}] ...", flush=True)
        t0 = time.time()
        Ls, Rs, Cs = [], [], []
        for s in seeds:
            rl, rr, rc, _ = run_bilateral(s, mu, p, T, thr)
            Ls.append(rl); Rs.append(rr); Cs.append(rc)
        dt = time.time() - t0
        bil_times.append(dt)
        report["bilateral"][f"{mu}"] = {
            "lobe_L": summarize(Ls),      # headline: same-seed control vs single
            "lobe_R": summarize(Rs),
            "combined": summarize(Cs),
        }
        print(f"  L collapse={summarize(Ls)['collapse_fraction']:.0%}  "
              f"combined={summarize(Cs)['collapse_fraction']:.0%}  [{dt:.1f}s]", flush=True)
        # checkpoint after every mu
        with open(ckpt, "w") as fh:
            json.dump(report, fh, indent=2)

    report["timing"]["bilateral_per_run_s"] = float(np.mean(bil_times) / nseed)
    report["timing"]["wall_time_ratio_bil_over_single"] = (
        (np.mean(bil_times) / nseed) / (t_single / nseed))

    # ---- controllability ----
    print("[controllability] ...", flush=True)
    report["controllability"] = controllability_check(p, T, thr)
    print("  ", report["controllability"], flush=True)

    with open(ckpt, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {ckpt}")

    verdict(report)
    return report


def verdict(report):
    s = report["single"]["collapse_fraction"]
    print("\n" + "=" * 64)
    print("VERDICT")
    print("=" * 64)
    print(f"single-field collapse fraction (baseline): {s:.0%}")
    print(f"{'mu':>6} | {'lobe-L':>7} | {'combined':>8} | {'vs single (L)':>13}")
    best = None
    for mu, d in report["bilateral"].items():
        L = d["lobe_L"]["collapse_fraction"]
        C = d["combined"]["collapse_fraction"]
        delta = L - s
        flag = "  <-- better" if L < s - 1e-9 else ("  (worse)" if L > s + 1e-9 else "  (==)")
        print(f"{mu:>6} | {L:>6.0%} | {C:>7.0%} | {delta:>+12.0%}{flag}")
        if best is None or L < best[1]:
            best = (mu, L)
    print("-" * 64)
    if best and best[1] < s - 1e-9:
        print(f"=> BEST mu = {best[0]} : lobe-L collapse {best[1]:.0%} vs single {s:.0%} "
              f"(reduction {s - best[1]:+.0%})")
    else:
        print("=> No mu reduced lobe-L collapse below the single-field baseline.")
    tr = report["timing"].get("wall_time_ratio_bil_over_single")
    if tr:
        print(f"compute cost: bilateral / single wall-time ratio = {tr:.2f}x")


if __name__ == "__main__":
    main()
