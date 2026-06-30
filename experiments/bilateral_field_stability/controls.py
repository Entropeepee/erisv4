"""
controls.py -- Adversarial controls for the bilateral-collapse "win"
====================================================================

The headline result (single 70% -> bilateral mu=0.1 lobe-L 30% collapse) is under
suspicion. This module runs the controls from the handoff to decide whether the
membrane effect is (a) a detector artifact, (b) reproducible by matched noise, or
(c) genuinely bilateral/mirror-specific.

Dynamics are REUSED VERBATIM from field_core (SingleField/BilateralField) and
metrics (CollapseMonitor/CollapseThresholds) -- nothing re-derived.

Each arm runs independently and writes results/controls/<key>.json with one record
per seed (collapsed, outcome, collapse_step, temporal_var_final, spatial_var_final,
kuramoto_final). Per-arm files => safe to run many arms in parallel processes.
Resumable: re-running an arm continues from the last checkpointed seed.

Arms (run via `python controls.py <arm> ...`):
  single     <sigma>                 single-field baseline / sigma-bump (T1,T2,T4)
  bilatL     <mu> <sigma>            true bilateral, lobe-L (also records RMS|phiL-phiR|)
  white      <mu> <sigma>            matched WHITE amplitude noise into a single field (T2)
  colored    <mu> <sigma>            matched COLORED amplitude noise (T2)
  sham_frozen   <mu> <sigma>         L coupled to a FROZEN partner snapshot (T3)
  sham_indep    <mu> <sigma>         L coupled to a free INDEPENDENT partner, no back-coupling (T3)
  sham_mutual   <mu> <sigma>         two MUTUALLY coupled lobes, NOT mirror-initialized (T3, isolates mirror-init)
  kick_lock     <sigma>             evolve, then zero novelty for 50 steps (push alive->lock), release (T4)
  kick_alive    <sigma>             evolve, then 5x novelty for 50 steps (push lock->alive), release (T4)
"""
from __future__ import annotations
import sys, os, json, copy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import field_core as cc
import metrics as ms

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "results", "controls")
os.makedirs(OUTDIR, exist_ok=True)

T_DEFAULT, N_DEFAULT = 800, 60
PHI_INIT, JITTER = 0.85, 0.04
CKPT_EVERY = 10
KICK_AT, KICK_LEN = 500, 50


def regime(**ov):
    p = cc.PDEParams()
    p.r_sat = 0.85; p.d_decay = 0.20
    p.sigma_noise = 0.004; p.sigma_phase = 0.004; p.omega_spread = 0.25
    for k, v in ov.items():
        setattr(p, k, v)
    return p


def THR():
    return ms.CollapseThresholds()


def _path(key):
    return os.path.join(OUTDIR, key + ".json")


def _load(key):
    p = _path(key)
    return json.load(open(p)) if os.path.exists(p) else {"key": key, "records": [], "meta": {}}


def _save(key, st):
    json.dump(st, open(_path(key), "w"), indent=1)


def _rec(seed, r):
    return {"seed": seed, "collapsed": bool(r.collapsed), "outcome": r.outcome,
            "collapse_step": r.collapse_step,
            "tvar": float(r.temporal_var_final), "svar": float(r.spatial_var_final),
            "kur": float(r.kuramoto_final)}


# --------------------------------------------------------------------------- #
#  per-seed evolvers (each returns metrics.RunResult; some also return extra)
# --------------------------------------------------------------------------- #
def ev_single(seed, p, T):
    mon = ms.CollapseMonitor(THR(), log_every=25)
    f = cc.SingleField(64, p, seed=seed, phi_init=PHI_INIT, phi_jitter=JITTER)
    for t in range(T):
        f.step(); mon.observe(t, f.phi, f.theta)
    return mon.finalize()


def ev_bilatL(seed, p, T, mu):
    f = cc.BilateralField(64, p, seed=seed, mu=mu, phi_init=PHI_INIT, phi_jitter=JITTER)
    mon = ms.CollapseMonitor(THR(), log_every=25)
    sq = 0.0
    for t in range(T):
        f.step()
        sq += float(np.mean((f.L.phi - f.R.phi) ** 2))   # for RMS|phiL-phiR|
        mon.observe(t, f.L.phi, f.L.theta)
    rms_diff = float(np.sqrt(sq / T))
    return mon.finalize(), rms_diff


def ev_matched(seed, p, T, amp, kind, off=777):
    """Inject matched amplitude noise via the coupling path:
       step_with_coupling(phi+nz, theta, mu=1) => injected increment = dt*nz.
       theta_other=theta makes the phase coupling term vanish (amplitude-only)."""
    f = cc.SingleField(64, p, seed=seed, phi_init=PHI_INIT, phi_jitter=JITTER)
    rng = np.random.default_rng(seed + off)
    mon = ms.CollapseMonitor(THR(), log_every=25)
    for t in range(T):
        nz = (rng.standard_normal(f.phi.shape) if kind == "white"
              else cc.colored_noise(f.phi.shape, rng, 3)) * amp
        f.step_with_coupling(f.phi + nz, f.theta, mu=1.0)
        mon.observe(t, f.phi, f.theta)
    return mon.finalize()


def ev_sham_frozen(seed, p, T, mu, off=20000):
    L = cc.SingleField(64, p, seed=seed, phi_init=PHI_INIT, phi_jitter=JITTER)
    P = cc.SingleField(64, p, seed=seed + off, phi_init=PHI_INIT, phi_jitter=JITTER)
    pphi, pth = P.phi.copy(), P.theta.copy()
    mon = ms.CollapseMonitor(THR(), log_every=25)
    for t in range(T):
        L.step_with_coupling(pphi, pth, mu)
        mon.observe(t, L.phi, L.theta)
    return mon.finalize()


def ev_sham_indep(seed, p, T, mu, off=20000):
    L = cc.SingleField(64, p, seed=seed, phi_init=PHI_INIT, phi_jitter=JITTER)
    P = cc.SingleField(64, p, seed=seed + off, phi_init=PHI_INIT, phi_jitter=JITTER)
    mon = ms.CollapseMonitor(THR(), log_every=25)
    for t in range(T):
        pphi, pth = P.phi.copy(), P.theta.copy()
        L.step_with_coupling(pphi, pth, mu)   # L <- P
        P.step()                              # P free; no back-coupling
        mon.observe(t, L.phi, L.theta)
    return mon.finalize()


def ev_sham_mutual(seed, p, T, mu, off=20000):
    """Two lobes MUTUALLY coupled (snapshot-before, like BilateralField) but NOT
    mirror-initialized: P is an ordinary independent field (theta_P != -theta_L,
    omega_P != -omega_L). Isolates the *mirror init* from the *mutual coupling*."""
    L = cc.SingleField(64, p, seed=seed, phi_init=PHI_INIT, phi_jitter=JITTER)
    P = cc.SingleField(64, p, seed=seed + off, phi_init=PHI_INIT, phi_jitter=JITTER)
    mon = ms.CollapseMonitor(THR(), log_every=25)
    for t in range(T):
        lp, lt = L.phi.copy(), L.theta.copy()
        pp, pt = P.phi.copy(), P.theta.copy()
        L.step_with_coupling(pp, pt, mu)
        P.step_with_coupling(lp, lt, mu)
        mon.observe(t, L.phi, L.theta)
    return mon.finalize()


def ev_kick(seed, p_base, T, mode):
    """Hysteresis probe. Evolve normally; during [KICK_AT, KICK_AT+KICK_LEN) apply a
    standardized transient, then RESTORE and continue. Records the post-kick tail
    temporal variance (steps 600-800) and the natural-vs-post outcome.
    mode='lock'  -> zero novelty during the kick (push toward standing-wave lock).
    mode='alive' -> 5x novelty during the kick (push toward incoherent/alive)."""
    p = copy.deepcopy(p_base)
    f = cc.SingleField(64, p, seed=seed, phi_init=PHI_INIT, phi_jitter=JITTER)
    mon = ms.CollapseMonitor(THR(), log_every=25)
    sig_n0, sig_p0 = p.sigma_noise, p.sigma_phase
    for t in range(T):
        if t == KICK_AT:
            if mode == "lock":
                p.sigma_noise = 0.0; p.sigma_phase = 0.0
            else:
                p.sigma_noise = sig_n0 * 5.0; p.sigma_phase = sig_p0 * 5.0
        if t == KICK_AT + KICK_LEN:
            p.sigma_noise = sig_n0; p.sigma_phase = sig_p0
        f.step(); mon.observe(t, f.phi, f.theta)
    r = mon.finalize()
    rec = _rec(seed, r)
    # post-kick tail temporal variance (final window ends at T, well after release)
    rec["post_outcome"] = r.outcome
    rec["post_tvar"] = float(r.temporal_var_final)
    return r, rec


# --------------------------------------------------------------------------- #
#  arm runner (resumable)
# --------------------------------------------------------------------------- #
def run_arm(arm, args, N, T):
    if arm == "single":
        sigma = float(args[0]); key = f"single_{sigma}"
        p = regime(sigma_noise=sigma, sigma_phase=sigma)
        fn = lambda s: ev_single(s, p, T)
    elif arm == "bilatL":
        mu, sigma = float(args[0]), float(args[1]); key = f"bilatL_{mu}_{sigma}"
        p = regime(sigma_noise=sigma, sigma_phase=sigma)
        fn = lambda s: ev_bilatL(s, p, T, mu)
    elif arm in ("white", "colored"):
        mu, sigma = float(args[0]), float(args[1]); key = f"{arm}_{mu}_{sigma}"
        p = regime(sigma_noise=sigma, sigma_phase=sigma)
        # amp = mu * RMS|phiL-phiR| measured in the true bilateral arm
        bkey = f"bilatL_{mu}_{sigma}"
        bst = _load(bkey)
        amp = bst.get("meta", {}).get("amp_for_match")
        if amp is None:
            raise SystemExit(f"need {bkey}.json (with measured RMS diff) first; run bilatL arm")
        fn = lambda s: ev_matched(s, p, T, amp, arm)
    elif arm == "sham_frozen":
        mu, sigma = float(args[0]), float(args[1]); key = f"sham_frozen_{mu}_{sigma}"
        p = regime(sigma_noise=sigma, sigma_phase=sigma)
        fn = lambda s: ev_sham_frozen(s, p, T, mu)
    elif arm == "sham_indep":
        mu, sigma = float(args[0]), float(args[1]); key = f"sham_indep_{mu}_{sigma}"
        p = regime(sigma_noise=sigma, sigma_phase=sigma)
        fn = lambda s: ev_sham_indep(s, p, T, mu)
    elif arm == "sham_mutual":
        mu, sigma = float(args[0]), float(args[1]); key = f"sham_mutual_{mu}_{sigma}"
        p = regime(sigma_noise=sigma, sigma_phase=sigma)
        fn = lambda s: ev_sham_mutual(s, p, T, mu)
    elif arm in ("kick_lock", "kick_alive"):
        sigma = float(args[0]); key = f"{arm}_{sigma}"
        p = regime(sigma_noise=sigma, sigma_phase=sigma)
        mode = "lock" if arm == "kick_lock" else "alive"
        fn = lambda s: ev_kick(s, p, T, mode)
    else:
        raise SystemExit(f"unknown arm {arm}")

    st = _load(key)
    recs = st["records"]
    done_seeds = {r["seed"] for r in recs}
    rms_accum = st.get("meta", {}).get("_rms_list", [])
    for s in range(N):
        if s in done_seeds:
            continue
        out = fn(s)
        if arm == "bilatL":
            r, rms = out
            rms_accum.append(rms)
            recs.append(_rec(s, r))
        elif arm in ("kick_lock", "kick_alive"):
            r, rec = out
            recs.append(rec)
        else:
            recs.append(_rec(s, out))
        if (s + 1) % CKPT_EVERY == 0:
            st["records"] = recs
            if arm == "bilatL":
                st.setdefault("meta", {})["_rms_list"] = rms_accum
            _save(key, st)
            print(f"[{key}] {len(recs)}/{N}", flush=True)
    st["records"] = recs
    st.setdefault("meta", {})
    st["meta"]["N"] = N; st["meta"]["T"] = T; st["meta"]["arm"] = arm; st["meta"]["args"] = args
    if arm == "bilatL":
        mean_rms = float(np.mean(rms_accum)) if rms_accum else None
        st["meta"]["mean_rms_diff"] = mean_rms
        st["meta"]["amp_for_match"] = float(float(args[0]) * mean_rms) if mean_rms else None
        st["meta"]["_rms_list"] = rms_accum
    st["meta"]["done"] = True
    _save(key, st)
    frac = float(np.mean([r["collapsed"] for r in recs]))
    print(f"[{key}] DONE collapse={frac:.0%} n={len(recs)}"
          + (f" mean_rms_diff={st['meta'].get('mean_rms_diff'):.4f} "
             f"amp={st['meta'].get('amp_for_match'):.5f}" if arm == "bilatL" else ""),
          flush=True)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    arm = sys.argv[1]
    # optional --N / --T at the end
    rest = sys.argv[2:]
    N, T = N_DEFAULT, T_DEFAULT
    pos = []
    i = 0
    while i < len(rest):
        if rest[i] == "--N":
            N = int(rest[i + 1]); i += 2
        elif rest[i] == "--T":
            T = int(rest[i + 1]); i += 2
        else:
            pos.append(rest[i]); i += 1
    run_arm(arm, pos, N, T)


if __name__ == "__main__":
    main()
