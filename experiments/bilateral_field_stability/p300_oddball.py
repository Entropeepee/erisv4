"""
p300_oddball.py -- P300 Level-1 (structural oddball), corrected per R1 (C1-C3).

A structural oddball: brief field perturbations injected into a RECEIVER lobe at a
controlled coupling angle Delta (0=standard, 45=meaningful deviant, 90=surprise), via the
egate membrane. NO semantics (that is Level 2, gated on kappa-wiring).

C1: the integrative response is the receiver's DISSIPATIVE re-solving, measured as the peak
    rate-of-change of the receiver's global phase-coherence order parameter over the post-
    stimulus window. NOT membrane transport, NOT conserved-amplitude redistribution.
C2: the TRANSPORT-ONLY matched-mass control is the primary discriminator (P4): the 45° gated
    input transports mass by construction, so "restructuring peaks at 45°" alone proves little.
    The load-bearing test is whether the real (phase-structured) 45° input restructures the
    receiver MORE than the same mass injected with the phase drive removed (gate_phase=False).
C3: four compute-matched arms separate "divided" from "has an E-gate":
    A = divided + egate, B = divided + diff, C = monolith (no partner). (D optional.)

Arms x streams x seeds. Threshold-free: raw per-trial restructuring per class. Honest null OK.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bifurcate as bf
from field_core import PDEParams, colored_noise

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUTDIR = os.path.join(RES, "p300")
os.makedirs(OUTDIR, exist_ok=True)
SIG = 0.016
W_STIM, W_POST, W_ISI, W_BASE = 12, 48, 36, 12
MU_PARTNER, MU_STIM = 0.9, 1.6
SEED_OFFSET = 20000


def regime(sigma=SIG):
    p = PDEParams(); p.r_sat = 0.85; p.d_decay = 0.20
    p.sigma_noise = sigma; p.sigma_phase = sigma; p.omega_spread = 0.25
    return p


def coherence(theta):
    return float(np.abs(np.mean(np.exp(1j * theta))))


class P300Rig:
    """Receiver lobe + optional partner; stimulus injection at controlled angle."""
    def __init__(self, arm, seed):
        p = regime()
        self.arm = arm
        self.R = bf.cc.SingleField(64, p, seed=seed, phi_init=0.85, phi_jitter=0.04)
        if arm in ("A_egate", "B_diff"):
            self.P = bf.cc.SingleField(64, p, seed=seed + SEED_OFFSET, phi_init=0.85, phi_jitter=0.04)
            self.P.omega0 = self.P.omega0 - 0.05   # mild detuning (distinct partner)
            self.R.omega0 = self.R.omega0 + 0.05
            self.partner_kind = "egate" if arm == "A_egate" else "diff"
        else:
            self.P = None
        # fixed structured stimulus (a "shape" to inject): amplitude pattern + phase TEXTURE.
        # The texture makes the per-cell relatedness vary around the controlled mean angle Delta,
        # so the gated kick is spatially heterogeneous -- a uniform offset would leave the global
        # coherence unchanged and produce no restructuring signal.
        rng = np.random.default_rng(seed + 777)
        self.stim_phi = np.clip(0.6 + 0.2 * colored_noise((64, 64), rng, 3), 0.05, 0.95)
        self.stim_texture = 0.45 * colored_noise((64, 64), np.random.default_rng(seed + 888), 2)

    def free_step(self):
        """No stimulus; the persistent partner membrane (the corpus callosum) couples R<->P
        for divided arms (egate for A, diff for B); monolith just steps."""
        if self.P is not None:
            rp, rt = self.R.phi.copy(), self.R.theta.copy()
            pp, pt = self.P.phi.copy(), self.P.theta.copy()
            self.R.step_with_coupling(pp, pt, MU_PARTNER, coupling_kind=self.partner_kind, gate_phase=True)
            self.P.step_with_coupling(rp, rt, MU_PARTNER, coupling_kind=self.partner_kind, gate_phase=True)
        else:
            self.R.step()

    def stim_step(self, stim_theta, mu_stim, gate_phase=True):
        """Inject the stimulus as an egate-gated membrane PRE-KICK on the receiver, then do the
        normal arm step (partner + PDE re-solving). gate_phase=False = transport-only (matched
        amplitude mass, NO phase drive) -- the C2 primary control. Returns the injected mass."""
        from field_core import coupling_gate, wrap_diff
        d = wrap_diff(stim_theta, self.R.theta)
        cg = coupling_gate(d, "egate")
        dt = self.R.p.dt
        amp_kick = dt * mu_stim * cg * (self.stim_phi - self.R.phi)
        self.R.phi = np.clip(self.R.phi + amp_kick, 0.0, self.R.p.B_max - 1e-4)
        if gate_phase:
            self.R.theta = (self.R.theta + dt * mu_stim * cg * d) % (2 * np.pi)
        self.free_step()
        return float(np.mean(np.abs(amp_kick)))

    def receiver_coherence(self):
        return coherence(self.R.theta)


def run_trial(rig, delta_deg, mu_stim, gate_phase=True):
    """Present one stimulus at angle delta; return the restructuring observable:
    peak |dC/dt| of receiver coherence over the post-stimulus window, minus the pre-stim
    baseline rate. Also returns peak |C - baseline| excursion."""
    base = [rig.receiver_coherence()]
    for _ in range(W_BASE):
        rig.free_step(); base.append(rig.receiver_coherence())
    base_C = float(np.mean(base[-W_BASE:]))
    base_rate = float(np.mean(np.abs(np.diff(base[-W_BASE:]))))
    # structured stimulus at controlled MEAN relatedness Delta: a fixed phase texture rotated so
    # circular-mean(wrap(stim_theta - R.theta)) = Delta. Per-cell relatedness varies about Delta.
    base_tex = rig.R.theta + rig.stim_texture
    cur_mean_rel = np.angle(np.mean(np.exp(1j * (base_tex - rig.R.theta))))
    offset = np.radians(delta_deg) - cur_mean_rel
    stim_theta = (base_tex + offset) % (2 * np.pi)
    C = [rig.receiver_coherence()]
    mass = 0.0
    for t in range(W_STIM):
        mass += rig.stim_step(stim_theta, mu_stim, gate_phase=gate_phase)
        C.append(rig.receiver_coherence())
    for t in range(W_POST):
        rig.free_step(); C.append(rig.receiver_coherence())
    C = np.array(C)
    rates = np.abs(np.diff(C))
    peak_rate = float(np.max(rates) - base_rate)
    peak_excursion = float(np.max(np.abs(C - base_C)))
    return {"delta": delta_deg, "peak_rate": peak_rate, "peak_excursion": peak_excursion,
            "mass": float(mass)}


def make_sequence(stream, n_trials, seed):
    rng = np.random.default_rng(seed + 12345)
    if stream == "oddball":
        choices, probs = [0.0, 45.0, 90.0], [0.8, 0.1, 0.1]
    else:  # equiprobable: 45 and 90 at 10% each, 8 fillers at 10% each
        fillers = [15.0, 25.0, 35.0, 55.0, 65.0, 75.0, 110.0, 130.0]
        choices = [45.0, 90.0] + fillers
        probs = [0.1, 0.1] + [0.8 / len(fillers)] * len(fillers)
    return list(rng.choice(choices, size=n_trials, p=probs))


def run_arm(arm, stream, seed, n_trials, mu_stim=MU_STIM, gate_phase=True, tag=""):
    rig = P300Rig(arm, seed)
    for _ in range(300):   # settle into the alive regime
        rig.free_step()
    seq = make_sequence(stream, n_trials, seed)
    recs = []
    for k, d in enumerate(seq):
        r = run_trial(rig, d, mu_stim, gate_phase=gate_phase)
        r["k"] = k; recs.append(r)
        for _ in range(W_ISI):
            rig.free_step()
    return recs


def aggregate(recs, classes=(0.0, 45.0, 90.0), metric="peak_excursion"):
    out = {}
    for c in classes:
        sub = [r[metric] for r in recs if abs(r["delta"] - c) < 1e-6]
        if sub:
            out[str(c)] = {"mean": float(np.mean(sub)), "sem": float(np.std(sub) / max(np.sqrt(len(sub)), 1)),
                           "n": len(sub), "mass": float(np.mean([r["mass"] for r in recs if abs(r["delta"]-c)<1e-6]))}
    return out


def main():
    a = sys.argv
    cmd = a[1] if len(a) > 1 else "run"
    def opt(n, d, c=float):
        return c(a[a.index(n) + 1]) if n in a else d
    seeds = int(opt("--seeds", 6, int)); ntr = int(opt("--ntrials", 200, int))
    if cmd == "run":
        arm = a[2]; stream = a[3]; gp = "--transport_only" not in a
        key = f"{arm}_{stream}" + ("_transportonly" if not gp else "")
        path = os.path.join(OUTDIR, key + ".json")
        st = json.load(open(path)) if os.path.exists(path) else {"key": key, "by_seed": {}}
        for s in range(seeds):
            if str(s) in st["by_seed"]:
                continue
            recs = run_arm(arm, stream, s, ntr, gate_phase=gp)
            st["by_seed"][str(s)] = recs
            json.dump(st, open(path, "w"))
            print(f"[{key}] seed {s} done ({len(recs)} trials)", flush=True)
        # pooled aggregate
        allr = [r for s in st["by_seed"].values() for r in s]
        agg = aggregate(allr, classes=(0.0, 45.0, 90.0) if stream == "oddball" else (45.0, 90.0))
        st["aggregate"] = agg; json.dump(st, open(path, "w"))
        print(f"[{key}] AGG " + " ".join(f"{int(float(c))}deg:{v['mean']:.4f}+-{v['sem']:.4f}(n{v['n']})"
                                          for c, v in agg.items()), flush=True)
    elif cmd == "ampsweep":
        # P3: sweep the 45deg stimulus amplitude (mu_stim); fit sigmoid vs line
        arm = "A_egate"
        res = {}
        for mu in [0.3, 0.6, 0.9, 1.2, 1.6, 2.0, 2.6]:
            vals = []
            for s in range(seeds):
                rig = P300Rig(arm, s)
                for _ in range(300):
                    rig.free_step()
                for _ in range(8):
                    r = run_trial(rig, 45.0, mu)
                    vals.append(r["peak_rate"])
                    for _ in range(W_ISI):
                        rig.free_step()
            res[mu] = {"mean": float(np.mean(vals)), "sem": float(np.std(vals) / np.sqrt(len(vals)))}
            print(f"  mu={mu}: {res[mu]['mean']:.4f}", flush=True)
        json.dump(res, open(os.path.join(OUTDIR, "ampsweep.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
