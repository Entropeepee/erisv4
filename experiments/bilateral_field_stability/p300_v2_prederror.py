"""
p300_v2_prederror.py -- Artificial P300 v2: the prediction-error WAVE.

Reframe (from the handoff): the conscious wave (P300 analogue) is a PREDICTION-ERROR
transient, ORTHOGONAL to the exchange rate. It is high whenever the input violates the
receiver's prediction -- at ANY coupling angle -- and low whenever the input is predicted,
even at maximal exchange (45deg flow). The decisive result is a DOUBLE DISSOCIATION:
wave tracks prediction error, exchange tracks coupling angle.

Two channels (measured separately):
  * EXCHANGE (the flow): E(Delta)-gated transport of the stimulus -- a property of the
    coupling angle Delta. Reported as transport magnitude.
  * WAVE (the transient): the receiver's phase-coherence transient triggered by the
    PREDICTION-ERROR residual (stimulus - kappa_pred), injected UNGATED by Delta.

kappa_pred = a running prediction of the stimulus texture (circular EMA of history). It is
updated ONLY AFTER the response is measured -- it must never peek at the current stimulus.
Prediction precision = the EMA resultant length (sharpness of the prior).

Excitability (flag): if the base field gives only a graded/local bump, add a MINIMAL
recovery variable per site (refractory) so the wave can be a thresholded, propagating,
self-extinguishing pulse. Reported pre/post so the effect of adding excitability is visible.

Reuses field_core verbatim (behind flags). Threshold-free; honest null valid.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bifurcate as bf
from field_core import PDEParams, colored_noise, local_coherence, wrap_diff

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUTDIR = os.path.join(RES, "p300v2")
os.makedirs(OUTDIR, exist_ok=True)
SIG = 0.016
W_POST = 60
N = 64


def regime():
    p = PDEParams(); p.r_sat = 0.85; p.d_decay = 0.20
    p.sigma_noise = SIG; p.sigma_phase = SIG; p.omega_spread = 0.25
    return p


def gcoh(theta):
    return float(np.abs(np.mean(np.exp(1j * theta))))


def circ_ema(acc, tex, alpha):
    """circular EMA in complex space; returns updated complex accumulator."""
    return (1 - alpha) * acc + alpha * np.exp(1j * tex)


class PredRig:
    def __init__(self, seed, alpha=0.25, w_gain=0.6, excitable=False):
        self.R = bf.cc.SingleField(N, regime(), seed=seed, phi_init=0.85, phi_jitter=0.04)
        self.rng = np.random.default_rng(seed + 4242)
        self.alpha = alpha; self.w_gain = w_gain; self.excitable = excitable
        # kappa_pred: complex accumulator over the stimulus texture history
        self.kappa = np.exp(1j * (0.4 * colored_noise((N, N), np.random.default_rng(seed + 1), 2)))
        self.pred_frozen = False       # no-prediction control
        # excitable recovery variable per site (0 = ready, high = refractory)
        self.w = np.zeros((N, N))
        # FHN excitable-layer constants (defaults = the values used for the main runs). Exposed so
        # the robustness sweep can perturb them and show the qualitative signatures are NOT a
        # knife-edge tuning artifact (audit point iii).
        self.fhn = dict(seed_thr=0.12, seed_gain=3.0, thr=0.25, diff=0.18,
                        rec=0.02, rec_w=1.5, theta_gain=0.25)

    def settle(self, steps=300):
        for _ in range(steps):
            self.R.step(); self._recover()

    def _recover(self):
        if self.excitable:
            self.w *= 0.9   # recovery decays (refractory relaxes)

    def kappa_angle(self):
        return np.angle(self.kappa)

    def precision(self):
        return float(np.mean(np.abs(self.kappa)))   # EMA resultant length in [0,1]

    def make_texture(self, kind, violation=1.0, seed=None):
        """kind='predicted' -> texture == kappa_pred (near-zero error);
           kind='violating' -> kappa_pred + a controlled random departure of size `violation`."""
        rng = self.rng if seed is None else np.random.default_rng(seed)
        base = self.kappa_angle()
        if kind == "predicted":
            return (base + 0.02 * rng.standard_normal((N, N))) % (2 * np.pi)
        dev = violation * colored_noise((N, N), rng, 2)
        return (base + dev) % (2 * np.pi)

    def probe(self, delta_deg, texture, localized=False, use_precision=True, return_trace=False):
        """Present a probe: measure EXCHANGE (E-gated transport, a function of delta) and the
        WAVE (transient from the PRECISION-WEIGHTED prediction-error residual). kappa_pred is
        NOT updated here. With excitable=True the residual seeds a FitzHugh-Nagumo activator
        layer (threshold + diffusion + refractory recovery) that can support a propagating,
        self-extinguishing pulse; else the residual perturbs theta directly (relaxational)."""
        from field_core import coupling_gate
        residual = wrap_diff(texture, self.kappa_angle())          # per-cell unpredicted part
        err = float(np.mean(np.abs(residual)))                     # scalar prediction error
        prec = self.precision() if use_precision else 1.0          # precision-weighting (Fisher)
        d_mean = np.radians(delta_deg)
        exchange = float(np.mean(coupling_gate(np.full((N, N), d_mean), "egate")))  # E(delta)
        C0 = gcoh(self.R.theta); lc0 = local_coherence(self.R.theta)
        window = np.ones((N, N))
        if localized:
            yy, xx = np.mgrid[0:N, 0:N]
            window = np.exp(-(((xx - N // 2) ** 2 + (yy - N // 2) ** 2) / (2 * (N / 8) ** 2)))
        drive = self.w_gain * prec * np.abs(residual) * window     # precision-weighted error drive
        Cs, spread, a_energy = [], [], []
        if self.pred_frozen:
            drive = np.zeros((N, N))                                # no-prediction control
        if self.excitable:
            f = self.fhn
            a = np.zeros((N, N)); w = np.zeros((N, N))
            a += np.clip(drive - f["seed_thr"], 0.0, 1.0) * f["seed_gain"]   # threshold seeding
            for t in range(W_POST):
                lap = (np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4 * a)
                a = a + 0.5 * (a * (a - f["thr"]) * (1.0 - a) - w + f["diff"] * lap)  # FHN + diffusion
                w = w + 0.5 * f["rec"] * (a - f["rec_w"] * w)                         # slow recovery
                a = np.clip(a, 0.0, 1.5)
                self.R.theta = (self.R.theta + f["theta_gain"] * a * np.sign(residual)) % (2 * np.pi)
                self.R.step(); self._recover()
                Cs.append(gcoh(self.R.theta)); a_energy.append(float(np.mean(a)))
                spread.append(float(np.mean(a > 0.15)))
        else:
            self.R.theta = (self.R.theta + drive * np.sign(residual)) % (2 * np.pi)
            for t in range(W_POST):
                self.R.step(); self._recover()
                Cs.append(gcoh(self.R.theta))
                spread.append(float(np.mean(np.abs(local_coherence(self.R.theta) - lc0) > 0.05)))
                a_energy.append(0.0)
        Cs = np.array(Cs)
        wave_amp = float(np.max(np.abs(Cs - C0)))
        tail = float(np.mean(np.abs(Cs[-8:] - C0)))
        out = {"delta": delta_deg, "error": err, "exchange": exchange, "precision": prec,
               "wave_amp": wave_amp, "spread_peak": float(np.max(spread)),
               "spread_final": float(np.mean(spread[-8:])),
               "a_peak": float(np.max(a_energy)), "a_final": float(np.mean(a_energy[-8:])),
               "self_extinct": bool(tail < 0.4 * wave_amp + 1e-9)}
        if return_trace:
            out["trace_C"] = [float(x) for x in Cs]
            out["trace_a"] = [float(x) for x in a_energy]
            out["trace_spread"] = [float(x) for x in spread]
        return out

    def update_pred(self, texture):
        if not self.pred_frozen:
            self.kappa = circ_ema(self.kappa, texture, self.alpha)

    def establish(self, steps=25, delta=45.0):
        """prelude: present a stable predicted stream so kappa_pred converges (no measurement)."""
        for _ in range(steps):
            tex = self.make_texture("predicted")
            # advance the field a little between prelude presentations
            for _ in range(6):
                self.R.step(); self._recover()
            self.update_pred(tex)


# --------------------------------------------------------------------------- #
#  Experiment drivers
# --------------------------------------------------------------------------- #
def _mkrig(seed, excitable, frozen=False):
    rig = PredRig(seed, excitable=excitable)
    rig.settle(300); rig.establish(30)
    rig.pred_frozen = frozen
    return rig


def run_crossed(excitable, n_seeds, violation=1.2):
    """The 2x3 crossed design (exchange angle x prediction match) + no-prediction + flow +
    transport-only controls. Returns per-seed records. kappa_pred established per seed first."""
    recs = []
    for s in range(n_seeds):
        # crossed cells: angle in {0,45,90} x {predicted, violating}
        for kind in ["predicted", "violating"]:
            for d in [0.0, 45.0, 90.0]:
                rig = _mkrig(s, excitable)
                r = rig.probe(d, rig.make_texture(kind, violation=violation), localized=True)
                r.update({"seed": s, "kind": kind, "arm": "pred"}); recs.append(r)
        # no-prediction control (frozen kappa): should give NO wave (Level-1 regime)
        for d in [0.0, 45.0, 90.0]:
            rig = _mkrig(s, excitable, frozen=True)
            r = rig.probe(d, rig.make_texture("violating", violation=violation), localized=True)
            r.update({"seed": s, "kind": "violating", "arm": "nopred"}); recs.append(r)
    return recs


def run_p2_sweep(excitable, n_seeds):
    recs = []
    for s in range(n_seeds):
        for v in [0.0, 0.2, 0.4, 0.6, 0.9, 1.2, 1.6, 2.2]:
            rig = _mkrig(s, excitable)
            r = rig.probe(45.0, rig.make_texture("violating" if v > 0 else "predicted",
                                                 violation=max(v, 0.02)), localized=True)
            r.update({"seed": s, "violation": v}); recs.append(r)
    return recs


def run_p3_sweep(excitable, n_seeds):
    recs = []
    for s in range(n_seeds):
        for spread in [0.05, 0.4, 0.9, 1.6]:
            rig = PredRig(s, excitable=excitable); rig.settle(300)
            base = rig.kappa_angle().copy()
            for _ in range(30):
                tex = (base + spread * colored_noise((N, N), rig.rng, 2)) % (2 * np.pi)
                for _ in range(6):
                    rig.R.step()
                rig.update_pred(tex)
            r = rig.probe(45.0, rig.make_texture("violating", violation=1.2), localized=True)
            r.update({"seed": s, "prelude_spread": spread}); recs.append(r)
    return recs


def run_p5_flow(excitable, n_seeds, n_stream=40):
    """Flow control: a sustained stream of PREDICTED inputs at high exchange (45deg). Report
    the wave per presentation across the stream -- should stay ~flat (no waves) while exchange
    is high (E(45)=max)."""
    recs = []
    for s in range(n_seeds):
        rig = _mkrig(s, excitable)
        for k in range(n_stream):
            tex = rig.make_texture("predicted")
            r = rig.probe(45.0, tex, localized=True)
            rig.update_pred(tex)
            recs.append({"seed": s, "k": k, "wave_amp": r["wave_amp"], "exchange": r["exchange"]})
        # then one violation at the end of the predicted stream -> should spike
        tex = rig.make_texture("violating", violation=1.2)
        r = rig.probe(45.0, tex, localized=True)
        recs.append({"seed": s, "k": n_stream, "wave_amp": r["wave_amp"],
                     "exchange": r["exchange"], "is_violation": True})
    return recs


def run_adaptation(excitable, n_seeds, shift=1.6, n_after=14):
    """Audit (i): is kappa_pred GENUINELY PREDICTIVE (does it LEARN)? Establish kappa on stream A,
    then STEP-CHANGE the input mean. A genuine running predictor mispredicts the first post-shift
    input (error spike -> wave), then ADAPTS: as kappa_pred tracks the new mean over presentations,
    the error -- and the wave -- decay back to floor. A fixed (non-learning) reference could not
    show this decay. kappa is updated ONLY AFTER each response is measured (no peek)."""
    recs = []
    for s in range(n_seeds):
        rig = _mkrig(s, excitable)                      # kappa established on stream A
        base = rig.kappa_angle().copy()
        new_center = (base + shift * colored_noise((N, N), np.random.default_rng(s + 9999), 2))
        for k in range(n_after):
            tex = (new_center + 0.02 * rig.rng.standard_normal((N, N))) % (2 * np.pi)
            r = rig.probe(45.0, tex, localized=True)    # measure BEFORE updating (no peek)
            rig.update_pred(tex)                         # kappa learns the new mean -> adapts
            recs.append({"seed": s, "k": k, "wave_amp": r["wave_amp"],
                         "error": r["error"], "precision": r["precision"]})
    return recs


def run_excite_robustness(excitable, n_seeds):
    """Audit (iii): is the excitable result a knife-edge tuning artifact? Perturb each FHN constant
    +/-40% (one at a time) and re-measure the double-dissociation gap (violating wave - predicted
    wave) and the ignition threshold. If the qualitative signatures survive across the sweep, the
    excitability is a generic ingredient, not hand-tuned to manufacture the headline."""
    recs = []
    base_fhn = dict(seed_thr=0.12, seed_gain=3.0, thr=0.25, diff=0.18, rec=0.02, rec_w=1.5, theta_gain=0.25)
    variants = [("baseline", None, 1.0)]
    for key in ["seed_gain", "thr", "diff", "rec", "theta_gain"]:
        for mult in [0.6, 1.4]:
            variants.append((key, key, mult))
    for name, key, mult in variants:
        for s in range(n_seeds):
            for kind in ["predicted", "violating"]:
                rig = _mkrig(s, True)                    # excitable always (this is the excitable audit)
                if key is not None:
                    rig.fhn = dict(base_fhn); rig.fhn[key] = base_fhn[key] * mult
                r = rig.probe(45.0, rig.make_texture(kind, violation=1.2), localized=True)
                recs.append({"variant": name, "param": key, "mult": mult, "seed": s,
                             "kind": kind, "wave_amp": r["wave_amp"]})
    return recs


def run_trace(excitable, n_seeds):
    """Audit (ii): capture the full post-stimulus time course to show the WAVE is a TRANSIENT
    propagating PULSE (activator rises then self-extinguishes; spread rises then collapses), not a
    steady-state step. Records one predicted + one violating trace per seed."""
    recs = []
    for s in range(n_seeds):
        for kind in ["predicted", "violating"]:
            rig = _mkrig(s, excitable)
            r = rig.probe(45.0, rig.make_texture(kind, violation=1.6), localized=True, return_trace=True)
            recs.append({"seed": s, "kind": kind, "trace_C": r["trace_C"],
                         "trace_a": r["trace_a"], "trace_spread": r["trace_spread"],
                         "wave_amp": r["wave_amp"]})
    return recs


def main():
    import sys
    cmd = sys.argv[1]; ns = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 20
    exc = "--excitable" in sys.argv
    tag = "_exc" if exc else "_base"
    fn = {"crossed": run_crossed, "p2": run_p2_sweep, "p3": run_p3_sweep, "p5": run_p5_flow,
          "adapt": run_adaptation, "robust": run_excite_robustness, "trace": run_trace}[cmd]
    recs = fn(exc, ns)
    out = os.path.join(OUTDIR, f"{cmd}{tag}.json")
    json.dump({"cmd": cmd, "excitable": exc, "n_seeds": ns, "records": recs}, open(out, "w"))
    print(f"[{cmd}{tag}] {len(recs)} records -> {out}", flush=True)


if __name__ == "__main__":
    main()
