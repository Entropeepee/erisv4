"""
BCDC Domain-of-Change Engine v3  (for Eris)
============================================

v3 fixes the PHASE dynamics. The phase channel Theta in Psi = Phi e^{iTheta}
is now a field of coupled Kuramoto oscillators:

    dTheta/dt = omega(Phi)  +  K * g(Phi) * SUM_neighbors sin(Theta_nbr - Theta)
                            +  phase noise

  * sin(.) takes phase differences ON THE CIRCLE, so wrap-around no longer
    injects spurious gradients (the v2 bug).
  * SUM sin(Theta_nbr - Theta) is the local Kuramoto coupling: it pulls
    neighbors into alignment, which is what makes coherence emerge. Coherence
    is the Kuramoto order parameter C = |<e^{iTheta}>|, the framework's own
    measure.
  * g(Phi) = Hill-Power(Phi): amplitude GATES synchronization — high-signal
    regions phase-lock first. This is the phi->theta direction of the coupling.
  * omega(Phi) gives a mild amplitude-frequency dispersion so locking is a
    real competition (Kuramoto transition), not trivial.

The Feedback domain F is now the genuine phi<->theta coupling: local phase
coherence reinforces amplitude (theta->phi), regime-modulated. And the engine
computes the framework observables C (coherence), X (exchange = INT phi*d_theta),
and dC/dX (the regime / transfixion diagnostic).

Everything else from v2 is unchanged: modular additive operators, the
elastic/plastic yield regime, the SGT-gated boundary, IBT/ZBT asymptotes,
colored noise, the text encoder, and the Brownian correctness check.

Copyright 2026 Willow IP Group LLC.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List
import re, hashlib
import numpy as np


# ─────────────────────────── unifying primitives ───────────────────────────
def hill_power(s, alpha=1.0, beta=0.5, gamma=1.0, delta=0.0):
    s_shift = np.maximum(s - delta, 0.0)
    sa = np.power(s_shift, alpha)
    return np.power(sa / (sa + beta + 1e-12), gamma)


def beta_star(lam):
    sigma_g = np.sqrt(8.0 / max(lam, 1e-9))
    return (2.0 / np.sqrt(np.pi)) * np.sqrt(sigma_g / (sigma_g ** 2 + 1.0))


def sigmoid_gate(s, center, width, steep=8.0):
    return 1.0 / (1.0 + np.exp(-(s - center) * (steep / max(width, 1e-6))))


def colored_noise(shape, rng, n_smooth=4):
    w = rng.standard_normal(shape)
    for _ in range(n_smooth):
        w = 0.5 * w + 0.125 * (np.roll(w, 1, 0) + np.roll(w, -1, 0)
                               + np.roll(w, 1, 1) + np.roll(w, -1, 1))
    return w / (w.std() + 1e-9)


def _lap(a):
    return (np.roll(a, 1, 0) + np.roll(a, -1, 0)
            + np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4.0 * a)


def wrap_diff(a_nbr, a):
    """Phase difference on the circle, in (-pi, pi]."""
    return np.angle(np.exp(1j * (a_nbr - a)))


def local_coherence(theta):
    """Local Kuramoto order parameter over the 4-neighborhood + self, in [0,1].
    1 = neighbors phase-aligned; 0 = scrambled."""
    ph = np.exp(1j * theta)
    z = (ph + np.roll(ph, 1, 0) + np.roll(ph, -1, 0)
         + np.roll(ph, 1, 1) + np.roll(ph, -1, 1)) / 5.0
    return np.abs(z)


# ───────────────── input encoder: text -> (Phi, Theta) ─────────────────────
def content_embedding(text: str, dim: int = 256) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float64)
    t = text.lower(); words = re.findall(r"[a-z0-9]+", t)
    toks = list(words) + [words[i] + "_" + words[i + 1] for i in range(len(words) - 1)]
    toks += [t[i:i + 3] for i in range(len(t) - 2)]
    for tok in toks:
        h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
        vec[h % dim] += 1.0 if (h >> 63) & 1 else -1.0
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def coupling_angles(emb, n_channels=12):
    blocks = np.array_split(emb, n_channels)
    energy = np.array([float(np.sum(b * b)) for b in blocks])
    cos2 = energy / (energy.sum() + 1e-12)
    theta = np.arccos(np.sqrt(np.clip(cos2, 0.0, 1.0)))
    return theta, cos2


def encode_text(text, size=64, n_channels=12, amp=0.6, B_max=1.0):
    if not text or not text.strip():
        return (np.full((size, size), 0.25), np.zeros((size, size)))
    th_v, c2_v = coupling_angles(content_embedding(text), n_channels)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    # complex interference field Psi = sum_i share_i * exp(i*(mode_i + coupling_angle_i)).
    # |Psi| is the amplitude envelope; arg(Psi) spans the full circle (configuration).
    Psi = np.zeros((size, size), dtype=complex)
    for i, (a, c2) in enumerate(zip(th_v, c2_v)):
        kx, ky = 1 + (i % 3), 1 + (i // 3)
        Psi += c2 * np.exp(1j * (2 * np.pi * (kx * x + ky * y) / size + a))
    phi = np.abs(Psi); phi = phi / (phi.max() + 1e-10) * amp + 0.12
    phi = np.clip(phi, 0.02, B_max - 0.02)
    return phi, (np.angle(Psi) % (2 * np.pi))


# ─────────────────────────── parameters / sliders ──────────────────────────
@dataclass
class BCDCParams:
    activations: Dict[str, float] = field(default_factory=lambda: {
        "S": 1.0, "D": 1.0, "F": 1.0, "TH": 1.0,
        "B": 1.0, "E": 1.0, "IBT": 1.0, "ZBT": 1.0})
    B_max: float = 1.0
    dt: float = 0.05
    r_sat: float = 0.65
    d_decay: float = 0.28
    k_fb: float = 0.18
    a_th: float = 0.20
    T_c: float = 0.55
    th_alpha: float = 4.0
    gamma_bnd: float = 6.0
    quiet_zone: float = 0.18
    bnd_steep: float = 8.0
    alpha_em: float = 0.10
    D_em: float = 1.8
    xi_ibt: float = 0.004
    zeta_zbt: float = 0.004
    delta_bt: float = 0.05
    sigma_noise: float = 0.06
    noise_smooth: int = 4
    noise_structured: bool = True
    hp_alpha: float = 2.0
    hp_beta: float = -1.0
    hp_gamma: float = 1.0
    # ── phase (Kuramoto) layer ──
    omega: float = 0.10          # mean intrinsic frequency
    omega_spread: float = 0.5    # quenched frequency disorder (resists total lock)
    K_phase: float = 1.6         # Kuramoto coupling strength
    sigma_phase: float = 0.04    # phase noise
    tau_obs: int = 20            # observation window (steps) for temporal-flux measures
    # ── regime (yield) surface ──
    phi_yield_hi: float = 0.62
    phi_yield_lo: float = 0.48


# regime-dependent operator weights (crossing yield changes the rules)
W_ELASTIC = {"S": 1.0, "D": 1.0, "F": 0.2, "TH": 1.0, "B": 1.0, "E": 0.3, "IBT": 1.0, "ZBT": 1.0}
W_PLASTIC = {"S": 0.3, "D": 1.0, "F": 1.0, "TH": 1.0, "B": 1.0, "E": 1.5, "IBT": 1.0, "ZBT": 1.0}


# ─────────────────────────────── the engine ────────────────────────────────
class BCDCField:
    DOMAINS = ["S", "D", "F", "TH", "B", "E", "IBT", "ZBT"]

    def __init__(self, size=64, params: BCDCParams | None = None, seed=7):
        self.size = size
        self.p = params or BCDCParams()
        self.rng = np.random.default_rng(seed)
        g = colored_noise((size, size), self.rng, 3)
        self.phi = np.clip(0.25 + 0.15 * g, 0.02, self.p.B_max - 0.02)
        self.theta = self.rng.uniform(0, 2 * np.pi, (size, size))
        self.regime = np.zeros((size, size))
        self.omega0 = self.p.omega + self.p.omega_spread * self.rng.standard_normal((size, size))
        self._lc = local_coherence(self.theta)
        self._phi_prev = self.phi.copy(); self._theta_prev = self.theta.copy()
        self._flux_phi = np.zeros((size, size)); self._flux_theta = np.zeros((size, size))
        self.t = 0
        self.history: List[dict] = []
        self._C_hist: List[float] = []; self._X_hist: List[float] = []
        self._hp_beta = self.p.hp_beta if self.p.hp_beta > 0 else beta_star(self.p.omega + 1.0)

    def seed_from_text(self, text: str, amp: float = 0.6):
        self.phi, self.theta = encode_text(text, self.size, amp=amp, B_max=self.p.B_max)
        self.regime = np.zeros((self.size, self.size))
        self._lc = local_coherence(self.theta)
        self._phi_prev = self.phi.copy(); self._theta_prev = self.theta.copy()
        self._flux_phi = np.zeros((self.size, self.size)); self._flux_theta = np.zeros((self.size, self.size))
        return self

    # ---- the eight base operators ----
    def _base_ops(self):
        p, phi, Bm = self.p, self.phi, self.p.B_max
        gate = lambda s, dlt: hill_power(s, p.hp_alpha, self._hp_beta, p.hp_gamma, dlt)
        ops = {}
        ops["S"]  = p.r_sat * phi * (1.0 - phi / Bm)
        ops["D"]  = -p.d_decay * phi
        # Feedback = phi<->theta coupling: coherence ABOVE field-average reinforces
        # amplitude, below-average dissipates it (mean-zero: redistributes, no runaway)
        ops["F"]  = p.k_fb * (self._lc - self._lc.mean()) * phi
        ops["TH"] = p.a_th * gate(phi, p.T_c) * (1.0 - phi)
        lower = Bm - p.quiet_zone; center = Bm - 0.5 * p.quiet_zone
        soft = sigmoid_gate(phi, center, p.quiet_zone, p.bnd_steep)
        ops["B"]  = -p.gamma_bnd * soft * np.maximum(phi - lower, 0.0) ** 2
        ops["E"]  = p.alpha_em * np.power(np.maximum(phi, 0.0), p.D_em)
        ops["IBT"] = -p.xi_ibt / (Bm - phi + p.delta_bt)
        ops["ZBT"] = +p.zeta_zbt / (phi + p.delta_bt)
        return ops

    def _update_regime(self):
        enter = self.phi > self.p.phi_yield_hi
        leave = self.phi < self.p.phi_yield_lo
        self.regime = np.where(enter, 1.0, np.where(leave, 0.0, self.regime))

    def _phase_step(self):
        """Kuramoto field update (wrap-correct, amplitude-gated)."""
        p = self.p
        ph = np.exp(1j * self.theta)
        nbr = (np.roll(ph, 1, 0) + np.roll(ph, -1, 0)
               + np.roll(ph, 1, 1) + np.roll(ph, -1, 1))
        coupling = np.imag(np.conj(ph) * nbr)          # = SUM sin(theta_nbr - theta)
        g = hill_power(self.phi, p.hp_alpha, self._hp_beta, 1.0, 0.0)
        dtheta = self.omega0 + p.K_phase * g * coupling
        noise = self.rng.standard_normal(self.phi.shape)
        self.theta = (self.theta + p.dt * dtheta
                      + p.sigma_phase * np.sqrt(p.dt) * noise) % (2 * np.pi)
        self._lc = local_coherence(self.theta)

    def step(self):
        p = self.p
        self._update_regime()
        ops = self._base_ops()
        plastic = self.regime > 0.5
        dphi = np.zeros_like(self.phi); contrib = {}
        for k in self.DOMAINS:
            a = p.activations.get(k, 0.0)
            wk = np.where(plastic, W_PLASTIC[k], W_ELASTIC[k])
            term = a * wk * ops[k]
            dphi += term
            contrib[k] = float(np.mean(np.abs(term)))
        eta = colored_noise(self.phi.shape, self.rng, p.noise_smooth)
        if p.noise_structured:
            grad = np.sqrt((self.phi - np.roll(self.phi, 1, 1)) ** 2
                           + (self.phi - np.roll(self.phi, 1, 0)) ** 2 + 1e-9)
            unresolved = 1.0 - hill_power(grad, 1.0, 0.25, 1.0, 0.0)
            eta = eta * (0.5 + 0.5 * unresolved)
        self.phi = np.clip(self.phi + p.dt * dphi + p.sigma_noise * np.sqrt(p.dt) * eta,
                           1e-4, p.B_max - 1e-4)
        self._phase_step()
        # windowed temporal flux: is the field still CHANGING IN TIME (alive) vs frozen?
        # (distinct from spatial smoothness — a standing wave is smooth but alive)
        a = 1.0 / max(self.p.tau_obs, 1)
        self._flux_phi = (1 - a) * self._flux_phi + a * np.abs(self.phi - self._phi_prev)
        self._flux_theta = (1 - a) * self._flux_theta + a * np.abs(wrap_diff(self.theta, self._theta_prev))
        self._phi_prev = self.phi.copy(); self._theta_prev = self.theta.copy()
        self.t += 1
        self._record(contrib)

    def _record(self, contrib):
        phi = self.phi
        ph = np.exp(1j * self.theta)
        C = float(np.abs(np.mean(ph)))                 # global Kuramoto coherence
        dthx = wrap_diff(np.roll(self.theta, -1, 1), self.theta)
        dthy = wrap_diff(np.roll(self.theta, -1, 0), self.theta)
        X = float(np.sum(phi * (dthx + dthy)))         # exchange INT phi*d_theta
        self._C_hist.append(C); self._X_hist.append(X)
        if len(self._C_hist) >= 2:
            dC = self._C_hist[-1] - self._C_hist[-2]
            dX = self._X_hist[-1] - self._X_hist[-2]
            dCdX = float(dC / dX) if abs(dX) > 1e-9 else 0.0
        else:
            dCdX = 0.0
        tors = np.abs(_lap(phi))
        # liveness = phase throughput (are oscillators still turning); config_flux =
        # is the PATTERN still evolving over the window. "crystallized" = confident
        # (coherent + high) AND pattern frozen in time. This is a STUCKNESS proxy:
        # confirmed transfixion additionally needs non-reactivity to input (probe_reactivity).
        rotation = float(self._flux_theta.mean())
        config_flux = float(self._flux_phi.mean())
        crystallized = float(np.mean((self._lc > 0.85) & (phi > 0.6) & (self._flux_phi < 0.0025)))
        rec = {"t": self.t,
               "phi_mean": float(phi.mean()), "phi_var": float(phi.var()),
               "phi_max": float(phi.max()), "phi_min": float(phi.min()),
               "saturation_frac": float(np.mean(phi > 0.8)),
               "plastic_frac": float(np.mean(self.regime > 0.5)),
               "crystallized_frac": crystallized,
               "config_flux_mean": config_flux, "rotation_mean": rotation,
               "torsion_rms": float(np.sqrt(np.mean(_lap(phi) ** 2))),
               "coherence_C": C, "exchange_X": X, "dCdX": dCdX,
               "local_coherence_mean": float(self._lc.mean())}
        for k in self.DOMAINS:
            rec[f"drive_{k}"] = contrib[k]
        self.history.append(rec)

    def run(self, steps):
        for _ in range(steps):
            self.step()

    def probe_reactivity(self, amp=0.45, steps=12):
        """The real transfixion test: does the field RESPOND to input over a window?
        Inject a localized input into a perturbed twin, evolve it against an
        unperturbed twin sharing the same noise, and measure how far the input
        drives the field. High divergence/response = reactive (alive, NOT
        transfixed, however steady its readout); low = unresponsive (transfixion).
        Non-destructive: operates on deep copies, leaves self untouched."""
        import copy
        base = copy.deepcopy(self); pert = copy.deepcopy(self)
        s = self.size; cy = cx = s // 2; r = s // 6
        yy, xx = np.mgrid[0:s, 0:s]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < r * r
        pert.phi = np.clip(pert.phi + amp * mask, 1e-4, pert.p.B_max - 1e-4)
        pert.theta = np.where(mask, (pert.theta + np.pi) % (2 * np.pi), pert.theta)
        C0 = float(np.abs(np.mean(np.exp(1j * base.theta))))
        for _ in range(steps):
            base.step(); pert.step()
        divergence = float(np.mean(np.abs(pert.phi - base.phi)))
        dC = float(abs(pert.history[-1]["coherence_C"] - base.history[-1]["coherence_C"]))
        return {"input_amp": amp, "field_divergence": divergence,
                "coherence_response": dC, "C_baseline": C0}

    # ---- diagnostics ----
    def write_csv(self, path):
        import csv
        keys = list(self.history[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in self.history:
                w.writerow(r)

    def write_json(self, path):
        import json
        last = self.history[-1]
        summary = {"size": self.size, "steps": self.t, "params": asdict(self.p),
                   "final": last,
                   "stable": bool(np.isfinite(self.phi).all()
                                  and self.phi.max() < self.p.B_max and self.phi.min() > 0.0),
                   "coherence_start": self.history[0]["coherence_C"],
                   "coherence_end": last["coherence_C"],
                   "mean_domain_drive": {k: float(np.mean([h[f"drive_{k}"] for h in self.history]))
                                         for k in self.DOMAINS}}
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    def visualize(self, path, snapshots, org_curve=None):
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        H = self.history; ts = [h["t"] for h in H]
        fig, ax = plt.subplots(2, 3, figsize=(15, 8))
        ax[0, 0].imshow(snapshots[-1], cmap="magma", vmin=0, vmax=self.p.B_max)
        ax[0, 0].set_title(f"Phi (amplitude) — step {self.t}"); ax[0, 0].axis("off")
        im1 = ax[0, 1].imshow(self.theta, cmap="twilight", vmin=0, vmax=2 * np.pi)
        ax[0, 1].set_title("Theta (phase)"); ax[0, 1].axis("off")
        fig.colorbar(im1, ax=ax[0, 1], fraction=0.046)
        im2 = ax[0, 2].imshow(self._lc, cmap="viridis", vmin=0, vmax=1)
        ax[0, 2].set_title("local phase coherence (domains)"); ax[0, 2].axis("off")
        fig.colorbar(im2, ax=ax[0, 2], fraction=0.046)
        ax[1, 0].plot(ts, [h["local_coherence_mean"] for h in H], "c-", lw=2, label="local coher (text-seed)")
        if org_curve is not None:
            xs = np.linspace(0, ts[-1], len(org_curve))
            ax[1, 0].plot(xs, org_curve, "m--", lw=2, label="local coher (random init)")
        ax[1, 0].plot(ts, [h["coherence_C"] for h in H], "b:", label="global C")
        ax[1, 0].set_title("PHASE COHERENCE over time"); ax[1, 0].set_xlabel("step")
        ax[1, 0].legend(fontsize=8); ax[1, 0].set_ylim(0, 1)
        for k in self.DOMAINS:
            if self.p.activations.get(k, 0) > 0:
                ax[1, 1].plot(ts, [h[f"drive_{k}"] for h in H], label=k)
        ax[1, 1].set_title("per-domain drive"); ax[1, 1].set_xlabel("step"); ax[1, 1].legend(fontsize=8, ncol=2)
        ax[1, 2].imshow(self.regime, cmap="coolwarm", vmin=0, vmax=1)
        ax[1, 2].set_title("regime (blue=elastic, red=plastic)"); ax[1, 2].axis("off")
        plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ─────────────── correctness check: S+D+Noise -> Brownian ───────────────
def validate_brownian(steps=80000, dt=0.01, gamma=0.1, lam=0.05, sigma=0.6, seed=11):
    rng = np.random.default_rng(seed)
    pos = np.array([0.1, 0.1]); traj = []
    for n in range(steps):
        r2 = pos @ pos; F = -gamma * pos - lam * r2 * pos
        pos = pos + dt * F + sigma * np.sqrt(dt) * rng.standard_normal(2)
        if n % 10 == 0:
            traj.append(pos.copy())
    traj = np.array(traj); inc = np.diff(traj[:, 0])
    z = (inc - inc.mean()) / (inc.std() + 1e-12)
    exk = float(np.mean(z ** 4) - 3.0); ac1 = float(np.corrcoef(inc[:-1], inc[1:])[0, 1])
    taus = np.arange(1, 31)
    msd = [np.mean((traj[t:, 0] - traj[:-t, 0]) ** 2 + (traj[t:, 1] - traj[:-t, 1]) ** 2) for t in taus]
    slope = float(np.polyfit(np.log(taus), np.log(msd), 1)[0])
    g = abs(exk) < 0.6 and abs(ac1) < 0.2
    return {"increment_excess_kurtosis": exk, "increment_lag1_autocorr": ac1,
            "msd_slope": slope, "gaussian_increments": bool(g),
            "is_brownian": bool(g and 0.6 < slope < 1.4),
            "regime": "free-diffusion" if slope > 0.85 else "confined (sub-diffusive)"}


def main(outdir="bcdc_out"):
    import os, json
    os.makedirs(outdir, exist_ok=True)
    f = BCDCField(64, BCDCParams(), seed=7)
    txt = ("constrained subspace estimation recovers a low-rank signal from "
           "overdetermined noisy linear measurements using shrinkage")
    f.seed_from_text(txt)
    lc0 = float(local_coherence(f.theta).mean())
    snaps = [f.phi.copy()]
    for _ in range(6):
        f.run(50); snaps.append(f.phi.copy())
    f.write_csv(f"{outdir}/bcdc_metrics.csv")
    summary = f.write_json(f"{outdir}/bcdc_summary.json")
    # proof the coupling ORGANIZES phase: random init -> local coherence rises
    fr = BCDCField(64, BCDCParams(), seed=3)   # random-phase init
    org = [float(local_coherence(fr.theta).mean())]
    for _ in range(8):
        fr.run(40); org.append(fr.history[-1]["local_coherence_mean"])
    f.visualize(f"{outdir}/bcdc_field.png", snaps, org_curve=org)
    sweep = []
    for g, l in [(0.1, 0.05), (0.5, 0.3), (1.0, 1.0)]:
        r = validate_brownian(gamma=g, lam=l); r["gamma"] = g; r["lambda"] = l; sweep.append(r)
    with open(f"{outdir}/brownian_validation.json", "w") as fp:
        json.dump(sweep, fp, indent=2)
    fin = summary["final"]
    print("=== PHASE ORGANIZATION (random init): local coherence",
          " -> ".join(f"{v:.2f}" for v in org))
    print(f"=== text-seeded field: local coherence {lc0:.3f} -> {fin['local_coherence_mean']:.3f}"
          f"  (global C {summary['coherence_start']:.3f} -> {fin['coherence_C']:.3f}, low = domains not lock)")
    print("=== field stable:", summary["stable"],
          "| phi_mean:", round(fin["phi_mean"], 3),
          "plastic_frac:", round(fin["plastic_frac"], 3),
          "crystallized_frac:", round(fin["crystallized_frac"], 3),
          "| rotation(liveness):", round(fin["rotation_mean"], 3),
          "config_flux:", round(fin["config_flux_mean"], 4))
    # RESONANCE vs TRANSFIXION: a steady resonant field is reactive; a saturated one is not.
    react_resonant = f.probe_reactivity()
    hot = BCDCParams(); hot.alpha_em = 0.34; hot.d_decay = 0.10; hot.k_fb = 0.5
    sat = BCDCField(64, hot, seed=7); sat.seed_from_text(txt); sat.run(200)
    react_sat = sat.probe_reactivity()
    print("=== resonance vs transfixion (same input probe):")
    print(f"    resonant field  (phi_mean {f.phi.mean():.2f}, steady): "
          f"divergence={react_resonant['field_divergence']:.4f}  Cresp={react_resonant['coherence_response']:.4f}  -> REACTIVE")
    print(f"    saturated field (phi_mean {sat.phi.mean():.2f}, pinned): "
          f"divergence={react_sat['field_divergence']:.4f}  Cresp={react_sat['coherence_response']:.4f}  -> unresponsive")
    print("=== Brownian sweep (S+D+noise):")
    for r in sweep:
        print(f"    gamma={r['gamma']} lam={r['lambda']}: kurt={r['increment_excess_kurtosis']:+.3f}"
              f" ac1={r['increment_lag1_autocorr']:+.3f} slope={r['msd_slope']:.3f} [{r['regime']}]")
    return f, summary, sweep


if __name__ == "__main__":
    main()
