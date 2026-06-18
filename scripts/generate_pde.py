import os

CODE = """"""
FRACTAL PDE — Field Evolution Substrate (BCDC Unified Engine v3)
================================================================

Replaced old advection PDE with the Kuramoto coupled-oscillator phase model
and 8-domain BCDC yield surface.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import hashlib
import time
import re
import numpy as np

from eris.config import xp, to_numpy, to_gpu, CONFIG
from eris.computation.activations import BVec, compute_bvec_from_field

# BGE-M3 ONNX integration (Placeholder / Hooks ready)
def get_bge_m3_embedding(text: str, dim: int = 256) -> np.ndarray:
    """Real ONNX BGE-M3 model hook."""
    # TODO: Load ONNX Runtime and run BGE-M3. 
    # For now, fallback to the hashed bag-of-words stopgap.
    vec = np.zeros(dim, dtype=np.float64)
    t = text.lower(); words = re.findall(r"[a-z0-9]+", t)
    toks = list(words) + [words[i] + "_" + words[i + 1] for i in range(len(words) - 1)]
    toks += [t[i:i + 3] for i in range(len(t) - 2)]
    for tok in toks:
        h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
        vec[h % dim] += 1.0 if (h >> 63) & 1 else -1.0
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec

# ─────────────────────────── unifying primitives (CuPy ported) ───────────────────────────
def hill_power(s, alpha=1.0, beta=0.5, gamma=1.0, delta=0.0):
    s_shift = xp.maximum(s - delta, 0.0)
    sa = xp.power(s_shift, alpha)
    return xp.power(sa / (sa + beta + 1e-12), gamma)

def beta_star(lam):
    sigma_g = np.sqrt(8.0 / max(lam, 1e-9))
    return (2.0 / np.sqrt(np.pi)) * np.sqrt(sigma_g / (sigma_g ** 2 + 1.0))

def sigmoid_gate(s, center, width, steep=8.0):
    return 1.0 / (1.0 + xp.exp(-(s - center) * (steep / max(width, 1e-6))))

def colored_noise_xp(shape, rng, n_smooth=4):
    w = to_gpu(rng.standard_normal(shape).astype(np.float32))
    for _ in range(n_smooth):
        w = 0.5 * w + 0.125 * (xp.roll(w, 1, 0) + xp.roll(w, -1, 0)
                               + xp.roll(w, 1, 1) + xp.roll(w, -1, 1))
    return w / (xp.std(w) + 1e-9)

def _lap(a):
    return (xp.roll(a, 1, 0) + xp.roll(a, -1, 0)
            + xp.roll(a, 1, 1) + xp.roll(a, -1, 1) - 4.0 * a)

def wrap_diff(a_nbr, a):
    """Phase difference on the circle, in (-pi, pi]."""
    return xp.angle(xp.exp(1j * (a_nbr - a)))

def local_coherence(theta):
    """Local Kuramoto order parameter over the 4-neighborhood + self, in [0,1]."""
    ph = xp.exp(1j * theta)
    z = (ph + xp.roll(ph, 1, 0) + xp.roll(ph, -1, 0)
         + xp.roll(ph, 1, 1) + xp.roll(ph, -1, 1)) / 5.0
    return xp.abs(z)

def coupling_angles(emb, n_channels=12):
    blocks = np.array_split(emb, n_channels)
    energy = np.array([float(np.sum(b * b)) for b in blocks])
    cos2 = energy / (energy.sum() + 1e-12)
    theta = np.arccos(np.sqrt(np.clip(cos2, 0.0, 1.0)))
    return theta, cos2

def encode_text(text, size=64, n_channels=12, amp=0.6, B_max=1.0):
    if not text or not text.strip():
        return (np.full((size, size), 0.25), np.zeros((size, size)))
    th_v, c2_v = coupling_angles(get_bge_m3_embedding(text), n_channels)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    Psi = np.zeros((size, size), dtype=complex)
    for i, (a, c2) in enumerate(zip(th_v, c2_v)):
        kx, ky = 1 + (i % 3), 1 + (i // 3)
        Psi += c2 * np.exp(1j * (2 * np.pi * (kx * x + ky * y) / size + a))
    phi = np.abs(Psi); phi = phi / (phi.max() + 1e-10) * amp + 0.12
    phi = np.clip(phi, 0.02, B_max - 0.02)
    return to_gpu(phi.astype(np.float32)), to_gpu((np.angle(Psi) % (2 * np.pi)).astype(np.float32))

def _enforce_dirichlet(field):
    field[0, :] = 0.0
    field[-1, :] = 0.0
    field[:, 0] = 0.0
    field[:, -1] = 0.0
    return field

@dataclass
class PDEParams:
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
    omega: float = 0.10
    omega_spread: float = 0.5
    K_phase: float = 1.6
    sigma_phase: float = 0.04
    tau_obs: int = 20
    phi_yield_hi: float = 0.62
    phi_yield_lo: float = 0.48

    # Eris repo specific modulators
    memory_tau: float = 10.0
    memory_coupling: float = 0.1
    attention_strength: float = 0.3

W_ELASTIC = {"S": 1.0, "D": 1.0, "F": 0.2, "TH": 1.0, "B": 1.0, "E": 0.3, "IBT": 1.0, "ZBT": 1.0}
W_PLASTIC = {"S": 0.3, "D": 1.0, "F": 1.0, "TH": 1.0, "B": 1.0, "E": 1.5, "IBT": 1.0, "ZBT": 1.0}

class FractalField:
    DOMAINS = ["S", "D", "F", "TH", "B", "E", "IBT", "ZBT"]

    def __init__(self, size: int = 64, params: Optional[PDEParams] = None, seed: int = 42):
        self.size = size
        self.params = params or PDEParams()
        self.p = self.params
        self.step_count: int = 0
        self.wall_time_start: float = time.time()
        self.rng = np.random.default_rng(seed)
        
        g = colored_noise_xp((size, size), self.rng, 3)
        self.phi = xp.clip(0.25 + 0.15 * g, 0.02, self.p.B_max - 0.02)
        self.theta = to_gpu(self.rng.uniform(0, 2 * np.pi, (size, size)).astype(np.float32))
        self.regime = xp.zeros((size, size), dtype=xp.float32)
        
        omega_base = self.p.omega + self.p.omega_spread * self.rng.standard_normal((size, size))
        self.omega0 = to_gpu(omega_base.astype(np.float32))
        
        self._lc = local_coherence(self.theta)
        self.phi_prev = xp.copy(self.phi)
        self.theta_prev = xp.copy(self.theta)
        self._flux_phi = xp.zeros((size, size), dtype=xp.float32)
        self._flux_theta = xp.zeros((size, size), dtype=xp.float32)
        
        # Legacy tracking for activation mapping
        self.tau = xp.zeros((size, size), dtype=xp.float32)
        
        # Eris modulators
        self.memory = xp.zeros((size, size), dtype=xp.float32)
        self.attention = xp.ones((size, size), dtype=xp.float32)
        
        self._coherence_history: list = []
        self._exchange_history: list = []
        self._dCdX_history: list = []
        self._hp_beta = self.p.hp_beta if self.p.hp_beta > 0 else beta_star(self.p.omega + 1.0)
        
        _enforce_dirichlet(self.phi)
        _enforce_dirichlet(self.phi_prev)

    def seed_from_text(self, text: str, use_frt: bool = False, amp: float = 0.6) -> None:
        # Unify seed_from_text directly to encode_text.
        phi_seed, theta_seed = encode_text(text, self.size, amp=amp, B_max=self.p.B_max)
        self.phi = phi_seed
        self.theta = theta_seed
        self.regime = xp.zeros((self.size, self.size), dtype=xp.float32)
        self._lc = local_coherence(self.theta)
        self.phi_prev = xp.copy(self.phi)
        self.theta_prev = xp.copy(self.theta)
        self._flux_phi = xp.zeros((self.size, self.size), dtype=xp.float32)
        self._flux_theta = xp.zeros((self.size, self.size), dtype=xp.float32)
        self.step_count = 0
        _enforce_dirichlet(self.phi)
        _enforce_dirichlet(self.phi_prev)
        self.tau = _lap(self.phi)

    def seed_from_fingerprint(self, fingerprint: str) -> None:
        self.seed_from_text(f"IDENTITY:{fingerprint}")

    def _base_ops(self):
        p, phi, Bm = self.p, self.phi, self.p.B_max
        gate = lambda s, dlt: hill_power(s, p.hp_alpha, self._hp_beta, p.hp_gamma, dlt)
        ops = {}
        ops["S"]  = p.r_sat * phi * (1.0 - phi / Bm)
        ops["D"]  = -p.d_decay * phi
        ops["F"]  = p.k_fb * (self._lc - xp.mean(self._lc)) * phi
        ops["TH"] = p.a_th * gate(phi, p.T_c) * (1.0 - phi)
        lower = Bm - p.quiet_zone; center = Bm - 0.5 * p.quiet_zone
        soft = sigmoid_gate(phi, center, p.quiet_zone, p.bnd_steep)
        ops["B"]  = -p.gamma_bnd * soft * xp.maximum(phi - lower, 0.0) ** 2
        ops["E"]  = p.alpha_em * xp.power(xp.maximum(phi, 0.0), p.D_em)
        ops["IBT"] = -p.xi_ibt / (Bm - phi + p.delta_bt)
        ops["ZBT"] = +p.zeta_zbt / (phi + p.delta_bt)
        return ops

    def _update_regime(self):
        enter = self.phi > self.p.phi_yield_hi
        leave = self.phi < self.p.phi_yield_lo
        self.regime = xp.where(enter, 1.0, xp.where(leave, 0.0, self.regime))

    def _phase_step(self):
        p = self.p
        ph = xp.exp(1j * self.theta)
        nbr = (xp.roll(ph, 1, 0) + xp.roll(ph, -1, 0)
               + xp.roll(ph, 1, 1) + xp.roll(ph, -1, 1))
        coupling = xp.imag(xp.conj(ph) * nbr)
        g = hill_power(self.phi, p.hp_alpha, self._hp_beta, 1.0, 0.0)
        dtheta = self.omega0 + p.K_phase * g * coupling
        noise = to_gpu(self.rng.standard_normal(self.phi.shape).astype(np.float32))
        self.theta = (self.theta + p.dt * dtheta
                      + p.sigma_phase * np.sqrt(p.dt) * noise) % (2 * np.pi)
        self._lc = local_coherence(self.theta)

    def step(self):
        p = self.p
        phi = self.phi
        theta = self.theta

        self.phi_prev = xp.copy(phi)
        _enforce_dirichlet(phi)
        self.tau = _lap(phi)

        # Eris modulators
        alpha_m = p.dt / max(p.memory_tau, p.dt)
        self.memory = (1.0 - alpha_m) * self.memory + alpha_m * phi
        
        gphi_x = phi - xp.roll(phi, 1, axis=1)
        gphi_y = phi - xp.roll(phi, 1, axis=0)
        grad_mag = xp.sqrt(gphi_x**2 + gphi_y**2 + 1e-10)
        grad_norm = grad_mag / xp.maximum(xp.max(grad_mag), xp.float32(1e-10))
        self.attention = 1.0 + p.attention_strength * (grad_norm - 0.5)

        self._update_regime()
        ops = self._base_ops()
        plastic = self.regime > 0.5
        dphi = xp.zeros_like(self.phi)
        for k in self.DOMAINS:
            a = p.activations.get(k, 0.0)
            wk = xp.where(plastic, W_PLASTIC[k], W_ELASTIC[k])
            term = a * wk * ops[k]
            dphi += term
            
        eta = colored_noise_xp(self.phi.shape, self.rng, p.noise_smooth)
        if p.noise_structured:
            grad = xp.sqrt((self.phi - xp.roll(self.phi, 1, 1)) ** 2
                           + (self.phi - xp.roll(self.phi, 1, 0)) ** 2 + 1e-9)
            unresolved = 1.0 - hill_power(grad, 1.0, 0.25, 1.0, 0.0)
            eta = eta * (0.5 + 0.5 * unresolved)

        memory_bias = p.memory_coupling * (self.memory - phi)
        
        # Apply Eris attention modulator to full phi evolution 
        # (combining BCDC engine dphi and Eris memory bias)
        phi_new = phi + p.dt * self.attention * (dphi + memory_bias) + p.sigma_noise * np.sqrt(p.dt) * eta
        self.phi = xp.clip(phi_new, 1e-4, p.B_max - 1e-4)
        
        self._phase_step()

        # Update temporal flux for liveness check
        a = 1.0 / max(self.p.tau_obs, 1)
        self._flux_phi = (1 - a) * self._flux_phi + a * xp.abs(self.phi - self.phi_prev)
        self._flux_theta = (1 - a) * self._flux_theta + a * xp.abs(wrap_diff(self.theta, self.theta_prev))
        self.theta_prev = xp.copy(self.theta)
        
        self.tau = _lap(self.phi)
        self._update_global_observables()
        self.step_count += 1

    def run(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.step()
            
    def compute_bvec(self) -> BVec:
        return compute_bvec_from_field(self.phi, self.theta, self.tau, self.phi_prev)

    def _update_global_observables(self) -> None:
        phi_np = to_numpy(self.phi)
        theta_np = to_numpy(self.theta)
        
        C = float(np.abs(np.mean(np.exp(1j * theta_np))))
        gtheta_x = wrap_diff(np.roll(theta_np, -1, 1), theta_np)
        gtheta_y = wrap_diff(np.roll(theta_np, -1, 0), theta_np)
        X = float(np.sum(phi_np * (gtheta_x + gtheta_y)))
        
        self._coherence_history.append(C)
        self._exchange_history.append(X)
        
        if len(self._coherence_history) >= 2:
            dC = self._coherence_history[-1] - self._coherence_history[-2]
            dX = self._exchange_history[-1] - self._exchange_history[-2]
            dCdX = dC / dX if abs(dX) > 1e-10 else 0.0
            self._dCdX_history.append(dCdX)

    @property
    def coherence(self) -> float:
        return self._coherence_history[-1] if self._coherence_history else 0.0

    @property
    def exchange(self) -> float:
        return self._exchange_history[-1] if self._exchange_history else 0.0

    @property
    def dCdX(self) -> float:
        return self._dCdX_history[-1] if self._dCdX_history else 0.0

    def detect_regime(self) -> str:
        if len(self._dCdX_history) < 5:
            return "warmup"
        recent_dCdX = self._dCdX_history[-5:]
        mean_abs = np.mean([abs(x) for x in recent_dCdX])
        recent_tau_rms = float(to_numpy(xp.sqrt(xp.mean(self.tau ** 2))))
        if mean_abs < 0.01 and recent_tau_rms < 0.01:
            return "transfixed"
        elif mean_abs > 0.5:
            return "plastic"
        else:
            return "elastic"

    def probe_reactivity(self, amp=0.45, steps=12):
        import copy
        base = copy.deepcopy(self); pert = copy.deepcopy(self)
        s = self.size; cy = cx = s // 2; r = s // 6
        yy, xx = np.mgrid[0:s, 0:s]
        mask = to_gpu((((xx - cx) ** 2 + (yy - cy) ** 2) < r * r).astype(np.float32))
        pert.phi = xp.clip(pert.phi + amp * mask, 1e-4, pert.p.B_max - 1e-4)
        pert.theta = xp.where(mask, (pert.theta + np.pi) % (2 * np.pi), pert.theta)
        C0 = float(xp.abs(xp.mean(xp.exp(1j * base.theta))))
        for _ in range(steps):
            base.step(); pert.step()
        divergence = float(to_numpy(xp.mean(xp.abs(pert.phi - base.phi))))
        dC = float(abs(pert._coherence_history[-1] - base._coherence_history[-1]))
        return {"input_amp": amp, "field_divergence": divergence,
                "coherence_response": dC, "C_baseline": C0}

    def snapshot(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "step_count": self.step_count,
            "phi": to_numpy(self.phi),
            "theta": to_numpy(self.theta),
            "tau": to_numpy(self.tau),
            "phi_prev": to_numpy(self.phi_prev),
            "memory": to_numpy(self.memory),
            "attention": to_numpy(self.attention),
            "coherence_history": self._coherence_history[-100:],
            "exchange_history": self._exchange_history[-100:],
            "dCdX_history": self._dCdX_history[-100:],
            "params": {
                "D": self.params.dt,
                "C_coupling": 0.5,
                "E_rate": 0.3,
                "D_decay": 0.05,
                "omega": 0.1,
                "lam": 0.5,
                "dt": self.params.dt,
                "memory_tau": self.params.memory_tau,
                "memory_coupling": self.params.memory_coupling,
                "attention_strength": self.params.attention_strength,
            },
        }

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "FractalField":
        params = PDEParams()
        field = cls(size=data["size"], params=params)
        field.step_count = data["step_count"]
        field.phi = to_gpu(data["phi"])
        field.theta = to_gpu(data["theta"])
        field.tau = to_gpu(data["tau"])
        field.phi_prev = to_gpu(data["phi_prev"])
        return field
"""

with open(os.path.join("C:\\Users\\david\\.gemini\\antigravity\\scratch\\erisv4\\eris\\field\\pde.py"), "w") as f:
    f.write(CODE)
