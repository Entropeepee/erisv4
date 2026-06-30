"""
FRACTAL PDE - Field Evolution Substrate (BCDC Unified Engine v3)
================================================================

Replaced old advection PDE with the Kuramoto coupled-oscillator phase model
and 8-domain BCDC yield surface.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import hashlib
import os
import time
import re
import numpy as np

from eris.config import xp, to_numpy, to_gpu, CONFIG
from eris.computation.activations import BVec, compute_bvec_from_field, bvec_distance

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

# ----------------------------- unifying primitives (CuPy ported) -----------------------------
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


def _soft_clamp(x, lo: float, hi: float, k: float = 40.0):
    """Bound φ into [lo, hi] with a SOFT amplitude CEILING (the handoff's Domain-6 boundary) and a
    hard non-negativity FLOOR. The ceiling is softplus-smoothed (C¹): ~identity inside the band,
    saturating gently near `hi` with NO gradient kink — so the saturation contour no longer injects a
    spurious edge into τ, and φ stays strictly below `hi` (the IBT pole is never reached). The floor
    is a plain max(·, lo): φ is a non-negative coherence amplitude (a hard physical constraint, not a
    tunable boundary), and it bites only in a quiescent region where the field is ~flat and τ≈0, so
    it dirties nothing — and it lets an unseeded field decay genuinely to lo, not float on a soft
    offset. `k` sets the ceiling sharpness."""
    x = xp.maximum(x, lo)                                          # hard non-negativity floor
    sp = xp.logaddexp(xp.asarray(0.0, dtype=x.dtype), k * (hi - x)) / k   # softplus(k·(hi−x))/k
    return hi - sp                                                 # smooth ceiling toward hi


# τ (torsion). The symbol contract (retrieval/field_interference.py) defines the field τ as the
# VORTICITY ∇ρ×∇θ — and that is what knowledge/frontends.py::torsion already computes. The PDE/FRT
# historically shipped a cheap proxy (Laplacian of the amplitude), which ignores phase entirely and
# discards the signed/chiral information that makes τ a distinct third channel (an ablation in the
# resonant-transfer-engine work measured ~28x better rotational/irrotational separation for the
# vorticity, 20.5 vs 0.73). The proxy was also DEGENERATE in practice: it pinned bvec C≈1.0 for
# every text (a constant wearing a variable's name), while the vorticity gives a content-varying
# C≈0.38–0.49. Default is now ON (canonical vorticity); set ERIS_TAU_VORTICITY=0 to reach the
# legacy Laplacian proxy for comparison.
_TAU_VORTICITY = os.environ.get("ERIS_TAU_VORTICITY", "1").strip().lower() in ("1", "on", "true", "yes")


def _vorticity(rho, theta):
    """Field torsion as the canonical vorticity τ = ∇ρ × ∇θ (2D scalar curl:
    dρ/dx·dθ/dy − dρ/dy·dθ/dx). Phase gradients go through wrap_diff so the 2π branch cut never
    injects spurious vorticity (the gauge-/wrap-safe form); tanh-stabilized like frontends.torsion.
    Roll-based + xp throughout, so the CuPy/NumPy path is preserved."""
    drho_x = rho - xp.roll(rho, 1, axis=1)
    drho_y = rho - xp.roll(rho, 1, axis=0)
    dth_x = wrap_diff(theta, xp.roll(theta, 1, axis=1))      # wrap-safe ∂θ/∂x
    dth_y = wrap_diff(theta, xp.roll(theta, 1, axis=0))      # wrap-safe ∂θ/∂y
    return xp.tanh(drho_x * dth_y - drho_y * dth_x)


def _compute_tau(rho, theta):
    """Dispatch τ: canonical vorticity ∇ρ×∇θ when ERIS_TAU_VORTICITY is on, else the legacy
    amplitude-Laplacian proxy (kept reachable for comparison)."""
    return _vorticity(rho, theta) if _TAU_VORTICITY else _lap(rho)

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

def encode_text(text, size=64, n_channels=12, amp=0.6, B_max=1.0, embedding=None):
    """Plane-wave seed from a text embedding. `embedding` is the seam (Phase-2 Step 4): pass a real
    semantic vector (e.g. eris.knowledge.embeddings.get_embedding(text), bge-m3) to seed on MEANING;
    when None it falls back to the in-module hashed bag-of-words stopgap (get_bge_m3_embedding)."""
    if not text or not text.strip():
        return (np.full((size, size), 0.25), np.zeros((size, size)))
    emb = get_bge_m3_embedding(text) if embedding is None else np.asarray(embedding, dtype=np.float64).ravel()
    th_v, c2_v = coupling_angles(emb, n_channels)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    Psi = np.zeros((size, size), dtype=complex)
    for i, (a, c2) in enumerate(zip(th_v, c2_v)):
        kx, ky = 1 + (i % 3), 1 + (i // 3)
        Psi += c2 * np.exp(1j * (2 * np.pi * (kx * x + ky * y) / size + a))
    phi = np.abs(Psi); phi = phi / (phi.max() + 1e-10) * amp + 0.12
    phi = np.clip(phi, 0.02, B_max - 0.02)
    return to_gpu(phi.astype(np.float32)), to_gpu((np.angle(Psi) % (2 * np.pi)).astype(np.float32))

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
        

    def seed_from_text(self, text: str, use_frt: bool = False, amp: float = 0.6) -> None:
        # `use_frt` is now honored (it was previously ignored): True → the fast FRT hash path
        # (text_to_field_arrays); False → the plane-wave encode_text path. Default False keeps the
        # current behavior for every caller.
        if use_frt:
            from eris.field.frt import text_to_field_arrays
            _p, _t = text_to_field_arrays(text, self.size)
            phi_seed, theta_seed = to_gpu(_p.astype(np.float32)), to_gpu(_t.astype(np.float32))
        else:
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
        self.tau = _compute_tau(self.phi, self.theta)

    def seed_from_fingerprint(self, fingerprint: str) -> None:
        self.seed_from_text(f"IDENTITY:{fingerprint}")

    def warm_reseed(self, text: str, blend: float = 0.7, amp: float = 0.6) -> None:
        """Tier 3 warm-start (CIP amortization, §0053): blend a new text seed INTO
        the current (warm) field state instead of overwriting it, so an apt prior
        from a previous turn carries over. `blend` is the weight of the new text:
        blend=1.0 reproduces a cold `seed_from_text`; smaller keeps more prior.
        Phase is blended on the circle (via unit vectors), amplitude linearly."""
        phi_seed, theta_seed = encode_text(text, self.size, amp=amp, B_max=self.p.B_max)
        b = float(blend)
        self.phi = (1.0 - b) * self.phi + b * phi_seed
        mixed = (1.0 - b) * xp.exp(1j * self.theta) + b * xp.exp(1j * theta_seed)
        self.theta = xp.angle(mixed) % (2 * np.pi)
        self.regime = xp.zeros((self.size, self.size), dtype=xp.float32)
        self._lc = local_coherence(self.theta)
        self.phi_prev = xp.copy(self.phi)
        self.theta_prev = xp.copy(self.theta)
        self._flux_phi = xp.zeros((self.size, self.size), dtype=xp.float32)
        self._flux_theta = xp.zeros((self.size, self.size), dtype=xp.float32)
        self.step_count = 0
        self.tau = _compute_tau(self.phi, self.theta)

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
        self.tau = _compute_tau(phi, theta)

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
        
        # If completely unseeded/inactive, gradual baseline decay to keep it quiet
        if not p.activations and xp.max(self.phi) < 0.5:
            phi_new *= 0.95
            
        # TORUS topology (no wall): the xp.roll stencils already wrap, so there is NO Dirichlet
        # edge to zero — that manufactured a hard φ discontinuity at the border and injected spurious
        # gradients/vorticity into τ. Removed. SPACE is bounded by the finite grid itself (the torus).
        #
        # AMPLITUDE boundary is a SOFT ceiling, not a hard clip. The Domain-6 soft restoring forces
        # already live in ops["B"]/["IBT"]/["ZBT"] (the −γ·σ·max(φ−lower,0)² ceiling + the soft
        # barriers near 0 and B_max). The old hard clip kinked the gradient at the saturation contour
        # (dirtying τ). Replace it with a smooth, differentiable clamp that is ~identity inside the
        # operating band and saturates softly — no kink — while keeping φ strictly inside (0, B_max)
        # so the IBT/ZBT barriers never reach their poles (stability the hard clip used to provide).
        self.phi = _soft_clamp(phi_new, 0.0, p.B_max - 1e-4)

        self._phase_step()

        # Update temporal flux for liveness check
        a = 1.0 / max(self.p.tau_obs, 1)
        self._flux_phi = (1 - a) * self._flux_phi + a * xp.abs(self.phi - self.phi_prev)
        self._flux_theta = (1 - a) * self._flux_theta + a * xp.abs(wrap_diff(self.theta, self.theta_prev))
        self.theta_prev = xp.copy(self.theta)
        
        self.tau = _compute_tau(self.phi, self.theta)
        self._update_global_observables()
        self.step_count += 1

    def run(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.step()

    def run_gated(self, monitor, max_steps: int, check_every: int = 4,
                  min_steps: int = 8) -> int:
        """Tier 2 (CIP early-termination for iterative solvers, §0069F).

        Evolve up to `max_steps`, but SUSPEND early once the trajectory has
        SETTLED — the change in global coherence over a `check_every` window
        drops to a low outlier below its own noise floor (judged by the shared
        CriticalityMonitor in "settle" mode). Returns the number of steps
        actually executed (the benchmark counts this).

        Safety (this signal is tiny — coherence sits ~0.04):
          • `min_steps` is a hard floor — never suspend before the field has
            done real work. Defaults to 8; callers pass CONFIG.orch_min_field_steps.
          • During the monitor's warmup the gate returns CONTINUE, so the first
            turns run the full `max_steps` exactly like `run()`.
        `run()` itself is untouched; this is an additive variant.
        """
        from eris.computation.criticality import Decision
        min_steps = max(1, min_steps)
        last_c = self.coherence
        executed = 0
        for _ in range(max_steps):
            self.step()
            executed += 1
            if executed < min_steps or executed % check_every != 0:
                continue
            c = self.coherence
            delta = abs(c - last_c)
            last_c = c
            decision, _ = monitor.observe("field_settle", delta, {"mode": "settle"})
            if decision == Decision.SUSPEND:
                break
        return executed

    def run_settled(self, max_steps: int, min_steps: int = 8, check_every: int = 4,
                    tol: float = 0.02) -> int:
        """Self-contained criticality early-stop (Stage-2 convergence-rate; the run_gated
        discipline WITHOUT a shared monitor, so it works for a fresh per-call field like the
        retrieval rerank, where a monitor would never leave warmup). Evolve up to `max_steps`
        but SUSPEND once the field has SETTLED — the coherence change over a `check_every`
        window falls below `tol` RELATIVE to the coherence level (the attractor is reached;
        more steps won't change what resonance measures). `min_steps` is a hard floor. Returns
        steps executed. `run()`/`run_gated()` are untouched — this is an additive variant."""
        min_steps = max(1, min_steps)
        last_c = self.coherence
        executed = 0
        for _ in range(max_steps):
            self.step()
            executed += 1
            if executed < min_steps or executed % check_every != 0:
                continue
            c = self.coherence
            if abs(c - last_c) <= tol * (abs(c) + 1e-9):
                break
            last_c = c
        return executed

    def run_gated_response(self, monitor, max_steps: int, check_every: int = 4,
                           min_steps: int = 8) -> int:
        """Tier 3: evolve the response field but SUSPEND once the response bvec
        stabilizes — the windowed bvec change drops to a low outlier below its
        floor (settle mode). Returns steps executed. Like run_gated but the
        settling signal is the bvec change, which is what feeds dissonance."""
        from eris.computation.criticality import Decision
        min_steps = max(1, min_steps)
        last_bvec = self.compute_bvec()
        executed = 0
        for _ in range(max_steps):
            self.step()
            executed += 1
            if executed < min_steps or executed % check_every != 0:
                continue
            cur = self.compute_bvec()
            change = bvec_distance(cur, last_bvec)
            last_bvec = cur
            decision, _ = monitor.observe("resp_settle", change, {"mode": "settle"})
            if decision == Decision.SUSPEND:
                break
        return executed
            
    def compute_bvec(self) -> BVec:
        return compute_bvec_from_field(self.phi, self.theta, self.tau, self.phi_prev)

    def _update_global_observables(self) -> None:
        C = float(xp.abs(xp.mean(xp.exp(1j * self.theta))))
        gtheta_x = wrap_diff(xp.roll(self.theta, -1, 1), self.theta)
        gtheta_y = wrap_diff(xp.roll(self.theta, -1, 0), self.theta)
        X = float(xp.sum(self.phi * (gtheta_x + gtheta_y)))
        
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

    def save_checkpoint(self, path: str) -> None:
        np.savez(path, phi=to_numpy(self.phi), theta=to_numpy(self.theta), step_count=self.step_count)

    @classmethod
    def load_checkpoint(cls, path: str, params: Optional[BCDCParams] = None) -> "FractalField":
        data = np.load(path)
        phi = data["phi"]
        size = phi.shape[0]
        field = cls(size=size, params=params)
        from eris.config import to_gpu
        field.phi = to_gpu(phi)
        field.theta = to_gpu(data["theta"])
        field.phi_prev = xp.copy(field.phi)
        field.step_count = int(data.get("step_count", 0))
        return field

    def detect_regime(self) -> str:
        """Internal information-processing regime — NOT a hallucination flag.

        Remediation Tier 1.2: the old logic used absolute thresholds
        (`mean_abs < 0.01 and tau_rms < 0.01`) tuned for a since-replaced
        advection PDE. On the BCDC Kuramoto engine tau_rms lives ~0.1 and never
        drops below 0.01, so "transfixed" was unreachable and the regime stayed
        pinned at "elastic". We now calibrate against THIS engine's own running
        dC/dX distribution (self-calibrating percentiles), so the regime varies
        meaningfully regardless of the engine's absolute scale.

        This signal tells you whether the field is stuck / under-coupled, not
        whether a claim is factually true — factual hallucination is a grounding
        failure handled in Tier 4 (knowledge), orthogonal to field coherence.
        """
        if len(self._dCdX_history) < 8:
            return "warmup"
        hist = [abs(x) for x in self._dCdX_history]
        cur = hist[-1]
        lo = float(np.percentile(hist, 20))
        hi = float(np.percentile(hist, 80))
        c_hist = self._coherence_history
        c_stable = (
            len(c_hist) >= 5 and
            float(np.std(c_hist[-5:])) < 0.25 * (float(np.mean(c_hist[-5:])) + 1e-9)
        )
        if cur <= lo and c_stable:
            return "transfixed"   # dC/dX in its own low tail + flat coherence = stuck / under-coupled
        elif cur >= hi:
            return "plastic"      # dC/dX in its own high tail = genuine restructuring
        else:
            return "elastic"      # mid-range = smooth incremental update

    def clone(self) -> "FractalField":
        import copy
        new_field = FractalField(size=self.size, params=self.p)
        new_field.phi = xp.copy(self.phi)
        new_field.theta = xp.copy(self.theta)
        new_field.phi_prev = xp.copy(self.phi_prev)
        new_field.theta_prev = xp.copy(self.theta_prev)
        new_field.tau = xp.copy(self.tau)
        new_field.memory = xp.copy(self.memory)
        new_field.attention = xp.copy(self.attention)
        new_field.regime = xp.copy(self.regime)
        new_field._flux_phi = xp.copy(self._flux_phi)
        new_field._flux_theta = xp.copy(self._flux_theta)
        new_field._lc = xp.copy(self._lc)
        new_field.step_count = self.step_count
        # NOTE: the real history attributes are _coherence_history /
        # _exchange_history / _dCdX_history. The previous clone() also copied
        # non-existent `_C_hist` / `_X_hist`, which raised AttributeError every
        # time clone() ran — and clone() runs on every turn via
        # probe_reactivity() -> the transfixion check. That crash is removed.
        new_field._coherence_history = list(self._coherence_history)
        new_field._exchange_history = list(self._exchange_history)
        new_field._dCdX_history = list(self._dCdX_history)
        return new_field

    def probe_reactivity(self, input_text: str = "", use_frt: bool = False, steps: int = 12):
        """The real transfixion test: does the field RESPOND to actual input over a window?"""
        base = self.clone()
        pert = self.clone()
        
        if input_text:
            pert.seed_from_text(input_text, use_frt=use_frt)
            
        C0 = float(xp.abs(xp.mean(xp.exp(1j * base.theta))))
        for _ in range(steps):
            base.step()
            pert.step()
            
        divergence = float(to_numpy(xp.mean(xp.abs(pert.phi - base.phi))))
        dC = float(abs(pert._coherence_history[-1] - base._coherence_history[-1]))
        return {"input_text": input_text[:20], "field_divergence": divergence,
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
