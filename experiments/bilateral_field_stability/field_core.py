"""
field_core.py -- Self-contained, faithful lift of the Eris FRACTAL/BCDC field
=============================================================================

Standalone numpy probe for the bilateral-stability experiment. The single-field
dynamics here are lifted directly from `eris/field/pde.py` (FractalField):

  * `_base_ops`        -> the 8-domain BLECD operator terms (S,D,F,TH,B,E,IBT,ZBT)
  * `_vorticity`       -> tau = grad rho x grad theta, wrap-safe phase diffs
  * the phi update     -> attention-modulated dphi + memory bias + structured noise
  * `_phase_step`      -> Kuramoto coupled-oscillator phase evolution
  * soft amplitude ceiling (the 'B' sigmoid boundary op) + hard non-negativity clip

Deviations from the repo, all deliberate and documented:
  * The torus is kept PERFECT -- all stencils are periodic (np.roll) and we do NOT
    apply the repo's `_enforce_dirichlet` edge zeroing. The brief explicitly asks
    to keep the torus; Dirichlet edges would inject boundary artefacts that confound
    a collapse measurement. (This is the only structural change to the dynamics.)
  * numpy only -- the repo's `xp`/`to_gpu` indirection (CuPy option) is dropped.
  * No LLM / embeddings / orchestrator. Seeding is a random colored-noise field.

The BilateralField adds a second mirror lobe coupled by a Robin/Newton-cooling
membrane of permeability mu. Nothing else changes.
"""
from __future__ import annotations
from dataclasses import dataclass, field as _dc_field
from typing import Dict, Optional
import numpy as np

# --------------------------------------------------------------------------- #
#  Primitives -- lifted verbatim (numpy form) from eris/field/pde.py
# --------------------------------------------------------------------------- #
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
    w = rng.standard_normal(shape).astype(np.float64)
    for _ in range(n_smooth):
        w = 0.5 * w + 0.125 * (np.roll(w, 1, 0) + np.roll(w, -1, 0)
                               + np.roll(w, 1, 1) + np.roll(w, -1, 1))
    return w / (np.std(w) + 1e-9)


def wrap_diff(a_nbr, a):
    """Phase difference on the circle, in (-pi, pi]. (pde.py:wrap_diff)"""
    return np.angle(np.exp(1j * (a_nbr - a)))


def vorticity(rho, theta):
    """tau = grad rho x grad theta  (2D scalar curl), wrap-safe, tanh-stabilized.
    Lifted from pde.py:_vorticity (the canonical ERIS_TAU_VORTICITY=1 path)."""
    drho_x = rho - np.roll(rho, 1, axis=1)
    drho_y = rho - np.roll(rho, 1, axis=0)
    dth_x = wrap_diff(theta, np.roll(theta, 1, axis=1))
    dth_y = wrap_diff(theta, np.roll(theta, 1, axis=0))
    return np.tanh(drho_x * dth_y - drho_y * dth_x)


def local_coherence(theta):
    """Local Kuramoto order over the 4-neighborhood + self, in [0,1]. (pde.py)"""
    ph = np.exp(1j * theta)
    z = (ph + np.roll(ph, 1, 0) + np.roll(ph, -1, 0)
         + np.roll(ph, 1, 1) + np.roll(ph, -1, 1)) / 5.0
    return np.abs(z)


def coupling_gate(delta, kind):
    """Per-cell membrane-transport gate as a function of the phase-relatedness
    angle delta = wrap(theta_other - theta_self).
      diff  -> 1                         (plain diffusive; one attractor = sameness)
      cos   -> cos^2(delta)              (maximal at alignment; fuses -- neg. control)
      egate -> cos^2(delta)*sin^2(delta) (= 1/4 sin^2(2 delta); the established
               coupling law E(theta): peaks at 45 deg, ZERO at 0 and 90 deg, so the
               membrane never rewards collapsing distinctions -- the 'keep sin' rule).
    """
    if kind == "diff":
        return 1.0
    if kind == "cos":
        return np.cos(delta) ** 2
    if kind == "egate":
        return (np.cos(delta) ** 2) * (np.sin(delta) ** 2)
    raise ValueError(f"unknown coupling kind {kind!r}")


# --------------------------------------------------------------------------- #
#  Parameters -- mirrors pde.py:PDEParams (defaults identical)
# --------------------------------------------------------------------------- #
@dataclass
class PDEParams:
    B_max: float = 1.0
    dt: float = 0.05
    r_sat: float = 0.65
    d_decay: float = 0.28
    k_fb: float = 0.18
    a_th: float = 0.20
    T_c: float = 0.55
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
    phi_yield_hi: float = 0.62
    phi_yield_lo: float = 0.48
    memory_tau: float = 10.0
    memory_coupling: float = 0.1
    attention_strength: float = 0.3
    # which BLECD domains are active (pde.py:PDEParams.activations defaults all 1.0)
    activations: Dict[str, float] = _dc_field(default_factory=lambda: {
        "S": 1.0, "D": 1.0, "F": 1.0, "TH": 1.0,
        "B": 1.0, "E": 1.0, "IBT": 1.0, "ZBT": 1.0})


# regime-dependent domain weights (pde.py:W_ELASTIC / W_PLASTIC)
W_ELASTIC = {"S": 1.0, "D": 1.0, "F": 0.2, "TH": 1.0, "B": 1.0, "E": 0.3, "IBT": 1.0, "ZBT": 1.0}
W_PLASTIC = {"S": 0.3, "D": 1.0, "F": 1.0, "TH": 1.0, "B": 1.0, "E": 1.5, "IBT": 1.0, "ZBT": 1.0}
DOMAINS = ["S", "D", "F", "TH", "B", "E", "IBT", "ZBT"]


# --------------------------------------------------------------------------- #
#  Single field
# --------------------------------------------------------------------------- #
class SingleField:
    """One coherence lobe. Faithful single-field evolver (pde.py:FractalField)."""

    def __init__(self, size: int = 64, params: Optional[PDEParams] = None,
                 seed: int = 42, phi_init: float = 0.25, phi_jitter: float = 0.15):
        self.size = size
        self.p = params or PDEParams()
        self.rng = np.random.default_rng(seed)
        self.step_count = 0

        g = colored_noise((size, size), self.rng, 3)
        self.phi = np.clip(phi_init + phi_jitter * g, 0.02, self.p.B_max - 0.02)
        self.theta = self.rng.uniform(0, 2 * np.pi, (size, size))
        self.regime = np.zeros((size, size))
        omega_base = self.p.omega + self.p.omega_spread * self.rng.standard_normal((size, size))
        self.omega0 = omega_base
        self._lc = local_coherence(self.theta)
        self.phi_prev = self.phi.copy()
        self.theta_prev = self.theta.copy()
        self.memory = np.zeros((size, size))
        self.attention = np.ones((size, size))
        self.tau = vorticity(self.phi, self.theta)
        self._hp_beta = self.p.hp_beta if self.p.hp_beta > 0 else beta_star(self.p.omega + 1.0)

    # --- BLECD operator terms (pde.py:_base_ops) ---------------------------- #
    def _base_ops(self):
        p, phi, Bm = self.p, self.phi, self.p.B_max
        gate = lambda s, dlt: hill_power(s, p.hp_alpha, self._hp_beta, p.hp_gamma, dlt)
        ops = {}
        ops["S"] = p.r_sat * phi * (1.0 - phi / Bm)
        ops["D"] = -p.d_decay * phi
        ops["F"] = p.k_fb * (self._lc - np.mean(self._lc)) * phi
        ops["TH"] = p.a_th * gate(phi, p.T_c) * (1.0 - phi)
        lower = Bm - p.quiet_zone
        center = Bm - 0.5 * p.quiet_zone
        soft = sigmoid_gate(phi, center, p.quiet_zone, p.bnd_steep)
        ops["B"] = -p.gamma_bnd * soft * np.maximum(phi - lower, 0.0) ** 2
        ops["E"] = p.alpha_em * np.power(np.maximum(phi, 0.0), p.D_em)
        ops["IBT"] = -p.xi_ibt / (Bm - phi + p.delta_bt)
        ops["ZBT"] = +p.zeta_zbt / (phi + p.delta_bt)
        return ops

    def _update_regime(self):
        enter = self.phi > self.p.phi_yield_hi
        leave = self.phi < self.p.phi_yield_lo
        self.regime = np.where(enter, 1.0, np.where(leave, 0.0, self.regime))

    def _dphi(self):
        """Amplitude drift term (everything inside dt*attention*(.) in pde.py:step),
        WITHOUT integration. Membrane coupling is added on top by the caller."""
        # Eris attention modulator
        gphi_x = self.phi - np.roll(self.phi, 1, axis=1)
        gphi_y = self.phi - np.roll(self.phi, 1, axis=0)
        grad_mag = np.sqrt(gphi_x ** 2 + gphi_y ** 2 + 1e-10)
        grad_norm = grad_mag / max(np.max(grad_mag), 1e-10)
        self.attention = 1.0 + self.p.attention_strength * (grad_norm - 0.5)

        self._update_regime()
        ops = self._base_ops()
        plastic = self.regime > 0.5
        dphi = np.zeros_like(self.phi)
        for k in DOMAINS:
            a = self.p.activations.get(k, 0.0)
            wk = np.where(plastic, W_PLASTIC[k], W_ELASTIC[k])
            dphi += a * wk * ops[k]
        memory_bias = self.p.memory_coupling * (self.memory - self.phi)
        return dphi, memory_bias

    def _phase_step(self, theta_other=None, mu=0.0, coupling_kind="diff", gate_phase=False):
        p = self.p
        ph = np.exp(1j * self.theta)
        nbr = (np.roll(ph, 1, 0) + np.roll(ph, -1, 0)
               + np.roll(ph, 1, 1) + np.roll(ph, -1, 1))
        coupling = np.imag(np.conj(ph) * nbr)
        g = hill_power(self.phi, p.hp_alpha, self._hp_beta, 1.0, 0.0)
        dtheta = self.omega0 + p.K_phase * g * coupling
        # membrane phase exchange (wrap-safe), Robin/Newton-cooling.
        # Plain diffusive by default; gate_phase=True applies the same E-gate as the
        # amplitude transport (the optional `egate_phase` variant).
        if theta_other is not None and mu != 0.0:
            d = wrap_diff(theta_other, self.theta)
            cg = coupling_gate(d, coupling_kind) if gate_phase else 1.0
            dtheta = dtheta + mu * cg * d
        noise = self.rng.standard_normal(self.phi.shape)
        self.theta = (self.theta + p.dt * dtheta
                      + p.sigma_phase * np.sqrt(p.dt) * noise) % (2 * np.pi)
        self._lc = local_coherence(self.theta)

    def step(self):
        """Standalone single-lobe step (no coupling)."""
        self.step_with_coupling(None, None, 0.0)

    def step_with_coupling(self, phi_other, theta_other, mu,
                           coupling_kind="diff", gate_phase=False):
        """One integration step. If phi_other/theta_other given and mu!=0, a membrane
        term mu*g(delta)*(other - self) is added to the amplitude update (and, with
        gate_phase, to the phase update). g is the per-cell coupling_gate of the
        phase-relatedness angle delta=wrap(theta_other-theta_self):
          coupling_kind='diff'  -> g=1               (plain diffusive; the mirror class)
          coupling_kind='cos'   -> g=cos^2(delta)    (fuses; negative control)
          coupling_kind='egate' -> g=cos^2 sin^2     (E-gated; vanishes at sameness)
        Defaults reproduce the original plain-diffusive coupling exactly."""
        p = self.p
        self.phi_prev = self.phi.copy()
        self.tau = vorticity(self.phi, self.theta)

        # memory EMA (pde.py:step)
        alpha_m = p.dt / max(p.memory_tau, p.dt)
        self.memory = (1.0 - alpha_m) * self.memory + alpha_m * self.phi

        dphi, memory_bias = self._dphi()

        # structured colored noise (pde.py:step)
        eta = colored_noise(self.phi.shape, self.rng, p.noise_smooth)
        if p.noise_structured:
            grad = np.sqrt((self.phi - np.roll(self.phi, 1, 1)) ** 2
                           + (self.phi - np.roll(self.phi, 1, 0)) ** 2 + 1e-9)
            unresolved = 1.0 - hill_power(grad, 1.0, 0.25, 1.0, 0.0)
            eta = eta * (0.5 + 0.5 * unresolved)

        phi_new = (self.phi + p.dt * self.attention * (dphi + memory_bias)
                   + p.sigma_noise * np.sqrt(p.dt) * eta)
        # --- membrane amplitude exchange, E-gated by phase relatedness ---
        self._last_transport = 0.0
        if phi_other is not None and mu != 0.0:
            d = wrap_diff(theta_other, self.theta)
            cg = coupling_gate(d, coupling_kind)
            transport = mu * cg * (phi_other - self.phi)
            phi_new = phi_new + p.dt * transport
            self._last_transport = float(np.mean(np.abs(transport)))  # ⟨mu·g·|Δφ|⟩

        # soft ceiling is the 'B' op above; here the hard clip (non-negativity
        # floor + amplitude ceiling) -- pde.py:step's xp.clip(.,0,B_max-1e-4)
        self.phi = np.clip(phi_new, 0.0, p.B_max - 1e-4)

        self._phase_step(theta_other=theta_other, mu=mu,
                         coupling_kind=coupling_kind, gate_phase=gate_phase)
        self.theta_prev = self.theta.copy()
        self.tau = vorticity(self.phi, self.theta)
        self.step_count += 1


# --------------------------------------------------------------------------- #
#  Bilateral field -- two mirror lobes joined by a permeable membrane
# --------------------------------------------------------------------------- #
class BilateralField:
    """Two coupled lobes (L,R). Mirror-symmetric init: phi_R = phi_L,
    theta_R = -theta_L  (=> grad^2 theta_L = -grad^2 theta_R, the bilateral
    derivation's ansatz). Each step both lobes exchange across a membrane of
    permeability mu. Lobe L is seeded with the SAME seed as the single-field
    baseline so single-vs-bilateral is a fair fight."""

    def __init__(self, size: int = 64, params: Optional[PDEParams] = None,
                 seed: int = 42, mu: float = 0.1,
                 phi_init: float = 0.25, phi_jitter: float = 0.15):
        self.mu = mu
        self.size = size
        # Lobe L: identical construction to the single baseline (same seed).
        self.L = SingleField(size, params, seed=seed,
                             phi_init=phi_init, phi_jitter=phi_jitter)
        # Lobe R: mirror of L. Independent noise stream (derived seed) so the
        # membrane has two genuinely distinct lobes to play off each other.
        self.R = SingleField(size, params, seed=seed + 10_000,
                             phi_init=phi_init, phi_jitter=phi_jitter)
        # impose mirror symmetry on the STATE (overwrite R's random init)
        self.R.phi = self.L.phi.copy()
        self.R.theta = (-self.L.theta) % (2 * np.pi)
        self.R.omega0 = -self.L.omega0          # mirrored intrinsic frequencies
        self.R.phi_prev = self.R.phi.copy()
        self.R.theta_prev = self.R.theta.copy()
        self.R._lc = local_coherence(self.R.theta)
        self.R.tau = vorticity(self.R.phi, self.R.theta)
        self.step_count = 0

    def step(self):
        # snapshot BEFORE either lobe updates, so the exchange is symmetric
        phi_L, theta_L = self.L.phi.copy(), self.L.theta.copy()
        phi_R, theta_R = self.R.phi.copy(), self.R.theta.copy()
        self.L.step_with_coupling(phi_R, theta_R, self.mu)
        self.R.step_with_coupling(phi_L, theta_L, self.mu)
        self.step_count += 1

    # combined readout: averaged descriptor across the two lobes
    @property
    def phi(self):
        return 0.5 * (self.L.phi + self.R.phi)

    @property
    def theta(self):
        # circular mean of the two lobe phases
        return np.angle(np.exp(1j * self.L.theta) + np.exp(1j * self.R.theta)) % (2 * np.pi)
