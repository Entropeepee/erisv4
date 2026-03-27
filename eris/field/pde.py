"""
FRACTAL PDE — Field Evolution Substrate
=========================================

Three coupled fields describe the information state:
    φ (phi)   : Coherence — how "sure" the system is at each point
    θ (theta) : Phase — what the system "thinks" at each point
    τ (tau)   : Torsion — where certainty is curving (contradictions)

The PDE system:
    ∂φ/∂t = D(1+C)∇²φ + E·max(0, φ - φ_prev) - D_decay·φ
    ∂θ/∂t = ω - λ(∇φ · ∇θ)
    τ = ∇²φ

Where:
    D       = base diffusion (how fast coherence spreads)
    C       = criticality coupling (amplifies diffusion near phase transitions)
    E       = emergence rate (positive novelty reinforcement)
    D_decay = decay constant (coherence naturally fades)
    ω       = intrinsic frequency (phase drift)
    λ       = advection coupling (phi gradients steer theta)

Boundary conditions: Dirichlet (phi=0 at edges). This is essential —
it creates the boundary pressure that feeds the B activation.

Uses cp.roll stencils for Laplacian (NOT cupyx.scipy.ndimage, which is
broken on CUDA 13.2). This is the same technique validated in the 32³
BLECD pure-TT engine.

Usage:
    from eris.field.pde import FractalField, PDEParams

    field = FractalField(size=64)
    field.seed_from_text("Hello world")
    for _ in range(50):
        field.step()
    bvec = field.compute_bvec()  # → BVec
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import hashlib
import time

from eris.config import xp, to_numpy, to_gpu, CONFIG
from eris.computation.activations import BVec, compute_bvec_from_field
import numpy as np


@dataclass
class PDEParams:
    """FRACTAL PDE parameters.

    Tune these to control field behavior. Defaults produce a
    stable reaction-diffusion system with rich dynamics.
    """
    D: float = 0.1          # Base diffusion coefficient
    C_coupling: float = 0.5  # Criticality amplification of diffusion
    E_rate: float = 0.3      # Emergence reinforcement strength
    D_decay: float = 0.05    # Natural coherence decay rate
    omega: float = 0.1       # Intrinsic phase frequency
    lam: float = 0.5         # Advection coupling (phi gradients → theta)
    dt: float = 0.01         # Timestep

    # Memory modulator (validated independent of Decay — March 2026)
    memory_tau: float = 10.0  # Forgetting timescale (steps). Higher = longer memory.
    memory_coupling: float = 0.1  # How strongly memory influences phi evolution

    # Attention modulator (validated independent of Boundary — March 2026)
    attention_strength: float = 0.3  # Gain field multiplier


def _laplacian_2d(field):
    """2D discrete Laplacian via cp.roll stencils.

    Uses the standard 5-point stencil:
        ∇²f ≈ f(x+1,y) + f(x-1,y) + f(x,y+1) + f(x,y-1) - 4f(x,y)

    cp.roll wraps around (periodic), but we zero out the boundary
    afterward to enforce Dirichlet conditions.
    """
    return (
        xp.roll(field, 1, axis=0) +
        xp.roll(field, -1, axis=0) +
        xp.roll(field, 1, axis=1) +
        xp.roll(field, -1, axis=1) -
        4.0 * field
    )


def _enforce_dirichlet(field):
    """Zero out boundary cells (Dirichlet boundary condition).

    This creates the boundary pressure that feeds the B domain activation.
    The field "wants" to be nonzero in the interior but is forced to zero
    at the edges — the resulting gradient is the constraint signal.
    """
    field[0, :] = 0.0
    field[-1, :] = 0.0
    field[:, 0] = 0.0
    field[:, -1] = 0.0
    return field


class FractalField:
    """FRACTAL PDE field state and evolution.

    Holds phi, theta, tau as 2D GPU arrays and implements the
    coupled PDE system. All heavy computation runs on GPU via CuPy;
    falls back to NumPy transparently.

    Parameters
    ----------
    size : int
        Grid resolution (NxN). Default 64. Use 32 for fast iteration,
        128 for high-fidelity, 256 for offline processing.
    params : PDEParams, optional
        PDE coefficients. Uses defaults if not provided.
    """

    def __init__(self, size: int = 64, params: Optional[PDEParams] = None):
        self.size = size
        self.params = params or PDEParams()
        self.step_count: int = 0
        self.wall_time_start: float = time.time()

        # Initialize fields — small random perturbation around zero
        rng = np.random.default_rng(42)
        self.phi = to_gpu(rng.uniform(0.0, 0.05, (size, size)).astype(np.float32))
        self.theta = to_gpu(rng.uniform(0.0, 2 * np.pi, (size, size)).astype(np.float32))
        self.phi_prev = xp.copy(self.phi)
        self.tau = xp.zeros((size, size), dtype=xp.float32)

        # Memory modulator: temporal persistence via exponential kernel
        # m(x,t) governed by τ_m · ∂m/∂t = -m + φ
        # Independent of Decay (validated March 2026 Alienware sims)
        self.memory = xp.zeros((size, size), dtype=xp.float32)

        # Attention modulator: multiplicative spatial gain field
        # A(x,t) = softmax(|∇φ|) — concentrates processing on high-gradient regions
        # Independent of Boundary (validated March 2026 Alienware sims)
        self.attention = xp.ones((size, size), dtype=xp.float32)

        # Global observables (dCdX derivation — the conservation law)
        self._coherence_history: list = []  # C(t) time series
        self._exchange_history: list = []   # X(t) time series
        self._dCdX_history: list = []       # dC/dX ratio time series

        # Enforce boundary conditions on initial state
        _enforce_dirichlet(self.phi)
        _enforce_dirichlet(self.phi_prev)

    def step(self) -> None:
        """Advance the PDE by one timestep.

        This is the core physics loop. Each call:
        1. Saves current phi as phi_prev
        2. Pre-zeros boundaries before Laplacian (prevents wrap contamination)
        3. Computes torsion (∇²φ)
        4. Updates memory modulator (exponential kernel)
        5. Computes attention gain field (gradient magnitude)
        6. Updates phi via reaction-diffusion with BLECD couplings + modulators
        7. Updates theta via advection
        8. Computes global observables C(t), X(t), dC/dX
        9. Clamps phi to [0, 1] and enforces boundary conditions
        """
        p = self.params
        phi = self.phi
        theta = self.theta

        # Save previous state (for E and D activations)
        self.phi_prev = xp.copy(phi)

        # Pre-zero boundaries BEFORE Laplacian to prevent wrap contamination
        # (cp.roll wraps around; pre-zeroing ensures the Laplacian at
        # interior boundary-adjacent cells sees zeros, not opposite-edge values)
        _enforce_dirichlet(phi)

        # Compute torsion (Laplacian of phi)
        self.tau = _laplacian_2d(phi)

        # ── Memory modulator ──────────────────────────────────────────
        # Exponential kernel: τ_m · ∂m/∂t = -m + φ
        # m integrates phi over time with forgetting timescale memory_tau.
        # This is orthogonal to Decay: Decay affects amplitude,
        # Memory affects trajectory correlation (subdiffusion at short times).
        alpha_m = p.dt / max(p.memory_tau, p.dt)  # EMA smoothing
        self.memory = (1.0 - alpha_m) * self.memory + alpha_m * phi

        # ── Attention modulator ───────────────────────────────────────
        # Multiplicative gain field: A(x,t) ∝ softmax(|∇φ|)
        # Concentrates processing on high-gradient regions.
        # This is the meditation connection: the "silent witness" choosing
        # where to project attention on the coupling sphere.
        gphi_x = phi - xp.roll(phi, 1, axis=1)
        gphi_y = phi - xp.roll(phi, 1, axis=0)
        grad_mag = xp.sqrt(gphi_x**2 + gphi_y**2 + 1e-10)
        # Normalize to [0, 1] range, then scale to [1-s, 1+s]
        grad_norm = grad_mag / xp.maximum(xp.max(grad_mag), xp.float32(1e-10))
        self.attention = 1.0 + p.attention_strength * (grad_norm - 0.5)

        # ── Phi evolution ─────────────────────────────────────────────
        # Criticality-modulated diffusion (near phase transitions, amplified)
        effective_D = p.D * (1.0 + p.C_coupling * xp.abs(self.tau))

        # Emergence term: reinforce positive novelty
        novelty = xp.maximum(0.0, phi - self.phi_prev)
        emergence = p.E_rate * novelty

        # Memory influence: past coherence patterns bias current evolution
        memory_bias = p.memory_coupling * (self.memory - phi)

        # Full phi update: diffusion + emergence - decay + memory, all × attention
        dphi = self.attention * (effective_D * self.tau + emergence - p.D_decay * phi + memory_bias)
        phi_new = phi + p.dt * dphi

        # ── Theta evolution ───────────────────────────────────────────
        gtheta_x = theta - xp.roll(theta, 1, axis=1)
        gtheta_y = theta - xp.roll(theta, 1, axis=0)
        advection = gphi_x * gtheta_x + gphi_y * gtheta_y
        dtheta = p.omega - p.lam * advection
        theta_new = theta + p.dt * dtheta

        # Clamp phi to [0, 1] and enforce boundary conditions
        self.phi = xp.clip(phi_new, 0.0, 1.0)
        _enforce_dirichlet(self.phi)

        # Wrap theta to [0, 2π]
        self.theta = theta_new % (2.0 * float(np.pi))

        # Recompute torsion from updated phi
        self.tau = _laplacian_2d(self.phi)

        # ── Global observables (dCdX conservation law) ────────────────
        self._update_global_observables()

        self.step_count += 1

    def run(self, n_steps: int) -> None:
        """Run multiple PDE steps."""
        for _ in range(n_steps):
            self.step()

    def compute_bvec(self) -> BVec:
        """Compute the current BFECDS activation vector from field state."""
        return compute_bvec_from_field(self.phi, self.theta, self.tau, self.phi_prev)

    # ─── Global Observables (dCdX Conservation Law) ───────────────────

    def _update_global_observables(self) -> None:
        """Compute and store C(t), X(t), dC/dX.

        From the dCdX derivation:
            C(t) = (1/L) |∫ e^{iθ(x,t)} dx|   (Kuramoto order parameter)
            X(t) = ∫ φ(x,t) · ∂_x θ(x,t) dx    (exchange)
            dC/dX = f(BLECD) + T_eff             (conservation law)

        C measures global phase alignment.
        X measures the flux of coherence through phase gradients.
        dC/dX tells you what regime the system is in:
            - Elastic: smooth coupling, incremental learning
            - Plastic: coupling exceeds κ, restructuring required
            - Incomprehensible: zero torsion despite nonzero input = transfixion
        """
        phi_np = to_numpy(self.phi)
        theta_np = to_numpy(self.theta)

        N = phi_np.size

        # C(t): Kuramoto order parameter — global phase coherence
        # Z = (1/N) Σ e^{iθ}, C = |Z|
        z = np.mean(np.exp(1j * theta_np))
        C = float(np.abs(z))

        # X(t): Exchange — ∫ φ · ∂_x θ dx (finite difference for gradient)
        gtheta_x = np.diff(theta_np, axis=1, prepend=theta_np[:, -1:])
        # Handle phase wrapping in gradient
        gtheta_x = np.angle(np.exp(1j * gtheta_x))  # wrap to [-π, π]
        X = float(np.sum(phi_np * gtheta_x))

        self._coherence_history.append(C)
        self._exchange_history.append(X)

        # dC/dX: finite difference of last two values
        if len(self._coherence_history) >= 2 and len(self._exchange_history) >= 2:
            dC = self._coherence_history[-1] - self._coherence_history[-2]
            dX = self._exchange_history[-1] - self._exchange_history[-2]
            dCdX = dC / dX if abs(dX) > 1e-10 else 0.0
            self._dCdX_history.append(dCdX)

    @property
    def coherence(self) -> float:
        """Current global coherence C(t) — Kuramoto order parameter."""
        return self._coherence_history[-1] if self._coherence_history else 0.0

    @property
    def exchange(self) -> float:
        """Current exchange X(t) — coherence flux through phase gradients."""
        return self._exchange_history[-1] if self._exchange_history else 0.0

    @property
    def dCdX(self) -> float:
        """Current dC/dX ratio — the conservation law diagnostic.

        From the Law of Change: ∂C/∂X = ∂X/∂C (reciprocity).
        From dCdX derivation: dC/dX = f(BLECD) + T_eff.

        Regime detection (information theory session):
            Elastic:          |dC/dX| moderate, smooth evolution
            Plastic:          |dC/dX| large, torsion-driven restructuring
            Incomprehensible: dC/dX ≈ 0 with nonzero input → transfixion
        """
        return self._dCdX_history[-1] if self._dCdX_history else 0.0

    def detect_regime(self) -> str:
        """Detect the current information-processing regime.

        Based on the conservation law and information theory framework:
        - Elastic: observer can absorb input without restructuring
        - Plastic: input demands restructuring (learning regime)
        - Transfixed: confident output with no genuine processing (hallucination)
        """
        if len(self._dCdX_history) < 5:
            return "warmup"

        recent_dCdX = self._dCdX_history[-5:]
        mean_abs = np.mean([abs(x) for x in recent_dCdX])
        recent_tau_rms = float(to_numpy(xp.sqrt(xp.mean(self.tau ** 2))))

        if mean_abs < 0.01 and recent_tau_rms < 0.01:
            return "transfixed"  # Zero coupling, zero processing → hallucination risk
        elif mean_abs > 0.5:
            return "plastic"     # High dC/dX → restructuring / genuine learning
        else:
            return "elastic"     # Moderate → smooth incremental update

    def seed_from_text(self, text: str, use_frt: bool = False) -> None:
        """Seed the phi field from text structure.

        Two pathways (dual-process architecture):
            use_frt=False (default): Character-level statistics → spatial geometry.
                Slower but captures more structure. Will be upgraded to
                coupling geometry when embedding model is available.
            use_frt=True: Fractal Rolling Tokenizer → hash → bit-slice.
                Instant, deterministic, no GPU. Originally built for
                GTX 970. Use for real-time chat or low-power hardware.

        The same text always produces the same initial field (deterministic).
        """
        if not text.strip():
            return

        if use_frt:
            from eris.field.frt import text_to_field_arrays
            phi_seed, theta_seed = text_to_field_arrays(text, size=self.size)
            self.phi = to_gpu(phi_seed)
            self.theta = to_gpu(theta_seed)
            self.phi_prev = xp.copy(self.phi)
            self.tau = _laplacian_2d(self.phi)
            self.step_count = 0
            return

        rng_seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(rng_seed)

        size = self.size

        # Character frequency spectrum → base energy
        chars = np.array([ord(c) for c in text[:1000]], dtype=np.float32)
        if len(chars) == 0:
            return

        # Normalized histogram of character codes (modular mapping to grid)
        hist, _ = np.histogram(chars % size, bins=size, range=(0, size))
        hist = hist.astype(np.float32)
        hist /= max(hist.max(), 1e-10)

        # Create 2D field from 1D spectrum via outer product with word-length signal
        words = text.split()
        word_lens = np.array([len(w) for w in words[:size]], dtype=np.float32)
        if len(word_lens) < size:
            word_lens = np.pad(word_lens, (0, size - len(word_lens)))
        word_lens = word_lens[:size]
        word_lens /= max(word_lens.max(), 1e-10)

        phi_seed = np.outer(word_lens, hist).astype(np.float32)

        # Scale to reasonable initial amplitude [0, 0.3]
        phi_seed *= 0.3

        # Add small noise for symmetry breaking
        phi_seed += rng.uniform(0, 0.02, (size, size)).astype(np.float32)

        self.phi = to_gpu(phi_seed)
        _enforce_dirichlet(self.phi)

        # Reset theta with text-derived phase
        theta_seed = rng.uniform(0, 2 * np.pi, (size, size)).astype(np.float32)
        self.theta = to_gpu(theta_seed)

        # Reset history
        self.phi_prev = xp.copy(self.phi)
        self.tau = _laplacian_2d(self.phi)
        self.step_count = 0

    def seed_from_fingerprint(self, fingerprint: str) -> None:
        """Seed from an identity fingerprint (hex string).

        Used for identity-driven seeding — the system's "self" pattern.
        Every Eris instance has a unique fingerprint that creates a
        characteristic field geometry.
        """
        self.seed_from_text(f"IDENTITY:{fingerprint}")

    # ─── Checkpointing ────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Serialize full field state for checkpoint/resume."""
        return {
            "size": self.size,
            "step_count": self.step_count,
            "phi": to_numpy(self.phi),
            "theta": to_numpy(self.theta),
            "tau": to_numpy(self.tau),
            "phi_prev": to_numpy(self.phi_prev),
            "memory": to_numpy(self.memory),
            "attention": to_numpy(self.attention),
            "coherence_history": self._coherence_history[-100:],  # Last 100 values
            "exchange_history": self._exchange_history[-100:],
            "dCdX_history": self._dCdX_history[-100:],
            "params": {
                "D": self.params.D,
                "C_coupling": self.params.C_coupling,
                "E_rate": self.params.E_rate,
                "D_decay": self.params.D_decay,
                "omega": self.params.omega,
                "lam": self.params.lam,
                "dt": self.params.dt,
                "memory_tau": self.params.memory_tau,
                "memory_coupling": self.params.memory_coupling,
                "attention_strength": self.params.attention_strength,
            },
        }

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "FractalField":
        """Restore from checkpoint."""
        params = PDEParams(**data["params"])
        field = cls(size=data["size"], params=params)
        field.step_count = data["step_count"]
        field.phi = to_gpu(data["phi"])
        field.theta = to_gpu(data["theta"])
        field.tau = to_gpu(data["tau"])
        field.phi_prev = to_gpu(data["phi_prev"])
        return field

    def save_checkpoint(self, path: str) -> None:
        """Save state to .npz file."""
        snap = self.snapshot()
        # Flatten params dict into the save
        params = snap.pop("params")
        np.savez_compressed(
            path,
            phi=snap["phi"],
            theta=snap["theta"],
            tau=snap["tau"],
            phi_prev=snap["phi_prev"],
            size=np.array([snap["size"]]),
            step_count=np.array([snap["step_count"]]),
            param_D=np.array([params["D"]]),
            param_C=np.array([params["C_coupling"]]),
            param_E=np.array([params["E_rate"]]),
            param_Dd=np.array([params["D_decay"]]),
            param_omega=np.array([params["omega"]]),
            param_lam=np.array([params["lam"]]),
            param_dt=np.array([params["dt"]]),
        )

    @classmethod
    def load_checkpoint(cls, path: str) -> "FractalField":
        """Load from .npz file."""
        data = np.load(path)
        params = PDEParams(
            D=float(data["param_D"].item()),
            C_coupling=float(data["param_C"].item()),
            E_rate=float(data["param_E"].item()),
            D_decay=float(data["param_Dd"].item()),
            omega=float(data["param_omega"].item()),
            lam=float(data["param_lam"].item()),
            dt=float(data["param_dt"].item()),
        )
        field = cls(size=int(data["size"].item()), params=params)
        field.step_count = int(data["step_count"].item())
        field.phi = to_gpu(data["phi"])
        field.theta = to_gpu(data["theta"])
        field.tau = to_gpu(data["tau"])
        field.phi_prev = to_gpu(data["phi_prev"])
        return field
