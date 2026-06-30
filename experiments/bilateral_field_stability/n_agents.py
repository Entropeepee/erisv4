"""
R2 -- N-lobe generalization (bifurcation n=2  <->  hive n=N).

NAgents: N distinct, non-mirror lobes (independent seeds/noise, spread intrinsic
frequencies), every coupled pair joined by an E-gated membrane (ring or all-to-all).
Reuses field_core.SingleField + the multi-neighbour `others` path verbatim.

Tests:
  nness   : settle, measure mean pairwise overlap angle; ATTRACTOR test (kick toward
            global sameness AND toward decoherence; does the interior config return?).
  budget  : conservative-limit check that the N-channel total Sum_i Sum_cells phi_i is
            conserved by the N membranes (R1 analog).
Sweep N=3,4,6 x mu; report the sustained-N band vs N.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bifurcate as bf
from field_core import PDEParams

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUTDIR = os.path.join(RES, "nagents")
os.makedirs(OUTDIR, exist_ok=True)
PHI_INIT, JITTER, SEED_OFFSET = 0.85, 0.04, 20000
SIG = 0.016
DOMAINS = ["S", "D", "F", "TH", "B", "E", "IBT", "ZBT"]


def regime(sigma=SIG, conservative=False):
    p = PDEParams(); p.r_sat = 0.85; p.omega_spread = 0.25
    if conservative:
        p.activations = {k: 0.0 for k in DOMAINS}; p.memory_coupling = 0.0
        p.sigma_noise = 0.0; p.sigma_phase = 0.0; p.d_decay = 0.0
    else:
        p.d_decay = 0.20; p.sigma_noise = sigma; p.sigma_phase = sigma
    return p


class NAgents:
    def __init__(self, size, params, N, mu, kind, delta_spread, seed,
                 gate_phase=True, topology="all"):
        self.lobes = [bf.cc.SingleField(size, params, seed=seed + i * SEED_OFFSET,
                                        phi_init=PHI_INIT, phi_jitter=JITTER) for i in range(N)]
        # spread intrinsic-frequency means linearly across the lobes (distinct, non-mirror)
        for i, f in enumerate(self.lobes):
            off = delta_spread * ((i / (N - 1)) - 0.5) if N > 1 else 0.0
            f.omega0 = f.omega0 + off
        self.N, self.mu, self.kind, self.gate_phase = N, mu, kind, gate_phase
        if topology == "ring":
            self.nbr = [[(i - 1) % N, (i + 1) % N] for i in range(N)]
        else:
            self.nbr = [[j for j in range(N) if j != i] for i in range(N)]

    def step(self):
        snaps = [(f.phi.copy(), f.theta.copy()) for f in self.lobes]
        for i, f in enumerate(self.lobes):
            others = [snaps[j] for j in self.nbr[i]]
            f.step_with_coupling(None, None, self.mu, coupling_kind=self.kind,
                                 gate_phase=self.gate_phase, others=others)

    def _angle(self, i, j):
        a, b = self.lobes[i], self.lobes[j]
        O = np.sum(a.phi * b.phi * np.exp(1j * (a.theta - b.theta)))
        nrm = np.sqrt(np.sum(a.phi ** 2)) * np.sqrt(np.sum(b.phi ** 2)) + 1e-12
        return float(np.degrees(np.arccos(np.clip(np.real(O) / nrm, -1, 1))))

    def pairwise_mean(self):
        angs = [self._angle(i, j) for i in range(self.N) for j in self.nbr[i] if j > i]
        return float(np.mean(angs)), float(np.min(angs)), float(np.max(angs))

    def total_amplitude(self):
        return float(sum(np.sum(f.phi) for f in self.lobes))

    def snapshot(self):
        return [bf.TwoAgents._snap_field(f) for f in self.lobes]

    def restore(self, snap):
        for f, s in zip(self.lobes, snap):
            bf.TwoAgents._restore_field(f, s)


def attractor_N(N, mu, kind, delta_spread, seed, topology="all",
                T_settle=1500, T_relax=1000):
    p = regime()
    na = NAgents(64, p, N, mu, kind, delta_spread, seed, gate_phase=True, topology=topology)
    for _ in range(T_settle):
        na.step()
    base = np.mean([na.pairwise_mean()[0] for _ in _peekN(na, 40)])
    snap = na.snapshot()

    def relax(perturb):
        na.restore(snap); perturb(na)
        vals = []
        for t in range(T_relax):
            na.step()
            if t % 10 == 0:
                vals.append(na.pairwise_mean()[0])
        return float(np.mean(vals[-10:]))

    # toward global sameness: all lobes take lobe-0's phase
    def kick_same(a):
        th0 = a.lobes[0].theta.copy()
        for f in a.lobes:
            f.theta = th0.copy()
    # toward decoherence: each lobe gets an independent uniform phase offset
    def kick_deco(a):
        for k, f in enumerate(a.lobes):
            f.theta = (f.theta + (k + 1) * np.pi / (a.N + 1)) % (2 * np.pi)
    after_same = relax(kick_same)
    after_deco = relax(kick_deco)
    return {"N": N, "mu": mu, "kind": kind, "topology": topology, "seed": seed,
            "theta_star": float(base), "after_same": after_same, "after_deco": after_deco}


def _peekN(na, n):
    for _ in range(n):
        na.step(); yield None


def test_budget(N, mu, kind="egate", delta_spread=0.2, seed=0, T=800):
    p = regime(conservative=True)
    na = NAgents(64, p, N, mu, kind, delta_spread, seed, gate_phase=True)
    Q0 = na.total_amplitude(); drift = []; spread = []
    for t in range(T):
        na.step(); drift.append((na.total_amplitude() - Q0) / Q0)
        spread.append(na.pairwise_mean()[0])
    return {"N": N, "kind": kind, "Q_max_drift": float(np.max(np.abs(drift)))}


def main():
    cmd = sys.argv[1]
    def opt(n, d, c=float):
        return c(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d
    if cmd == "attractor":
        N = int(opt("--N", 3, int)); mu = opt("--mu", 0.9); kind = sys.argv[sys.argv.index("--kind")+1] if "--kind" in sys.argv else "egate"
        ds = opt("--ds", 0.2); topo = sys.argv[sys.argv.index("--topo")+1] if "--topo" in sys.argv else "all"
        nseed = int(opt("--seeds", 8, int))
        key = f"natt_{kind}_N{N}_m{mu}_ds{ds}_{topo}"
        path = os.path.join(OUTDIR, key + ".json")
        st = json.load(open(path)) if os.path.exists(path) else {"records": []}
        done = {r["seed"] for r in st["records"]}
        for s in range(nseed):
            if s in done:
                continue
            st["records"].append(attractor_N(N, mu, kind, ds, s, topology=topo))
            json.dump(st, open(path, "w"))
        rs = st["records"]
        ts = np.mean([r["theta_star"] for r in rs]); sa = np.mean([r["after_same"] for r in rs])
        sd = np.mean([r["after_deco"] for r in rs])
        # interior-return pass rate
        pr = np.mean([(abs(r["after_same"]-r["after_deco"])<15 and 15<(r["after_same"]+r["after_deco"])/2<80) for r in rs])
        print(f"[{key}] theta*={ts:.1f} <-same={sa:.1f} <-deco={sd:.1f} interior-return={pr:.0%} n={len(rs)}", flush=True)
    elif cmd == "budget":
        for N in [3, 4, 6]:
            r = test_budget(N, opt("--mu", 0.9))
            print(f"  N={N} egate conservative: Q max drift = {r['Q_max_drift']:.2e}", flush=True)


if __name__ == "__main__":
    main()
