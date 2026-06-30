"""
analyze_bifurcate.py -- delta/mu regime map + egate-vs-cos contrast.

Reads results/bifurcate/<kind>_gp_d<delta>_m<mu>_s<sigma>.json. Produces:
  * per-cell raw table (mean theta_LR, std, both-alive%, transport, class)
  * delta/mu heatmaps of theta_LR for egate vs cos (the headline map)
  * sustained-cell count per arm + theta_LR-vs-mu curves at each delta
  * stability (within-run std of theta_LR) -- a real attractor has low drift
All threshold-free where it matters (theta_LR is an angle, not an eps-cut).
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CDIR = os.path.join(HERE, "results", "bifurcate")
RES = os.path.join(HERE, "results")
SIG = "0.016"
DELTAS = [0.05, 0.1, 0.2, 0.4]
MUS = [0.4, 0.6, 0.9, 1.3, 1.8]


def load(kind, d, m, gp=True):
    key = f"{kind}{'_gp' if gp else ''}_d{d}_m{m}_s{SIG}"
    p = os.path.join(CDIR, key + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def cell_stats(kind, d, m):
    j = load(kind, d, m)
    if not j or not j["records"]:
        return None
    r = j["records"]
    return {
        "n": len(r),
        "mean_theta": float(np.mean([x["mean_theta_LR"] for x in r])),
        "std_theta_within": float(np.mean([x["std_theta_LR"] for x in r])),  # drift (stability)
        "std_theta_across": float(np.std([x["mean_theta_LR"] for x in r])),   # seed spread
        "both_alive": float(np.mean([x["both_alive"] for x in r])),
        "transport": float(np.mean([x["mean_transport_post"] for x in r])),
        "class": Counter(x["class"] for x in r).most_common(1)[0][0],
        "class_dist": dict(Counter(x["class"] for x in r)),
        "thetas": [x["mean_theta_LR"] for x in r],
    }


def main():
    print("=" * 88)
    print(f"BIFURCATE delta/mu map  (sigma={SIG}, gate_phase=True)")
    print("=" * 88)
    summary = {"egate": {}, "cos": {}, "diff": {}, "iso": {}}

    for kind in ["egate", "cos"]:
        print(f"\n### {kind.upper()}  (mean theta_LR deg | within-run drift std | both-alive | class)")
        print(f"{'mu\\delta':>9} | " + " | ".join(f"d={d}" for d in DELTAS))
        for m in MUS:
            cells = [cell_stats(kind, d, m) for d in DELTAS]
            row = f"{m:>9} | "
            for c in cells:
                if c:
                    row += f"{c['mean_theta']:5.0f}/{c['std_theta_within']:4.1f}/{c['both_alive']*100:3.0f}% |"
                else:
                    row += "   --     |"
            print(row)
            for d, c in zip(DELTAS, cells):
                if c:
                    summary[kind][f"d{d}_m{m}"] = c

    # sustained-cell tally
    print("\n### Sustained-cell tally (class=='sustained', both-alive, interior theta, live transport)")
    for kind in ["egate", "cos"]:
        sus = [(d, m) for d in DELTAS for m in MUS
               if (c := cell_stats(kind, d, m)) and c["class"] == "sustained"
               and c["both_alive"] >= 0.75 and c["transport"] > 1e-5]
        print(f"  {kind:6s}: {len(sus)} sustained cells -> {sus}")
        summary.setdefault("sustained", {})[kind] = sus

    # diff/iso references
    print("\n### references")
    for m in MUS:
        c = cell_stats("diff", 0.1, m)
        if c:
            print(f"  diff d=0.1 m={m}: theta={c['mean_theta']:.0f} class={c['class']}")
    for d in DELTAS:
        c = cell_stats("iso", d, 0.0)
        if c:
            print(f"  iso  d={d}: theta={c['mean_theta']:.0f} class={c['class']}")

    json.dump(summary, open(os.path.join(RES, "bifurcate_summary.json"), "w"), indent=1, default=float)
    make_plots()
    print("\nwrote results/bifurcate_summary.json + plots")


def make_plots():
    # heatmaps of mean theta_LR for egate vs cos
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, kind in zip(axes, ["egate", "cos"]):
        grid = np.full((len(MUS), len(DELTAS)), np.nan)
        for i, m in enumerate(MUS):
            for j, d in enumerate(DELTAS):
                c = cell_stats(kind, d, m)
                if c:
                    grid[i, j] = c["mean_theta"]
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=120)
        ax.set_xticks(range(len(DELTAS))); ax.set_xticklabels(DELTAS)
        ax.set_yticks(range(len(MUS))); ax.set_yticklabels(MUS)
        ax.set_xlabel("detuning delta"); ax.set_title(f"{kind}: mean theta_LR (deg)")
        for i, m in enumerate(MUS):
            for j, d in enumerate(DELTAS):
                c = cell_stats(kind, d, m)
                if c:
                    mark = {"sustained": "S", "fusion": "F", "segregation": "X"}[c["class"]]
                    ax.text(j, i, f"{grid[i,i if False else j]:.0f}\n{mark}", ha="center", va="center",
                            color="w" if grid[i, j] < 60 else "k", fontsize=8)
    axes[0].set_ylabel("coupling mu")
    fig.colorbar(im, ax=axes, label="theta_LR (deg): 0=fusion, 90=orthogonal")
    fig.suptitle("delta/mu regime map: S=sustained interior, F=fusion (theta->0), X=segregation")
    fig.savefig(os.path.join(RES, "bifurcate_map.png"), dpi=130, bbox_inches="tight")

    # theta_LR vs mu curves at each delta
    fig2, axes2 = plt.subplots(1, len(DELTAS), figsize=(16, 4), sharey=True)
    for ax, d in zip(axes2, DELTAS):
        for kind, col in [("egate", "navy"), ("cos", "crimson"), ("diff", "gray")]:
            xs, ys = [], []
            for m in MUS:
                c = cell_stats(kind, d, m)
                if c:
                    xs.append(m); ys.append(c["mean_theta"])
            if xs:
                ax.plot(xs, ys, "o-", color=col, label=kind, lw=2)
        ax.axhspan(20, 75, color="lightgreen", alpha=0.2)
        ax.set_title(f"delta={d}"); ax.set_xlabel("mu"); ax.grid(alpha=0.3)
    axes2[0].set_ylabel("mean theta_LR (deg)"); axes2[0].legend()
    fig2.suptitle("theta_LR vs mu (green band = sustained interior). egate plateaus; cos jumps through.")
    fig2.savefig(os.path.join(RES, "bifurcate_curves.png"), dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
