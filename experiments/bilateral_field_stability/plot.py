"""Plot collapse-fraction vs mu (bilateral lobe-L) against the single-field
baseline. Reads results/sweep_<tag>.json. Saves results/collapse_vs_mu_<tag>.png."""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


def main(tag="full"):
    with open(os.path.join(OUT, f"sweep_{tag}.json")) as fh:
        r = json.load(fh)

    s = r["single"]["collapse_fraction"]
    mus = sorted(float(k) for k in r["bilateral"].keys())
    L = [r["bilateral"][f"{m}"]["lobe_L"]["collapse_fraction"] for m in mus]
    C = [r["bilateral"][f"{m}"]["combined"]["collapse_fraction"] for m in mus]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhline(s, color="crimson", ls="--", lw=2,
               label=f"single field (baseline) = {s:.0%}")
    ax.plot(mus, L, "o-", color="navy", lw=2, ms=8,
            label="bilateral lobe-L (same-seed control)")
    ax.plot(mus, C, "s:", color="gray", lw=1.5, ms=6, alpha=0.7,
            label="bilateral combined (averaged readout)")

    # shade any mu where bilateral beats single
    for i, m in enumerate(mus):
        if L[i] < s - 1e-9:
            ax.axvspan(m - 0.012, m + 0.012, color="lightgreen", alpha=0.25, zorder=0)

    ax.set_xlabel("membrane permeability  μ")
    ax.set_ylabel("collapse fraction within T steps")
    ax.set_title(f"Bilateral coherence-field stability  (n={r['config']['nseed']} seeds, "
                 f"T={r['config']['T']})")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    n = r["config"]["nseed"]
    for m, l in zip(mus, L):
        ax.annotate(f"{l:.0%}", (m, l), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color="navy")
    fig.tight_layout()
    p = os.path.join(OUT, f"collapse_vs_mu_{tag}.png")
    fig.savefig(p, dpi=130)
    print("wrote", p)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
