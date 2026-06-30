"""
analyze.py -- consume results/controls/*.json and produce the adversarial verdict.

T1: eps_lock sensitivity of the single-vs-bilateral gap.
T4: unimodality (Hartigan dip + GMM 1-vs-2 BIC) and hysteresis (kick persistence).
T2: matched white/colored noise + sigma-bump arms vs single & bilateral.
T3: sham partners (frozen / independent / mutual-non-mirror) vs TRUE bilateral,
    each with a two-proportion z-test against true bilateral (the discriminator).

Raw tables only; no decorative pass/fail. Writes results/controls_summary.json and
plots into results/.
"""
import os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CDIR = os.path.join(HERE, "results", "controls")
RES = os.path.join(HERE, "results")
THR_EPS_LOCK, THR_EPS_FLAT = 3e-5, 2e-4


def load(key):
    p = os.path.join(CDIR, key + ".json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def recs(key):
    st = load(key)
    return st["records"] if st else None


def frac_collapsed(key):
    r = recs(key)
    return (np.mean([x["collapsed"] for x in r]), len(r)) if r else (None, 0)


def tvars(key):
    r = recs(key)
    return np.array([x["tvar"] for x in r]) if r else None


def outcomes(key):
    from collections import Counter
    r = recs(key)
    return dict(Counter(x["outcome"] for x in r)) if r else None


def five_num(a):
    a = np.asarray(a)
    return (float(a.min()), float(np.percentile(a, 25)), float(np.median(a)),
            float(np.percentile(a, 75)), float(a.max()))


def ztest(x1, n1, x2, n2):
    """two-proportion two-sided z-test. x=count collapsed."""
    if n1 == 0 or n2 == 0:
        return None, None
    p1, p2 = x1 / n1, x2 / n2
    pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    pval = math.erfc(abs(z) / math.sqrt(2))
    return z, pval


def proxy_collapse(key, eps_lock=THR_EPS_LOCK, eps_flat=THR_EPS_FLAT):
    """Reproduce the LOCK condition from FINAL-state values (tvar<eps_lock AND
    svar>eps_flat). Approximation of the real n_consec detector; tracks it in this
    regime where structure persists."""
    r = recs(key)
    if not r:
        return None
    return float(np.mean([(x["tvar"] < eps_lock and x["svar"] > eps_flat) for x in r]))


# --------------------------------------------------------------------------- #
def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def main():
    summary = {}
    SB = "single_0.007"
    BL = "bilatL_0.1_0.007"

    # ---------- headline reproduction ----------
    section("HEADLINE (this pipeline, N=60, sigma=0.007)")
    fs, ns = frac_collapsed(SB)
    fb, nb = frac_collapsed(BL)
    print(f"single  sigma=0.007       collapse = {fs:.0%}  (n={ns})  outcomes={outcomes(SB)}")
    print(f"bilateral mu=0.1 lobe-L   collapse = {fb:.0%}  (n={nb})  outcomes={outcomes(BL)}")
    z, pv = ztest(round(fs * ns), ns, round(fb * nb), nb)
    print(f"single vs bilateral: z={z:.2f}  p={pv:.2e}")
    blmeta = load(BL)["meta"]
    print(f"mean RMS|phiL-phiR| = {blmeta.get('mean_rms_diff'):.4f}  "
          f"=> matched-noise amp = {blmeta.get('amp_for_match'):.5f}")
    summary["headline"] = {"single": fs, "bilateral_L": fb, "z": z, "p": pv,
                           "mean_rms_diff": blmeta.get("mean_rms_diff"),
                           "amp": blmeta.get("amp_for_match")}

    # ---------- T1: eps_lock sweep ----------
    section("T1 -- eps_lock sensitivity (gap = single - bilateral collapse %)")
    tv_s, tv_b = tvars(SB), tvars(BL)
    sv_s = np.array([x["svar"] for x in recs(SB)])
    sv_b = np.array([x["svar"] for x in recs(BL)])
    print(f"{'eps_lock':>10} | {'single%':>8} | {'bilat%':>7} | {'gap':>5}")
    t1 = []
    for el in np.linspace(2.0e-5, 4.0e-5, 11):
        s = float(np.mean((tv_s < el) & (sv_s > THR_EPS_FLAT)))
        b = float(np.mean((tv_b < el) & (sv_b > THR_EPS_FLAT)))
        print(f"{el:10.2e} | {s*100:8.0f} | {b*100:7.0f} | {(s-b)*100:5.0f}")
        t1.append({"eps_lock": el, "single": s, "bilat": b, "gap": s - b})
    summary["T1_epslock"] = t1
    # tvar five-number summaries
    print("\ntemporal_var_final five-number summary (min q25 med q75 max):")
    for k in [SB, BL]:
        print(f"  {k:24s} {tuple(round(v,3) for v in (np.array(five_num(tvars(k)))*1e5))}  (x1e-5)")

    # ---------- T4 unimodality ----------
    section("T4 -- unimodality of temporal_var_final (Hartigan dip + GMM BIC)")
    try:
        import diptest
        have_dip = True
    except Exception:
        have_dip = False
    from sklearn.mixture import GaussianMixture
    uni = {}
    for sig in ["0.006", "0.007", "0.008"]:
        k = f"single_{sig}"
        a = tvars(k)
        if a is None:
            continue
        x = (a * 1e5).reshape(-1, 1)
        g1 = GaussianMixture(1, random_state=0).fit(x)
        g2 = GaussianMixture(2, random_state=0).fit(x)
        bic1, bic2 = g1.bic(x), g2.bic(x)
        dip = diptest.diptest(a) if have_dip else (None, None)
        fc, _ = frac_collapsed(k)
        print(f"single sigma={sig}: collapse={fc:.0%}  dip_p={dip[1] if dip[1] is not None else 'NA'}"
              f"  BIC(1)={bic1:.1f} BIC(2)={bic2:.1f}  -> "
              f"{'BIMODAL-favored' if bic2 < bic1 - 6 else 'unimodal-favored'}")
        uni[sig] = {"collapse": fc, "dip_p": dip[1], "bic1": float(bic1), "bic2": float(bic2),
                    "five_num_x1e5": [round(v, 3) for v in (np.array(five_num(a)) * 1e5)]}
    summary["T4_unimodality"] = uni

    # ---------- T4 hysteresis ----------
    section("T4 -- hysteresis (kick persistence vs natural outcome, per seed)")
    nat = {x["seed"]: x["outcome"] for x in recs(SB)}
    for kick in ["kick_lock", "kick_alive"]:
        kk = f"{kick}_0.007"
        rk = recs(kk)
        if not rk:
            print(f"  {kk}: (missing)"); continue
        # natural class split
        flips = 0; n_consider = 0; detail = {"AliveToLock": 0, "LockToAlive": 0}
        for x in rk:
            s = x["seed"]; natc = nat.get(s)
            post = "LOCK" if x["collapsed"] else "ALIVE"
            natc2 = "LOCK" if natc != "ALIVE" else "ALIVE"
            if kick == "kick_lock" and natc2 == "ALIVE":
                n_consider += 1
                if post == "LOCK":
                    flips += 1; detail["AliveToLock"] += 1
            if kick == "kick_alive" and natc2 == "LOCK":
                n_consider += 1
                if post == "ALIVE":
                    flips += 1; detail["LockToAlive"] += 1
        frac = flips / n_consider if n_consider else float("nan")
        print(f"  {kick}: of {n_consider} seeds naturally "
              f"{'ALIVE' if kick=='kick_lock' else 'LOCK'}, "
              f"{flips} flipped & persisted ({frac:.0%})")
        summary.setdefault("T4_hysteresis", {})[kick] = {
            "n_consider": n_consider, "flipped": flips, "frac": frac}

    # ---------- T2 matched noise + sigma bump ----------
    section("T2 -- matched-noise & sigma-bump controls (collapse %)")
    print(f"  {'arm':28s} {'collapse%':>9}  outcomes")
    t2 = {}
    for k in ["single_0.007", "white_0.1_0.007", "colored_0.1_0.007",
              "single_0.0075", "single_0.008", "single_0.0085", "single_0.009",
              BL]:
        f, n = frac_collapsed(k)
        if f is None:
            continue
        print(f"  {k:28s} {f*100:8.0f}%  {outcomes(k)}")
        t2[k] = f
    summary["T2_matched"] = t2

    # ---------- T3 sham partners (decisive) ----------
    section("T3 -- sham partners vs TRUE bilateral (z-test EACH arm vs bilateral)")
    fb, nb = frac_collapsed(BL)
    xb = round(fb * nb)
    arms = [("single_0.007", "single baseline"),
            ("colored_0.1_0.007", "matched colored noise"),
            ("sham_frozen_0.1_0.007", "sham: frozen partner"),
            ("sham_indep_0.1_0.007", "sham: independent free partner"),
            ("sham_mutual_0.1_0.007", "sham: mutual NON-mirror partner"),
            (BL, "TRUE bilateral (mirror, mutual)")]
    print(f"  {'arm':34s} {'collapse%':>9} {'vs bilat z':>11} {'p':>10}")
    t3 = {}
    for k, label in arms:
        f, n = frac_collapsed(k)
        if f is None:
            print(f"  {label:34s}    (missing)"); continue
        if k == BL:
            print(f"  {label:34s} {f*100:8.0f}%   {'-':>10} {'-':>10}")
        else:
            z, p = ztest(round(f * n), n, xb, nb)
            print(f"  {label:34s} {f*100:8.0f}%  {z:>10.2f} {p:>10.2e}")
            t3[k] = {"collapse": f, "z_vs_bilat": z, "p_vs_bilat": p}
    summary["T3_sham"] = t3

    json.dump(summary, open(os.path.join(RES, "controls_summary.json"), "w"), indent=2)
    print("\nwrote results/controls_summary.json")
    make_plots(summary)


def make_plots(summary):
    # (1) unimodality: tvar histograms for single at 3 sigmas
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, sig in zip(axes, ["0.006", "0.007", "0.008"]):
        a = tvars(f"single_{sig}")
        if a is None:
            continue
        ax.hist(a * 1e5, bins=15, color="steelblue", edgecolor="k", alpha=0.8)
        ax.axvline(THR_EPS_LOCK * 1e5, color="crimson", ls="--", lw=2, label="eps_lock=3e-5")
        fc, _ = frac_collapsed(f"single_{sig}")
        ax.set_title(f"single sigma={sig}  (collapse {fc:.0%})")
        ax.set_xlabel("temporal_var_final  (x1e-5)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("seed count")
    fig.suptitle("T4: is collapse a threshold cut through a UNIMODAL cluster?")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "controls_unimodality.png"), dpi=130)

    # (2) collapse-fraction bar chart across all arms
    order = [("single_0.007", "single"),
             ("bilatL_0.1_0.007", "TRUE bilateral"),
             ("white_0.1_0.007", "matched white"),
             ("colored_0.1_0.007", "matched colored"),
             ("sham_frozen_0.1_0.007", "sham frozen"),
             ("sham_indep_0.1_0.007", "sham indep"),
             ("sham_mutual_0.1_0.007", "sham mutual\n(non-mirror)"),
             ("single_0.008", "single sigma=.008")]
    labels, vals, cols = [], [], []
    for k, lab in order:
        f, _ = frac_collapsed(k)
        if f is None:
            continue
        labels.append(lab); vals.append(f * 100)
        cols.append("navy" if k == "bilatL_0.1_0.007" else
                    ("crimson" if k == "single_0.007" else "gray"))
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    ax2.bar(range(len(vals)), vals, color=cols, edgecolor="k")
    sb, _ = frac_collapsed("single_0.007")
    ax2.axhline(sb * 100, color="crimson", ls="--", lw=1.5, alpha=0.7)
    for i, v in enumerate(vals):
        ax2.annotate(f"{v:.0f}%", (i, v), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=9)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("collapse fraction %")
    ax2.set_title("Adversarial controls: collapse fraction by arm (mu=0.1, sigma=0.007, N=60)")
    ax2.set_ylim(0, 105)
    fig2.tight_layout()
    fig2.savefig(os.path.join(RES, "controls_bars.png"), dpi=130)
    print("wrote results/controls_unimodality.png, results/controls_bars.png")


if __name__ == "__main__":
    main()
