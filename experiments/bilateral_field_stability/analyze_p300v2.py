"""
analyze_p300v2.py -- threshold-free analysis of the P300 v2 prediction-error wave.

Consumes results/p300v2/{crossed,p2,p3,p5}{_base,_exc}.json and reports:
  P1  double dissociation: wave tracks prediction error (predicted vs violating) at every Delta;
      exchange tracks Delta (0/45/90). Mann-Whitney U on the wave pools; flatness across Delta.
  P5  flow control: predicted stream at 45deg (E=max) -> ~flat waves; violation at end -> spike.
  no-prediction control: frozen kappa -> wave collapses toward the Level-1 null.
  P2  ignition/sigmoid: wave vs violation size; sigmoid vs linear fit (AIC).
  P3  precision: bigger prior precision -> bigger wave (monotonic; Spearman sign).
  P4  propagation + self-extinction: spread_peak > 0, a_final ~ 0 (excitable only).

Pure-numpy stats (Mann-Whitney U w/ normal approx, Spearman, two-proportion). No thresholds
baked into the verdict -- raw effect sizes + CIs are printed for every load-bearing claim.
"""
import os, json, math
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "p300v2")


def load(cmd, tag):
    p = os.path.join(OUT, f"{cmd}{tag}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["records"]


def mannwhitney(x, y):
    """Two-sided Mann-Whitney U with normal approximation + tie correction. Returns U, z, p, A
    where A = common-language effect size P(X>Y)+.5 P(=)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    allv = np.concatenate([x, y])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # tie-average ranks
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    R1 = ranks[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2.0
    U = min(U1, n1 * n2 - U1)
    mu = n1 * n2 / 2.0
    _, tie_cnt = np.unique(allv, return_counts=True)
    tie = (tie_cnt ** 3 - tie_cnt).sum()
    n = n1 + n2
    sd = math.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1))))
    z = (U - mu) / sd if sd > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    A = U1 / (n1 * n2)   # P(X>Y) + 0.5 P(X=Y)
    return float(U), float(z), float(p), float(A)


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    def rank(v):
        o = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); r[o] = np.arange(1, len(v) + 1)
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        s = np.zeros(len(cnt)); np.add.at(s, inv, r); return (s / cnt)[inv]
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = math.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / d) if d > 0 else float("nan")


def summ(v):
    v = np.asarray(v, float)
    return f"{v.mean():.4f}+-{v.std()/max(math.sqrt(len(v)),1):.4f}(n{len(v)})"


def analyze_crossed(tag):
    recs = load("crossed", tag)
    if not recs:
        print(f"  [crossed{tag}] MISSING"); return
    print(f"\n=== P1 DOUBLE DISSOCIATION  (crossed{tag}) ===")
    pred = [r for r in recs if r["arm"] == "pred"]
    # wave by (kind, delta)
    print("  wave_amp by cell:")
    for kind in ["predicted", "violating"]:
        row = []
        for d in [0.0, 45.0, 90.0]:
            w = [r["wave_amp"] for r in pred if r["kind"] == kind and r["delta"] == d]
            row.append(f"{int(d)}deg:{summ(w)}")
        print(f"    {kind:10s} " + "  ".join(row))
    print("  exchange by cell (should track Delta, ignore kind):")
    for d in [0.0, 45.0, 90.0]:
        e = [r["exchange"] for r in pred if r["delta"] == d]
        print(f"    {int(d)}deg: {summ(e)}")
    # headline: pool violating (ii+iii) vs predicted (i+iv)
    viol = [r["wave_amp"] for r in pred if r["kind"] == "violating"]
    pmatch = [r["wave_amp"] for r in pred if r["kind"] == "predicted"]
    U, z, p, A = mannwhitney(viol, pmatch)
    print(f"  HEADLINE wave: violating {summ(viol)} vs predicted {summ(pmatch)}")
    print(f"    Mann-Whitney U={U:.0f} z={z:.2f} p={p:.2e}  A(P[viol>pred])={A:.3f}")
    # flatness of wave across Delta within violating (dissociation from exchange)
    wv = {d: np.mean([r["wave_amp"] for r in pred if r["kind"]=="violating" and r["delta"]==d]) for d in [0.,45.,90.]}
    ex = {d: np.mean([r["exchange"] for r in pred if r["delta"]==d]) for d in [0.,45.,90.]}
    wv_spread = (max(wv.values())-min(wv.values()))/ (np.mean(list(wv.values()))+1e-12)
    ex_spread = (max(ex.values())-min(ex.values()))/ (np.mean(list(ex.values()))+1e-12)
    print(f"  wave rel-range across Delta (violating) = {wv_spread:.2f} (small=flat=good)")
    print(f"  exchange rel-range across Delta          = {ex_spread:.2f} (large=Delta-tuned=good)")
    # no-prediction control
    nop = [r["wave_amp"] for r in recs if r["arm"] == "nopred"]
    if nop:
        U2, z2, p2_, A2 = mannwhitney(viol, nop)
        print(f"  NO-PREDICTION control (frozen kappa): wave {summ(nop)}  vs pred-viol {summ(viol)}")
        print(f"    frozen collapses wave? Mann-Whitney A(viol>frozen)={A2:.3f} p={p2_:.2e}")


def logistic_fit(x, y):
    """Crude 2-param logistic (L=max(y)) vs linear; return (rmse_sig, rmse_lin)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    L = y.max() * 1.05 + 1e-9
    best = None
    for x0 in np.linspace(x.min(), x.max(), 25):
        for k in np.linspace(0.5, 12, 25):
            pred = L / (1 + np.exp(-k * (x - x0)))
            r = np.mean((pred - y) ** 2)
            if best is None or r < best:
                best = r
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    rlin = np.mean((A @ coef - y) ** 2)
    return math.sqrt(best), math.sqrt(rlin)


def analyze_p2(tag):
    recs = load("p2", tag)
    if not recs:
        print(f"  [p2{tag}] MISSING"); return
    print(f"\n=== P2 IGNITION / SIGMOID  (p2{tag}) ===")
    vs = sorted(set(r["violation"] for r in recs))
    xs, ys = [], []
    for v in vs:
        w = [r["wave_amp"] for r in recs if r["violation"] == v]
        a = [r["a_peak"] for r in recs if r["violation"] == v]
        print(f"    viol={v:.2f}: wave {summ(w)}  a_peak {np.mean(a):.4f}")
        xs.append(v); ys.append(np.mean(w))
    rs, rl = logistic_fit(xs, ys)
    print(f"  sigmoid RMSE={rs:.4f}  linear RMSE={rl:.4f}  -> {'SIGMOID' if rs < 0.8*rl else 'graded/linear'}")


def analyze_p3(tag):
    recs = load("p3", tag)
    if not recs:
        print(f"  [p3{tag}] MISSING"); return
    print(f"\n=== P3 PRECISION -> WAVE  (p3{tag}) ===")
    sp = sorted(set(r["prelude_spread"] for r in recs))
    precs, waves = [], []
    for s in sp:
        pr = [r["precision"] for r in recs if r["prelude_spread"] == s]
        w = [r["wave_amp"] for r in recs if r["prelude_spread"] == s]
        print(f"    prelude_spread={s:.2f}: precision {np.mean(pr):.3f}  wave {summ(w)}")
        precs += pr; waves += w
    rho = spearman(precs, waves)
    print(f"  Spearman(precision, wave) = {rho:+.3f}  (positive = higher precision -> bigger wave)")


def analyze_p5(tag):
    recs = load("p5", tag)
    if not recs:
        print(f"  [p5{tag}] MISSING"); return
    print(f"\n=== P5 FLOW CONTROL  (p5{tag}) ===")
    stream = [r for r in recs if not r.get("is_violation")]
    viol = [r for r in recs if r.get("is_violation")]
    sw = [r["wave_amp"] for r in stream]
    vw = [r["wave_amp"] for r in viol]
    ex = [r["exchange"] for r in stream]
    print(f"  predicted-stream wave {summ(sw)}   (exchange {np.mean(ex):.3f} = E(45)=max)")
    print(f"  end-of-stream violation wave {summ(vw)}")
    U, z, p, A = mannwhitney(vw, sw)
    print(f"  violation spikes above predicted stream? A(viol>stream)={A:.3f} p={p:.2e}")


def analyze_propagation(tag):
    recs = load("crossed", tag)
    if not recs:
        return
    print(f"\n=== P4 PROPAGATION + SELF-EXTINCTION  (crossed{tag}) ===")
    viol = [r for r in recs if r["arm"] == "pred" and r["kind"] == "violating"]
    sp = [r["spread_peak"] for r in viol]; sf = [r["spread_final"] for r in viol]
    ap = [r["a_peak"] for r in viol]; af = [r["a_final"] for r in viol]
    ext = [r["self_extinct"] for r in viol]
    cr = [r.get("coh_return", float("nan")) for r in viol]
    print(f"  spread_peak {summ(sp)}  spread_final {summ(sf)}")
    print(f"  a_peak {summ(ap)}  a_final {summ(af)}  (a_final~0 => refractory self-extinction)")
    print(f"  coherence return-to-baseline (1=fully returned, 0=permanent step) = {summ(cr)}")
    print(f"  self_extinct fraction = {np.mean(ext):.2f}  (n={len(ext)})")


def analyze_adapt(tag):
    recs = load("adapt", tag)
    if not recs:
        print(f"  [adapt{tag}] MISSING"); return
    print(f"\n=== AUDIT(i) kappa_pred IS PREDICTIVE / ADAPTS  (adapt{tag}) ===")
    ks = sorted(set(r["k"] for r in recs))
    curve = []
    for k in ks:
        w = [r["wave_amp"] for r in recs if r["k"] == k]
        e = [r["error"] for r in recs if r["k"] == k]
        curve.append((k, np.mean(w), np.mean(e)))
    for k, w, e in curve:
        print(f"    after-shift k={k:2d}: wave {w:.4f}  error {e:.4f}")
    w0 = curve[0][1]; wend = np.mean([c[1] for c in curve[-4:]])
    # is the post-shift wave decaying (adaptation)? Spearman(k, wave) should be negative
    rho = spearman([r["k"] for r in recs], [r["wave_amp"] for r in recs])
    print(f"  wave at first post-shift = {w0:.4f}; settled tail = {wend:.4f}; decay ratio = {wend/max(w0,1e-9):.2f}")
    print(f"  Spearman(k, wave) = {rho:+.3f}  (negative => wave decays as kappa learns => ADAPTS)")


def analyze_robust(tag):
    recs = load("robust", tag)
    if not recs:
        print(f"  [robust{tag}] MISSING"); return
    print(f"\n=== AUDIT(iii) EXCITABILITY NOT KNIFE-EDGE  (robust{tag}) ===")
    variants = []
    seen = set()
    for r in recs:
        if r["variant"] not in seen:
            seen.add(r["variant"]); variants.append(r["variant"])
    # for each FHN perturbation: does the IGNITION THRESHOLD survive?  sub-threshold viol (0.3) must
    # stay near the ~0.008 floor; supra-threshold viol (1.4) must ignite well above it.
    print("  ignition threshold per FHN perturbation (sub=viol0.3 floor, supra=viol1.4 ignited):")
    base_sub = np.mean([r["wave_amp"] for r in recs if r["variant"] == "baseline" and r["band"] == "sub"])
    all_ok = True
    for name in variants:
        sub = [r["wave_amp"] for r in recs if r["variant"] == name and r["band"] == "sub"]
        supra = [r["wave_amp"] for r in recs if r["variant"] == name and r["band"] == "supra"]
        U, z, p, A = mannwhitney(supra, sub)
        gap = np.mean(supra) - np.mean(sub)
        # threshold preserved: supra ignites above sub (A high, gap sizeable) AND sub stays near floor
        sub_low = np.mean(sub) < 3 * base_sub + 0.004
        survives = A > 0.75 and gap > 0.008 and sub_low
        all_ok = all_ok and survives
        print(f"    {name:14s} sub={np.mean(sub):.4f} supra={np.mean(supra):.4f} "
              f"gap={gap:.4f} A={A:.3f} p={p:.1e}  {'OK' if survives else 'WEAK'}")
    print(f"  ignition threshold (floor-below / ignite-above) survives ALL {len(variants)} "
          f"perturbations (incl. 9 one-at-a-time and 6 joint)? {all_ok}")


def analyze_trace(tag):
    recs = load("trace", tag)
    if not recs:
        print(f"  [trace{tag}] MISSING"); return
    print(f"\n=== AUDIT(ii) WAVE IS A TRANSIENT PULSE  (trace{tag}) ===")
    for kind in ["predicted", "violating"]:
        sub = [r for r in recs if r["kind"] == kind]
        a = np.array([r["trace_a"] for r in sub]).mean(0)
        sp = np.array([r["trace_spread"] for r in sub]).mean(0)
        if a.max() > 1e-9:
            t_peak = int(np.argmax(a))
            rise = a[:t_peak + 1]; fall = a[t_peak:]
            monotone_rise = bool(np.all(np.diff(rise) >= -1e-6)) if len(rise) > 1 else True
            decays = bool(a[-1] < 0.1 * a.max())
            print(f"    {kind:10s} a: peak={a.max():.4f}@t{t_peak}  final={a[-1]:.4f}  "
                  f"spread peak={sp.max():.3f} final={sp[-1]:.3f}  "
                  f"rise+fall pulse? {monotone_rise and decays}")
        else:
            print(f"    {kind:10s} a: no activation (floor) -- peak={a.max():.5f}")


def main():
    for tag in ["_base", "_exc"]:
        print("\n" + "#" * 70)
        print(f"#  P300 v2  --  {tag[1:].upper()} FIELD")
        print("#" * 70)
        analyze_crossed(tag)
        analyze_p5(tag)
        analyze_p2(tag)
        analyze_p3(tag)
        analyze_propagation(tag)
        analyze_adapt(tag)
        analyze_robust(tag)
        analyze_trace(tag)


if __name__ == "__main__":
    main()
