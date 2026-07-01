"""Analyze P300 Level-1: P1 (division+gate), P2 (E vs novelty), P4 (transport-only),
P5 (gate vs division), P3 (sigmoid), + oddball-vs-equiprobable adaptation. Threshold-free."""
import json, os, numpy as np
from scipy.stats import mannwhitneyu
P="results/p300"; M="peak_excursion"
def load(key):
    p=f"{P}/{key}.json"
    if not os.path.exists(p): return None
    d=json.load(open(p)); return [r for s in d["by_seed"].values() for r in s]
def cls(recs,c): return np.array([r[M] for r in recs if abs(r["delta"]-c)<1e-6])
def mw(a,b):
    if len(a)<3 or len(b)<3: return (np.nan,np.nan)
    U,p=mannwhitneyu(a,b,alternative='two-sided'); return float(np.median(a)-np.median(b)),float(p)

A=load("A_egate_oddball"); B=load("B_diff_oddball"); C=load("C_monolith_oddball")
Aeq=load("A_egate_equiprobable"); Ato=load("A_egate_oddball_transportonly")

print("=== per-arm mean %s by Delta (oddball) ==="%M)
for name,r in [("A egate",A),("B diff",B),("C monolith",C)]:
    if r: print(f"  {name:11s}: " + " ".join(f"{int(d)}d={cls(r,d).mean():.4f}(n{len(cls(r,d))})" for d in [0,45,90]))

def report(label, a, b, aname, bname):
    d,p=mw(a,b); print(f"  {label}: {aname} med={np.median(a):.4f} vs {bname} med={np.median(b):.4f} | diff={d:+.4f} MW p={p:.2e}")

print("\n=== P1 (division+gate: A > C at 45) ==="); report("P1", cls(A,45), cls(C,45), "A45","C45")
print("=== P5 (gate vs division: A > B at 45) ==="); report("P5", cls(A,45), cls(B,45), "A45","B45")
print("\n=== P2 (E vs novelty: A peaks at 45, not 90) ===")
a45,a90,a0=cls(A,45),cls(A,90),cls(A,0)
print(f"  A: 0d={a0.mean():.4f} 45d={a45.mean():.4f} 90d={a90.mean():.4f} -> peak at {[0,45,90][np.argmax([a0.mean(),a45.mean(),a90.mean()])]}deg")
report("  A45 vs A90", a45, a90, "45","90")
print("  B: peak at %ddeg (novelty control)"%([0,45,90][np.argmax([cls(B,0).mean(),cls(B,45).mean(),cls(B,90).mean()])]))
print("\n=== P4 (transport-only, full-window -- see caveat) ===")
if Ato: report("P4", cls(A,45), cls(Ato,45), "A45-real","A45-transportonly")
print("\n=== oddball vs equiprobable (adaptation: rare 45 > frequent 45?) ===")
if Aeq: report("odd-vs-equi", cls(A,45), cls(Aeq,45), "A45-oddball","A45-equiprob")
# P3 sigmoid
if os.path.exists(f"{P}/ampsweep.json"):
    aw=json.load(open(f"{P}/ampsweep.json")); mus=sorted(float(k) for k in aw); ys=[aw[str(m) if str(m) in aw else m]["mean"] for m in mus] if False else [aw[k]["mean"] for k in sorted(aw,key=float)]
    print("\n=== P3 amplitude sweep (45deg) ==="); print("  mu:",[round(m,1) for m in mus]); print("  resp:",[round(y,4) for y in ys])
