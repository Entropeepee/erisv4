"""Analyze T-B (aliveness at lock edge) and T-D (N>=20 attractor pass-rates)."""
import json, glob, os, numpy as np, math
from collections import defaultdict
CDIR="results/bifurcate"

def binom_ci(k,n,z=1.96):
    if n==0: return (0,0)
    p=k/n; half=z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n); c=(p+z*z/(2*n))/(1+z*z/n)
    return (max(0,c-half),min(1,c+half))

def ztest(k1,n1,k2,n2):
    p1,p2=k1/n1,k2/n2; pool=(k1+k2)/(n1+n2); se=math.sqrt(pool*(1-pool)*(1/n1+1/n2)) or 1e-12
    z=(p1-p2)/se; return z, math.erfc(abs(z)/math.sqrt(2))

print("="*70); print("T-B: ALIVENESS AT THE LOCK EDGE (N=20, per-lobe)"); print("="*70)
print(f"{'sigma':>6} {'arm':>6} | {'both_alive':>10} {'L_lock':>7} {'R_lock':>7} {'theta_LR':>8}")
tb=defaultdict(dict)
for f in sorted(glob.glob(f"{CDIR}/aliveedge_*.json")):
    d=json.load(open(f)); m=d["meta"]; r=d["records"]
    arm=m_kind=f.split("aliveedge_")[1].split("_d")[0]; sig=m["sigma"]
    ba=np.mean([x["both_alive"] for x in r]); Lc=np.mean([x["L_collapsed"] for x in r])
    Rc=np.mean([x["R_collapsed"] for x in r]); th=np.mean([x["mean_theta_LR"] for x in r])
    tb[sig][arm]={"both_alive":ba,"L":Lc,"R":Rc,"theta":th,"n":len(r)}
for sig in sorted(tb):
    for arm in ["iso","diff","cos","egate"]:
        if arm in tb[sig]:
            c=tb[sig][arm]; print(f"{sig:>6} {arm:>6} | {c['both_alive']:>9.0%} {c['L']:>6.0%} {c['R']:>6.0%} {c['theta']:>8.0f}")
    # decisive: egate vs diff both_alive
    if "egate" in tb[sig] and "diff" in tb[sig]:
        e=tb[sig]["egate"]; di=tb[sig]["diff"]
        ke=round(e["both_alive"]*e["n"]); kd=round(di["both_alive"]*di["n"])
        z,p=ztest(ke,e["n"],kd,di["n"])
        print(f"   -> egate both_alive {e['both_alive']:.0%} vs diff {di['both_alive']:.0%}: z={z:.2f} p={p:.2e}")
    print()

print("="*70); print("T-D: N>=20 ATTRACTOR PASS-RATES (fixed code)"); print("="*70)
cells=defaultdict(list)
for f in glob.glob(f"{CDIR}/attractor_*_seed*.json"):
    b=os.path.basename(f)
    if "_s0.016_" not in b: continue
    key="_".join(b.split("_")[1:4])  # kind_dX_mY
    cells[key].append(json.load(open(f)))
def is_interior_return(r):
    sa,so=r["after_same_kick"],r["after_orth_kick"]
    return (abs(sa-so)<12) and (20<(sa+so)/2<75)
print(f"{'cell':>20} {'N':>3} {'pass-rate (interior from both)':>32} {'95% CI':>16}")
passrates={}
for key in sorted(cells):
    rs=cells[key]; n=len(rs); k=sum(is_interior_return(r) for r in rs)
    lo,hi=binom_ci(k,n); passrates[key]=(k,n)
    print(f"{key:>20} {n:>3} {k}/{n} = {k/n:>6.0%}{'':>14} [{lo:.2f},{hi:.2f}]")
# egate vs cos aggregate pass-rate
eg=[v for kk,v in passrates.items() if kk.startswith("egate")]
co=[v for kk,v in passrates.items() if kk.startswith("cos")]
if eg and co:
    ek=sum(k for k,n in eg); en=sum(n for k,n in eg); ck=sum(k for k,n in co); cn=sum(n for k,n in co)
    z,p=ztest(ek,en,ck,cn)
    print(f"\negate pass-rate {ek}/{en}={ek/en:.0%} vs cos {ck}/{cn}={ck/cn:.0%}: z={z:.2f} p={p:.2e}")
