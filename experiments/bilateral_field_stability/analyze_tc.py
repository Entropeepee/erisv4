"""T-C robustness: egate vs cos attractor pass-rate across sigma in {0.013,0.016,0.020}."""
import glob, json, os, numpy as np, math
from collections import defaultdict
def interior_return(r): 
    sa,so=r["after_same_kick"],r["after_orth_kick"]; return abs(sa-so)<12 and 20<(sa+so)/2<75
def ci(k,n,z=1.96):
    if n==0:return(0,0)
    p=k/n;h=z*math.sqrt(p*(1-p)/n+z*z/4/n/n)/(1+z*z/n);c=(p+z*z/2/n)/(1+z*z/n);return(max(0,c-h),min(1,c+h))
cells=defaultdict(list)
for f in glob.glob("results/bifurcate/attractor_*_seed*.json"):
    b=os.path.basename(f)
    for sig in ["0.013","0.016","0.02"]:
        if f"_s{sig}_" in b:
            kind=b.split("_")[1]; mu=b.split("_m")[1].split("_")[0]
            cells[(sig,kind,mu)].append(json.load(open(f)))
print("T-C ROBUSTNESS: interior-attractor pass-rate (return-from-both, 20<theta<75) vs sigma")
print(f"{'sigma':>6} {'kind':>6} {'mu':>4} {'pass':>10} {'mean theta*':>11} {'95% CI':>14}")
for (sig,kind,mu) in sorted(cells):
    rs=cells[(sig,kind,mu)]; n=len(rs); k=sum(interior_return(r) for r in rs)
    mt=np.mean([(r["after_same_kick"]+r["after_orth_kick"])/2 for r in rs]); lo,hi=ci(k,n)
    print(f"{sig:>6} {kind:>6} {mu:>4} {k}/{n}={k/n:>4.0%} {mt:>11.1f} [{lo:.2f},{hi:.2f}]")
