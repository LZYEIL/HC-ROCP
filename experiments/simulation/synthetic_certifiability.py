#!/usr/bin/env python3
"""Controlled study of certify-vs-reject behavior under known risks."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from shared.paths import result_path

import json
from pathlib import Path
import numpy as np
from scipy.stats import beta as beta_dist

def upper(k,n,confidence):
    """Return the one-sided Clopper-Pearson upper bound for a binomial count."""
    return 1.0 if k==n else float(beta_dist.ppf(confidence,k+1,n-k))
def trial(rng,true_risk,n,delta,confidence):
    """Draw a binomial error count and apply a single-policy harm gate."""
    k=int(rng.binomial(n,true_risk)); u=upper(k,n,confidence)
    return {'k':k,'upper':u,'certified':u<=delta}
def main():
    """Run the command-line entry point."""
    rng=np.random.default_rng(20260821); delta=.01; confidence=.95; repeats=5000; out={}
    for n in [50,100,200,300,500,750,1000,1500,2000,3000,5000]:
        for risk in [.001,.005,.01,.03]:
            rows=[trial(rng,risk,n,delta,confidence) for _ in range(repeats)]
            out[f'n={n},risk={risk}']={'n':n,'true_risk':risk,'delta':delta,'confidence':confidence,'certification_rate':float(np.mean([r['certified'] for r in rows])),'mean_upper':float(np.mean([r['upper'] for r in rows]))}
    p=result_path('synthetic_certifiability.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    for k,v in out.items():print(k,round(v['certification_rate'],3),round(v['mean_upper'],4))
if __name__=='__main__':main()
