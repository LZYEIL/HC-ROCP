#!/usr/bin/env python3
"""Run all standard baselines on matched ten-seed calibration splits."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from shared.paths import OUTPUTS_DIR, load_config

import json, sys
from pathlib import Path
import numpy as np, yaml
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from experiments.evaluation.evaluate_baselines import load, run

def main():
    """Run and write per-seed baseline metrics."""
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',nargs='+',type=int,default=list(range(20260815,20260825))); ap.add_argument('--dataset',default='all'); ap.add_argument('--output',default=str(OUTPUTS_DIR / 'baselines_multiseed_10.json')); a=ap.parse_args()
    root=load_config((ROOT/'configs/experiments.yaml')); out={'protocol':{'seeds':a.seeds,'calibration':'same three-way split; coverage arm used for baseline calibration'},'datasets':{}}
    if a.dataset != 'all' and a.dataset not in root['datasets']:
        ap.error('unknown dataset; choose all, covid19, isic2019, or eyepacs_to_aptos')
    if len(set(a.seeds)) != len(a.seeds) or any(seed < 0 or seed >= 2**32 - 1 for seed in a.seeds):
        ap.error('seeds must be unique integers between 0 and 2**32 - 2')
    names=root['datasets'] if a.dataset=='all' else {a.dataset:root['datasets'][a.dataset]}
    for name,cfg in names.items():
        p,y=load(cfg['output_dir'],cfg['calibration_split']); rows=[]
        for seed in a.seeds:
            construct,rem=train_test_split(np.arange(len(y)),test_size=2/3,random_state=seed,stratify=y)
            cov,safe=train_test_split(rem,test_size=.5,random_state=seed+1,stratify=y[rem])
            result=run(name,cfg,float(root['alpha']),(p[cov],y[cov]))
            for m in result['results']:
                rows.append({'seed':seed,**{k:m.get(k) for k in ('method','coverage','harmful_action_rate','mean_realized_loss','mean_set_size')}})
            print(name,seed,flush=True)
        out['datasets'][name]={'runs':rows}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(out,indent=2)+'\n'); print('wrote',a.output)
if __name__=='__main__': main()
