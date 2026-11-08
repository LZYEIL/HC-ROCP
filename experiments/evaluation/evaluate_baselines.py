#!/usr/bin/env python3
"""Evaluate the frozen standard baselines on the held-out test split.

Calibration uses the full predefined split unless an explicit calibration
subset is supplied by the multi-seed runner. Test data are used only for reporting.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from shared.paths import load_config

import argparse, json, sys
from pathlib import Path
import numpy as np, yaml

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from baselines.rocp import RiskOptimalConformalPredictor
from baselines.rac import RACBaseline
from baselines.las import LASConformal
from baselines.aps import APSConformal
from baselines.saps import SAPSConformal
from baselines.mondrian_aps import MondrianAPS
from baselines.utility_directed import Score3CortesConformal
from baselines.action_only import expected_loss_actions, always_safe_actions
from baselines.raps import OfficialRAPSConformal
from baselines.socop import OfficialSOCOPConformal
from baselines.crc import fit_official_crc_action
from baselines._shared import threshold_policy_actions

def load(outdir, split):
    """Load frozen probabilities and labels from the named NPZ split."""
    d=np.load(Path(outdir)/f"probabilities_{split}.npz", allow_pickle=False)
    return np.asarray(d['probabilities'],float), np.asarray(d['labels'],int)

def metric(name, sets, actions, p, y, loss, severe, harmful):
    """Compute evaluation metrics."""
    y=np.asarray(y,int); a=None if actions is None else np.asarray(actions,int)
    out={'method':name,'coverage':None,'mean_set_size':None,'empty_set_rate':None,
         'mean_realized_loss':None,'harmful_action_rate':None,
         'critical_count':int(np.sum(np.isin(y,severe)))}
    if sets is not None:
        out['coverage']=float(np.mean([int(t) in s for t,s in zip(y,sets)]))
        out['mean_set_size']=float(np.mean([len(s) for s in sets]))
        out['empty_set_rate']=float(np.mean([len(s)==0 for s in sets]))
    if a is not None:
        out['mean_realized_loss']=float(np.mean(loss[y,a])); m=np.isin(y,severe)
        out['harmful_action_rate']=float(np.mean(np.isin(a[m],harmful))) if np.any(m) else None
        out['action_counts']={str(i):int(np.sum(a==i)) for i in range(loss.shape[1])}
    return out

def set_actions(pred, sets, alpha, n):
    """Choose robust-loss actions for sets, using the full label set when a set is empty."""
    full=set(range(n)); return [int(pred.action_and_certificate(s if s else full,alpha=alpha,num_labels=n)[0]) for s in sets]

def run(name,cfg,alpha, calibration_override=None):
    """Calibrate comparison methods and evaluate them on the fixed test split."""
    od=Path(cfg['output_dir']); cp,cy=load(od,cfg['calibration_split']); tp,ty=load(od,cfg['test_split'])
    if calibration_override is not None: cp, cy = calibration_override
    loss=np.asarray(cfg['loss_matrix'],float); k=len(cfg['actions']); acts=list(range(k)); rocp=RiskOptimalConformalPredictor(acts,loss)
    res=[]; beta=rocp.calibrate_beta(cp,cy,alpha); ss=[rocp.predict_set(p,beta) for p in tp]
    res.append(metric('ROCP',ss,set_actions(rocp,ss,alpha,loss.shape[0]),tp,ty,loss,cfg['severe_labels'],cfg['harmful_actions']))
    utility=float(np.max(loss))-loss; rac=RACBaseline(acts,utility); rb=rac.calibrate_beta(cp,cy,alpha); rs=[rac.predict_set(p,rb) for p in tp]
    res.append(metric('RAC',rs,[int(rac.a_RAC(s if s else set(range(k)))[0]) for s in rs],tp,ty,loss,cfg['severe_labels'],cfg['harmful_actions']))
    scorers={'LAS':LASConformal(acts),'APS':APSConformal(acts),'RandomizedRAPS':OfficialRAPSConformal(),'RandomizedSAPS':SAPSConformal(),'MondrianAPS':MondrianAPS(),'SOCOP':OfficialSOCOPConformal(alpha=alpha,penalty_lambda=.0625),'UtilityDirectedCP':Score3CortesConformal(acts,loss,alpha)}
    for nm,sc in scorers.items():
        sc.calibrate(cp,cy,alpha); z=sc.predict_sets(tp) if hasattr(sc,'predict_sets') else [sc.predict_set(p) for p in tp]
        res.append(metric(nm,z,set_actions(rocp,z,alpha,loss.shape[0]),tp,ty,loss,cfg['severe_labels'],cfg['harmful_actions']))
    b,_=expected_loss_actions(tp,loss); res.append(metric('BayesAction',None,b,tp,ty,loss,cfg['severe_labels'],cfg['harmful_actions']))
    aa=always_safe_actions(tp,loss,cfg['harmful_actions'])
    res.append(metric('AlwaysSafeAction',None,aa,tp,ty,loss,cfg['severe_labels'],cfg['harmful_actions']))
    crc=fit_official_crc_action(cp,cy,loss,cfg['severe_labels'],cfg['harmful_actions'],np.linspace(0,1,101),delta=.01)
    cc=threshold_policy_actions(tp,loss,cfg['severe_labels'],cfg['harmful_actions'],crc['threshold'])
    row=metric('CRC_official_action_adapter',None,cc,tp,ty,loss,cfg['severe_labels'],cfg['harmful_actions']); row['calibration']=crc; res.append(row)
    return {'dataset':name,'selection_data':{'calibration':cfg['calibration_split'],'test':cfg['test_split']},'alpha':alpha,'num_test':len(ty),'results':res}

def main():
    """Run the command-line entry point."""
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',default='all',choices=['all','isic2019','covid19','eyepacs_to_aptos']); ap.add_argument('--output-suffix',default='standard_baselines_test.json'); ap.add_argument('--write-dir',default=None, help='directory for result JSON (input probabilities remain in config output_dir)'); a=ap.parse_args()
    root=load_config((ROOT/'configs/experiments.yaml')); names=list(root['datasets']) if a.dataset=='all' else [a.dataset]
    for n in names:
        r=run(n,root['datasets'][n],float(root['alpha'])); p=(Path(a.write_dir)/f'{n}_{a.output_suffix}' if a.write_dir else Path(root['datasets'][n]['output_dir'])/a.output_suffix); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n'); print(json.dumps(r,indent=2,sort_keys=True),flush=True)
if __name__=='__main__': main()
