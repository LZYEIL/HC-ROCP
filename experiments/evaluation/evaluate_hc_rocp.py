#!/usr/bin/env python3
"""Run the unified evidence-aware Certifiable HC-ROCP protocol."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from shared.paths import load_config, result_path

import json,sys
import math
from pathlib import Path
import numpy as np,yaml
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from core.decision_problem import FiniteDecisionProblem
from core.hc_rocp import CertifiableHCROCP,automatic_groups
def load(od,s):
 """Load frozen probabilities and labels from the named NPZ split."""
 d=np.load(Path(od)/f'probabilities_{s}.npz');return d['probabilities'].astype(float),d['labels'].astype(int)
def test_metrics(m,p,y,b,g,l,c):
 """Compute held-out coverage, loss, severe-action errors and set sizes."""
 s=m.sets(p,b,g);a=m.actions(p,b,g);z=np.isin(y,c['severe_labels']);h=np.isin(a,c['harmful_actions']);return {'coverage':float(np.mean([int(t) in q for t,q in zip(y,s)])),'loss':float(np.mean(l[y,a])),'harmful_rate':float(np.mean(h[z])),'harm_count':int(h[z].sum()),'n_severe':int(z.sum()),'set_size':float(np.mean([len(q) for q in s]))}
def main():
 """Run the command-line entry point."""
 root=load_config((ROOT/'configs/experiments.yaml'));out={}
 for name,c in root['datasets'].items():
  od=Path(c['output_dir']);p,y=load(od,c['calibration_split']);tp,ty=load(od,c['test_split']);construct,remain=train_test_split(np.arange(len(y)),test_size=2/3,random_state=root['calibration_split_seed'],stratify=y);cov,safe=train_test_split(remain,test_size=.5,random_state=root['calibration_split_seed']+1,stratify=y[remain]);l=np.asarray(c['loss_matrix'],float);m=CertifiableHCROCP(FiniteDecisionProblem(l,alpha=root['alpha']),c['severe_labels'],c['harmful_actions']);gammas=np.r_[0.,np.geomspace(root['gamma_min_positive'],root['gamma_max'],root['gamma_log_points'])];coverage_targets=[c['coverage_target'],.975,.99];betas=sorted(set(round(m.calibrate_beta(p[construct],y[construct],float(g),t),10) for t in coverage_targets for g in gammas));groups=automatic_groups(c['severe_labels'],y[safe],root['harm_delta']);fit=m.fit_certifiable(p[cov],y[cov],p[safe],y[safe],betas,gammas,c['coverage_target'],groups,l,.95);sel=fit['selected'];fit['test']=test_metrics(m,tp,ty,sel['beta'],sel['gamma'],l,c);fit['splits']={'construct':len(construct),'coverage_certification':len(cov),'safety_certification':len(safe)};fit['candidate_coverage_targets']=coverage_targets;out[name]=fit;print(name,json.dumps({'global_level':fit['global_level'],'counts':fit['counts'],'selected':{'beta':sel['beta'],'gamma':sel['gamma']},'subgroups':fit['subgroups'],'test':fit['test']}))
 for fit in out.values():
  eta=fit['multiplicity']['per_test_error']
  fit['zero_error_sample_requirements']={str(delta):math.ceil(math.log(eta)/math.log1p(-delta)) for delta in (.01,.02,.05)}
 q=result_path('certifiable_hc_rocp.json');q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
