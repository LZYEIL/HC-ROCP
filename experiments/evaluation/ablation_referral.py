#!/usr/bin/env python3
"""Refer-action ablation: does adding a 'refer to human' action improve the loss
while keeping the severe-harm rate within budget?

Adds a fixed-cost 'refer' column to the loss matrix (refer is never harmful),
reruns the joint HC-ROCP protocol, and reports test metrics and action counts.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from shared.paths import load_config, result_path

import json, sys
from pathlib import Path
import numpy as np, yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.decision_problem import FiniteDecisionProblem
from core.hc_rocp import CertifiableHCROCP, automatic_groups


def load(od, s):
    """Load frozen probabilities and labels from the named NPZ split."""
    d = np.load(Path(od) / f"probabilities_{s}.npz")
    return d["probabilities"].astype(float), d["labels"].astype(int)


def test_metrics(m, p, y, b, g, l, c):
    """Compute held-out coverage, loss, severe-action errors and set sizes."""
    s = m.sets(p, b, g)
    a = m.actions(p, b, g)
    z = np.isin(y, c["severe_labels"])
    h = np.isin(a, c["harmful_actions"])
    return {
        "coverage": float(np.mean([int(t) in q for t, q in zip(y, s)])),
        "loss": float(np.mean(l[y, a])),
        "harmful_rate": float(np.mean(h[z])),
        "harm_count": int(h[z].sum()),
        "n_severe": int(z.sum()),
        "set_size": float(np.mean([len(q) for q in s])),
        "action_counts": {int(ai): int((a == ai).sum()) for ai in np.unique(a)},
    }


def run(root, c, loss_matrix):
    """Fit and evaluate HC-ROCP under the supplied COVID-19 action-loss matrix."""
    od = Path(c["output_dir"])
    p, y = load(od, c["calibration_split"])
    tp, ty = load(od, c["test_split"])
    construct, remain = train_test_split(np.arange(len(y)), test_size=2 / 3,
                                         random_state=root["calibration_split_seed"], stratify=y)
    cov, safe = train_test_split(remain, test_size=.5,
                                 random_state=root["calibration_split_seed"] + 1, stratify=y[remain])
    l = np.asarray(loss_matrix, float)
    m = CertifiableHCROCP(FiniteDecisionProblem(l, alpha=root["alpha"]),
                          c["severe_labels"], c["harmful_actions"])
    gammas = np.r_[0., np.geomspace(root["gamma_min_positive"], root["gamma_max"],
                                    root["gamma_log_points"])]
    coverage_targets = [c["coverage_target"], .975, .99]
    betas = sorted(set(round(m.calibrate_beta(p[construct], y[construct], float(g), t), 10)
                       for t in coverage_targets for g in gammas))
    groups = automatic_groups(c["severe_labels"], y[safe], root["harm_delta"])
    fit = m.fit_certifiable(p[cov], y[cov], p[safe], y[safe], betas, gammas,
                            c["coverage_target"], groups, l, .95)
    sel = fit["selected"]
    subs = [{"name": s["name"], "n": s["n"], "harm_count": s["harm_count"],
             "upper": s["upper"], "certified": s["certified"]} for s in fit["subgroups"]]
    return {
        "global_level": fit["global_level"],
        "selected": {"beta": sel["beta"], "gamma": sel["gamma"]},
        "subgroups": subs,
        "test": test_metrics(m, tp, ty, sel["beta"], sel["gamma"], l, c),
    }


def main():
    """Run the command-line entry point."""
    root = load_config((ROOT / "configs/experiments.yaml"))
    c = dict(root["datasets"]["covid19"])
    base = np.asarray(c["loss_matrix"], float)

    results = {}
    results["original_4action"] = run(root, c, base)
    for cost in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        lm = np.hstack([base, np.full((base.shape[0], 1), float(cost))])
        results[f"refer_{cost}"] = run(root, c, lm)

    out = {}
    for k, v in results.items():
        t = v["test"]
        out[k] = v
        print(f"{k:16s} level={v['global_level']:24s} "
              f"beta={v['selected']['beta']:.3f} gamma={v['selected']['gamma']:.3f} "
              f"cov={t['coverage']:.4f} loss={t['loss']:.4f} "
              f"harm={t['harmful_rate']*100:.3f}% set_size={t['set_size']:.3f} "
              f"actions={t['action_counts']}")

    q = result_path("refer_ablation.json")
    q.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("saved", q)


if __name__ == "__main__":
    main()
