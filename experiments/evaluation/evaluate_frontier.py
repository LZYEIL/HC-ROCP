#!/usr/bin/env python3
"""Safety-utility frontier: Pareto envelope of the 2-D (beta, gamma) family.

Reproduces the candidate family from evaluate_hc_rocp.py, evaluates its policies
on the held-out test split, and extracts the nondominated harm-loss envelope.
This descriptive comparison does not change the calibration-selected policy.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from shared.paths import OUTPUTS_DIR, load_config

import argparse, json, sys
from pathlib import Path
import numpy as np, yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.decision_problem import FiniteDecisionProblem
from core.policy import HarmConstrainedROCP


def load(od, split):
    """Load frozen probabilities and labels from the named NPZ split."""
    d = np.load(Path(od) / f"probabilities_{split}.npz")
    return d["probabilities"].astype(float), d["labels"].astype(int)


def evaluate(m, p, y, b, g, loss, severe, harm):
    """Compute held-out coverage, loss, severe-action errors and set sizes."""
    sets = m.sets(p, b, g)
    a = m.actions(p, b, g)
    sev = np.isin(y, severe)
    hm = np.isin(a, harm)
    return {
        "coverage": float(np.mean([int(t) in s for t, s in zip(y, sets)])),
        "loss": float(np.mean(loss[y, a])),
        "harmful_rate": float(np.mean(hm[sev])) if sev.any() else None,
        "set_size": float(np.mean([len(s) for s in sets])),
    }


def main():
    """Run the command-line entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="covid19")
    ap.add_argument("--out-dir", default=str(OUTPUTS_DIR))
    a = ap.parse_args()

    c = load_config((ROOT / "configs/experiments.yaml"))
    cfg = c["datasets"][a.dataset]
    od = Path(cfg["output_dir"])
    cp, cy = load(od, cfg["calibration_split"])
    tp, ty = load(od, cfg["test_split"])
    loss = np.asarray(cfg["loss_matrix"], float)
    m = HarmConstrainedROCP(FiniteDecisionProblem(loss, alpha=c["alpha"]),
                            cfg["severe_labels"], cfg["harmful_actions"])

    construct, _ = train_test_split(np.arange(len(cy)), test_size=2 / 3,
                                    random_state=c["calibration_split_seed"], stratify=cy)
    gammas = np.r_[0., np.geomspace(c["gamma_min_positive"], c["gamma_max"],
                                    c["gamma_log_points"])]
    targets = [cfg["coverage_target"], 0.975, 0.99]
    betas = sorted(set(round(m.calibrate_beta(cp[construct], cy[construct], float(g), t), 10)
                       for t in targets for g in gammas))

    rows = []
    for b in betas:
        for g in gammas:
            ev = evaluate(m, tp, ty, b, float(g), loss, cfg["severe_labels"],
                          cfg["harmful_actions"])
            rows.append({"beta": float(b), "gamma": float(g), **ev})

    pts = [r for r in rows if r["harmful_rate"] is not None]
    envelope = []
    for r in pts:
        dominated = any(
            q["harmful_rate"] <= r["harmful_rate"] and q["loss"] <= r["loss"]
            and (q["harmful_rate"] < r["harmful_rate"] or q["loss"] < r["loss"])
            for q in pts)
        if not dominated:
            envelope.append(r)
    envelope.sort(key=lambda r: r["harmful_rate"])

    out = Path(a.out_dir) / f"{a.dataset}_frontier.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"family": rows, "envelope": envelope}, indent=2, sort_keys=True) + "\n")
    print(a.dataset, "family:", len(rows), "envelope:", len(envelope))


if __name__ == "__main__":
    main()
