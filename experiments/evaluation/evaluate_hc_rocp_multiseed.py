#!/usr/bin/env python3
"""Ten-seed stability analysis for the complete Certifiable HC-ROCP protocol."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from shared.paths import OUTPUTS_DIR, load_config

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.hc_rocp import CertifiableHCROCP, automatic_groups
from core.decision_problem import FiniteDecisionProblem


def load(output_dir, split):
    """Load frozen probabilities and labels from the named NPZ split."""
    data = np.load(Path(output_dir) / f"probabilities_{split}.npz", allow_pickle=False)
    return np.asarray(data["probabilities"], float), np.asarray(data["labels"], int)


def evaluate(model, probabilities, labels, beta, gamma, loss, config):
    """Compute held-out coverage, loss, severe-action errors and set sizes."""
    sets = model.sets(probabilities, beta, gamma)
    actions = model.actions(probabilities, beta, gamma)
    severe = np.isin(labels, config["severe_labels"])
    harmful = np.isin(actions, config["harmful_actions"])
    return {
        "coverage": float(np.mean([int(y) in s for y, s in zip(labels, sets)])),
        "loss": float(np.mean(loss[labels, actions])),
        "harmful_rate": float(np.mean(harmful[severe])) if np.any(severe) else None,
        "harm_count": int(harmful[severe].sum()),
        "n_severe": int(severe.sum()),
        "set_size": float(np.mean([len(s) for s in sets])),
    }


def one_seed(probabilities, labels, test_p, test_y, config, root_config, seed, delta):
    """Run construction, certification, selection and test evaluation for one split seed."""
    construct, remaining = train_test_split(
        np.arange(len(labels)), test_size=2 / 3, random_state=seed, stratify=labels
    )
    coverage, safety = train_test_split(
        remaining, test_size=0.5, random_state=seed + 1, stratify=labels[remaining]
    )
    loss = np.asarray(config["loss_matrix"], float)
    model = CertifiableHCROCP(
        FiniteDecisionProblem(loss, alpha=float(root_config["alpha"])),
        config["severe_labels"], config["harmful_actions"],
    )
    gammas = np.r_[
        0.0,
        np.geomspace(
            root_config["gamma_min_positive"],
            root_config["gamma_max"],
            root_config["gamma_log_points"],
        ),
    ]
    coverage_targets = [float(config["coverage_target"]), 0.975, 0.99]
    betas = sorted({
        round(model.calibrate_beta(probabilities[construct], labels[construct], float(g), target), 10)
        for target in coverage_targets for g in gammas
    })
    groups = automatic_groups(config["severe_labels"], labels[safety], delta)
    fit = model.fit_certifiable(
        probabilities[coverage], labels[coverage],
        probabilities[safety], labels[safety],
        betas, gammas, float(config["coverage_target"]), groups, loss, 0.95,
    )
    selected = fit["selected"]
    return {
        "seed": int(seed),
        "split_sizes": {"construct": len(construct), "coverage": len(coverage), "safety": len(safety)},
        "num_beta_candidates": len(betas),
        "num_candidates": fit["num_candidates"],
        "global_level": fit["global_level"],
        "counts": fit["counts"],
        "selected": {"beta": selected["beta"], "gamma": selected["gamma"]},
        "test": evaluate(model, test_p, test_y, selected["beta"], selected["gamma"], loss, config),
    }


def summarize(rows):
    """Summarize test metrics and evidence levels across calibration seeds."""
    keys = ["coverage", "loss", "harmful_rate", "set_size"]
    metrics = {}
    for key in keys:
        values = np.asarray([row["test"][key] for row in rows if row["test"][key] is not None], float)
        if not len(values):
            metrics[key] = {"mean": None, "std": None, "min": None, "max": None}
            continue
        metrics[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return {
        "test_metrics": metrics,
        "selected_beta": [row["selected"]["beta"] for row in rows],
        "selected_gamma": [row["selected"]["gamma"] for row in rows],
        "guarantee_level_counts": {
            level: sum(row["global_level"] == level for row in rows)
            for level in sorted({row["global_level"] for row in rows})
        },
    }


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["all", "isic2019", "covid19", "eyepacs_to_aptos"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20260815, 20260825)))
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--output", default=str(OUTPUTS_DIR / "certifiable_multiseed_10.json"))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**32 - 1 for seed in args.seeds):
        parser.error("seeds must be unique integers between 0 and 2**32 - 2")
    if not 0 < args.delta < 1:
        parser.error("delta must lie strictly between 0 and 1")
    config = load_config((ROOT / "configs/experiments.yaml"))
    names = list(config["datasets"]) if args.dataset == "all" else [args.dataset]
    output = {"protocol": {"seeds": args.seeds, "delta": args.delta, "coverage_candidate_targets": [0.95, 0.975, 0.99]}, "datasets": {}}
    for name in names:
        dataset_config = config["datasets"][name]
        probabilities, labels = load(dataset_config["output_dir"], dataset_config["calibration_split"])
        test_p, test_y = load(dataset_config["output_dir"], dataset_config["test_split"])
        rows = []
        for seed in args.seeds:
            row = one_seed(probabilities, labels, test_p, test_y, dataset_config, config, seed, args.delta)
            rows.append(row)
            print(name, seed, row["global_level"], row["selected"], row["test"], flush=True)
        output["datasets"][name] = {"runs": rows, "summary": summarize(rows)}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
