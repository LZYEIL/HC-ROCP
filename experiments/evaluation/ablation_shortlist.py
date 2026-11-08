#!/usr/bin/env python3
"""Construct-only candidate shortlisting for Certifiable HC-ROCP.

The full (beta, gamma) family is generated and ranked using only the construct
split.  For each pre-specified shortlist size L, the finalist pairs are frozen
before the disjoint coverage and safety splits are inspected.  Certification
therefore pays multiplicity for L candidates, not for the original Cartesian
family.  Each L is reported as a separate protocol; the script never selects
an L after observing certification or test results.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from shared.paths import OUTPUTS_DIR, load_config

import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.hc_rocp import (  # noqa: E402
    CertifiableHCROCP,
    automatic_groups,
)
from core.decision_problem import FiniteDecisionProblem  # noqa: E402

OUT = OUTPUTS_DIR / "construct_shortlist_ablation.json"
SHORTLIST_SIZES = (1, 5, 10, 20, 50, 100, 250, 500, 1000)


def load(output_dir, split):
    """Load frozen probabilities and labels from the named NPZ split."""
    data = np.load(Path(output_dir) / f"probabilities_{split}.npz")
    return data["probabilities"].astype(float), data["labels"].astype(int)


def split_calibration(labels, seed):
    """Create disjoint stratified construct, coverage and safety splits."""
    indices = np.arange(len(labels))
    construct, remaining = train_test_split(
        indices, test_size=2 / 3, random_state=seed, stratify=labels,
    )
    coverage, safety = train_test_split(
        remaining, test_size=0.5, random_state=seed + 1,
        stratify=labels[remaining],
    )
    return construct, coverage, safety


def build_full_pairs(model, probabilities, labels, gammas, targets):
    """Construct the Cartesian beta/gamma family using only construct data."""
    betas = sorted({
        round(model.calibrate_beta(probabilities, labels, float(gamma),
                                   float(target)), 10)
        for target in targets for gamma in gammas
    })
    return [(float(beta), float(gamma)) for beta in betas for gamma in gammas]


def construct_only_ranking(model, probabilities, labels, pairs, loss_matrix,
                           target_coverage, delta):
    """Rank pairs without accessing either certification split.

    The pre-specified lexicographic key uses the same add-one estimates as
    the expected-risk rung of the main protocol: (covered+1)/(n+1) for
    coverage and (harm+1)/(n_H+1) for severe harm.  It prioritizes coverage
    feasibility, then severe-harm feasibility, then utility.  Numeric
    beta/gamma fields provide deterministic tie-breaking.  Construct
    statistics are proposal scores, not certificates.
    """
    p = np.asarray(probabilities, float)
    y = np.asarray(labels, int)
    loss = np.asarray(loss_matrix, float)
    indicators = model.problem.table.indicators
    risks = model.problem.table.risks
    actions_by_set = model.problem.table.actions
    mass = p @ indicators.T
    severe_mass = p[:, model.severe_labels].sum(axis=1)
    severe = np.isin(y, model.severe_labels)
    n_severe = int(severe.sum())
    rows = []
    for beta, gamma in pairs:
        objective = (
            risks[None, :] - beta * mass
            + gamma * severe_mass[:, None] * model.harm_indicator[None, :]
        )
        set_indices = np.argmin(objective, axis=1)
        actions = actions_by_set[set_indices]
        covered = int(np.sum(indicators[set_indices, y]))
        coverage = covered / len(y)
        coverage_expected = (covered + 1) / (len(y) + 1)
        harm_count = int(np.sum(np.isin(actions[severe], model.harmful_actions)))
        harm_rate = harm_count / n_severe if n_severe else 1.0
        harm_expected = (harm_count + 1) / (n_severe + 1) if n_severe else 1.0
        policy_loss = float(np.mean(loss[y, actions]))
        rows.append({
            "beta": beta,
            "gamma": gamma,
            "construct_coverage": coverage,
            "construct_coverage_expected": coverage_expected,
            "construct_harm_count": harm_count,
            "construct_n_severe": n_severe,
            "construct_harm_rate": harm_rate,
            "construct_harm_expected": harm_expected,
            "construct_loss": policy_loss,
            "rank_key": [
                max(0.0, target_coverage - coverage_expected),
                max(0.0, harm_expected - delta),
                policy_loss,
                beta,
                gamma,
            ],
        })
    return sorted(rows, key=lambda row: tuple(row["rank_key"]))


def test_metrics(model, probabilities, labels, beta, gamma, loss_matrix,
                 severe_labels, harmful_actions):
    """Compute held-out coverage, loss, severe-action errors and set sizes."""
    sets = model.sets(probabilities, beta, gamma)
    actions = model.actions(probabilities, beta, gamma)
    severe = np.isin(labels, severe_labels)
    harmful = np.isin(actions, harmful_actions)
    return {
        "coverage": float(np.mean([int(y) in s for y, s in zip(labels, sets)])),
        "loss": float(np.mean(loss_matrix[labels, actions])),
        "harmful_rate": float(np.mean(harmful[severe])),
        "harm_count": int(harmful[severe].sum()),
        "n_severe": int(severe.sum()),
        "set_size": float(np.mean([len(s) for s in sets])),
    }


def summarize_fit(fit, model, test_p, test_y, loss, config):
    """Collect the selected policy, evidence bounds and held-out metrics."""
    selected = fit["selected"]
    global_group = selected["groups"][0]
    return {
        "shortlist_size": fit["num_candidates"],
        "tests_per_candidate": fit["multiplicity"]["tests_per_pair"],
        "per_test_error": fit["multiplicity"]["per_test_error"],
        "global_level": fit["global_level"],
        "counts": fit["counts"],
        "selected": {
            "beta": selected["beta"],
            "gamma": selected["gamma"],
            "coverage_empirical": selected["coverage_empirical"],
            "coverage_lower": selected["coverage_lower"],
            "selection_loss": selected["selection_loss"],
            "global_harm_count": global_group["harm_count"],
            "global_n_severe": global_group["n"],
            "global_harm_empirical": global_group["empirical"],
            "global_harm_upper": global_group["upper"],
            "global_harm_upper_minus_empirical": (
                global_group["upper"] - global_group["empirical"]
            ),
        },
        "subgroups": fit["subgroups"],
        "test": test_metrics(
            model, test_p, test_y, selected["beta"], selected["gamma"], loss,
            config["severe_labels"], config["harmful_actions"],
        ),
    }


def minimum_n_zero_harm(delta, per_test_error):
    """Return the minimum zero-error sample count needed to pass the harm gate."""
    return int(math.ceil(math.log(per_test_error) / math.log(1.0 - delta)))


def main():
    """Run the command-line entry point."""
    root = load_config((ROOT / "configs/experiments.yaml"))
    config = root["datasets"]["covid19"]
    p, y = load(config["output_dir"], config["calibration_split"])
    test_p, test_y = load(config["output_dir"], config["test_split"])
    construct, coverage, safety = split_calibration(
        y, int(root["calibration_split_seed"]),
    )
    loss = np.asarray(config["loss_matrix"], float)
    model = CertifiableHCROCP(
        FiniteDecisionProblem(loss, alpha=float(root["alpha"])),
        config["severe_labels"], config["harmful_actions"],
    )
    gammas = np.r_[
        0.0,
        np.geomspace(root["gamma_min_positive"], root["gamma_max"],
                     root["gamma_log_points"]),
    ]
    targets = [config["coverage_target"], 0.975, 0.99]
    pairs = build_full_pairs(model, p[construct], y[construct], gammas, targets)
    ranked = construct_only_ranking(
        model, p[construct], y[construct], pairs, loss,
        float(config["coverage_target"]), float(root["harm_delta"]),
    )
    groups = automatic_groups(
        config["severe_labels"], y[safety], float(root["harm_delta"]),
    )

    protocols = []
    for shortlist_size in (*SHORTLIST_SIZES, len(pairs)):
        finalists = ranked[:shortlist_size]
        finalist_pairs = [(row["beta"], row["gamma"]) for row in finalists]
        fit = model.fit_candidate_pairs(
            p[coverage], y[coverage], p[safety], y[safety], finalist_pairs,
            float(config["coverage_target"]), groups, loss, 0.95,
        )
        summary = summarize_fit(fit, model, test_p, test_y, loss, config)
        summary["protocol"] = (
            "full_family" if shortlist_size == len(pairs)
            else f"construct_only_top_{shortlist_size}"
        )
        summary["minimum_safety_n_if_zero_harm"] = minimum_n_zero_harm(
            float(root["harm_delta"]), summary["per_test_error"],
        )
        protocols.append(summary)

    result = {
        "dataset": "covid19",
        "delta": float(root["harm_delta"]),
        "family_confidence": 0.95,
        "full_family_size": len(pairs),
        "shortlist_sizes_pre_specified": list(SHORTLIST_SIZES),
        "ranking": {
            "data": "construct split only",
            "key": [
                "max(0, target_coverage - construct_coverage_expected)",
                "max(0, construct_harm_expected - delta)",
                "construct_loss",
                "beta",
                "gamma",
            ],
            "certification_or_test_data_used": False,
        },
        "splits": {
            "construct": len(construct),
            "coverage_certification": len(coverage),
            "safety_certification": len(safety),
            "safety_severe": int(np.isin(y[safety], config["severe_labels"]).sum()),
        },
        "interpretation": (
            "Each shortlist size is a separate pre-specified protocol; no "
            "shortlist size is selected after certification or test evaluation."
        ),
        "protocols": protocols,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for row in protocols:
        selected = row["selected"]
        print(
            row["protocol"],
            "L=", row["shortlist_size"],
            "level=", row["global_level"],
            "cov_lb=", round(selected["coverage_lower"], 4),
            "harm_ub=", round(selected["global_harm_upper"], 4),
            "k/n=", f'{selected["global_harm_count"]}/{selected["global_n_severe"]}',
            "loss=", round(row["test"]["loss"], 4),
            "test_harm=", round(row["test"]["harmful_rate"], 4),
        )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
