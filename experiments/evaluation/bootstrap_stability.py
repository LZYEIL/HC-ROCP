#!/usr/bin/env python3
"""Nested bootstrap that refits Certifiable HC-ROCP for every resample.

The calibration sample is resampled within label strata.  The complete
construct/coverage/safety split, candidate construction, certification and
selection protocol is then rerun.  The held-out test sample is bootstrapped
independently for evaluation.  Thus intervals include both calibration and
test-sampling variability.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from shared.paths import OUTPUTS_DIR, load_config

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.evaluation.evaluate_hc_rocp_multiseed import load, one_seed


def stratified_bootstrap_indices(labels, rng):
    """Resample indices within each label, preserving the observed class counts."""
    labels = np.asarray(labels, int)
    chunks = []
    for value in np.unique(labels):
        indices = np.flatnonzero(labels == value)
        chunks.append(rng.choice(indices, size=len(indices), replace=True))
    result = np.concatenate(chunks)
    rng.shuffle(result)
    return result


def interval(values):
    """Summarize bootstrap draws by their mean, sample standard deviation and percentile interval."""
    values = np.asarray([x for x in values if x is not None], float)
    if not len(values):
        return {"mean": None, "standard_error": None, "percentile_95ci": [None, None]}
    return {
        "mean": float(values.mean()),
        "standard_error": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "percentile_95ci": [float(x) for x in np.percentile(values, [2.5, 97.5])],
    }


def summarize_rows(rows):
    """Summarize bootstrap metrics, selected parameters and evidence-level counts."""
    metrics = {
        key: interval([row["test"][key] for row in rows])
        for key in ["coverage", "loss", "harmful_rate", "set_size"]
    }
    return {
        "metrics": metrics,
        "guarantee_level_counts": dict(sorted(Counter(row["global_level"] for row in rows).items())),
        "beta": interval([row["selected"]["beta"] for row in rows]),
        "gamma": interval([row["selected"]["gamma"] for row in rows]),
    }


def run_dataset(name, config, root_config, repeats, seed, delta):
    """Refit and evaluate HC-ROCP for each calibration/test bootstrap resample."""
    probabilities, labels = load(config["output_dir"], config["calibration_split"])
    test_p, test_y = load(config["output_dir"], config["test_split"])
    seed_sequence = np.random.SeedSequence(seed)
    children = seed_sequence.spawn(repeats)
    rows = []
    for repeat, child in enumerate(children):
        rng = np.random.default_rng(child)
        calibration_indices = stratified_bootstrap_indices(labels, rng)
        test_indices = rng.integers(0, len(test_y), size=len(test_y))
        split_seed = int(rng.integers(0, 2**31 - 2))
        fitted = one_seed(
            probabilities[calibration_indices], labels[calibration_indices],
            test_p[test_indices], test_y[test_indices],
            config, root_config, split_seed, delta,
        )
        rows.append({
            "repeat": repeat,
            "split_seed": split_seed,
            "global_level": fitted["global_level"],
            "selected": fitted["selected"],
            "test": fitted["test"],
        })
    return {
        "num_repeats": repeats,
        "summary": summarize_rows(rows),
        "runs": rows,
    }


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["all", "isic2019", "covid19", "eyepacs_to_aptos"])
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--jobs", type=int, default=9, help="maximum CPU workers across dataset chunks")
    parser.add_argument("--output", default=str(OUTPUTS_DIR / "certifiable_nested_bootstrap.json"))
    args = parser.parse_args()
    if args.repeats < 1 or args.jobs < 1 or args.seed < 0:
        parser.error("repeats and jobs must be positive; seed must be nonnegative")
    if not 0 < args.delta < 1:
        parser.error("delta must lie strictly between 0 and 1")
    root_config = load_config((ROOT / "configs/experiments.yaml"))
    names = list(root_config["datasets"]) if args.dataset == "all" else [args.dataset]
    output = {
        "protocol": {
            "repeats": args.repeats,
            "seed": args.seed,
            "jobs": args.jobs,
            "delta": args.delta,
            "calibration_resampling": "within-label stratified pairs bootstrap; full protocol refit",
            "test_resampling": "independent nonparametric pairs bootstrap",
            "coverage_candidate_targets": [0.95, 0.975, 0.99],
        },
        "datasets": {},
    }
    if args.jobs == 1:
        for offset, name in enumerate(names):
            output["datasets"][name] = run_dataset(
                name, root_config["datasets"][name], root_config,
                args.repeats, args.seed + offset * 100000, args.delta,
            )
            print(name, output["datasets"][name]["summary"], flush=True)
    else:
        chunks_per_dataset = max(1, args.jobs // len(names))
        tasks = []
        for dataset_offset, name in enumerate(names):
            counts = [args.repeats // chunks_per_dataset] * chunks_per_dataset
            for index in range(args.repeats % chunks_per_dataset):
                counts[index] += 1
            for chunk, count in enumerate(counts):
                if count:
                    tasks.append((name, chunk, count, args.seed + dataset_offset * 100000 + chunk * 10000))
        collected = {name: [] for name in names}
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(tasks))) as executor:
            future_to_task = {
                executor.submit(
                    run_dataset, name, root_config["datasets"][name], root_config,
                    count, chunk_seed, args.delta,
                ): (name, chunk) for name, chunk, count, chunk_seed in tasks
            }
            for future in as_completed(future_to_task):
                name, chunk = future_to_task[future]
                result = future.result()
                for row in result["runs"]:
                    row["chunk"] = chunk
                collected[name].extend(result["runs"])
                print(name, f"chunk {chunk} complete ({len(result['runs'])} repeats)", flush=True)
        for name in names:
            rows = sorted(collected[name], key=lambda row: (row["chunk"], row["repeat"]))
            for repeat, row in enumerate(rows):
                row["repeat"] = repeat
            output["datasets"][name] = {
                "num_repeats": len(rows),
                "summary": summarize_rows(rows),
                "runs": rows,
            }
            print(name, output["datasets"][name]["summary"], flush=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
