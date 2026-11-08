#!/usr/bin/env python3
"""Global false-certification simulation: three settings, 400 trials per point.

Simulates binomial harm counts and exact Clopper--Pearson gates. Coverage and
data-dependent candidate construction are not simulated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.stats import beta as beta_dist

from shared.paths import OUTPUTS_DIR

EPS = 0.05
DELTA = 0.01
TRUE_HARM = 1.1 * DELTA
G = 4
F_LEAK = 25  # expanded-search pool multiplier


def cp_upper(k, n, conf):
    """Exact one-sided Clopper--Pearson upper bounds for binomial counts."""
    k = np.asarray(k, float)
    out = np.ones_like(k)
    m = k < n
    out[m] = beta_dist.ppf(conf, k[m] + 1.0, n - k[m])
    return out


def global_false_counts(rng, candidates, n_h, eta, trials):
    # Cache the exact CP bound for every possible count rather than recomputing
    # the same beta quantile for each candidate in every repetition.
    bounds = cp_upper(np.arange(n_h + 1), n_h, 1.0 - eta)
    counts = rng.binomial(n_h, TRUE_HARM, size=(trials, candidates))
    return int((bounds[counts] <= DELTA).any(axis=1).sum())


def run(trials=400, seed=20261108):
    if trials < 1:
        raise ValueError('trials must be positive')
    m = 100
    sizes = [50, 100, 200, 300, 500, 800, 1000, 1500, 2000]
    settings = {
        'Full': {'candidates': m, 'eta': EPS / (m * (1 + G))},
        'Reduced correction': {'candidates': m, 'eta': EPS / m},
        'Expanded search': {'candidates': F_LEAK * m, 'eta': EPS / (m * (1 + G))},
    }
    streams = np.random.SeedSequence(seed).spawn(len(settings) * len(sizes))
    counts, curves = {}, {}
    for i, (name, spec) in enumerate(settings.items()):
        counts[name] = [global_false_counts(
            np.random.default_rng(streams[i * len(sizes) + j]),
            spec['candidates'], n, spec['eta'], trials)
            for j, n in enumerate(sizes)]
        curves[name] = [k / trials for k in counts[name]]
    return {
        'experiment1': {'M': m, 'n_H': sizes, 'n_trials': trials,
                        'false_counts': counts, 'curves': curves, 'settings': settings},
        'provenance': {'seed': seed, 'numpy_version': np.__version__,
                       'delta': DELTA, 'true_harm': TRUE_HARM, 'epsilon': EPS, 'G': G,
                       'event': 'at least one false global-harm claim',
                       'rng': 'one spawned SeedSequence stream per setting/sample size',
                       'run_type': 'fresh run, no pooling or selection of runs',
                       'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path,
                        default=OUTPUTS_DIR / 'synthetic_protocol_study_uniform_400.json')
    parser.add_argument('--trials', type=int, default=400)
    parser.add_argument('--seed', type=int, default=20261108)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'Refusing to overwrite {args.output}')
    result = run(args.trials, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps(result, indent=2))
