#!/usr/bin/env python3
"""Ten-seed harm-cost sensitivity using the three-way calibration protocol.

Only severe-label x harmful-action entries are scaled. Each factor is a
separate, fixed sensitivity scenario, not selected on certification/test data.
Exact ROCP calibrates on the coverage split, as in Table 10.
Previously generated comparison results are optional consistency checks.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from shared.paths import OUTPUTS_DIR, load_config

import argparse
import copy
import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.hc_rocp import CertifiableHCROCP, automatic_groups
from core.decision_problem import FiniteDecisionProblem
from core.policy import HarmConstrainedROCP
from experiments.evaluation.evaluate_hc_rocp_multiseed import evaluate, load

DATASETS = ('covid19', 'isic2019', 'eyepacs_to_aptos')
SEEDS = tuple(range(20260815, 20260825))
FACTORS = (0.5, 1.0, 2.0)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def split_indices(labels, seed):
    construct, remaining = train_test_split(
        np.arange(len(labels)), test_size=2 / 3, random_state=seed, stratify=labels)
    coverage, safety = train_test_split(
        remaining, test_size=0.5, random_state=seed + 1, stratify=labels[remaining])
    assert len(np.unique(np.concatenate((construct, coverage, safety)))) == len(labels)
    return {'construct': construct, 'coverage': coverage, 'safety': safety}


def scaled_config(config, factor):
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError('cost factors must be finite and positive')
    result = copy.deepcopy(config)
    loss = np.asarray(config['loss_matrix'], float).copy()
    loss[np.ix_(config['severe_labels'], config['harmful_actions'])] *= factor
    result['loss_matrix'] = loss.tolist()
    return result


def test_metrics(model, p, y, beta, gamma, loss, config):
    result = evaluate(model, p, y, beta, gamma, loss, config)
    actions = model.actions(p, beta, gamma)
    result['n_test'] = len(y)
    result['action_counts'] = {
        str(a): int(np.sum(actions == a)) for a in range(loss.shape[1])}
    return result


def run_setting(p, y, tp, ty, config, root, seed, factor):
    c = scaled_config(config, factor)
    loss = np.asarray(c['loss_matrix'], float)
    arms = split_indices(y, seed)
    construct, coverage, safety = (arms[k] for k in ('construct', 'coverage', 'safety'))
    model = CertifiableHCROCP(
        FiniteDecisionProblem(loss, alpha=float(root['alpha'])),
        c['severe_labels'], c['harmful_actions'])
    gammas = np.r_[0.0, np.geomspace(
        root['gamma_min_positive'], root['gamma_max'], root['gamma_log_points'])]
    targets = [float(c['coverage_target']), 0.975, 0.99]
    betas = sorted({
        round(model.calibrate_beta(p[construct], y[construct], float(g), t), 10)
        for t in targets for g in gammas})
    groups = automatic_groups(c['severe_labels'], y[safety], root['harm_delta'])
    fit = model.fit_certifiable(
        p[coverage], y[coverage], p[safety], y[safety], betas, gammas,
        c['coverage_target'], groups, loss, 0.95)
    selected = fit['selected']
    exact = HarmConstrainedROCP(
        FiniteDecisionProblem(loss, alpha=float(root['alpha'])),
        c['severe_labels'], c['harmful_actions'])
    beta0 = exact.calibrate_beta(p[coverage], y[coverage], 0.0, c['coverage_target'])
    # Test labels enter reporting only after both policies have been frozen.
    hc_test = test_metrics(model, tp, ty, selected['beta'], selected['gamma'], loss, c)
    exact_test = test_metrics(exact, tp, ty, beta0, 0.0, loss, c)
    return {
        'seed': int(seed), 'harm_cost_factor': float(factor), 'loss_matrix': loss.tolist(),
        'split_audit': {
            name: {'n': len(ids),
                   'indices_sha256': hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest(),
                   'class_counts': np.bincount(y[ids], minlength=loss.shape[0]).tolist()}
            for name, ids in arms.items()},
        'candidate_grid': {'beta': betas, 'gamma': gammas.tolist(), 'coverage_targets': targets},
        'hc_rocp': {'beta': selected['beta'], 'gamma': selected['gamma'],
                    'test': hc_test, 'certification': fit},
        'exact_rocp_gamma0': {'beta': beta0, 'gamma': 0.0, 'test': exact_test}}


def assert_close(actual, expected, label):
    if isinstance(expected, (bool, str)) or expected is None:
        if actual != expected:
            raise AssertionError(f'{label}: {actual!r} != {expected!r}')
    elif not np.isclose(actual, expected, rtol=1e-12, atol=1e-12):
        raise AssertionError(f'{label}: {actual!r} != {expected!r}')


def check_paper_anchor(dataset, row, paper, paired):
    """Compare the 1x-cost setting against explicitly supplied reference results."""
    seed = row['seed']
    reference = next(r for r in paper['datasets'][dataset]['runs'] if r['seed'] == seed)
    ours = row['hc_rocp']
    fit = ours['certification']
    for key in ('beta', 'gamma'):
        assert_close(ours[key], reference['selected'][key], f'{dataset}/{seed}/{key}')
    assert_close(fit['global_level'], reference['global_level'], 'level')
    assert_close(fit['num_candidates'], reference['num_candidates'], 'family size')
    assert len(row['candidate_grid']['beta']) == reference['num_beta_candidates']
    assert fit['counts'] == reference['counts']
    for key, value in reference['test'].items():
        assert_close(ours['test'][key], value, f'{dataset}/{seed}/test/{key}')
    for name in ('construct', 'coverage', 'safety'):
        assert row['split_audit'][name]['n'] == reference['split_sizes'][name]
    for method, key in (('HCROCP', 'hc_rocp'), ('ExactROCP', 'exact_rocp_gamma0')):
        ref = paired[(dataset, method, seed)]
        for csv_key, test_key in (('coverage', 'coverage'), ('harm', 'harmful_rate'), ('loss', 'loss')):
            assert_close(row[key]['test'][test_key], float(ref[csv_key]),
                         f'{dataset}/{seed}/{method}/{csv_key}')


def summarize(rows):
    output = {}
    for method in ('hc_rocp', 'exact_rocp_gamma0'):
        metrics = {}
        for metric in ('coverage', 'loss', 'harmful_rate', 'set_size'):
            values = np.asarray([r[method]['test'][metric] for r in rows
                                 if r[method]['test'][metric] is not None], float)
            if not len(values):
                metrics[metric] = {'mean': None, 'std': None, 'min': None, 'max': None}
                continue
            metrics[metric] = {
                'mean': float(values.mean()),
                'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                'min': float(values.min()), 'max': float(values.max())}
        output[method] = {'test_metrics': metrics}
    output['hc_rocp']['evidence_level_counts'] = dict(sorted(Counter(
        r['hc_rocp']['certification']['global_level'] for r in rows).items()))
    output['hc_rocp']['subgroup_certified_seed_counts'] = {
        name: sum(s['certified'] for r in rows for s in r['hc_rocp']['certification']['subgroups']
                  if s['name'] == name)
        for name in sorted({s['name'] for r in rows for s in r['hc_rocp']['certification']['subgroups']})}
    return output


def write_outputs(output, json_path, csv_path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + '\n')
    fields = ['dataset', 'harm_cost_factor', 'method', 'seed', 'coverage', 'harm', 'loss',
              'set_size', 'harm_count', 'n_severe', 'beta', 'gamma', 'evidence_level']
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset, data in output['datasets'].items():
            for row in data['runs']:
                for method, key in (('HCROCP', 'hc_rocp'), ('ExactROCP', 'exact_rocp_gamma0')):
                    record = row[key]
                    metric = record['test']
                    writer.writerow({
                        'dataset': dataset, 'harm_cost_factor': row['harm_cost_factor'],
                        'method': method, 'seed': row['seed'], 'coverage': metric['coverage'],
                        'harm': metric['harmful_rate'], 'loss': metric['loss'],
                        'set_size': metric['set_size'], 'harm_count': metric['harm_count'],
                        'n_severe': metric['n_severe'], 'beta': record['beta'], 'gamma': record['gamma'],
                        'evidence_level': record['certification']['global_level']
                        if method == 'HCROCP' else 'not_applicable'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--factors', nargs='+', type=float, default=list(FACTORS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--output', type=Path,
                        default=OUTPUTS_DIR / 'loss_sensitivity_certifiable_10.json')
    parser.add_argument('--csv-output', type=Path,
                        default=OUTPUTS_DIR / 'loss_sensitivity_certifiable_10.csv')
    parser.add_argument('--reference-dir', type=Path,
                        help='Optional directory with ten-seed results for 1x consistency checks')
    args = parser.parse_args()
    if args.output.resolve() == args.csv_output.resolve():
        parser.error('JSON and CSV outputs must use different paths')
    for values in (args.datasets, args.factors, args.seeds):
        if len(values) != len(set(values)):
            parser.error('datasets, factors, and seeds must each be unique')
    if not set(args.seeds).issubset(SEEDS):
        parser.error('use the established ten calibration seeds for paper-anchor checks')
    if any(not np.isfinite(f) or f <= 0 for f in args.factors):
        parser.error('cost factors must be finite and positive')
    for path in (args.output, args.csv_output):
        if path.exists():
            parser.error(f'refusing to overwrite existing result: {path}')
    config_path = ROOT / 'configs/experiments.yaml'
    root = load_config(config_path)
    paper = paired = None
    reference_paths = []
    if args.reference_dir is not None:
        paper_path = args.reference_dir / 'certifiable_multiseed_10.json'
        paired_path = args.reference_dir / 'paired_comparison_10.csv'
        for path in (paper_path, paired_path):
            if not path.is_file():
                parser.error(f'Missing requested reference file: {path}')
        paper = json.loads(paper_path.read_text())
        with paired_path.open() as handle:
            paired = {(r['dataset'], r['method'], int(r['seed'])): r for r in csv.DictReader(handle)}
        assert paper['protocol']['delta'] == root['harm_delta']
        reference_paths = [paper_path, paired_path]
    source_paths = [
        Path(__file__), config_path,
        ROOT / 'core/decision_problem.py', ROOT / 'core/policy.py',
        ROOT / 'core/hc_rocp.py',
        ROOT / 'experiments/evaluation/evaluate_hc_rocp_multiseed.py']
    output = {
        'schema_version': 2, 'complete': False,
        'protocol': {
            'seeds': args.seeds, 'harm_cost_factors': args.factors,
            'alpha': float(root['alpha']), 'delta': float(root['harm_delta']),
            'family_confidence': 0.95, 'coverage_candidate_targets': [0.95, 0.975, 0.99],
            'hc_calibration': 'unchanged construct/coverage/safety three-way protocol',
            'exact_calibration': 'gamma=0; beta fitted on the same coverage arm as Table 10',
            'scaled_entries': 'severe_labels x harmful_actions only',
            'factor_selection': 'separate fixed sensitivity scenarios; no selection across factors',
            'test_design': 'fixed test sets; only calibration partitions vary'},
        'provenance': {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
            'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
            'reference_sha256': {str(p.resolve()): sha256(p) for p in reference_paths},
            'inputs': {}},
        'checks': {'paper_anchor_enabled': paper is not None,
                   'paper_anchor_settings_checked': 0}, 'datasets': {}}
    start = time.perf_counter()
    for dataset in args.datasets:
        c = root['datasets'][dataset]
        paths = [Path(c['output_dir']) / f"probabilities_{c[k]}.npz"
                 for k in ('calibration_split', 'test_split')]
        output['provenance']['inputs'][dataset] = {str(p): sha256(p) for p in paths}
        p, y = load(c['output_dir'], c['calibration_split'])
        tp, ty = load(c['output_dir'], c['test_split'])
        rows = []
        for factor in args.factors:
            for seed in args.seeds:
                tick = time.perf_counter()
                row = run_setting(p, y, tp, ty, c, root, seed, factor)
                row['elapsed_seconds'] = time.perf_counter() - tick
                if factor == 1.0 and paper is not None:
                    check_paper_anchor(dataset, row, paper, paired)
                    output['checks']['paper_anchor_settings_checked'] += 1
                rows.append(row)
                print(json.dumps({
                    'dataset': dataset, 'factor': factor, 'seed': seed,
                    'seconds': round(row['elapsed_seconds'], 3),
                    'level': row['hc_rocp']['certification']['global_level'],
                    'test': row['hc_rocp']['test']}, allow_nan=False), flush=True)
        output['datasets'][dataset] = {
            'runs': rows,
            'summaries': [
                {'harm_cost_factor': factor,
                 **summarize([r for r in rows if r['harm_cost_factor'] == factor])}
                for factor in args.factors]}
        write_outputs(output, args.output, args.csv_output)
    output['elapsed_seconds'] = time.perf_counter() - start
    output['complete'] = True
    write_outputs(output, args.output, args.csv_output)
    print(f"COMPLETE: {args.output}; {args.csv_output}; {output['elapsed_seconds']:.1f} seconds",
          flush=True)


if __name__ == '__main__':
    main()
