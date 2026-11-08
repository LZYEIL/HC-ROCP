#!/usr/bin/env python3
"""Complete the exact baseline and report paired ten-seed comparisons."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from shared.paths import OUTPUTS_DIR, load_config

import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import yaml
from scipy.stats import ttest_rel
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.evaluation.evaluate_hc_rocp_multiseed import evaluate, load
from core.decision_problem import FiniteDecisionProblem
from core.policy import HarmConstrainedROCP

METHODS = [('ROCP', 'ROCP (release solver)'), ('ExactROCP', 'ROCP (exact, matched)'),
           ('RAC', 'RAC'), ('LAS', 'LAS'), ('APS', 'APS'),
           ('RandomizedRAPS', 'RAPS (rand., official)'),
           ('RandomizedSAPS', 'SAPS (rand., formula)'), ('MondrianAPS', 'Mondrian APS'),
           ('SOCOP', 'SOCOP (official)'), ('UtilityDirectedCP', 'Utility-directed CP (score adapter)'),
           ('BayesAction', 'Bayes action'), ('AlwaysSafeAction', 'Always-safe action'),
           ('CRC_official_action_adapter', 'CRC action (official)'), ('HCROCP', 'Certifiable HC-ROCP')]
DATASETS = ['covid19', 'isic2019', 'eyepacs_to_aptos']


def load_baseline_records(directory):
    """Accept the default all-dataset run or explicitly saved per-dataset runs."""
    combined = directory / 'baselines_multiseed_10.json'
    if combined.is_file():
        datasets = json.loads(combined.read_text())['datasets']
        missing = set(DATASETS) - datasets.keys()
        if missing:
            raise ValueError(f"{combined} is missing datasets: {sorted(missing)}; run evaluate_baselines_multiseed.py without --dataset")
        return datasets
    return {ds: json.loads((directory / f'baselines_{ds}_10.json').read_text())['datasets'][ds]
            for ds in DATASETS}

def holm(pvalues):
    """Apply step-down Holm adjustment to a finite family."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted

def main():
    """Assemble a complete CSV, then derive every reported statistic from it."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, default=OUTPUTS_DIR)
    parser.add_argument('--output-dir', type=Path, default=OUTPUTS_DIR)
    args = parser.parse_args()
    artifacts = args.input_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config((ROOT / 'configs/experiments.yaml'))
    hc = json.loads((artifacts / 'certifiable_multiseed_10.json').read_text())
    seeds = hc['protocol']['seeds']
    assert seeds == list(range(20260815, 20260825))
    baselines = load_baseline_records(artifacts)
    rows = []
    size_rows = []
    for ds in DATASETS:
        for row in baselines[ds]['runs']:
            rows.append(dict(dataset=ds, method=row['method'], seed=row['seed'],
                             coverage=row['coverage'], harm=row['harmful_action_rate'], loss=row['mean_realized_loss']))
            size_rows.append(dict(dataset=ds, method=row['method'], seed=row['seed'],
                                  set_size=row['mean_set_size']))
        c = cfg['datasets'][ds]
        p, y = load(c['output_dir'], c['calibration_split'])
        tp, ty = load(c['output_dir'], c['test_split'])
        loss = np.asarray(c['loss_matrix'], float)
        model = HarmConstrainedROCP(FiniteDecisionProblem(loss, alpha=cfg['alpha']), c['severe_labels'], c['harmful_actions'])
        for seed in seeds:
            _, remaining = train_test_split(np.arange(len(y)), test_size=2/3, random_state=seed, stratify=y)
            cov, _ = train_test_split(remaining, test_size=.5, random_state=seed+1, stratify=y[remaining])
            beta = model.calibrate_beta(p[cov], y[cov], 0., c['coverage_target'])
            t = evaluate(model, tp, ty, beta, 0., loss, c)
            rows.append(dict(dataset=ds, method='ExactROCP', seed=seed, coverage=t['coverage'], harm=t['harmful_rate'], loss=t['loss']))
            size_rows.append(dict(dataset=ds, method='ExactROCP', seed=seed, set_size=t['set_size']))
        for row in hc['datasets'][ds]['runs']:
            t = row['test']
            rows.append(dict(dataset=ds, method='HCROCP', seed=row['seed'], coverage=t['coverage'], harm=t['harmful_rate'], loss=t['loss']))
            size_rows.append(dict(dataset=ds, method='HCROCP', seed=row['seed'], set_size=t['set_size']))
    with (args.output_dir / 'set_sizes_10.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['dataset', 'method', 'mean', 'std'])
        writer.writeheader()
        for ds in DATASETS:
            for method, _ in METHODS:
                group = [r for r in size_rows if r['dataset'] == ds and r['method'] == method]
                assert sorted(r['seed'] for r in group) == seeds
                sizes = [r['set_size'] for r in group if r['set_size'] is not None]
                assert len(sizes) in (0, len(seeds))
                writer.writerow(dict(dataset=ds, method=method,
                                     mean=float(np.mean(sizes)) if sizes else 'N/A',
                                     std=float(np.std(sizes, ddof=1)) if sizes else 'N/A'))
    csv_path = args.output_dir / 'paired_comparison_10.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['dataset', 'method', 'seed', 'coverage', 'harm', 'loss'])
        writer.writeheader()
        writer.writerows(rows)
    with csv_path.open() as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 420
    report = {}
    for ds in DATASETS:
        values = {}
        report[ds] = {}
        for method, _ in METHODS:
            group = sorted([r for r in records if r['dataset'] == ds and r['method'] == method], key=lambda r: int(r['seed']))
            assert [int(r['seed']) for r in group] == seeds
            values[method] = {metric: np.array([float(r[metric]) if r[metric] else np.nan for r in group]) for metric in ['coverage', 'harm', 'loss']}
            report[ds][method] = {}
        for metric, mei in [('coverage', .01), ('harm', .01), ('loss', .1)]:
            ours = values['HCROCP'][metric]
            names, pvalues = [], []
            for method, _ in METHODS:
                v = values[method][metric]
                if np.isnan(v).all():
                    report[ds][method][metric] = None
                    continue
                assert np.isfinite(v).all()
                report[ds][method][metric] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)), mark='')
                if method == 'HCROCP' or metric == 'coverage':
                    continue
                diff = ours - v
                if np.all(diff == 0):
                    pvalue = 1.
                elif np.all(diff == diff[0]):
                    pvalue = 0.
                else:
                    pvalue = float(ttest_rel(ours, v).pvalue)
                names.append(method)
                pvalues.append(pvalue)
            for method, p, pa in zip(names, pvalues, holm(pvalues)):
                effect = float((ours - values[method][metric]).mean()) * (1 if metric == 'coverage' else -1)
                mark = ('dagger' if effect >= mei else 'ddagger' if effect <= -mei else '') if pa < .05 else ''
                report[ds][method][metric].update(p=p, p_holm=float(pa), improvement=effect, mei=mei, mark=mark)
    (args.output_dir / 'paired_statistics_10.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    lines = []
    for method, label in METHODS:
        cells = []
        for ds in DATASETS:
            for metric in ['coverage', 'harm', 'loss']:
                item = report[ds][method][metric]
                if item is None:
                    cells.append('--')
                    continue
                scale, digits = (100, 2) if metric == 'harm' else (1, 3)
                suffix = ('^{\\' + item['mark'] + '}') if item['mark'] else ''
                cells.append(f"${item['mean']*scale:.{digits}f}\\pm{item['std']*scale:.{digits}f}{suffix}$")
        lines.append(label + ' & ' + ' & '.join(cells) + r'\\')
    (args.output_dir / 'paired_table_rows.tex').write_text('\n'.join(lines) + '\n')
    print(json.dumps({ds: {m: report[ds][m] for m in ['ExactROCP','HCROCP','ROCP']} for ds in DATASETS}, indent=2))

if __name__ == '__main__':
    main()
