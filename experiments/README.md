# Experiments

Run commands below from the repository root, with the environment in
[the main README](../README.md) activated. Real-data evaluations use the frozen
classifier probabilities bundled in `outputs/`; no images or trained weights
are needed. Simulations need neither these files nor weights.

## Folders

| Folder | Purpose |
|---|---|
| `covid19/` | COVID-19 data preparation, classifier training, and probability export. |
| `isic2019/` | ISIC 2019 data preparation, classifier training, and probability export. |
| `eyepacs_to_aptos/` | EyePACS-to-APTOS data preparation, classifier training, and probability export. |
| `evaluation/` | Method comparisons and supplementary experiments on frozen probabilities. |
| `simulation/` | Synthetic certification-power and false-certification experiments. |

## Dataset folders

Each of the three dataset folders contains the same two entry points:

- `prepare_manifest.py`: builds the image/label manifest and fixed data splits.
- `run.py`: selects classifier training (`train`) or probability export (`export`).

```bash
python experiments/covid19/run.py train
python experiments/covid19/run.py export
```

Replace `covid19` with `isic2019` or `eyepacs_to_aptos` for the other settings.
The matching `configs/train_<dataset>.yaml` is selected automatically.
Use `train --resume` to resume training, or `--config` for a custom YAML.
Resume requires the training-state checkpoint `last.pt`; downloaded `best.pt`
weights are for inference. Resume restores optimization and early-stopping state,
but does not replay the uninterrupted run's data-augmentation random stream.
Export uses `checkpoints/<dataset>/best.pt` if `--checkpoint` is omitted.
Training saves weights to the same directory and refuses to overwrite existing
weights without `--resume`. Training and export create missing manifests automatically.
Download fine-tuned weights from
[Hugging Face](https://huggingface.co/LZYEIL/HC-ROCP-Assets) and follow the
[filename mapping](../checkpoints/README.md#our-fine-tuned-classifiers).
Then skip training and run `export` directly; no official initialization
checkpoint is required for export.
Preparation accepts `--data-dir`, `--output`, and explicit `--overwrite`.
For images outside the repository, set `HC_ROCP_DATASETS_DIR` to the same
root used for preparation; generated manifests retain portable `data/...` paths.
These entry points support `--help`.

Common image loading, preprocessing, classifier training, and probability export
live in the top-level [shared/](../shared/README.md) package. Normally use the
dataset entry points above; they call those implementations automatically.

## Evaluation

These scripts use frozen probabilities and do not retrain classifiers.
Shared decision costs, actions, and protocol settings are in `configs/experiments.yaml`.
`ablation_protocol.py` defines its fixed diagnostic settings in the script.

| File in `evaluation/` | Purpose |
|---|---|
| `evaluate_hc_rocp.py` | Full HC-ROCP on the reference calibration split, including subgroup details. |
| `evaluate_hc_rocp_multiseed.py` | HC-ROCP across calibration seeds. |
| `evaluate_baselines.py` | Baselines on the reference calibration split. |
| `evaluate_baselines_multiseed.py` | Baselines across matched calibration seeds. |
| `evaluate_ltt.py` | Action-only LTT comparison. |
| `evaluate_frontier.py` | Evaluates candidates for the harm–loss frontier. |
| `ablation_protocol.py` | Split, correction, grid, coverage-target, and sample-count diagnostics. |
| `ablation_shortlist.py` | Construct-only candidate screening on COVID-19. |
| `ablation_referral.py` | COVID-19 referral-action cost sweep. |
| `sensitivity_cost.py` | Repeats evaluation with scaled harm costs. |
| `sensitivity_budget.py` | Compares certification at different harm budgets. |
| `bootstrap_stability.py` | Nested bootstrap with recalibration and evaluation. |
| `summarize_comparison.py` | Paired ten-seed statistics, set-size summaries, and table rows. |
| `__init__.py` | Python package marker. |

`evaluate_baselines.py` uses the full calibration split by default. The
multi-seed runner instead calibrates baselines on each seed's coverage split.

To run the ten-seed comparison for all three settings and summarize it:

```bash
python experiments/evaluation/evaluate_hc_rocp_multiseed.py
python experiments/evaluation/evaluate_baselines_multiseed.py
python experiments/evaluation/summarize_comparison.py
```

The multi-seed runners also accept `--dataset covid19` for a single setting.
The summary command expects all three datasets and the default ten seeds.
It reads the combined `baselines_multiseed_10.json` by default, with support for
separate `baselines_<dataset>_10.json` files when no combined file is present.
For a supplementary experiment, run its script directly, for example:

```bash
python experiments/evaluation/ablation_shortlist.py
python experiments/evaluation/bootstrap_stability.py
```

For the paper's bootstrap settings, use the default all-dataset run with 500
repetitions, seed 20260821, and 9 workers. Dataset ordering and worker chunking
determine the random streams; changing `--dataset` or `--jobs` changes the draws.

`evaluate_hc_rocp.py`, `evaluate_ltt.py`, `sensitivity_budget.py`, and the three
`ablation_*.py` scripts run without arguments, using fixed settings and output
paths. They do not implement `--dataset` or `--help`. Other evaluation scripts
support `--help`.

Run cost sensitivity independently with:

```bash
python experiments/evaluation/sensitivity_cost.py
```

It writes `loss_sensitivity_certifiable_10.json` and
`loss_sensitivity_certifiable_10.csv` under the output root and refuses to
overwrite existing files. Use `--output` and `--csv-output` for other locations.
Optionally pass `--reference-dir outputs` to check the 1x-cost
settings against a completed ten-seed comparison.

## Simulation

| File in `simulation/` | Purpose |
|---|---|
| `synthetic_certifiability.py` | Single-policy certification probability versus sample size and true error rate (5,000 repetitions). |
| `simulate_protocol_study_uniform.py` | Current three-setting false-certification experiment, with 400 repetitions per point. |

```bash
python experiments/simulation/synthetic_certifiability.py
python experiments/simulation/simulate_protocol_study_uniform.py
```

Use the uniform runner for the paper's current false-certification results.
It requires a new output file and will not overwrite an existing one.
The default is `outputs/synthetic_protocol_study_uniform_400.json`, with 400
repetitions and seed 20261108. Use `--output` to choose another destination.

## Inputs and outputs

- Dataset manifests, probabilities, and training logs default to `outputs/<dataset>/`.
- Downloaded and newly trained weights use `checkpoints/<dataset>/`.
- Experiment results and diagnostics default to files directly in `outputs/`;
  no previous results are required to start. `HC_ROCP_OUTPUTS_DIR` changes the
  root for both dataset files and results. Explicit output arguments take precedence.
  When overriding the root, copy the bundled dataset folders there to reuse
  the paper's frozen inputs.
- Several scripts overwrite their default output files. Use a fresh output path
  where supported, as in the examples above, to preserve existing results.

External data/probability locations can be configured with the environment variables
in [the main README](../README.md). The scripts do not download datasets or weights.

## Paper-to-script mapping

Paths below are relative to `experiments/`. Results are written to `outputs/`
unless a dataset subdirectory is listed. Figures 1–2 are the case illustration
and conceptual overview; figure rendering and the manuscript are not bundled.

| Paper content | Script / source | Numerical output |
|---|---|---|
| Reference policy and subgroup evidence (Table 2) | `evaluation/evaluate_hc_rocp.py` | `certifiable_hc_rocp.json` |
| Ten-seed comparisons (Tables 3, 10; set sizes in Section 7.5) | The three comparison commands above | `paired_comparison_10.csv`, `paired_statistics_10.json`, `set_sizes_10.csv` |
| COVID-19 trade-off (Figure 3) | `evaluation/evaluate_frontier.py`, `evaluation/evaluate_baselines.py`, and the reference-policy script | `covid19_frontier.json`, `covid19/standard_baselines_test.json`, `certifiable_hc_rocp.json` |
| Budgets, subgroups, LTT (Figures 4, 5(b), 7; Table 9) | `evaluation/sensitivity_budget.py` | `delta_sensitivity.json` |
| Zero-error sample requirements (Table 4) | `evaluation/evaluate_hc_rocp.py` | `certifiable_hc_rocp.json` (`zero_error_sample_requirements`) |
| Synthetic power and real-data count diagnostic (Figure 5(a)) | `simulation/synthetic_certifiability.py`, `evaluation/ablation_protocol.py` | `synthetic_certifiability.json`, `ablation_study.json` (`subsample`) |
| Cost sensitivity (Figure 6) | `evaluation/sensitivity_cost.py` | `loss_sensitivity_certifiable_10.json` and `.csv` |
| Protocol and coverage-target diagnostics (Tables 11, 13) | `evaluation/ablation_protocol.py` | `ablation_study.json` (`table5` and `table6` keys, respectively) |
| Candidate shortlisting (Table 12) | `evaluation/ablation_shortlist.py` | `construct_shortlist_ablation.json` |
| Nested bootstrap (Section 7.6, Appendix F) | `evaluation/bootstrap_stability.py` | `certifiable_nested_bootstrap.json` |
| False-certification simulation (Figure 8) | `simulation/simulate_protocol_study_uniform.py` | `synthetic_protocol_study_uniform_400.json` |
| Referral (Table 14) | `evaluation/ablation_referral.py` | `refer_ablation.json` |
| Data splits, loss matrices, training settings (Tables 6–8) | Released manifests, `configs/experiments.yaml`, `configs/train_*.yaml` | Split counts in manifests; expert costs and training settings in YAML |

Tables 1 and 5 define the reporting rules and notation; they are not additional
experiments. No cached results outside this repository are required.
