# HC-ROCP

Code for **Certify, Report, or Refuse: Harm-Constrained Conformal Prediction
for Clinical Actions**.

## Repository layout

```text
HC_ROCP2/
├── core/           # our set/action construction, selection, and certification
├── baselines/      # self-contained comparison implementations and adapters
├── experiments/    # dataset-specific entry points, evaluations, and simulations
├── shared/         # data loading, preprocessing, classifier training, and utilities
├── configs/        # classifier and decision-problem YAML configurations
├── data/           # APTOS_2019, COVID, EyePACS, ISIC_2019 (download separately)
├── checkpoints/    # pretrained/fine-tuned weight instructions; no weights bundled
├── outputs/        # bundled frozen inputs; generated results are ignored
└── environment.yml
```

The full HC-ROCP protocol is implemented in
`core/hc_rocp.py`, with finite-set construction in
`core/decision_problem.py` and `core/policy.py`. Protocol diagnostics
are under `experiments/evaluation/`. Shared classifier code is in
`shared/backbone.py`; it is not part of the training-free wrapper.
See [shared/README.md](shared/README.md) for the common training and data utilities.
See [experiments/README.md](experiments/README.md) for the experiment folders and entry points.

This repository reproduces numerical results; figure-rendering code is not included.

## Setup

From the repository root, create the `hc-rocp` Conda environment with the
dependencies listed in `environment.yml`, then activate it:

```bash
conda env create -f environment.yml
conda activate hc-rocp
```

The YAML pins direct dependencies for Linux x86_64 and PyTorch with CUDA 12.4.
Conda and pip install the transitive dependencies automatically. GPU training
requires a compatible NVIDIA driver on the host.

Baseline code is included by method under `baselines/`; no separate baseline
repository download is required. Only the needed definitions and adapters are
included. See [baseline documentation](baselines/README.md) for a brief guide
to the method folders.

## Data and weights

Frozen classifier probabilities and split manifests are bundled in `outputs/`.
Wrapper evaluations need no additional data or weight downloads.
Original images and trained weights are not bundled.
Dataset sources and required files are described in each dataset's README under `data/`.
For optional command-line downloads from Kaggle, install `python -m pip install kaggle==2.2.4`
and configure your Kaggle credentials; browser downloads do not require this tool.
Download our fine-tuned weights from
[Hugging Face](https://huggingface.co/LZYEIL/HC-ROCP-Assets).
See [checkpoints/README.md](checkpoints/README.md) for file placement and the
separate official initialization weights needed only for training.

Place downloaded resources in these locations; the default commands need no
path arguments or YAML edits. Extract images and label files in the exact layout
shown in each dataset README (placing a ZIP archive alone is not sufficient).
All required destination directories are included, including the image and label
subdirectories inside `data/`. Merge extracted files into those directories;
do not add another archive-name directory above them. `.gitkeep` files only
preserve empty folders in Git and can be left in place.

| Resource | Location |
|---|---|
| Original datasets | `data/COVID/`, `data/ISIC_2019/`, `data/EyePACS/`, `data/APTOS_2019/` |
| Official initialization weights (training only) | `checkpoints/convnextv2_tiny_22k_224_ema.pt` |
| Our fine-tuned weights | `checkpoints/<dataset>/best.pt` |
| Frozen probabilities and manifests (included) | `outputs/<dataset>/` |

Here `<dataset>` is `covid19`, `isic2019`, or `eyepacs_to_aptos`.
The weight bundle contains three dataset-named `.pt` files. Place each in its
matching `checkpoints/<dataset>/` folder and rename it to `best.pt`; see the
[filename mapping](checkpoints/README.md#our-fine-tuned-classifiers).
The six calibration/test
probability files and three manifests are already included in `outputs/`.
Original images come from the dataset providers, not the weight bundle.
Use of the dataset-derived files remains subject to the original dataset terms.

All default paths are anchored to this repository's location, not the current
working directory or the author's machine. For non-default external storage only, set:

```bash
export HC_ROCP_DATASETS_DIR=/absolute/path/to/datasets
export HC_ROCP_MODELS_DIR=/absolute/path/to/checkpoints
export HC_ROCP_OUTPUTS_DIR=/absolute/path/to/experiment_outputs
```

The dataset root must contain `APTOS_2019/`, `COVID/`, `EyePACS/`, and
`ISIC_2019/` (case-sensitive). Relative override values resolve against this
repository root. Explicit absolute paths in custom YAML configurations also
work. Released manifests use portable `data/...` image paths.
These settings do not copy or relocate data.
Relative YAML paths, `--config`, and `--checkpoint` resolve from the repository
root. Explicit relative input/output CLI paths (such as `--output` or
`--input-dir`) resolve from the current working directory.

## Reproduction

`configs/` contains four files, with no subdirectories:

- `train_covid19.yaml`, `train_isic2019.yaml`, and `train_eyepacs_to_aptos.yaml`:
  classifier training and probability export.
- `experiments.yaml`: shared actions, loss matrices, data splits, and parameters
  for HC-ROCP and baseline experiments.

Using the bundled frozen probabilities, run the full ten-seed comparison:

```bash
python experiments/evaluation/evaluate_hc_rocp_multiseed.py
python experiments/evaluation/evaluate_baselines_multiseed.py
python experiments/evaluation/summarize_comparison.py
```

No images or checkpoints are needed for these three commands. Use the bundled frozen probabilities
for exact numerical comparisons with the paper; re-exporting on different
hardware can introduce floating-point differences.

To export probabilities from our downloaded COVID-19 weights and the original images:

```bash
python experiments/covid19/run.py export
```

To train your own COVID-19 classifier from the official initialization weights:

```bash
python experiments/covid19/run.py train
python experiments/covid19/run.py export
```

Replace `covid19` with `isic2019` or `eyepacs_to_aptos` for the other settings.
Training and export generate a missing manifest automatically; an existing
manifest is preserved. Training requires a GPU and will not overwrite existing
weights unless explicitly resumed. Follow [experiments/README.md](experiments/README.md) for matched
baseline runs, paired statistics, sensitivity analyses, and simulation.
The [paper-to-script mapping](experiments/README.md#paper-to-script-mapping)
also covers the main-text and appendix numerical results.

## Outputs

See [outputs/README.md](outputs/README.md). The reference manifests and probabilities
are included under `outputs/<dataset>/`; training and export also write there.
Model weights are saved under
`checkpoints/<dataset>/`. Experiment results and diagnostics go directly to `outputs/`.

Results are generated by the experiment scripts and are not bundled.
Run the HC-ROCP and baseline evaluations before the statistical summary command.
Cost sensitivity runs independently; comparisons against an existing ten-seed
run are optional. Git ignores generated files except the nine bundled input
files; re-exporting probabilities to those same paths changes tracked files.
