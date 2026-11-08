# HC-ROCP

Code for **Certify, Report, or Refuse: Harm-Constrained Conformal Prediction
for Clinical Actions**.

## Quick start

The frozen classifier probabilities and split manifests are included.
No images, model weights, or separate baseline repositories are needed to run
the wrapper evaluations.

```bash
git clone https://github.com/LZYEIL/HC-ROCP.git
cd HC-ROCP
conda env create -f environment.yml
conda activate hc-rocp

python experiments/evaluation/evaluate_hc_rocp_multiseed.py
python experiments/evaluation/evaluate_baselines_multiseed.py
python experiments/evaluation/summarize_comparison.py
```

The commands run the ten-seed comparison across all three settings and summarize
the results in `outputs/`. Run them in order. If you downloaded a ZIP instead,
start from the extracted repository root and skip `git clone` and `cd` above.

The environment targets Linux x86_64 with PyTorch and CUDA 12.4.
See the [experiment guide](experiments/README.md) for supplementary runs and the
[paper-to-script mapping](experiments/README.md#paper-to-script-mapping).
The repository reproduces numerical results; figure-rendering code is not included.

## Repository layout

```text
HC-ROCP/
├── core/           # HC-ROCP construction, selection, and certification
├── baselines/      # self-contained comparison methods
├── experiments/    # dataset entry points, evaluations, and simulations
├── shared/         # data loading, classifier training, and utilities
├── configs/        # training and decision-problem configurations
├── data/           # folders for separately downloaded datasets
├── checkpoints/    # folders for separately downloaded model weights
├── outputs/        # bundled frozen inputs and generated results
└── environment.yml
```

Component and dataset READMEs describe their files and use.
The main method is `core/hc_rocp.py:CertifiableHCROCP`; see
[core](core/README.md), [baselines](baselines/README.md),
[shared utilities](shared/README.md), and [configurations](configs/README.md).

## Images and weights (optional)

These are needed only for classifier training or exporting predictions from images.

| Resource | Download and placement |
|---|---|
| Original images and labels | Follow the dataset guides: [COVID-19](data/COVID/README.md), [ISIC 2019](data/ISIC_2019/README.md), [EyePACS](data/EyePACS/README.md), [APTOS 2019](data/APTOS_2019/README.md). |
| Our fine-tuned weights | Download from [Hugging Face](https://huggingface.co/LZYEIL/HC-ROCP-Assets), then use the [filename mapping](checkpoints/README.md#our-fine-tuned-classifiers). |
| Official initialization weights (training only) | Follow the [backbone instructions](checkpoints/README.md#official-pretrained-backbone). |

Extract datasets into the documented folders, without an extra archive-name
directory. Empty destination folders are preserved by `.gitkeep` files.
Dataset-derived files remain subject to the original dataset terms.

After placing the COVID-19 images and fine-tuned weights, export probabilities with:

```bash
python experiments/covid19/run.py export
```

To train from the official initialization weights instead:

```bash
python experiments/covid19/run.py train
python experiments/covid19/run.py export
```

Replace `covid19` with `isic2019` or `eyepacs_to_aptos` for the other settings.
GPU training requires a compatible NVIDIA driver. Use the bundled probabilities
for numerical comparison with the paper; re-exporting can introduce floating-point
differences. Export also replaces the tracked probability files at its output paths.
See [output handling](outputs/README.md) before running new classifier experiments.
