# Experiment outputs

This folder includes the nine frozen input files needed for the paper's wrapper
evaluations (about 8.5 MB). No images or weights are needed for those evaluations.
Scripts also write their results here; generated files are ignored by Git except
the nine input paths listed below.

| Location | Contents |
|---|---|
| `covid19/`, `isic2019/`, `eyepacs_to_aptos/` | Manifests, frozen probabilities, training history, and metrics |
| Files directly in `outputs/` | Experiment results, diagnostics, simulations, and statistical tables (JSON/CSV/LaTeX) |

No experiment results are bundled. Run the evaluations before summarizing their
outputs; see [the experiment guide](../experiments/README.md) for the order and commands.
Use fresh `--output` paths where available to avoid overwriting previous runs.

`HC_ROCP_OUTPUTS_DIR` overrides the output root for both dataset files and results.
Explicit command-line output paths take precedence.
If you override the root, copy the three bundled dataset folders there first
to use the reference inputs. The override does not move files automatically.

Model weights (`best.pt`/`last.pt`) belong in `checkpoints/<dataset>/`, not here.
The released fine-tuned weights are on
[Hugging Face](https://huggingface.co/LZYEIL/HC-ROCP-Assets);
see [file placement](../checkpoints/README.md#our-fine-tuned-classifiers).

Bundled inputs:

| Folder | Files |
|---|---|
| `covid19/` | `manifest.csv`, `probabilities_calibration.npz`, `probabilities_test.npz` |
| `isic2019/` | `manifest.csv`, `probabilities_calibration.npz`, `probabilities_official_test_known.npz` |
| `eyepacs_to_aptos/` | `manifest.csv`, `probabilities_target_calibration.npz`, `probabilities_target_test.npz` |

Each probability file contains `probabilities`, `labels`, and `sample_ids`.
Validation and unknown-class exports are not required for the paper's wrapper experiments.
These dataset-derived files remain subject to the original dataset terms.
Training/export can replace the tracked manifests or probabilities; use a
separate output root for new classifier runs to preserve the reference inputs.
