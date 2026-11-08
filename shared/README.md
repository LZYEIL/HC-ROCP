# Shared utilities

Common code used by the dataset and evaluation entry points in `experiments/`.
The HC-ROCP method itself is in `core/`.

| File | Purpose |
|---|---|
| `dataset.py` | Load images and labels; apply preprocessing and training augmentation. |
| `backbone.py` | Build the ConvNeXt V2 classifier. |
| `train_classifier.py` | Train and validate classifiers; save checkpoints. |
| `export_probabilities.py` | Export frozen class probabilities. |
| `dataset_cli.py` | Route dataset preparation, training, and export commands. |
| `paths.py` | Resolve data, configuration, checkpoint, and output paths. |
| `utils.py` | Random seeds, classification metrics, and JSON saving. |
| `__init__.py` | Python package marker. |

Use the dataset entry points described in [experiments/README.md](../experiments/README.md).
For example, `python experiments/covid19/run.py train` calls the shared trainer.
Manifests, probabilities, and logs go to `outputs/<dataset>/`; model weights go to
`checkpoints/<dataset>/`. Missing manifests are prepared automatically by training and export.
Fine-tuned weights are available on
[Hugging Face](https://huggingface.co/LZYEIL/HC-ROCP-Assets);
see [file placement](../checkpoints/README.md#our-fine-tuned-classifiers).
