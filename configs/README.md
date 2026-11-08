# Configurations

| File | Purpose |
|---|---|
| `experiments.yaml` | Shared task definitions, loss matrices, and calibration settings for HC-ROCP and baselines. |
| `train_covid19.yaml` | COVID-19 classifier training and probability export. |
| `train_isic2019.yaml` | ISIC 2019 classifier training and probability export. |
| `train_eyepacs_to_aptos.yaml` | EyePACS classifier training and probability export for the EyePACS-to-APTOS experiment. |

Default paths are already configured: images in `data/`, weights in
`checkpoints/`, and probabilities/results in `outputs/`. No YAML edits are
needed when using the documented folder layout. `checkpoint_dir` controls
where task weights are loaded and saved; `output_dir` controls probabilities and logs.
Fine-tuned weights are available on
[Hugging Face](https://huggingface.co/LZYEIL/HC-ROCP-Assets);
follow the [placement instructions](../checkpoints/README.md#our-fine-tuned-classifiers).
