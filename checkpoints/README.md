# Model weights

Place downloaded model weights here. The dataset directories contain `.gitkeep`
placeholders so they remain visible in Git; weight files are ignored.

```text
checkpoints/
├── convnextv2_tiny_22k_224_ema.pt  # download from Meta
├── covid19/best.pt               # our fine-tuned classifier
├── isic2019/best.pt
└── eyepacs_to_aptos/best.pt
```

The `.pt` files shown above are download destinations, not bundled files.

## Official pretrained backbone

The classifier uses the ConvNeXt V2-Tiny ImageNet-22k checkpoint
`convnextv2_tiny_22k_224_ema.pt`. Obtain the matching checkpoint from the official
[Meta ConvNeXt V2 repository](https://github.com/facebookresearch/ConvNeXt-V2)
and place it at:

```text
checkpoints/convnextv2_tiny_22k_224_ema.pt
```

Set `HC_ROCP_MODELS_DIR` to use another checkpoint directory, or supply an
absolute `pretrained_checkpoint` in the classifier YAML. A different backbone,
resolution, or pretraining checkpoint is not an exact reproduction.

## Our fine-tuned classifiers

Download the three weights from
[LZYEIL/HC-ROCP-Assets](https://huggingface.co/LZYEIL/HC-ROCP-Assets).
Place each at the destination below, renaming it to `best.pt`:

| Downloaded file | Destination in this repository |
|---|---|
| [covid19_convnextv2_tiny.pt](https://huggingface.co/LZYEIL/HC-ROCP-Assets/resolve/main/covid19_convnextv2_tiny.pt) | `checkpoints/covid19/best.pt` |
| [isic2019_convnextv2_tiny.pt](https://huggingface.co/LZYEIL/HC-ROCP-Assets/resolve/main/isic2019_convnextv2_tiny.pt) | `checkpoints/isic2019/best.pt` |
| [eyepacs_to_aptos_convnextv2_tiny.pt](https://huggingface.co/LZYEIL/HC-ROCP-Assets/resolve/main/eyepacs_to_aptos_convnextv2_tiny.pt) | `checkpoints/eyepacs_to_aptos/best.pt` |

After placing the matching weight file above and extracting the original images,
run (no path arguments required):

```bash
python experiments/covid19/run.py export
```

Replace `covid19` with `isic2019` or `eyepacs_to_aptos` as needed.
The exporter reads `checkpoints/<dataset>/best.pt` and creates a missing manifest
automatically. It does not need the official initialization checkpoint.
Training uses that official checkpoint and saves `best.pt` and `last.pt` under
`checkpoints/<dataset>/`. For other locations, use `export --checkpoint` for
inference weights or a custom YAML for training paths.

## Frozen probabilities

Matching frozen probability arrays and split manifests are already bundled in
`outputs/<dataset>/`. HC-ROCP experiments need no weight download or image-model
inference. See [the input file list](../outputs/README.md).

Download only trusted weight files: the exporter loads PyTorch checkpoint dictionaries.
