"""ConvNeXt V2-Tiny construction for training or full-checkpoint inference."""

from pathlib import Path

import timm


def build_convnextv2_tiny(num_classes: int, checkpoint: str | None = None, drop_path_rate: float = 0.1):
    """Optionally initialize from Meta weights; full-checkpoint export needs none."""
    if checkpoint is None:
        # A full fine-tuned state dict supplies every parameter during export.
        return timm.create_model(
            "convnextv2_tiny", pretrained=False,
            num_classes=int(num_classes), drop_path_rate=float(drop_path_rate),
        )
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing official checkpoint: {checkpoint_path}")
    # timm's ConvNeXt checkpoint filter converts the official Meta key names.
    return timm.create_model(
        "convnextv2_tiny",
        pretrained=True,
        pretrained_cfg_overlay={"file": str(checkpoint_path)},
        num_classes=int(num_classes),
        drop_path_rate=float(drop_path_rate),
    )
