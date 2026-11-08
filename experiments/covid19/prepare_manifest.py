#!/usr/bin/env python3
"""Create deterministic stratified COVID image splits (never masks)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.dataset_cli import manifest_arguments, save_manifest

import pandas as pd
from sklearn.model_selection import train_test_split


def build_manifest(data_dir):
    """Build the manifest using the fixed split rules and random seeds."""
    DATA = Path(data_dir) / "COVID"
    classes = ["Normal", "Viral Pneumonia", "COVID", "Lung_Opacity"]
    rows = []
    for label, name in enumerate(classes):
        images = sorted((DATA / name / "images").glob("*.png"))
        if not images:
            raise FileNotFoundError(f"No PNG images in {DATA / name / 'images'}; extract the dataset as described in data/COVID/README.md")
        for path in images:
            rows.append(
                {
                    "sample_id": path.stem,
                    "path": str(path),
                    "label": label,
                    "class_name": name,
                }
            )
    d = pd.DataFrame(rows)
    train, rest = train_test_split(
        d, test_size=0.4, random_state=20260816, stratify=d.label
    )
    val, rest = train_test_split(
        rest, test_size=0.75, random_state=20260817, stratify=rest.label
    )
    cal, test = train_test_split(
        rest, test_size=2 / 3, random_state=20260818, stratify=rest.label
    )
    parts = []
    for name, frame in [
        ("train", train),
        ("validation", val),
        ("calibration", cal),
        ("test", test),
    ]:
        frame = frame.copy()
        frame["split"] = name
        parts.append(frame)
    m = pd.concat(parts).sort_values(["split", "sample_id"])
    return m


def main(argv=None):
    """Prepare and save the dataset manifest using the requested paths."""
    args = manifest_arguments("covid19", __doc__, argv)
    save_manifest(build_manifest(args.data_dir), args)


if __name__ == "__main__":
    main()
