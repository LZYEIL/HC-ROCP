#!/usr/bin/env python3
"""EyePACS source train/val plus labeled APTOS target cal/val/test manifest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.dataset_cli import manifest_arguments, save_manifest

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def build_manifest(data_dir):
    """Build the manifest using the fixed split rules and random seeds."""
    EYE = Path(data_dir) / "EyePACS"
    APTOS = Path(data_dir) / "APTOS_2019"
    e = pd.read_csv(EYE / "labels/trainLabels.csv").rename(
        columns={"image": "sample_id", "level": "label"}
    )
    e["path"] = e.sample_id.map(lambda x: str(EYE / "train_images/train" / f"{x}.jpeg"))
    e["group"] = e.sample_id.str.replace(r"_(left|right)$", "", regex=True)
    fold = np.full(len(e), -1)
    cv = StratifiedGroupKFold(10, shuffle=True, random_state=20260816)
    for k, (_, idx) in enumerate(cv.split(e, e.label, e.group)):
        fold[idx] = k
    e["split"] = np.where(fold == 0, "source_validation", "source_train")
    a = pd.read_csv(APTOS / "train.csv").rename(
        columns={"id_code": "sample_id", "diagnosis": "label"}
    )
    a["path"] = a.sample_id.map(lambda x: str(APTOS / "train_images" / f"{x}.png"))
    a["group"] = a.sample_id
    cal, rest = train_test_split(
        a, test_size=0.6, random_state=20260816, stratify=a.label
    )
    val, test = train_test_split(
        rest, test_size=2 / 3, random_state=20260817, stratify=rest.label
    )
    parts = [e]
    for name, frame in [
        ("target_calibration", cal),
        ("target_validation", val),
        ("target_test", test),
    ]:
        frame = frame.copy()
        frame["split"] = name
        parts.append(frame)
    m = pd.concat(parts, ignore_index=True)
    exists = m.path.map(lambda value: Path(value).exists())
    if not exists.all():
        raise RuntimeError(f"Missing {sum(~exists)} images")
    return m[["sample_id", "path", "label", "split", "group"]]


def main(argv=None):
    """Prepare and save the dataset manifest using the requested paths."""
    args = manifest_arguments("eyepacs_to_aptos", __doc__, argv)
    save_manifest(build_manifest(args.data_dir), args)


if __name__ == "__main__":
    main()
