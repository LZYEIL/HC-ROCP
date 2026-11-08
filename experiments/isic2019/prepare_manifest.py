#!/usr/bin/env python3
"""Create lesion-grouped ISIC train/calibration/validation and official-test manifest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.dataset_cli import manifest_arguments, save_manifest

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def build_manifest(data_dir):
    """Build the manifest using the fixed split rules and random seeds."""
    DATA = Path(data_dir) / "ISIC_2019"
    labels = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    gt = pd.read_csv(DATA / "ISIC_2019_Training_GroundTruth.csv")
    meta = pd.read_csv(DATA / "ISIC_2019_Training_Metadata.csv")
    d = gt.merge(meta[["image", "lesion_id"]], on="image", how="left")
    d["label"] = d[labels].values.argmax(1)
    d["group"] = d.lesion_id.fillna(d.image)
    d["split"] = "train"
    fold = np.full(len(d), -1)
    cv = StratifiedGroupKFold(10, shuffle=True, random_state=20260816)
    for k, (_, idx) in enumerate(cv.split(d, d.label, d.group)):
        fold[idx] = k
    d.loc[fold == 0, "split"] = "validation"
    d.loc[np.isin(fold, [1, 2]), "split"] = "calibration"
    d["path"] = d.image.map(
        lambda x: str(DATA / "train/ISIC_2019_Training_Input" / f"{x}.jpg")
    )
    d["sample_id"] = d.image
    test = pd.read_csv(DATA / "ISIC_2019_Test_GroundTruth.csv")
    test["label"] = test[labels].values.argmax(1)
    test["split"] = np.where(
        test.UNK == 1, "official_test_unknown", "official_test_known"
    )
    test.loc[test.UNK == 1, "label"] = -1
    test["path"] = test.image.map(
        lambda x: str(DATA / "test/ISIC_2019_Test_Input" / f"{x}.jpg")
    )
    test["sample_id"] = test.image
    manifest = pd.concat(
        [
            d[["sample_id", "path", "label", "split", "group"]],
            test.assign(group=test.image)[
                ["sample_id", "path", "label", "split", "group"]
            ],
        ],
        ignore_index=True,
    )
    if not manifest.path.map(lambda value: Path(value).exists()).all():
        raise RuntimeError("ISIC manifest contains missing images")
    return manifest


def main(argv=None):
    """Prepare and save the dataset manifest using the requested paths."""
    args = manifest_arguments("isic2019", __doc__, argv)
    save_manifest(build_manifest(args.data_dir), args)


if __name__ == "__main__":
    main()
