#!/usr/bin/env python3
"""Export frozen probabilities for every configured non-training split."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from shared.paths import load_config, checkpoint_directory, data_path
from shared.dataset_cli import ensure_manifest

import argparse, sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from shared.backbone import build_convnextv2_tiny
from shared.dataset import ManifestDataset
from shared.utils import classification_metrics, write_json


def main():
    """Run the command-line entry point."""
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True)
    ap.add_argument("--checkpoint", type=Path,
                    help="Fine-tuned checkpoint; defaults to checkpoints/<dataset>/best.pt")
    args=ap.parse_args()
    cfg=load_config(Path(args.config)); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out=Path(cfg["output_dir"])
    weight_path=data_path(args.checkpoint) if args.checkpoint else checkpoint_directory(cfg)/"best.pt"
    if not weight_path.is_file():
        raise FileNotFoundError(f"Place the fine-tuned checkpoint at {weight_path}, or run this dataset's train command first")
    checkpoint=torch.load(weight_path,map_location="cpu",weights_only=False)
    ensure_manifest(cfg)
    out.mkdir(parents=True, exist_ok=True)
    model=build_convnextv2_tiny(cfg["num_classes"],drop_path_rate=0.0)
    model.load_state_dict(checkpoint["model"]); model.to(device).eval(); metrics={}
    for split in cfg["export_splits"]:
        ds=ManifestDataset(cfg["manifest"],split,False,cfg["image_size"])
        loader=DataLoader(ds,batch_size=cfg["batch_size"],shuffle=False,num_workers=cfg["num_workers"],pin_memory=True)
        labels=[]; ids=[]; probs=[]
        with torch.inference_mode():
            for images,target,sample_id in loader:
                logits=model(images.to(device,non_blocking=True))
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(f"Non-finite classifier logits in split {split}; export aborted")
                probs.append(torch.softmax(logits,1).float().cpu().numpy())
                labels.extend(target.numpy().tolist()); ids.extend(sample_id)
        probabilities=np.concatenate(probs); labels=np.asarray(labels,dtype=np.int64); ids=np.asarray(ids)
        np.savez_compressed(out/f"probabilities_{split}.npz",probabilities=probabilities,labels=labels,sample_ids=ids)
        if np.all(labels>=0): metrics[split]=classification_metrics(labels,probabilities)
    write_json(out/"export_metrics.json",metrics); print(metrics)


if __name__=="__main__": main()
