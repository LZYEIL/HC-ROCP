#!/usr/bin/env python3
"""Fine-tune ConvNeXt V2-Tiny using the selected experiment configuration."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from shared.paths import load_config, checkpoint_directory
from shared.dataset_cli import ensure_manifest

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.backbone import build_convnextv2_tiny
from shared.dataset import ManifestDataset
from shared.utils import classification_metrics, seed_everything, write_json


@torch.inference_mode()
def evaluate(model, loader, criterion, device, amp_dtype):
    """Compute classifier metrics and loss on the validation split."""
    model.eval(); labels=[]; probs=[]; total=0.0
    for images, target, _ in loader:
        images=images.to(device, non_blocking=True); target=target.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype):
            logits=model(images); loss=criterion(logits,target)
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError("Non-finite logits/loss encountered during validation")
        total += float(loss) * len(target); labels.extend(target.cpu().tolist())
        probs.append(torch.softmax(logits,1).float().cpu().numpy())
    return classification_metrics(labels,np.concatenate(probs),total/len(loader.dataset))


def main():
    """Run the command-line entry point."""
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--resume",action="store_true"); args=ap.parse_args()
    cfg=load_config(Path(args.config)); seed_everything(int(cfg["seed"]))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("GPU is required for classifier training")
    out=Path(cfg["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    checkpoint_dir=checkpoint_directory(cfg)
    if not args.resume and any((checkpoint_dir/name).exists() for name in ("best.pt", "last.pt")):
        raise FileExistsError(f"Weights already exist in {checkpoint_dir}; use --resume or a new checkpoint_dir in a custom config")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ensure_manifest(cfg)
    train_ds=ManifestDataset(cfg["manifest"],cfg["train_split"],True,cfg["image_size"])
    val_ds=ManifestDataset(cfg["manifest"],cfg["validation_split"],False,cfg["image_size"])
    if len(train_ds) < cfg["batch_size"] or len(val_ds) == 0:
        raise ValueError("Training requires at least one full batch and a nonempty validation split")
    loader_args=dict(batch_size=cfg["batch_size"],num_workers=cfg["num_workers"],pin_memory=True,persistent_workers=cfg["num_workers"]>0)
    if cfg.get("balanced_sampler", False):
        train_counts=np.bincount(train_ds.frame.label.astype(int),minlength=cfg["num_classes"])
        sample_weights=(1.0/np.maximum(train_counts,1))[train_ds.frame.label.astype(int)]
        sampler=WeightedRandomSampler(torch.as_tensor(sample_weights,dtype=torch.double),num_samples=len(train_ds),replacement=True)
        train_loader=DataLoader(train_ds,sampler=sampler,shuffle=False,drop_last=True,**loader_args)
    else:
        train_loader=DataLoader(train_ds,shuffle=True,drop_last=True,**loader_args)
    val_loader=DataLoader(val_ds,shuffle=False,drop_last=False,**loader_args)
    counts=np.bincount(train_ds.frame.label.astype(int),minlength=cfg["num_classes"])
    class_weights=1.0/np.sqrt(np.maximum(counts,1)); class_weights/=class_weights.mean()
    if cfg.get("disable_class_weights", False): class_weights=np.ones_like(class_weights)
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(class_weights,dtype=torch.float32,device=device),label_smoothing=cfg["label_smoothing"])
    initialization=None if args.resume else cfg["pretrained_checkpoint"]
    model=build_convnextv2_tiny(cfg["num_classes"],initialization,cfg["drop_path_rate"]).to(device)
    resume_state=None; start_epoch=1; history=[]; best=-1.0; stale=0
    if args.resume:
        resume_path=checkpoint_dir/"last.pt" if (checkpoint_dir/"last.pt").is_file() else checkpoint_dir/"best.pt"
        if not resume_path.is_file():
            raise FileNotFoundError(f"No checkpoint available to resume in {checkpoint_dir}")
        resume_state=torch.load(resume_path,map_location="cpu",weights_only=False)
        if not all(key in resume_state for key in ("optimizer", "scheduler", "scaler")):
            raise ValueError("Resume requires last.pt with optimizer, scheduler and scaler states; best.pt is for inference")
        model.load_state_dict(resume_state["model"])
        start_epoch=int(resume_state["epoch"])+1
        if (out/"history.json").is_file():
            history=json.loads((out/"history.json").read_text())
            latest_recorded=max((int(row["epoch"]) for row in history), default=-1)
            if latest_recorded!=int(resume_state["epoch"]):
                raise RuntimeError("Checkpoint is not from the latest recorded epoch; refusing an unsafe resume")
            best=max(float(row["validation"]["balanced_accuracy"]) for row in history)
            best_epoch=next(int(row["epoch"]) for row in history if float(row["validation"]["balanced_accuracy"]) == best)
            stale=int(resume_state["epoch"])-best_epoch
        elif "best_balanced_accuracy" not in resume_state or "stale_epochs" not in resume_state:
            raise RuntimeError("This checkpoint requires its history.json to recover best-score and early-stopping state")
        if "best_balanced_accuracy" in resume_state:
            best=float(resume_state["best_balanced_accuracy"])
            stale=int(resume_state["stale_epochs"])
        print(f"Resuming from {resume_path} at epoch {start_epoch}",flush=True)
    head_names=tuple(cfg.get("head_parameter_prefixes", ["head."]))
    head_params=[p for name,p in model.named_parameters() if name.startswith(head_names)]
    backbone_params=[p for name,p in model.named_parameters() if not name.startswith(head_names)]
    if not head_params or not backbone_params:
        raise RuntimeError("Could not separate classifier head and backbone parameters")
    backbone_lr=float(cfg.get("backbone_learning_rate",cfg["learning_rate"]))
    head_lr=float(cfg.get("head_learning_rate",cfg["learning_rate"]))
    optimizer=torch.optim.AdamW([
        {"params":backbone_params,"lr":backbone_lr,"name":"backbone"},
        {"params":head_params,"lr":head_lr,"name":"head"},
    ],weight_decay=cfg["weight_decay"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg["epochs"],eta_min=backbone_lr*0.01)
    precision=str(cfg.get("precision","fp16")).lower()
    if precision not in {"fp32","fp16","bf16"}:
        raise ValueError(f"Unsupported precision: {precision}")
    amp_dtype={"fp32":None,"fp16":torch.float16,"bf16":torch.bfloat16}[precision]
    if precision=="bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; use precision: fp32")
    scaler=torch.amp.GradScaler("cuda",enabled=precision=="fp16")
    if resume_state is not None and "optimizer" in resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        scaler.load_state_dict(resume_state["scaler"])
    freeze_epochs=int(cfg.get("head_only_epochs",0))
    for epoch in range(start_epoch,cfg["epochs"]+1):
        if stale >= cfg["early_stopping_patience"]:
            break
        backbone_trainable=epoch>freeze_epochs
        for parameter in backbone_params: parameter.requires_grad_(backbone_trainable)
        if backbone_trainable:
            model.train()
        else:
            # Keep the frozen feature extractor deterministic while fitting only
            # the newly initialized task head.
            model.eval(); model.head.train()
        running=0.0
        for images,target,_ in train_loader:
            images=images.to(device,non_blocking=True); target=target.to(device,non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=amp_dtype is not None,dtype=amp_dtype):
                logits=model(images); loss=criterion(logits,target)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite logits/loss at epoch {epoch}; training aborted")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg.get("gradient_clip_norm",1.0)),error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update(); running+=float(loss)*len(target)
        scheduler.step(); val=evaluate(model,val_loader,criterion,device,amp_dtype)
        row={"epoch":epoch,"stage":"head_only" if not backbone_trainable else "full_finetune","train_loss":running/(len(train_loader)*cfg["batch_size"]),"learning_rates":{group["name"]:group["lr"] for group in optimizer.param_groups},"last_gradient_norm":float(grad_norm),"validation":val}
        history.append(row); write_json(out/"history.json",history); print(row,flush=True)
        state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),"config":cfg,"epoch":epoch,"validation":val,"class_counts":counts.tolist()}
        score=val["balanced_accuracy"]
        if score>best:
            best=score; stale=0
            torch.save({"model":model.state_dict(),"config":cfg,"epoch":epoch,"validation":val,"class_counts":counts.tolist()},checkpoint_dir/"best.pt")
            write_json(out/"best_metrics.json",row)
        else:
            stale+=1
        state.update(best_balanced_accuracy=best, stale_epochs=stale)
        torch.save(state,checkpoint_dir/"last.pt")
        if stale>=cfg["early_stopping_patience"]: break
    completed_epoch=int(history[-1]["epoch"]) if history else start_epoch-1
    write_json(out/"training_complete.json",{"best_balanced_accuracy":best,"epochs_completed":completed_epoch,"device":str(device)})


if __name__=="__main__": main()
