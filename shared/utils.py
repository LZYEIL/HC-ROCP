"""Reproducibility and metric helpers."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


def seed_everything(seed: int):
    """Seed Python, NumPy and PyTorch and request deterministic cuDNN kernels."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def classification_metrics(labels, probabilities, loss=None):
    """Report accuracy, recalls and AUC; undefined metrics are None."""
    labels = np.asarray(labels, dtype=int); probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    pred = probabilities.argmax(1)
    result = {
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "per_class_recall": {},
    }
    for label in range(probabilities.shape[1]):
        mask = labels == label
        result["per_class_recall"][str(label)] = float(np.mean(pred[mask] == label)) if mask.any() else None
    result["macro_ovr_auc"] = None
    if probabilities.shape[1] >= 2 and len(np.unique(labels)) == probabilities.shape[1]:
        scores = probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities
        try:
            auc = float(roc_auc_score(labels, scores, labels=np.arange(probabilities.shape[1]), multi_class="ovr", average="macro"))
            result["macro_ovr_auc"] = auc if np.isfinite(auc) else None
        except ValueError:
            pass
    if loss is not None: result["loss"] = float(loss)
    return result


def write_json(path, value):
    """Write indented JSON, creating the destination's parent directories."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
