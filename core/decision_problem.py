"""Exact finite-label decision problem used by HC-ROCP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate_indices(values, upper, name, *, nonempty=True):
    """Validate a one-dimensional sequence of nonnegative integer indices."""
    values = np.asarray(values)
    if values.ndim != 1 or (nonempty and values.size == 0):
        raise ValueError(
            f"{name} must be a {'nonempty ' if nonempty else ''}1D sequence"
        )
    if values.dtype.kind not in "iuf" or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite integer indices")
    if np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError(f"{name} must contain nonnegative integer indices")
    if upper is not None and np.any(values >= upper):
        raise ValueError(f"{name} indices must be less than {upper}")
    return values.astype(int)


@dataclass(frozen=True)
class SetDecisionTable:
    """Nonempty label subsets and their minimum robust risks and actions."""

    indicators: np.ndarray
    risks: np.ndarray
    actions: np.ndarray
    masks: np.ndarray


class FiniteDecisionProblem:
    """Compute set risks and actions from a prespecified label-by-action loss matrix."""

    def __init__(self, loss_matrix, alpha: float = 0.05):
        """Enumerate nonempty prediction sets at miscoverage level alpha."""
        self.loss_matrix = np.asarray(loss_matrix, dtype=float)
        if self.loss_matrix.ndim != 2 or 0 in self.loss_matrix.shape:
            raise ValueError("loss_matrix must be nonempty [num_labels, num_actions]")
        if not np.all(np.isfinite(self.loss_matrix)):
            raise ValueError("loss_matrix must contain only finite values")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0,1)")
        self.alpha = float(alpha)
        self.num_labels, self.num_actions = self.loss_matrix.shape
        if self.num_labels > 16:
            raise ValueError("Exact set enumeration is restricted to at most 16 labels")
        self.table = self._build_table()
        self._mask_to_index = {
            int(mask): index for index, mask in enumerate(self.table.masks)
        }

    def _build_table(self):
        """Minimize robust loss over actions for every set; ties use the first action."""
        masks = np.arange(1, 1 << self.num_labels, dtype=np.int64)
        indicators = np.asarray(
            [
                [(int(mask) >> label) & 1 for label in range(self.num_labels)]
                for mask in masks
            ],
            dtype=float,
        )
        risks = np.empty(len(masks), dtype=float)
        actions = np.empty(len(masks), dtype=int)
        universe = np.arange(self.num_labels)
        for index, indicator in enumerate(indicators):
            inside = np.flatnonzero(indicator)
            outside = np.setdiff1d(universe, inside, assume_unique=True)
            values = []
            for action in range(self.num_actions):
                in_loss = float(np.max(self.loss_matrix[inside, action]))
                out_loss = (
                    in_loss
                    if len(outside) == 0
                    else float(np.max(self.loss_matrix[outside, action]))
                )
                values.append(in_loss + self.alpha * max(out_loss - in_loss, 0.0))
            actions[index] = int(np.argmin(values))
            risks[index] = float(np.min(values))
        return SetDecisionTable(indicators, risks, actions, masks)

    def mask(self, label_set) -> int:
        """Encode a nonempty collection of label indices as a bit mask."""
        labels = _validate_indices(list(label_set), self.num_labels, "label_set")
        mask = 0
        for label in labels:
            label = int(label)
            mask |= 1 << label
        return mask

    def set_index(self, label_set) -> int:
        """Return the row index of a prediction set in the decision table."""
        return self._mask_to_index[self.mask(label_set)]

    def action(self, label_set) -> int:
        """Return the action minimizing the supplied set's robust loss."""
        return int(self.table.actions[self.set_index(label_set)])

    def risk(self, label_set) -> float:
        """Return the supplied set's minimum robust loss."""
        return float(self.table.risks[self.set_index(label_set)])
