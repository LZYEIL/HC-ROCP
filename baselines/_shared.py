"""Shared calibration and threshold-action helpers for comparison methods."""
from typing import Sequence, Set, Optional
import numpy as np

class _BaseConformalScorer:
    """
    Base helper for split conformal prediction with a scalar score.

    Subclasses must implement _score(x_probs, label) returning a numeric
    nonconformity score. Lower scores are “better”; the conformal set is
    {y : score(x, y) <= q_hat}.
    """

    def __init__(self, actions: Sequence[object]) -> None:
        self.actions = list(actions)
        self.q_hat: Optional[float] = None

    def calibrate(self, cal_probs: Sequence[Sequence[float]], cal_labels: Sequence[int], alpha: float) -> float:
        """
        Store the k=ceil((n+1)(1-alpha)) score, or infinity when k exceeds n.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")
        if len(cal_probs) != len(cal_labels) or len(cal_labels) == 0:
            raise ValueError("Calibration probabilities and labels must have equal, nonzero lengths")
        probs_array = np.asarray(cal_probs, dtype=float)
        labels_array = np.asarray(cal_labels)
        if (probs_array.ndim != 2 or probs_array.shape[1] == 0
                or not np.isfinite(probs_array).all() or np.any(probs_array < 0)
                or not np.allclose(probs_array.sum(axis=1), 1.0, atol=1e-6)):
            raise ValueError("Calibration probabilities must be finite, nonnegative rows summing to one")
        if (labels_array.ndim != 1 or not np.isfinite(labels_array).all()
                or np.any(labels_array != labels_array.astype(int))
                or np.any(labels_array < 0) or np.any(labels_array >= probs_array.shape[1])):
            raise ValueError("Calibration labels must be integer indices within the probability columns")
        scores = []
        for probs, y in zip(cal_probs, cal_labels):
            scores.append(self._score(np.asarray(probs, dtype=float), int(y)))
        scores_arr = np.asarray(scores, dtype=float)
        if not np.isfinite(scores_arr).all():
            raise ValueError("Calibration scores must be finite")
        scores_sorted = np.sort(scores_arr)
        n = len(scores_sorted)
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        self.q_hat = float(scores_sorted[k - 1]) if k <= n else float("inf")
        return self.q_hat

    def predict_set(self, x_probs: Sequence[float]) -> Set[int]:
        """
        Return prediction set {y: score(x, y) <= q_hat}.
        """
        if self.q_hat is None:
            raise RuntimeError("Call calibrate() before predict_set().")
        probs = np.asarray(x_probs, dtype=float)
        return {y for y in range(len(probs)) if self._score(probs, y) <= self.q_hat}

    # to be implemented by subclasses
    def _score(self, x_probs, label):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def _bayes_actions(probabilities, loss_matrix):
    """Choose the action minimizing expected loss for each probability row."""
    return np.argmin(
        np.asarray(probabilities, float) @ np.asarray(loss_matrix, float), axis=1
    )


def threshold_policy_actions(
    probabilities, loss_matrix, severe_labels, harmful_actions, threshold
):
    """Replace harmful Bayes actions with safe actions when severe mass reaches the threshold."""
    probabilities = np.asarray(probabilities, float)
    loss_matrix = np.asarray(loss_matrix, float)
    actions = _bayes_actions(probabilities, loss_matrix)
    severe_mass = probabilities[:, np.asarray(severe_labels, int)].sum(axis=1)
    unsafe = (severe_mass >= float(threshold)) & np.isin(
        actions, harmful_actions
    )
    safe = np.flatnonzero(
        ~np.isin(np.arange(loss_matrix.shape[1]), harmful_actions)
    )
    if not len(safe):
        raise ValueError("Threshold policies require at least one non-harmful action")
    expected = probabilities @ loss_matrix
    replacements = safe[np.argmin(expected[:, safe], axis=1)]
    actions = actions.copy()
    actions[unsafe] = replacements[unsafe]
    return actions
