"""Bayes and always-safe action baselines."""
import numpy as np

def expected_loss_actions(probabilities, loss_matrix):
    """Return the Bayes action and expected loss for every frozen prediction."""
    probabilities = np.asarray(probabilities, dtype=float)
    loss_matrix = np.asarray(loss_matrix, dtype=float)
    expected = probabilities @ loss_matrix
    return np.argmin(expected, axis=1), expected


def always_safe_actions(probabilities, loss_matrix, harmful_actions):
    """Choose the least-expected-loss action among the prespecified safe ones."""
    safe = np.flatnonzero(~np.isin(np.arange(loss_matrix.shape[1]), harmful_actions))
    expected = probabilities @ loss_matrix
    return safe[np.argmin(expected[:, safe], axis=1)]
