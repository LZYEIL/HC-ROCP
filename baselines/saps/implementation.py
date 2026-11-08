"""SAPS baseline: algorithm and experiment interface."""
import numpy as np

def randomized_saps_scores(probabilities, uniforms, penalty=0.02):
    """Randomized SAPS scores using only rank and the largest probability."""
    probabilities = np.asarray(probabilities, dtype=float)
    uniforms = np.asarray(uniforms, dtype=float).reshape(-1, 1)
    ranks = 1 + np.sum(
        probabilities[:, :, None] > probabilities[:, None, :], axis=1
    )
    top = np.max(probabilities, axis=1, keepdims=True)
    return np.where(
        ranks == 1,
        uniforms * top,
        top + float(penalty) * (ranks - 2 + uniforms),
    )


class _RandomizedScoreSet:
    """Split-conformal adapter for randomized all-label score functions.

    One independent U(0,1) variate is used per example, as in randomized
    APS/RAPS/SAPS.  Separate fixed streams are used for calibration and
    prediction so results are reproducible and test randomness cannot affect
    the calibrated quantile.
    """
    def __init__(self, score_matrix_fn, calibration_seed=20260815, prediction_seed=20260816):
        """Initialize the object."""
        self.score_matrix_fn = score_matrix_fn
        self.calibration_seed = int(calibration_seed)
        self.prediction_seed = int(prediction_seed)
        self.q = None
        self._prediction_rng = None

    def calibrate(self, probs, labels, alpha):
        """Fit the finite-sample quantile and reset the independent prediction stream."""
        probs = np.asarray(probs, float); labels = np.asarray(labels, int)
        rng = np.random.default_rng(self.calibration_seed)
        scores = self.score_matrix_fn(probs, rng.random(len(probs)))
        true_scores = scores[np.arange(len(labels)), labels]
        rank = int(np.ceil((len(true_scores) + 1) * (1 - alpha)))
        self.q = float("inf") if rank > len(true_scores) else float(np.partition(true_scores, rank - 1)[rank - 1])
        self._prediction_rng = np.random.default_rng(self.prediction_seed)
        return self.q

    def predict_sets(self, probs):
        """Generate predictions for the inputs."""
        if self.q is None: raise RuntimeError("calibrate first")
        probs = np.asarray(probs, float)
        scores = self.score_matrix_fn(probs, self._prediction_rng.random(len(probs)))
        return [set(np.flatnonzero(row <= self.q).tolist()) for row in scores]

    def predict_set(self, p):
        """Generate predictions for the inputs."""
        return self.predict_sets(np.asarray(p, float)[None, :])[0]

class RandomizedSAPSConformal(_RandomizedScoreSet):
    """Faithful randomized SAPS score from Huang et al."""
    def __init__(self, penalty=0.02, calibration_seed=20260815, prediction_seed=20260816):
        """Initialize the object."""
        self.penalty = float(penalty)
        super().__init__(
            lambda p, u: randomized_saps_scores(p, u, self.penalty),
            calibration_seed, prediction_seed,
        )

SAPSConformal = RandomizedSAPSConformal
