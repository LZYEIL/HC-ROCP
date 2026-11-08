"""Class-conditional APS with a calibration threshold per label."""
import numpy as np

def _aps_score(p, y):
    """Sum probabilities strictly larger than the true label's probability."""
    return float(np.sum(p[p > p[y]]))

class MondrianAPS:
    """Class-conditional APS with one conformal quantile per true class."""
    def __init__(self):
        """Initialize class-conditional quantiles."""
        self.q = None
    def calibrate(self, probs, labels, alpha):
        """Fit a conformal score threshold within each true-label stratum."""
        self.q = {}
        for y in range(probs.shape[1]):
            scores = np.asarray([_aps_score(p, y0) for p, y0 in zip(probs[labels == y], labels[labels == y])], float)
            if len(scores) == 0: self.q[y] = float("inf"); continue
            k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
            self.q[y] = float("inf") if k > len(scores) else float(np.sort(scores)[k - 1])
        return self.q
    def predict_set(self, p):
        """Generate predictions for the inputs."""
        if self.q is None: raise RuntimeError("calibrate first")
        return {y for y in range(len(p)) if _aps_score(np.asarray(p), y) <= self.q[y]}
