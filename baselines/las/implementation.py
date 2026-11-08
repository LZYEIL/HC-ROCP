"""Least-ambiguous set score with split-conformal calibration."""
from baselines._shared import _BaseConformalScorer

class LASConformal(_BaseConformalScorer):
    """
    LAS (Least Ambiguous Set, Sadinle et al. 2019): 1 - p_true.
    """

    def _score(self, x_probs, label):
        return 1.0 - float(x_probs[label])
