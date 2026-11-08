"""Adaptive prediction sets with split-conformal calibration."""
from baselines._shared import _BaseConformalScorer

class APSConformal(_BaseConformalScorer):
    """
    APS (Adaptive Prediction Sets, Romano et al. 2020):
    sum_{y' : p(y'|x) > p(label|x)} p(y'|x).
    """

    def _score(self, x_probs, label):
        p_y = float(x_probs[label])
        mask = x_probs > p_y
        mask[label] = False
        return float(x_probs[mask].sum())
