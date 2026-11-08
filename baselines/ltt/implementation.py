"""LTT baseline: algorithm and experiment interface."""
import numpy as np
from scipy.stats import binom
from baselines._shared import threshold_policy_actions

def hb_p_value(r_hat,n,alpha):
    if n <= 0 or int(n) != n or not 0 <= r_hat <= 1 or not 0 < alpha < 1:
        raise ValueError("HB requires a positive integer n, a rate in [0, 1], and alpha in (0, 1)")
    bentkus_p_value = np.e * binom.cdf(np.ceil(n*r_hat),n,alpha)
    def h1(y,mu):
        if y == 0:
            return -np.log1p(-mu)
        return y * np.log(y/mu) + (1-y) * np.log((1-y)/(1-mu))
    hoeffding_p_value = np.exp(-n*h1(min(r_hat,alpha),alpha))
    return min(1.0,bentkus_p_value,hoeffding_p_value)


def fit_official_ltt_action(
    probabilities,
    labels,
    loss_matrix,
    severe_labels,
    harmful_actions,
    thresholds,
    delta=0.01,
    family_error=0.05,
):
    """Official LTT HB p-values with a prespecified Bonferroni family test."""
    probabilities = np.asarray(probabilities, float)
    labels = np.asarray(labels, int)
    thresholds = np.asarray(thresholds, float)
    if thresholds.ndim != 1 or not thresholds.size or not np.isfinite(thresholds).all():
        raise ValueError("LTT thresholds must be a nonempty finite sequence")
    if not 0 < delta < 1 or not 0 < family_error < 1:
        raise ValueError("delta and family_error must lie strictly between 0 and 1")
    severe_mask = np.isin(labels, severe_labels)
    n = int(severe_mask.sum())
    rows = []
    for threshold in thresholds:
        actions = threshold_policy_actions(
            probabilities,
            loss_matrix,
            severe_labels,
            harmful_actions,
            threshold,
        )
        harm = np.isin(actions[severe_mask], harmful_actions)
        empirical = float(harm.mean()) if n else 1.0
        p_value = float(hb_p_value(empirical, n, float(delta))) if n else 1.0
        rows.append(
            {
                "threshold": float(threshold),
                "harm_count": int(harm.sum()),
                "harm_n": n,
                "harm_empirical": empirical,
                "p_value": p_value,
                "certified": p_value < float(family_error) / len(thresholds),
                "loss": float(
                    np.mean(np.asarray(loss_matrix, float)[labels, actions])
                ),
            }
        )
    feasible = [row for row in rows if row["certified"]]
    selected = min(feasible, key=lambda row: row["loss"]) if feasible else None
    return {
        "rows": rows,
        "selected": selected,
        "num_certified": len(feasible),
        "family_error": float(family_error),
        "implementation": "official LTT HB p-value + Bonferroni adapter",
    }


def policy_actions(p, loss, severe, harmful, threshold):
    """Return actions from the calibrated threshold policy."""
    return threshold_policy_actions(p, loss, severe, harmful, threshold)
def fit_ltt_style(p,y,loss,severe,harmful,thresholds,delta=.01,family_confidence=.95):
    """Fit the LTT threshold family at the requested family confidence level."""
    return fit_official_ltt_action(
        p,y,loss,severe,harmful,thresholds,delta=delta,
        family_error=1-float(family_confidence),
    )
