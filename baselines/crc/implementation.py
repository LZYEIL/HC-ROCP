"""CRC baseline: algorithm and experiment interface."""
import numpy as np
from baselines._shared import threshold_policy_actions

def get_lhat(calib_loss_table, lambdas, alpha, B=1):
    n = calib_loss_table.shape[0]
    rhat = calib_loss_table.mean(axis=0)
    lhat_idx = max(np.argmax(((n/(n+1)) * rhat + B/(n+1) ) >= alpha) - 1, 0) # Can't be -1.
    return lambdas[lhat_idx]


def fit_official_crc_action(
    probabilities,
    labels,
    loss_matrix,
    severe_labels,
    harmful_actions,
    thresholds,
    delta=0.01,
):
    """Adapt official CRC ``get_lhat`` to the severe-harm action loss."""
    probabilities = np.asarray(probabilities, float)
    labels = np.asarray(labels, int)
    thresholds = np.asarray(thresholds, float)
    if thresholds.ndim != 1 or not thresholds.size or not np.isfinite(thresholds).all() or np.any(np.diff(thresholds) <= 0):
        raise ValueError("CRC thresholds must be a nonempty, strictly increasing finite sequence")
    if not 0 < delta < 1:
        raise ValueError("delta must lie strictly between 0 and 1")
    severe_mask = np.isin(labels, severe_labels)
    if not severe_mask.any():
        raise ValueError("CRC calibration requires at least one severe case")
    loss_columns = []
    for threshold in thresholds:
        actions = threshold_policy_actions(
            probabilities,
            loss_matrix,
            severe_labels,
            harmful_actions,
            threshold,
        )
        loss_columns.append(
            np.isin(actions[severe_mask], harmful_actions).astype(float)
        )
    loss_table = np.column_stack(loss_columns)
    corrected = (
        n := loss_table.shape[0]
    ) / (n + 1) * loss_table.mean(axis=0) + 1 / (n + 1)
    # get_lhat assumes the ordered path eventually crosses the target. If the
    # entire path is controlled (as on EyePACS), the least conservative final
    # policy is the mathematically correct boundary solution.
    if np.all(corrected < float(delta)):
        selected = float(thresholds[-1])
    else:
        selected = float(
            get_lhat(
                loss_table, thresholds, alpha=float(delta), B=1
            )
        )
    idx = int(np.flatnonzero(np.isclose(thresholds, selected))[0])
    return {
        "threshold": selected,
        "calibration_harm": float(loss_table[:, idx].mean()),
        "n_severe": int(severe_mask.sum()),
        "implementation": "official CRC get_lhat adapter",
    }
