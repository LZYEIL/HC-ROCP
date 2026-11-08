"""RAPS baseline: algorithm and experiment interface."""
import numpy as np

def gcq(scores, tau, I, ordered, cumsum, penalties, randomized, allow_zero_sets):
    penalties_cumsum = np.cumsum(penalties, axis=1)
    sizes_base = ((cumsum + penalties_cumsum) <= tau).sum(axis=1) + 1  # First boundary rank.
    sizes_base = np.minimum(sizes_base, scores.shape[1]) # Cap at the number of labels.

    if randomized:
        V = np.zeros(sizes_base.shape)
        for i in range(sizes_base.shape[0]):
            V[i] = 1/ordered[i,sizes_base[i]-1] * \
                    (tau-(cumsum[i,sizes_base[i]-1]-ordered[i,sizes_base[i]-1])-penalties_cumsum[0,sizes_base[i]-1]) # Convert the rank to a zero-based index.

        sizes = sizes_base - (np.random.random(V.shape) >= V).astype(int)
    else:
        sizes = sizes_base

    if tau == 1.0:
        sizes[:] = cumsum.shape[1] # Preserve the release routine's full-set special case at tau=1.

    if not allow_zero_sets:
        sizes[sizes == 0] = 1 # Optional nonempty-set constraint; the adapter allows empty sets.

    S = list()

    # Construct S from equation (5)
    for i in range(I.shape[0]):
        S = S + [I[i,0:sizes[i]],]

    return S

def get_tau(score, target, I, ordered, cumsum, penalty, randomized, allow_zero_sets): # For one example
    idx = np.where(I==target)
    tau_nonrandom = cumsum[idx]

    if not randomized:
        return tau_nonrandom + penalty[0]
    
    U = np.random.random()

    if idx == (0,0):
        if not allow_zero_sets:
            return tau_nonrandom + penalty[0]
        else:
            return U * tau_nonrandom + penalty[0] 
    else:
        return U * ordered[idx] + cumsum[(idx[0],idx[1]-1)] + (penalty[0:(idx[1][0]+1)]).sum()


class OfficialRAPSConformal:
    """RAPS using the authors' ``get_tau`` and ``gcq`` numerical routines.

    Probabilities are already frozen by the common experimental protocol, so
    the adapter does not refit the image model or add temperature scaling.
    ``k_reg`` and ``penalty_lambda`` use the fixed constructor defaults unless overridden.
    """

    def __init__(
        self,
        penalty_lambda: float = 0.01,
        k_reg: int = 1,
        calibration_seed: int = 20260815,
        prediction_seed: int = 20260816,
    ):
        """Initialize the object."""
        self.penalty_lambda = float(penalty_lambda)
        self.k_reg = int(k_reg)
        self.calibration_seed = int(calibration_seed)
        self.prediction_seed = int(prediction_seed)
        self.q = None
        self.num_classes = None

    @staticmethod
    def _sort_sum(probabilities):
        """Sort each probability row and compute cumulative mass in that order."""
        order = np.argsort(-probabilities, axis=1, kind="mergesort")
        ordered = np.take_along_axis(probabilities, order, axis=1)
        return order, ordered, np.cumsum(ordered, axis=1)

    def calibrate(self, probabilities, labels, alpha):
        """Fit the empirical quantile of randomized RAPS calibration scores."""
        probabilities = np.asarray(probabilities, float)
        labels = np.asarray(labels, int)
        self.num_classes = probabilities.shape[1]
        penalties = np.zeros((1, self.num_classes), float)
        penalties[:, self.k_reg :] = self.penalty_lambda
        order, ordered, cumulative = self._sort_sum(probabilities)
        state = np.random.get_state()
        np.random.seed(self.calibration_seed)
        try:
            # Convert each per-row score explicitly to a scalar for NumPy 2.
            scores = np.asarray(
                [
                    np.asarray(
                        get_tau(
                            probabilities[i : i + 1],
                            int(labels[i]),
                            order[i : i + 1],
                            ordered[i : i + 1],
                            cumulative[i : i + 1],
                            penalties[0],
                            randomized=True,
                            allow_zero_sets=True,
                        )
                    ).item()
                    for i in range(len(labels))
                ],
                float,
            )
        finally:
            np.random.set_state(state)
        # Use the higher empirical quantile.
        try:
            self.q = float(np.quantile(scores, 1 - alpha, method="higher"))
        except TypeError:  # NumPy < 1.22 compatibility
            self.q = float(
                np.quantile(scores, 1 - alpha, interpolation="higher")
            )
        return self.q

    def predict_sets(self, probabilities):
        """Generate predictions for the inputs."""
        if self.q is None:
            raise RuntimeError("calibrate first")
        probabilities = np.asarray(probabilities, float)
        penalties = np.zeros((1, probabilities.shape[1]), float)
        penalties[:, self.k_reg :] = self.penalty_lambda
        order, ordered, cumulative = self._sort_sum(probabilities)
        state = np.random.get_state()
        np.random.seed(self.prediction_seed)
        try:
            sets = gcq(
                probabilities,
                self.q,
                I=order,
                ordered=ordered,
                cumsum=cumulative,
                penalties=penalties,
                randomized=True,
                allow_zero_sets=True,
            )
        finally:
            np.random.set_state(state)
        return [set(np.asarray(s, int).tolist()) for s in sets]
