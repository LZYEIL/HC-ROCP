"""Prediction-set and action selection with beta/gamma calibration helpers."""

from __future__ import annotations

import numpy as np

from .decision_problem import _validate_indices


class HarmConstrainedROCP:
    """Select prediction sets with coverage and harmful-action penalties.

    For frozen probabilities p(x), the selected set minimizes

        rho(S) - beta p(S|x) + gamma p(Y_H|x) 1{a(S) in A_H},

    Here rho is the robust set risk and a is its minimizing action.
    Setting gamma=0 recovers the ROCP set objective.
    """

    def __init__(self, problem, severe_labels, harmful_actions):
        """Use prespecified severe labels and harmful actions to define the error event."""
        self.problem = problem
        self.severe_labels = _validate_indices(
            severe_labels, problem.num_labels, "severe_labels"
        )
        self.harmful_actions = _validate_indices(
            harmful_actions, problem.num_actions, "harmful_actions", nonempty=False
        )
        if len(np.unique(self.severe_labels)) != len(self.severe_labels):
            raise ValueError("severe_labels must not contain duplicates")
        self.harm_indicator = np.isin(
            problem.table.actions, self.harmful_actions
        ).astype(float)

    def _probabilities(self, probabilities, name="probabilities"):
        """Validate finite probability rows without modifying or renormalizing them."""
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] != self.problem.num_labels:
            raise ValueError(f"{name} must have shape [num_cases, num_labels]")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise ValueError(f"{name} must contain finite, nonnegative probabilities")
        if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-8):
            raise ValueError(f"{name} rows must sum to one")
        return probabilities

    def _calibration_data(self, probabilities, labels, name):
        """Validate a nonempty calibration split with one label per probability row."""
        probabilities = self._probabilities(probabilities, f"{name} probabilities")
        labels = _validate_indices(labels, self.problem.num_labels, f"{name} labels")
        if len(probabilities) != len(labels):
            raise ValueError(f"{name} probabilities and labels must have equal lengths")
        return probabilities, labels

    @staticmethod
    def _parameter(value, name):
        """Validate a finite, nonnegative penalty parameter."""
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return value

    def set_indices(self, probabilities, beta, gamma):
        """Return decision-table row indices; objective ties use the first row."""
        probabilities = self._probabilities(probabilities)
        beta = self._parameter(beta, "beta")
        gamma = self._parameter(gamma, "gamma")
        mass = probabilities @ self.problem.table.indicators.T
        severe_mass = probabilities[:, self.severe_labels].sum(axis=1)
        objective = (
            self.problem.table.risks[None, :]
            - float(beta) * mass
            + float(gamma) * severe_mass[:, None] * self.harm_indicator[None, :]
        )
        return np.argmin(objective, axis=1)

    def actions(self, probabilities, beta, gamma):
        """Return the selected actions."""
        indices = self.set_indices(probabilities, beta, gamma)
        return self.problem.table.actions[indices]

    def sets(self, probabilities, beta, gamma):
        """Return the selected prediction sets."""
        indices = self.set_indices(probabilities, beta, gamma)
        return [set(np.flatnonzero(self.problem.table.indicators[i])) for i in indices]

    def calibrate_beta(
        self, probabilities, labels, gamma, target_coverage, iterations=25
    ):
        """Search for beta meeting empirical coverage at a fixed gamma.

        Double the upper endpoint up to the search limit, then bisect. The
        returned endpoint is not a coverage certificate and may miss the target
        if the search limit is reached without finding a feasible endpoint.
        """
        probabilities, labels = self._calibration_data(
            probabilities, labels, "coverage"
        )
        gamma = self._parameter(gamma, "gamma")
        if not 0 < target_coverage <= 1:
            raise ValueError("target_coverage must be in (0, 1]")
        if not isinstance(iterations, (int, np.integer)) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        mass = probabilities @ self.problem.table.indicators.T
        severe_mass = probabilities[:, self.severe_labels].sum(axis=1)
        base_objective = (
            self.problem.table.risks[None, :]
            + float(gamma) * severe_mass[:, None] * self.harm_indicator[None, :]
        )

        def coverage(beta):
            """Compute empirical coverage."""
            idx = np.argmin(base_objective - float(beta) * mass, axis=1)
            return float(np.mean(self.problem.table.indicators[idx, labels]))

        if coverage(0.0) >= target_coverage:
            return 0.0
        low, high = 0.0, 1.0
        while coverage(high) < target_coverage and high < 1e7:
            low, high = high, high * 2.0
        for _ in range(iterations):
            mid = (low + high) / 2.0
            if coverage(mid) >= target_coverage:
                high = mid
            else:
                low = mid
        return float(high)
