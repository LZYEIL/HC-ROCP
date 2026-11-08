"""RAC baseline using a label-by-action utility matrix."""
from typing import Dict, Iterable, List, Sequence, Set, Tuple
import numpy as np

class RACBaseline:
    """
    RAC baseline for discrete labels using a utility matrix U[y, a].
    """

    def __init__(self, actions: Sequence[object], utility_matrix: Sequence[Sequence[float]]) -> None:
        self.actions: List[object] = list(actions)
        self.action_to_index: Dict[object, int] = {a: i for i, a in enumerate(self.actions)}
        self.utility_matrix = np.asarray(utility_matrix, dtype=float)
        if self.utility_matrix.ndim != 2:
            raise ValueError("utility_matrix must be a 2D array [num_labels, num_actions].")
        if self.utility_matrix.shape[1] != len(self.actions):
            raise ValueError("utility_matrix second dimension must match number of actions.")

    def _utility(self, action: object, true_label: int) -> float:
        a_idx = self.action_to_index[action]
        return float(self.utility_matrix[true_label, a_idx])

    def hbtheta_and_arg(self, t: float, x_probs: Sequence[float]) -> Tuple[float, object]:
        """
        Compute h_btheta(x,t) and argmax action (utilities-based).
        """
        x_probs = np.asarray(x_probs, dtype=float)
        num_labels = len(x_probs)
        labels = np.arange(num_labels)

        best_u = -np.inf
        best_action = None

        for action in self.actions:
            utilities_for_a = np.array([self._utility(action, y) for y in labels], dtype=float)
            unique_util_vals = np.unique(utilities_for_a)
            unique_util_vals = np.sort(unique_util_vals)[::-1]

            candidate_u = -np.inf
            for u_candidate in unique_util_vals:
                mask = utilities_for_a >= u_candidate
                sum_prob = np.sum(x_probs[mask])
                if sum_prob >= t:
                    candidate_u = u_candidate
                    break

            if candidate_u > best_u:
                best_u = candidate_u
                best_action = action

        if best_action is None:
            raise RuntimeError("Failed to compute hbtheta_and_arg.")
        return float(best_u), best_action

    def compute_g_hat(self, x_probs: Sequence[float], beta: float) -> float:
        """
        Compute g_hat(x, beta) = argmax_s [ beta*s + h_btheta(x, s) ].
        """
        x_probs = np.asarray(x_probs, dtype=float)
        num_labels = len(x_probs)

        action_thresholds: Dict[object, List[Tuple[float, float]]] = {}
        all_s_candidates = {0.0, 1.0}

        for action in self.actions:
            utilities_for_a = np.array(
                [self._utility(action, y) for y in range(num_labels)], dtype=float
            )
            unique_utils = np.unique(utilities_for_a)
            unique_utils = np.sort(unique_utils)[::-1]

            pairs: List[Tuple[float, float]] = []
            for u_candidate in unique_utils:
                mask = utilities_for_a >= u_candidate
                t_val = float(x_probs[mask].sum())
                pairs.append((float(u_candidate), t_val))
                all_s_candidates.add(t_val)

            action_thresholds[action] = pairs

        s_candidates = sorted(all_s_candidates)
        s_star = 0.0
        best_score = -np.inf

        for s in s_candidates:
            best_u_for_s = -np.inf
            for action, pairs in action_thresholds.items():
                feasible_u = -np.inf
                for u_candidate, t_val in pairs:
                    if t_val >= s:
                        feasible_u = u_candidate
                        break
                if feasible_u > best_u_for_s:
                    best_u_for_s = feasible_u

            phi_s = beta * s + best_u_for_s
            if phi_s > best_score:
                best_score = phi_s
                s_star = s

        return float(s_star)

    def get_conformal_set(
        self, x_probs: Sequence[float], s_value: float
    ) -> Set[int]:
        """
        Construct the RAC conformal set for a given x and s.
        """
        x_probs = np.asarray(x_probs, dtype=float)
        num_labels = len(x_probs)
        best_u, best_action = self.hbtheta_and_arg(s_value, x_probs)


        u_vec = np.array([self._utility(best_action, y) for y in range(num_labels)], dtype=float)
        return set(np.where(u_vec >= best_u)[0])

    def predict_set(self, x_probs: Sequence[float], beta: float) -> Set[int]:
        s_star = self.compute_g_hat(x_probs, beta)
        return self.get_conformal_set(x_probs, s_star)

    def coverage_for_beta(
        self,
        calib_probs: Sequence[Sequence[float]],
        calib_labels: Sequence[int],
        beta: float,
    ) -> float:
        hits = 0
        n = len(calib_probs)
        for x_probs, y in zip(calib_probs, calib_labels):
            c_star = self.predict_set(x_probs, beta)
            if int(y) in c_star:
                hits += 1
        return hits / max(n, 1)

    def calibrate_beta(
        self,
        calib_probs: Sequence[Sequence[float]],
        calib_labels: Sequence[int],
        alpha: float,
        max_iter: int = 20,
        beta_init: float = 1.0,
        beta_max: float = 1e6,
    ) -> float:
        """
        Find beta (q) using doubling + bisection, matching RAC code.

        Notes:
        - Empirical coverage may plateau below the target (1 - alpha) for some
          datasets/utilities. To avoid an infinite loop, we cap the doubling
          search at beta_max; if the target is still unreachable, we return the
          capped beta.
        """
        target = 1.0 - alpha
        cov0 = self.coverage_for_beta(calib_probs, calib_labels, 0.0)
        if cov0 >= target:
            return 0.0

        beta_low = 0.0
        beta_high = float(beta_init)
        cov = self.coverage_for_beta(calib_probs, calib_labels, beta_high)
        while cov < target and beta_high < beta_max:
            beta_low = beta_high
            beta_high = min(beta_high * 2.0, beta_max)
            cov = self.coverage_for_beta(calib_probs, calib_labels, beta_high)

        if cov < target:
            return float(beta_high)

        for _ in range(max_iter):
            beta_mid = 0.5 * (beta_low + beta_high)
            cov = self.coverage_for_beta(calib_probs, calib_labels, beta_mid)
            if cov >= target:
                beta_high = beta_mid
            else:
                beta_low = beta_mid

        return float(0.5 * (beta_low + beta_high))

    def a_RAC(self, label_set: Iterable[int]) -> Tuple[object, float]:
        """
        Max-min action rule: argmax_a min_{y in S} U[y, a].
        Returns (best_action, best_min_utility).
        """
        label_set = list(label_set)
        if not label_set:
            raise ValueError("label_set must be non-empty.")

        best_action = None
        best_min_u = -np.inf

        for action in self.actions:
            a_idx = self.action_to_index[action]
            utilities = self.utility_matrix[label_set, a_idx]
            min_u = float(np.min(utilities))
            if min_u > best_min_u:
                best_min_u = min_u
                best_action = action

        if best_action is None:
            raise RuntimeError("Failed to compute a_RAC.")
        return best_action, best_min_u
