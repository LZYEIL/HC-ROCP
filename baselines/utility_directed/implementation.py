"""Utility-directed score adapter for prediction sets."""
from typing import Sequence
import numpy as np
from baselines._shared import _BaseConformalScorer

class Score3CortesConformal(_BaseConformalScorer):
    """
    Cortés-Gómez et al. (ICLR'25) greedy score (score-3).

    Steps per x:
      1) Build permutation sigma_x with greedy rule (their Eq. 2):
         start S=∅; while labels remain, pick y maximizing
            (M - L(S∪{y})) / (1 - p_y)
         subject to p_y <= alpha - p(S). If no feasible y remains, append
         the remaining labels in descending probability order.
      2) Nonconformity score s(x,y) = sum_{i <= k(y)} p(sigma_x(i))
         (APS-style cumulative mass along the greedy order).

    For nonempty sets, we use the risk-averse certificate as the set loss:
      L(S) = min_a [ ell_in(a,S) + alpha * (ell_out(a,S) - ell_in(a,S))_+ ]
    where ell_in(a,S) = max_{y in S} loss[y,a],
          ell_out(a,S) = max_{y not in S} loss[y,a] (0 if S=Y).
    The empty-set loss is defined as zero. M is max loss over all (y,a)
    and is only used to form the greedy ratio.
    """

    def __init__(
        self,
        actions: Sequence[object],
        loss_matrix: Sequence[Sequence[float]],
        alpha: float,
    ) -> None:
        super().__init__(actions)
        self.loss_matrix = np.asarray(loss_matrix, dtype=float)
        if self.loss_matrix.ndim != 2:
            raise ValueError("loss_matrix must be 2D [num_labels, num_actions].")
        if self.loss_matrix.shape[1] != len(self.actions):
            raise ValueError("loss_matrix second dimension must match actions.")
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0,1).")
        self.alpha = float(alpha)
        self.M = float(np.max(self.loss_matrix))

    def _set_loss(self, subset: np.ndarray) -> float:
        """
        Risk-averse set loss L(S).
        """
        if subset.size == 0:
            return 0.0
        num_labels = self.loss_matrix.shape[0]
        all_labels = np.arange(num_labels)
        comp = np.setdiff1d(all_labels, subset, assume_unique=True)

        best_val = np.inf
        for a_idx in range(len(self.actions)):
            ell_in = float(np.max(self.loss_matrix[subset, a_idx])) if subset.size else 0.0
            ell_out = float(np.max(self.loss_matrix[comp, a_idx])) if comp.size else 0.0
            val = ell_in + self.alpha * max(ell_out - ell_in, 0.0)
            if val < best_val:
                best_val = val
        return best_val

    def _greedy_order(self, x_probs: np.ndarray) -> np.ndarray:
        """
        Build sigma_x using greedy rule with probability budget (phase 1),
        then append remaining labels in descending probability order (phase 2).
        """
        K = len(x_probs)
        remaining = np.ones(K, dtype=bool)
        order = []
        p_sum = 0.0
        # Phase 1: constrained greedy
        while len(order) < K:
            feasible_mask = (x_probs <= self.alpha - p_sum) & remaining
            if not feasible_mask.any():
                break
            candidates = np.where(feasible_mask)[0]

            best_ratio = -np.inf
            best_y = candidates[0]
            for y in candidates:
                s_temp = np.array(order + [y], dtype=int)
                l_val = self._set_loss(s_temp)
                ratio = (self.M - l_val) / max(1e-12, 1.0 - float(x_probs[y]))
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_y = y

            order.append(int(best_y))
            remaining[best_y] = False
            p_sum += float(x_probs[best_y])

        # Phase 2: append any leftover labels in descending probability (fixed tie-break).
        if remaining.any():
            tail = np.where(remaining)[0]
            tail_sorted = tail[np.argsort(-x_probs[tail], kind="mergesort")]
            order.extend(tail_sorted.tolist())

        return np.array(order, dtype=int)

    def _score(self, x_probs, label):
        x_probs = np.asarray(x_probs, dtype=float)
        sigma = self._greedy_order(x_probs)
        pos = int(np.where(sigma == label)[0][0])
        return float(np.sum(x_probs[sigma[: pos + 1]]))

    def calibrate(self, cal_probs: Sequence[Sequence[float]], cal_labels: Sequence[int], alpha: float) -> float:
        # enforce alpha consistency with the greedy construction
        if abs(alpha - self.alpha) > 1e-12:
            raise ValueError("Alpha for calibrate must match the alpha used in the greedy score.")
        return super().calibrate(cal_probs, cal_labels, alpha)
