"""SOCOP baseline: algorithm and experiment interface."""
import numpy as np
from abc import ABC, abstractmethod
from numba import jit

class PredictionSetMethod(ABC):
    """Abstract base class for different conformal prediction methods."""
    
    def __init__(self, alpha: float):
        self.alpha = alpha
    
    @abstractmethod
    def compute_conformity_score(self, p_hat: np.ndarray, target: int) -> float:
        """Compute conformity score for calibration."""
        pass
    
    @abstractmethod
    def compute_prediction_set(self, p_hat: np.ndarray, threshold: float) -> np.ndarray:
        """Compute prediction set given probabilities and threshold."""
        pass
    
    @abstractmethod
    def is_conformity_score(self) -> bool:
        """Return True if this method uses conformity scores (higher = more conforming).
        Return False if this method uses non-conformity scores (higher = less conforming)."""
        pass
    
    def needs_calibration(self) -> bool:
        """Return True if this method needs conformity score calibration."""
        return True

class SOCOP(PredictionSetMethod):
    """
    Singleton-Optimized Conformal Prediction using the geometric Convex Hull approach.
    """
    def __init__(self, alpha: float, penalty_lambda: float = 0.2, use_numba: bool = False):
        super().__init__(alpha)
        if penalty_lambda <= 0:
            raise ValueError("penalty_lambda must be positive for SOCOP.")
        self.penalty_lambda = penalty_lambda
        self.use_numba = use_numba

        if self.use_numba:
            # Warm-up call to compile the function
            _ = _compute_lower_convex_hull_numba(np.array([0.5, 0.3, 0.2]), self.penalty_lambda)

    def _compute_lower_convex_hull(self, p_hat_sorted: np.ndarray):
        """
        Computes the lower convex hull of the point set P_k = (S_k, g_k).
        """
        if self.use_numba:
            return _compute_lower_convex_hull_numba(p_hat_sorted, self.penalty_lambda)
        else:
            K = len(p_hat_sorted)
            
            S = np.zeros(K + 1)
            S[1:] = np.cumsum(p_hat_sorted)
            g = (np.arange(K + 1) > 1).astype(float) + self.penalty_lambda * np.arange(K + 1)
            
            lower_hull_indices = []
            for k in range(K + 1):
                while len(lower_hull_indices) >= 2:
                    i = lower_hull_indices[-1]
                    j = lower_hull_indices[-2]
                    cross_product = (S[i] - S[j]) * (g[k] - g[i]) - (S[k] - S[i]) * (g[i] - g[j])
                    
                    if cross_product <= 1e-9:
                        lower_hull_indices.pop()
                    else:
                        break 
                lower_hull_indices.append(k)
                
            return np.array(lower_hull_indices), S, g

    def compute_conformity_score(self, p_hat: np.ndarray, target: int) -> float:
        """
        Compute non-conformity score for SOCOP.
        """
        sorted_indices = np.argsort(p_hat)[::-1]
        p_sorted = p_hat[sorted_indices]
        try:
            i = np.where(sorted_indices == target)[0][0] + 1
        except IndexError:
            return np.inf

        hull_indices, S, g = self._compute_lower_convex_hull(p_sorted)

        found_j = -1
        for j, v_j in enumerate(hull_indices):
            if v_j >= i:
                found_j = j
                break
        
        if found_j <= 0: 
             if len(hull_indices) > 1 and hull_indices[0] < i <= hull_indices[1]:
                 found_j = 1
             else: 
                 return 0.0

        v_j = hull_indices[found_j]
        v_prev = hull_indices[found_j - 1]
        
        delta_S = S[v_j] - S[v_prev]
        
        return np.inf if delta_S < 1e-12 else (g[v_j] - g[v_prev]) / delta_S

    def compute_prediction_set(self, p_hat: np.ndarray, threshold: float) -> np.ndarray:
        """
        Compute the prediction set for SOCOP.
        """
        sorted_indices = np.argsort(p_hat)[::-1]
        p_sorted = p_hat[sorted_indices]

        hull_indices, S, g = self._compute_lower_convex_hull(p_sorted)
        k_final = hull_indices[0] 
        for j in range(1, len(hull_indices)):
            v_prev = hull_indices[j - 1]
            v_curr = hull_indices[j]
            
            delta_S = S[v_curr] - S[v_prev]
            slope = np.inf if delta_S < 1e-12 else (g[v_curr] - g[v_prev]) / delta_S

            if slope <= threshold:
                k_final = v_curr
            else:
                break
        
        return sorted_indices[:k_final]
    
    
    def is_conformity_score(self) -> bool:
        return False  

@jit(nopython=True, cache=True)
def _compute_lower_convex_hull_numba(p_hat_sorted: np.ndarray, penalty_lambda: float):
    K = len(p_hat_sorted)
    S = np.zeros(K + 1, dtype=np.float64)
    g = np.zeros(K + 1, dtype=np.float64)
    
    current_S = 0.0
    for k in range(K):
        current_S += p_hat_sorted[k]
        S[k+1] = current_S
    for k in range(K + 1):
        g[k] = (1.0 if k > 1 else 0.0) + penalty_lambda * k

    lower_hull_indices_arr = np.zeros(K + 1, dtype=np.int64)
    hull_size = 0
    for k in range(K + 1):
        while hull_size >= 2:
            i = lower_hull_indices_arr[hull_size - 1]
            j = lower_hull_indices_arr[hull_size - 2]
            cross_product = (S[i] - S[j]) * (g[k] - g[i]) - (S[k] - S[i]) * (g[i] - g[j])
            if cross_product <= 1e-9:
                hull_size -= 1
            else:
                break
        lower_hull_indices_arr[hull_size] = k
        hull_size += 1
    return lower_hull_indices_arr[:hull_size], S, g


class OfficialSOCOPConformal:
    """Split-conformal adapter executing the official pure-Python SOCOP class."""

    def __init__(self, alpha=0.05, penalty_lambda=0.0625):
        """Initialize the object."""
        implementation = SOCOP
        self.method = implementation(
            alpha=float(alpha),
            penalty_lambda=float(penalty_lambda),
            use_numba=False,
        )
        self.q = None

    def calibrate(self, probabilities, labels, alpha):
        """Fit the finite-sample quantile of SOCOP nonconformity scores."""
        probabilities = np.asarray(probabilities, float)
        labels = np.asarray(labels, int)
        scores = np.asarray(
            [
                self.method.compute_conformity_score(p, int(y))
                for p, y in zip(probabilities, labels)
            ],
            float,
        )
        rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
        self.q = (
            float("inf")
            if rank > len(scores)
            else float(np.partition(scores, rank - 1)[rank - 1])
        )
        return self.q

    def predict_set(self, probability):
        """Generate predictions for the inputs."""
        if self.q is None:
            raise RuntimeError("calibrate first")
        return set(
            np.asarray(
                self.method.compute_prediction_set(
                    np.asarray(probability, float), self.q
                ),
                int,
            ).tolist()
        )
