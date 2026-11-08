"""Evidence-aware hierarchical certification for joint beta/gamma HC-ROCP."""

from dataclasses import dataclass
import numpy as np
from scipy.stats import beta as beta_dist

from .decision_problem import _validate_indices
from .policy import HarmConstrainedROCP


@dataclass(frozen=True)
class SafetyGroup:
    """Severe-label group and its maximum permitted conditional error rate."""

    name: str
    labels: tuple
    delta: float


def automatic_groups(severe_labels, safety_labels, delta, min_count=1):
    """Add the global severe group, then labels with at least min_count cases.

    The count rule uses labels only, not policy errors. The global group is
    always retained, including when no severe cases are present.
    """
    y = _validate_indices(safety_labels, None, "safety_labels", nonempty=False)
    severe = tuple(
        int(x) for x in _validate_indices(severe_labels, None, "severe_labels")
    )
    if len(set(severe)) != len(severe):
        raise ValueError("severe_labels must not contain duplicates")
    if not 0 <= delta <= 1:
        raise ValueError("delta must be in [0, 1]")
    if not isinstance(min_count, (int, np.integer)) or min_count < 0:
        raise ValueError("min_count must be a nonnegative integer")
    groups = [SafetyGroup("all_severe", severe, float(delta))]
    for lab in severe:
        if int(np.sum(y == lab)) >= int(min_count):
            groups.append(SafetyGroup(f"label_{lab}", (lab,), float(delta)))
    return groups


class CertifiableHCROCP(HarmConstrainedROCP):
    """Test candidate policies and select by evidence level, then safety-split loss.

    Callers must fix the candidate family independently of the disjoint coverage
    and safety splits. The first SafetyGroup must contain all severe labels;
    later groups specify severe subgroups. Array validation cannot check split
    independence or the sampling assumptions needed for the statistical bounds.
    """

    @staticmethod
    def _upper(k, n, confidence):
        """Return a one-sided Clopper-Pearson upper bound (one if n is zero)."""
        if n == 0 or k == n:
            return 1.0
        return float(beta_dist.ppf(confidence, k + 1, n - k))

    @staticmethod
    def _lower(k, n, confidence):
        """Return a one-sided Clopper-Pearson lower bound (zero if n is zero)."""
        if n == 0 or k == 0:
            return 0.0
        return float(beta_dist.ppf(1 - confidence, k, n - k + 1))

    def _fit_pairs(
        self,
        coverage_p,
        coverage_y,
        safety_p,
        safety_y,
        pairs,
        target_coverage,
        groups,
        loss_matrix,
        family_confidence=0.95,
    ):
        """Apply adjusted coverage and harm gates to a fixed candidate family."""
        cp, cy = self._calibration_data(coverage_p, coverage_y, "coverage")
        sp, sy = self._calibration_data(safety_p, safety_y, "safety")
        loss = np.asarray(loss_matrix, float)
        if loss.shape != self.problem.loss_matrix.shape or not np.all(
            np.isfinite(loss)
        ):
            raise ValueError(
                "loss_matrix must be finite and match the decision problem's shape"
            )
        if not 0 < target_coverage <= 1:
            raise ValueError("target_coverage must be in (0, 1]")
        if not 0 < family_confidence < 1:
            raise ValueError("family_confidence must be in (0, 1)")
        groups = list(groups)
        if not groups:
            raise ValueError("groups must start with the global severe-label group")
        names = set()
        severe = set(self.severe_labels)
        for index, group in enumerate(groups):
            if not isinstance(group, SafetyGroup):
                raise ValueError("groups must contain SafetyGroup instances")
            labels = _validate_indices(
                group.labels, self.problem.num_labels, "group labels"
            )
            if len(set(labels)) != len(labels) or not set(labels).issubset(severe):
                raise ValueError("group labels must be distinct severe labels")
            if index == 0 and set(labels) != severe:
                raise ValueError("the first group must contain all severe labels")
            if not isinstance(group.name, str) or not group.name or group.name in names:
                raise ValueError("group names must be nonempty and unique")
            if not 0 <= group.delta <= 1:
                raise ValueError("group delta must be in [0, 1]")
            names.add(group.name)
        pairs = [
            (self._parameter(b, "beta"), self._parameter(g, "gamma")) for b, g in pairs
        ]
        if not pairs:
            raise ValueError("candidate pair family must be nonempty")
        tests_per_pair = 1 + len(groups)
        err = (1 - family_confidence) / max(1, len(pairs) * tests_per_pair)
        conf = 1 - err
        rows = []
        # Cache set masses and severe-label probabilities shared by every candidate.
        indicators = self.problem.table.indicators
        risks = self.problem.table.risks
        table_actions = self.problem.table.actions
        harm_indicator = self.harm_indicator
        cp_mass = cp @ indicators.T
        sp_mass = sp @ indicators.T
        cp_severe = cp[:, self.severe_labels].sum(axis=1)
        sp_severe = sp[:, self.severe_labels].sum(axis=1)
        for b, g in pairs:
            cp_objective = (
                risks[None, :]
                - b * cp_mass
                + g * cp_severe[:, None] * harm_indicator[None, :]
            )
            cp_indices = np.argmin(cp_objective, axis=1)
            covered = int(np.sum(indicators[cp_indices, cy]))
            cov_emp = covered / len(cy)
            cov_exp = (covered + 1) / (len(cy) + 1)
            cov_lb = self._lower(covered, len(cy), conf)
            sp_objective = (
                risks[None, :]
                - b * sp_mass
                + g * sp_severe[:, None] * harm_indicator[None, :]
            )
            acts = table_actions[np.argmin(sp_objective, axis=1)]
            stats = []
            for group in groups:
                mask = np.isin(sy, group.labels)
                n = int(mask.sum())
                k = int(np.sum(np.isin(acts[mask], self.harmful_actions))) if n else 0
                expected = (k + 1) / (n + 1) if n else 1.0
                upper = self._upper(k, n, conf)
                stats.append(
                    {
                        "name": group.name,
                        "labels": list(group.labels),
                        "n": n,
                        "harm_count": k,
                        "empirical": k / n if n else None,
                        "expected_bound": expected,
                        "upper": upper,
                        "delta": group.delta,
                        "empirical_feasible": bool(n and k / n <= group.delta),
                        "expected_feasible": bool(n and expected <= group.delta),
                        "certified": bool(n and upper <= group.delta),
                    }
                )
            global_s = stats[0]
            selection_loss = float(np.mean(loss[sy, acts]))
            rows.append(
                {
                    "beta": b,
                    "gamma": g,
                    "coverage_empirical": cov_emp,
                    "coverage_expected": cov_exp,
                    "coverage_lower": cov_lb,
                    "coverage_empirical_feasible": cov_emp >= target_coverage,
                    "coverage_certified": cov_lb >= target_coverage,
                    "groups": stats,
                    "selection_loss": selection_loss,
                    "global_empirical_feasible": cov_emp >= target_coverage
                    and global_s["empirical_feasible"],
                    "global_expected_feasible": cov_exp >= target_coverage
                    and global_s["expected_feasible"],
                    "global_certified": cov_lb >= target_coverage
                    and global_s["certified"],
                }
            )

        def best(key):
            """Choose minimum loss among candidates passing the specified criterion."""
            z = [r for r in rows if r[key]]
            return min(z, key=lambda r: r["selection_loss"]) if z else None

        selected = (
            best("global_certified")
            or best("global_expected_feasible")
            or best("global_empirical_feasible")
        )
        if selected is None:
            selected = min(
                rows,
                key=lambda r: (
                    max(0, target_coverage - r["coverage_empirical"]),
                    (
                        r["groups"][0]["empirical"]
                        if r["groups"][0]["empirical"] is not None
                        else 1
                    ),
                    r["selection_loss"],
                ),
            )
            level = "infeasible_fallback"
        elif selected["global_certified"]:
            level = "high_confidence_certified"
        elif selected["global_expected_feasible"]:
            level = "expected_risk_fallback"
        else:
            level = "empirical_fallback"
        subgroup_status = []
        for s in selected["groups"][1:]:
            subgroup_status.append(
                {
                    **s,
                    "decision": (
                        "subgroup_certificate"
                        if s["certified"]
                        else "fallback_to_global"
                    ),
                    "fallback_level": level if not s["certified"] else None,
                }
            )
        return {
            "selected": selected,
            "global_level": level,
            "subgroups": subgroup_status,
            "num_candidates": len(pairs),
            "multiplicity": {
                "family_confidence": family_confidence,
                "tests_per_pair": tests_per_pair,
                "per_test_error": err,
            },
            "counts": {
                "global_certified": sum(r["global_certified"] for r in rows),
                "global_expected_feasible": sum(
                    r["global_expected_feasible"] for r in rows
                ),
                "global_empirical_feasible": sum(
                    r["global_empirical_feasible"] for r in rows
                ),
            },
        }

    def fit_candidate_pairs(
        self,
        coverage_p,
        coverage_y,
        safety_p,
        safety_y,
        candidate_pairs,
        target_coverage,
        groups,
        loss_matrix,
        family_confidence=0.95,
    ):
        """Certify an explicitly pre-specified finite family of (beta, gamma) pairs.

        This entry point is useful when an independent construction split has
        reduced a larger Cartesian grid to a shortlist.  Validity requires the
        supplied pairs to be fixed without inspecting ``coverage_*`` or
        ``safety_*``; the method then corrects only over the supplied family.
        """
        return self._fit_pairs(
            coverage_p,
            coverage_y,
            safety_p,
            safety_y,
            candidate_pairs,
            target_coverage,
            groups,
            loss_matrix,
            family_confidence,
        )

    def fit_certifiable(
        self,
        coverage_p,
        coverage_y,
        safety_p,
        safety_y,
        beta_grid,
        gamma_grid,
        target_coverage,
        groups,
        loss_matrix,
        family_confidence=0.95,
    ):
        """Test the Cartesian beta/gamma grid and return the selected policy report.

        Fix both grids before examining coverage or safety outcomes. Reports
        distinguish certification from descriptive fallbacks; groups with no
        observed severe cases cannot pass their harm gate.
        """
        beta_grid = list(beta_grid)
        gamma_grid = list(gamma_grid)
        pairs = [(float(b), float(g)) for b in beta_grid for g in gamma_grid]
        return self._fit_pairs(
            coverage_p,
            coverage_y,
            safety_p,
            safety_y,
            pairs,
            target_coverage,
            groups,
            loss_matrix,
            family_confidence,
        )


__all__ = ["SafetyGroup", "automatic_groups", "CertifiableHCROCP"]
