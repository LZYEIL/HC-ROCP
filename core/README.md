# HC-ROCP core

The algorithm uses prespecified actions and an expert-specified loss matrix.

| File | Role |
|---|---|
| `decision_problem.py` | Compute each possible prediction set's risk and corresponding action from the loss matrix. |
| `policy.py` | Select a set and action for a case using its predicted probabilities and beta/gamma parameters. |
| `hc_rocp.py` | Test candidate policies on calibration data, select a policy, and report evidence status. |

Main entry point: `hc_rocp.py:CertifiableHCROCP`. Experiment scripts are in `experiments/`.

Calibration inputs must be nonempty, with finite probability rows summing to one
and zero-based class labels. Supply disjoint coverage and safety splits, fix the
candidate family independently, and place the global severe-label group first.

`policy.py` also calibrates beta at a fixed gamma. Certification and evidence
reporting are handled by `CertifiableHCROCP`.
