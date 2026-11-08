# Baselines

Self-contained comparison methods operating on frozen classifier probabilities.
Install the root Python environment; no upstream baseline repositories are needed.

| Folder | Method |
|---|---|
| `rocp/` | Risk-optimal prediction sets and actions using a loss matrix. |
| `rac/` | Utility-based prediction sets and action selection. |
| `las/` | Least-ambiguous sets using the score `1 − p(y)`. |
| `aps/` | Adaptive prediction sets based on cumulative class probabilities. |
| `raps/` | Randomized APS with a penalty for labels beyond a chosen rank. |
| `saps/` | Randomized prediction sets using label ranks and the largest probability. |
| `mondrian_aps/` | APS with a separate calibration threshold for each class. |
| `socop/` | Singleton-optimized conformal prediction. |
| `crc/` | Conformal risk control adapted to severe-action-error loss. |
| `ltt/` | Learn-then-Test with Hoeffding–Bentkus p-values and Bonferroni correction. |
| `utility_directed/` | Utility-directed CP score adapter using the decision loss matrix. |
| `action_only/` | Bayes and always-safe action rules; neither outputs prediction sets. |

Each folder contains `implementation.py` (algorithm and experiment interface)
and `__init__.py` (imports). `_shared.py` holds calibration helpers used by
APS, LAS, and utility-directed CP, plus the threshold-action rule used by CRC and LTT.
