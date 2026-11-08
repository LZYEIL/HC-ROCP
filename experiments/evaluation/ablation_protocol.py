#!/usr/bin/env python3
"""Ablation study for Certifiable HC-ROCP.

Uses frozen COVID-19 probabilities, with an additional ISIC subgroup diagnostic:

  * Table 11 -- component ablations (single-split leakage, two-split designs,
                reduced correction, fixed-beta, gamma=0, sparse/dense grid);
  * Table 13 -- coverage-target (alpha) sensitivity;
  * Figure 5(a) -- post-selection severe-case count diagnostic.

Reuses the same classifier probabilities and CertifiableHCROCP implementation.
Variants change the split design, correction, candidate grid, parameter selection,
or coverage target. The count diagnostic subsamples a fixed selected policy.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from shared.paths import OUTPUTS_DIR

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.decision_problem import FiniteDecisionProblem
from core.hc_rocp import (
    CertifiableHCROCP, automatic_groups,
)

OUT = OUTPUTS_DIR

LOSS = [[0, 8, 8, 6], [10, 0, 7, 3], [10, 7, 0, 2], [9, 6, 6, 0]]
SEVERE = [1, 2, 3]
HARMFUL = [0]
ALPHA0 = 0.05
DELTA = 0.01
EPS = 0.05
SEED = 20260815
COVID_OD = str(OUTPUTS_DIR / "covid19")
ISIC_OD = str(OUTPUTS_DIR / "isic2019")
GAMMAS_FULL = np.r_[0.0, np.geomspace(1e-4, 1e6, 41)]


def load(od: str, split: str):
    """Load frozen probabilities and labels from the named NPZ split."""
    d = np.load(f"{od}/probabilities_{split}.npz")
    return d["probabilities"].astype(float), d["labels"].astype(int)


def make_method(loss=LOSS, severe=SEVERE, harmful=HARMFUL, alpha0=ALPHA0):
    """Construct HC-ROCP with the specified loss matrix and severe/harmful definitions."""
    return CertifiableHCROCP(FiniteDecisionProblem(np.asarray(loss, float), alpha=alpha0),
                             severe, harmful)


def test_metrics(m, p, y, b, g, loss):
    """Compute held-out coverage, loss, severe-action errors and set sizes."""
    s = m.sets(p, b, g); a = m.actions(p, b, g)
    z = np.isin(y, SEVERE); h = np.isin(a, HARMFUL)
    return {
        "coverage": float(np.mean([int(t) in q for t, q in zip(y, s)])),
        "loss": float(np.mean(loss[y, a])),
        "harmful_rate": float(np.mean(h[z])) if z.any() else None,
        "harm_count": int(h[z].sum()),
        "n_severe": int(z.sum()),
        "set_size": float(np.mean([len(q) for q in s])),
    }


def build_betas(m, cp, cy, gammas, targets):
    """Calibrate and deduplicate beta candidates across gamma values and coverage targets."""
    return sorted(set(round(m.calibrate_beta(cp, cy, float(g), float(t)), 10)
                      for t in targets for g in gammas))


def run_variant(m, cst_p, cst_y, cov_p, cov_y, saf_p, saf_y, gammas, targets,
                target_cov, groups, family_confidence=0.95):
    """Construct and test one COVID-19 protocol variant and return its selected policy."""
    betas = build_betas(m, cst_p, cst_y, gammas, targets)
    fit = m.fit_certifiable(cov_p, cov_y, saf_p, saf_y, betas, gammas, target_cov,
                            groups, np.asarray(LOSS, float), family_confidence)
    sel = fit["selected"]
    return {"beta": sel["beta"], "gamma": sel["gamma"], "level": fit["global_level"],
            "num_candidates": fit["num_candidates"], "groups": sel["groups"],
            "subgroups": fit["subgroups"], "counts": fit["counts"],
            "per_test_error": fit["multiplicity"]["per_test_error"]}


def standard_splits(cal_y):
    """Create the fixed stratified construct, coverage and safety splits."""
    ids = np.arange(len(cal_y))
    construct, remain = train_test_split(ids, test_size=2 / 3, random_state=SEED,
                                         stratify=cal_y)
    cov, safe = train_test_split(remain, test_size=0.5, random_state=SEED + 1,
                                 stratify=cal_y[remain])
    return construct, cov, safe


def main():
    """Run the command-line entry point."""
    OUT.mkdir(parents=True, exist_ok=True)
    cp, cy = load(COVID_OD, "calibration")
    tp, ty = load(COVID_OD, "test")
    m = make_method()
    loss = np.asarray(LOSS, float)

    construct, cov, safe = standard_splits(cy)
    full_ids = np.arange(len(cy))

    # ---- Component ablations (all on COVID-19) ----
    table5 = []

    # Row 1: Full (ours)
    groups = automatic_groups(SEVERE, cy[safe], DELTA)
    r = run_variant(m, cp[construct], cy[construct], cp[cov], cy[cov],
                    cp[safe], cy[safe], GAMMAS_FULL, [0.95, 0.975, 0.99], 0.95, groups)
    table5.append({"row": "Full (ours)", **r,
                   "test": test_metrics(m, tp, ty, r["beta"], r["gamma"], loss)})

    # Row 2: naive single split (construct == coverage == safety == all data)
    r = run_variant(m, cp, cy, cp, cy, cp, cy, GAMMAS_FULL, [0.95, 0.975, 0.99],
                    0.95, groups)
    table5.append({"row": "Naive single-split (leakage)", **r,
                   "test": test_metrics(m, tp, ty, r["beta"], r["gamma"], loss)})

    # Row 3: two-split A (construct+coverage merged 2/3, safety 1/3)
    merged, safeA = train_test_split(full_ids, test_size=1 / 3, random_state=SEED,
                                     stratify=cy)
    groupsA = automatic_groups(SEVERE, cy[safeA], DELTA)
    r = run_variant(m, cp[merged], cy[merged], cp[merged], cy[merged],
                    cp[safeA], cy[safeA], GAMMAS_FULL, [0.95, 0.975, 0.99], 0.95, groupsA)
    table5.append({"row": "Two-split A (construct=coverage)", **r,
                   "test": test_metrics(m, tp, ty, r["beta"], r["gamma"], loss)})

    # Row 4: two-split B (construct 1/3, coverage+safety merged 2/3)
    constructB, mergedB = train_test_split(full_ids, test_size=2 / 3, random_state=SEED,
                                           stratify=cy)
    groupsB = automatic_groups(SEVERE, cy[mergedB], DELTA)
    r = run_variant(m, cp[constructB], cy[constructB], cp[mergedB], cy[mergedB],
                    cp[mergedB], cy[mergedB], GAMMAS_FULL, [0.95, 0.975, 0.99], 0.95, groupsB)
    table5.append({"row": "Two-split B (coverage=safety)", **r,
                   "test": test_metrics(m, tp, ty, r["beta"], r["gamma"], loss)})

    # Row 5: remove the group factor (eta = eps/M instead of eps/[M(1+G)]).
    #       via family_confidence = 1 - eps*(1+G) so that per-test error = eps/M.
    G = 1 + len(groups)
    fc_no_bonf = 1.0 - EPS * G
    r = run_variant(m, cp[construct], cy[construct], cp[cov], cy[cov],
                    cp[safe], cy[safe], GAMMAS_FULL, [0.95, 0.975, 0.99], 0.95, groups,
                    family_confidence=fc_no_bonf)
    table5.append({"row": "No Bonferroni (eta=eps/M)", **r,
                   "test": test_metrics(m, tp, ty, r["beta"], r["gamma"], loss)})

    # Row 6: fixed-beta sweep gamma (beta fixed at gamma=0 calibration value)
    b0 = m.calibrate_beta(cp[construct], cy[construct], 0.0, 0.95)
    sev_s = np.isin(cy[safe], SEVERE); nH = int(sev_s.sum())
    k_allowed = max(0, int(np.floor(DELTA * (nH + 1))) - 1)
    gammas = GAMMAS_FULL
    fixed_beta_cov = None; fixed_beta_loss = None; fixed_beta_test = None; fixed_gamma = None
    for g in gammas:
        a = m.actions(cp[safe], b0, g)
        k = int(np.sum(np.isin(a[sev_s], HARMFUL)))
        if k <= k_allowed:
            tm = test_metrics(m, tp, ty, b0, g, loss)
            fixed_beta_cov = tm["coverage"]; fixed_beta_loss = tm["loss"]
            fixed_beta_test = tm; fixed_gamma = float(g)
            break
    # most-conservative beta over the grid
    bstar = max(m.calibrate_beta(cp[construct], cy[construct], float(g), 0.95)
                for g in gammas)
    bstar_loss = None; bstar_cov = None; bstar_test = None; bstar_gamma = None
    for g in gammas:
        a = m.actions(cp[safe], bstar, g)
        k = int(np.sum(np.isin(a[sev_s], HARMFUL)))
        if k <= k_allowed:
            tm = test_metrics(m, tp, ty, bstar, g, loss)
            bstar_loss = tm["loss"]; bstar_cov = tm["coverage"]
            bstar_test = tm; bstar_gamma = float(g)
            break
    table5.append({"row": "Fixed-beta sweep gamma",
                   "fixed_beta0": {"beta0": b0, "coverage_collapse": fixed_beta_cov,
                                   "loss": fixed_beta_loss, "gamma": fixed_gamma,
                                   "test": fixed_beta_test},
                   "fixed_betastar": {"betastar": bstar, "loss": bstar_loss,
                                      "coverage": bstar_cov, "gamma": bstar_gamma,
                                      "test": bstar_test}})

    # Row 7: gamma = 0 exact ROCP (beta calibrated at gamma=0, no harm dual).
    # This diagnostic fits beta on 2/3 of calibration observations, as stated
    # in Appendix F (not the ten-seed comparison's coverage-only split).
    _ci, _si = train_test_split(np.arange(len(cy)), test_size=1 / 3,
                                random_state=SEED, stratify=cy)
    b_g0 = m.calibrate_beta(cp[_ci], cy[_ci], 0.0, 0.95)
    table5.append({"row": "gamma=0 (exact ROCP)",
                   "test": test_metrics(m, tp, ty, b_g0, 0.0, loss),
                   "beta": b_g0, "gamma": 0.0})

    # Rows 8/9: sparse vs dense grid (family size M)
    for label, gs, tgs in [
        ("Sparse grid (M~50)", np.r_[0.0, np.geomspace(1e-4, 1e6, 4)], [0.95, 0.99]),
        ("Dense grid (M~200)", np.r_[0.0, np.geomspace(1e-4, 1e6, 8)], [0.95, 0.975, 0.99]),
    ]:
        r = run_variant(m, cp[construct], cy[construct], cp[cov], cy[cov],
                        cp[safe], cy[safe], gs, tgs, 0.95, groups)
        table5.append({"row": label, **r,
                       "test": test_metrics(m, tp, ty, r["beta"], r["gamma"], loss)})

    # ---- Coverage-target (alpha) sensitivity ----
    # The beta candidate grid uses the SAME additive coverage margins as the
    # main protocol ({target, target+0.025, target+0.04}); for alpha=0.05 this
    # reproduces the exact main-protocol grid {0.95, 0.975, 0.99}.
    table6 = []
    for alpha in [0.10, 0.05, 0.01]:
        target = 1.0 - alpha
        tgts = [target, min(target + 0.025, 1.0), min(target + 0.04, 1.0)]
        r = run_variant(m, cp[construct], cy[construct], cp[cov], cy[cov],
                        cp[safe], cy[safe], GAMMAS_FULL, tgts, target, groups)
        table6.append({"alpha": alpha, "target": target, **r,
                       "test": test_metrics(m, tp, ty, r["beta"], r["gamma"], loss)})

    # ---- subsample : real-data certification rate vs n_H ----
    # Fix the full-protocol selected policy, then subsample severe cases from the
    # safety split (without replacement) and test whether a single-test 95% CP
    # upper bound certifies harm <= delta, as in Figure 5(a).
    full_fit = run_variant(m, cp[construct], cy[construct], cp[cov], cy[cov],
                           cp[safe], cy[safe], GAMMAS_FULL, [0.95, 0.975, 0.99], 0.95, groups)
    bsel, gsel = full_fit["beta"], full_fit["gamma"]
    sev_idx = np.flatnonzero(np.isin(cy[safe], SEVERE))
    acts = m.actions(cp[safe], bsel, gsel)
    harm_severe = np.isin(acts[sev_idx], HARMFUL)
    nH_full = len(sev_idx)
    subsample = []
    rng = np.random.default_rng(20260822)
    for nh in [50, 100, 200, nH_full]:
        rate = 0.0
        nrep = 20 if nh < nH_full else 1
        for _ in range(nrep):
            if nh >= nH_full:
                sub = np.arange(nH_full)
            else:
                sub = rng.choice(nH_full, size=nh, replace=False)
            k = int(harm_severe[sub].sum())
            upper = beta_dist.ppf(0.95, k + 1, nh - k) if k < nh else 1.0
            rate += float(upper <= DELTA)
        subsample.append({"n_H": int(nh), "n_repeats": nrep,
                          "certification_rate": float(rate / nrep)})

    # ---- ISIC diagnostic with the subgroup multiplicity factor removed ----
    ip, iy = load(ISIC_OD, "calibration")
    im = CertifiableHCROCP(FiniteDecisionProblem(
        np.asarray([[10, 4, 1, 0], [0, 1, 3, 6], [7, 2, 0, 4], [5, 1, 2, 4],
                    [0, 1, 3, 6], [0, 1, 3, 6], [1, 1, 3, 6], [9, 3, 1, 0]], float),
        alpha=0.05), [0, 7], [0])
    ic, icov, isaf = standard_splits(iy)
    igroups = automatic_groups([0, 7], iy[isaf], 0.05)  # delta=5% for the SCC demo
    igs = np.r_[0.0, np.geomspace(1e-4, 1e6, 41)]
    ibetas = build_betas(im, ip[ic], iy[ic], igs, [0.95, 0.975, 0.99])
    # corrected (Bonferroni) at delta=5%
    fit_corrected = im.fit_certifiable(ip[icov], iy[icov], ip[isaf], iy[isaf],
                                       ibetas, igs, 0.95, igroups,
                                       np.asarray([[10, 4, 1, 0], [0, 1, 3, 6], [7, 2, 0, 4], [5, 1, 2, 4],
                                                   [0, 1, 3, 6], [0, 1, 3, 6], [1, 1, 3, 6], [9, 3, 1, 0]], float),
                                       0.95)
    # Candidate-only correction at delta=5%.
    Gi = 1 + len(igroups)
    fit_nobonf = im.fit_certifiable(ip[icov], iy[icov], ip[isaf], iy[isaf],
                                    ibetas, igs, 0.95, igroups,
                                    np.asarray([[10, 4, 1, 0], [0, 1, 3, 6], [7, 2, 0, 4], [5, 1, 2, 4],
                                                [0, 1, 3, 6], [0, 1, 3, 6], [1, 1, 3, 6], [9, 3, 1, 0]], float),
                                    1.0 - EPS * Gi)
    def scc_cert(fit):
        """Extract the selected policy's SCC certification flag, upper bound and sample count."""
        for s in fit["subgroups"]:
            if s["name"] == "label_7":
                return s["certified"], s["upper"], s["n"]
        return None, None, None
    scc_corr = scc_cert(fit_corrected)
    scc_nobonf = scc_cert(fit_nobonf)
    isic_nobonf = {"scc_corrected": {"certified": scc_corr[0], "upper": scc_corr[1], "n": scc_corr[2]},
                   "scc_no_bonferroni": {"certified": scc_nobonf[0], "upper": scc_nobonf[1], "n": scc_nobonf[2]}}

    result = {"table5": table5, "table6": table6, "subsample": subsample,
              "isic_no_bonferroni": isic_nobonf,
              "config": {"loss": LOSS, "severe": SEVERE, "harmful": HARMFUL,
                         "delta": DELTA, "eps": EPS, "seed": SEED}}
    (OUT / "ablation_study.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    # compact console summary
    for t in table5:
        tm = t.get("test")
        print("TABLE5", t["row"], "level=", t.get("level"),
              "beta=", round(t.get("beta", 0), 3) if isinstance(t.get("beta"), float) else t.get("beta"),
              "test=", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in tm.items()} if tm else None,
              "| fixed=", t.get("fixed_beta0"), t.get("fixed_betastar"))
    for t in table6:
        print("TABLE6 alpha=", t["alpha"], "level=", t["level"],
              "test=", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in t["test"].items()})
    print("SUBSAMPLE", subsample)
    print("ISIC_NOBONF", isic_nobonf)


if __name__ == "__main__":
    main()
