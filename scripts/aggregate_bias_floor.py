"""Stage 3: aggregate the bias-floor sweep + heterogeneity measurements into the
Delta(alpha, delta) floor estimates and check consistency against the two theoretical
predictions:

  1. Shi et al. 2025 (arXiv:2503.16337): epsilon_bzt = Omega(rho^0.5 * delta^0.5 * zeta),
     rho = aggregator's own empirically-measured robustness coefficient c, RE-MEASURED at
     this study's own delta_byz in {0.1, 0.2} (scripts/measure_rho_matched.py), replacing
     the earlier plug-in that reused a delta_byz=0.25 measurement -- a disclosed transfer
     assumption a prior review round flagged as unverified.
     Since this is a LOWER bound (Omega), we cannot test exact equality -- only that (a)
     measured bias is non-decreasing in delta at fixed alpha (monotonicity), and (b) the
     measured bias is broadly consistent with a sqrt(delta)*zeta scaling shape via a
     log-log regression slope check (expected slope ~0.5 in delta, ~1.0 in zeta if the
     bound's scaling is tight; we do NOT expect an exact match, only a positive,
     order-of-magnitude-consistent relationship, since Omega bounds are not tight rates
     in general and this is exploratory, not confirmatory).

  2. Allouah/Gaucher-family bound (arXiv:2602.03329, building on Allouah et al. 2024's
     (G,B)-heterogeneity floor): stated for STRONGLY CONVEX losses, which our CNNs are
     NOT -- so this is checked only DIRECTIONALLY/QUALITATIVELY (does bias increase in
     both delta and heterogeneity, matching the bound's f/(n-(2+B^2)f) and G-dependence
     shape), explicitly disclosed as outside this bound's proven regime, in the same
     spirit as the rest of this project's stress-tests of paper assumptions beyond their
     own validated conditions.

Statistical testing:
  - CONFIRMATORY family: paired Wilcoxon (by seed) of delta>0 vs delta=0 accuracy/loss at
    each (dataset, alpha), testing whether a real (non-noise) Byzantine bias floor exists
    at all. 2 datasets x 3 alphas x 2 delta-levels (byz=2,4) x 2 metrics = 24 tests,
    Bonferroni-corrected within this family (alpha/24).
  - EXPLORATORY family: the scaling-consistency regression (log bias vs log(sqrt(delta)*
    zeta)), reported with its own R^2/slope and NOT correction-tested for significance,
    since with only 3 alphas x 2 nonzero deltas = 6 points per dataset this is manifestly
    underpowered for a precision-fit claim and is reported as suggestive only.
"""
import os
import sys
import json
import itertools

import numpy as np
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "bias_floor")
OUT_PATH = os.path.join(RESULTS_DIR, "aggregate_report.json")

ALPHAS = [100.0, 0.5, 0.1]
BYZ_COUNTS = [0, 2, 4]
N_SEEDS = 10  # scaled up from 5 per the code-review's power critique
DATASETS = ["mnist", "cifar10"]


def load_rho_matched():
    """rho re-measured at THIS study's own delta_byz values (n_honest=20), replacing the
    earlier plug-in that reused a delta_byz=0.25 measurement for delta_byz in {0.1,0.2}."""
    path = os.path.join(RESULTS_DIR, "rho_matched.json")
    with open(path) as f:
        data = json.load(f)
    return {2: data["delta_0.1"]["c_empirical"], 4: data["delta_0.2"]["c_empirical"]}


def load_sweep_cell(dataset, alpha, n_byz):
    accs, losses = [], []
    for seed in range(N_SEEDS):
        path = os.path.join(RESULTS_DIR, f"sweep__{dataset}__alpha{alpha}__byz{n_byz}__seed{seed}.json")
        with open(path) as f:
            r = json.load(f)
        accs.append(r["final_test_acc"])
        losses.append(r["final_test_loss"])
    return np.array(accs), np.array(losses)


def load_zeta(dataset, alpha):
    path = os.path.join(RESULTS_DIR, f"zeta__{dataset}__alpha{alpha}.json")
    with open(path) as f:
        return json.load(f)["zeta_mean"]


def main():
    report = {"cells": [], "confirmatory_tests": [], "scaling_regression": {}}
    rho_by_nbyz = load_rho_matched()

    for dataset in DATASETS:
        zeta_by_alpha = {alpha: load_zeta(dataset, alpha) for alpha in ALPHAS}

        acc0_by_alpha, loss0_by_alpha = {}, {}
        for alpha in ALPHAS:
            acc0, loss0 = load_sweep_cell(dataset, alpha, 0)
            acc0_by_alpha[alpha] = acc0
            loss0_by_alpha[alpha] = loss0

        log_bias, log_pred = [], []

        for alpha in ALPHAS:
            for n_byz in [2, 4]:
                acc_d, loss_d = load_sweep_cell(dataset, alpha, n_byz)
                acc0, loss0 = acc0_by_alpha[alpha], loss0_by_alpha[alpha]
                delta_byz = n_byz / 20.0

                bias_acc = float(acc0.mean() - acc_d.mean())    # positive = accuracy drop from Byzantine presence
                bias_loss = float(loss_d.mean() - loss0.mean())  # positive = loss increase

                w_acc = stats.wilcoxon(acc0, acc_d, alternative="greater") if not np.allclose(acc0, acc_d) else None
                w_loss = stats.wilcoxon(loss_d, loss0, alternative="greater") if not np.allclose(loss_d, loss0) else None

                cell = {
                    "dataset": dataset, "alpha": alpha, "n_byz": n_byz, "delta_byz": delta_byz,
                    "zeta": zeta_by_alpha[alpha],
                    "acc_delta0_mean": float(acc0.mean()), "acc_delta0_std": float(acc0.std()),
                    "acc_delta_mean": float(acc_d.mean()), "acc_delta_std": float(acc_d.std()),
                    "loss_delta0_mean": float(loss0.mean()), "loss_delta_mean": float(loss_d.mean()),
                    "bias_acc": bias_acc, "bias_loss": bias_loss,
                    "wilcoxon_acc_p": float(w_acc.pvalue) if w_acc else None,
                    "wilcoxon_loss_p": float(w_loss.pvalue) if w_loss else None,
                }
                report["cells"].append(cell)
                if w_acc:
                    report["confirmatory_tests"].append({
                        "dataset": dataset, "alpha": alpha, "n_byz": n_byz, "metric": "acc", "p": float(w_acc.pvalue),
                    })
                if w_loss:
                    report["confirmatory_tests"].append({
                        "dataset": dataset, "alpha": alpha, "n_byz": n_byz, "metric": "loss", "p": float(w_loss.pvalue),
                    })

                pred_shape = np.sqrt(rho_by_nbyz[n_byz]) * np.sqrt(delta_byz) * zeta_by_alpha[alpha]
                if bias_loss > 0 and pred_shape > 0:
                    log_bias.append(np.log(bias_loss))
                    log_pred.append(np.log(pred_shape))

        if len(log_bias) >= 3:
            slope, intercept, r, p, se = stats.linregress(log_pred, log_bias)
            report["scaling_regression"][dataset] = {
                "n_points": len(log_bias), "slope": float(slope), "intercept": float(intercept),
                "r_value": float(r), "r_squared": float(r ** 2), "p_value": float(p),
                "note": "log(bias_loss) ~ slope*log(sqrt(c)*sqrt(delta)*zeta)+intercept; "
                        "slope near 1.0 (with positive r) is consistent with the Shi et al. "
                        "Omega(rho^0.5 delta^0.5 zeta) scaling shape; this is exploratory given n<6.",
            }

    n_conf = len(report["confirmatory_tests"])
    bonferroni_alpha = 0.05 / n_conf if n_conf else None
    report["bonferroni_correction"] = {"family_size": n_conf, "corrected_alpha": bonferroni_alpha}
    report["n_surviving_bonferroni"] = sum(
        1 for t in report["confirmatory_tests"] if bonferroni_alpha and t["p"] < bonferroni_alpha
    )

    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {OUT_PATH}")
    print(f"Confirmatory family size: {n_conf}, Bonferroni alpha: {bonferroni_alpha:.5f}")
    print(f"Tests surviving Bonferroni correction: {report['n_surviving_bonferroni']}/{n_conf}")
    for dataset, reg in report["scaling_regression"].items():
        print(f"[{dataset}] scaling regression: slope={reg['slope']:.3f} r^2={reg['r_squared']:.3f} "
              f"p={reg['p_value']:.4f} (n={reg['n_points']})")


if __name__ == "__main__":
    main()
