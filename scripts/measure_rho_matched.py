"""Re-measures trimmed-mean's empirical robustness coefficient c AT THIS STUDY'S OWN
delta_byz values (n_honest=20, f in {2,4} -> delta_byz=f/n_honest in {0.1, 0.2}), rather
than reusing the existing tests/test_robust_aggregators.py measurement at n_honest=16,
f=4 (delta_byz=0.25). Addresses the code-review finding that the bias-floor study's
scaling-check x-axis depended on an untested delta_byz=0.25 -> {0.1,0.2} transfer
assumption for its plug-in rho.

Same randomized-trial protocol as tests/test_robust_aggregators.py (Gaussian/uniform/
Student-t honest distributions x constant-large/mean-shift/sign-flip Byzantine strategies
x 5 magnitudes), just re-parametrized to n_honest=20 and this study's own f values.
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from robust_aggregators import apply_ragg

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "bias_floor")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_PATH = os.path.join(RESULTS_DIR, "rho_matched.json")


def make_honest(n_honest, d, dist, generator):
    if dist == "gaussian":
        return torch.randn(n_honest, d, generator=generator)
    if dist == "uniform":
        return (torch.rand(n_honest, d, generator=generator) - 0.5) * 4
    if dist == "student_t":
        z = torch.randn(n_honest, d, generator=generator)
        chi2 = torch.sum(torch.randn(n_honest, 3, generator=generator) ** 2, dim=1, keepdim=True)
        return z / torch.sqrt(chi2 / 3)
    raise ValueError(dist)


def make_byzantine(strategy, f, d, honest, magnitude):
    if f == 0:
        return torch.zeros(0, d)
    if strategy == "constant_large":
        return torch.full((f, d), float(magnitude))
    if strategy == "mean_shift":
        mean_honest = honest.mean(dim=0)
        return mean_honest.unsqueeze(0).repeat(f, 1) + magnitude
    if strategy == "sign_flip":
        return -magnitude * honest[:f]
    raise ValueError(strategy)


def empirical_ratio(agg_name, honest, byz, f):
    n_honest = honest.shape[0]
    xbar = honest.mean(dim=0)
    sum_sq = torch.sum((honest - xbar) ** 2).item()
    vectors = torch.cat([honest, byz], dim=0)
    ragg_out = apply_ragg(agg_name, vectors, f)
    lhs = torch.sum((ragg_out - xbar) ** 2).item()
    if f == 0 or sum_sq < 1e-12:
        return None
    delta_byz = f / n_honest
    denom = (delta_byz / n_honest) * sum_sq
    return lhs / denom if denom > 1e-12 else float("inf")


def measure_c(n_honest, f, d=20):
    generator = torch.Generator().manual_seed(0)
    dists = ["gaussian", "uniform", "student_t"]
    strategies = ["constant_large", "mean_shift", "sign_flip"]
    magnitudes = [1.0, 10.0, 1e3, 1e6, 1e9]
    max_ratio = 0.0
    for dist in dists:
        honest = make_honest(n_honest, d, dist, generator)
        for strategy in strategies:
            for mag in magnitudes:
                byz = make_byzantine(strategy, f, d, honest, mag)
                ratio = empirical_ratio("trimmed_mean", honest, byz, f)
                if ratio is not None and torch.isfinite(torch.tensor(ratio)):
                    max_ratio = max(max_ratio, ratio)
    return max_ratio


def main():
    n_honest = 20
    results = {}
    for f, label in [(2, "delta_0.1"), (4, "delta_0.2")]:
        c = measure_c(n_honest, f)
        delta_byz = f / n_honest
        results[label] = {"n_honest": n_honest, "f": f, "delta_byz": delta_byz, "c_empirical": c}
        print(f"delta_byz={delta_byz}: c_empirical={c:.4f}", flush=True)

    with open(OUT_PATH, "w") as f_out:
        json.dump(results, f_out, indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
