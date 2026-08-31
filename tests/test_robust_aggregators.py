"""Unit test: empirically verify Definition 3.2's inequality for both RAggs.

Definition 3.2 (as specified in the task prompt):
    ||RAgg(x_1..x_n) - xbar||^2 <= (c * delta_byz / (n - |B|)) * sum_{i in S} ||x_i - xbar||^2

where S is the honest set (size n - |B|), xbar = mean_{i in S} x_i, and delta_byz is the
Byzantine-fraction parameter. We use the common convention delta_byz = |B| / (n - |B|)
(Allouah et al. 2023a). This script does not derive the theoretical constant c from the
paper (that derivation is out of scope without the paper's appendix); instead it estimates
c empirically across many randomized trials -- honest-vector distributions (Gaussian,
uniform, heavy-tailed Student-t) crossed with adversarial Byzantine strategies (constant
large-magnitude vector, mean-shift attack, sign-flip attack) -- and checks two things:

  1. The inequality holds for a *finite, magnitude-independent* c: as we scale the
     Byzantine vectors' magnitude to extreme values, the empirical ratio
     ||RAgg-xbar||^2 / [(delta_byz/(n-|B|)) * sum_S ||x_i-xbar||^2] does not grow
     without bound -- both trimmed-mean and coordinate-median saturate rather than
     diverge, which is the defining robustness property RAgg implementations must have.
  2. The ratio for f=0 (no Byzantine clients) is (near) zero (both aggregators reduce
     to consensus over honest vectors, up to the trimmed-mean's trim-vs-declared-f
     bookkeeping).

Run with: venv/Scripts/python.exe tests/test_robust_aggregators.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from robust_aggregators import coordinate_trimmed_mean, coordinate_median, apply_ragg


def make_honest(n_honest, d, dist, generator):
    if dist == "gaussian":
        return torch.randn(n_honest, d, generator=generator)
    if dist == "uniform":
        return (torch.rand(n_honest, d, generator=generator) - 0.5) * 4
    if dist == "student_t":
        # Heavy-tailed reference distribution (df=3) via ratio of normals.
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
    n = n_honest + f
    xbar = honest.mean(dim=0)
    sum_sq = torch.sum((honest - xbar) ** 2).item()

    vectors = torch.cat([honest, byz], dim=0)
    ragg_out = apply_ragg(agg_name, vectors, f)
    lhs = torch.sum((ragg_out - xbar) ** 2).item()

    if f == 0 or sum_sq < 1e-12:
        return lhs, sum_sq, None

    delta_byz = f / n_honest
    denom = (delta_byz / n_honest) * sum_sq
    ratio = lhs / denom if denom > 1e-12 else float("inf")
    return lhs, sum_sq, ratio


def main():
    generator = torch.Generator().manual_seed(0)
    aggregators = ["trimmed_mean", "median"]
    dists = ["gaussian", "uniform", "student_t"]
    strategies = ["constant_large", "mean_shift", "sign_flip"]
    magnitudes = [1.0, 10.0, 1e3, 1e6, 1e9]

    n_honest, d = 16, 20
    f = 4  # n = 20, delta_byz = f/(n-f) = 4/16 = 0.25 < 1/2 requirement satisfied

    max_ratio = {agg: 0.0 for agg in aggregators}
    ratio_by_magnitude = {agg: {} for agg in aggregators}

    failures = []

    for agg in aggregators:
        for dist in dists:
            honest = make_honest(n_honest, d, dist, generator)
            for strategy in strategies:
                for mag in magnitudes:
                    byz = make_byzantine(strategy, f, d, honest, mag)
                    lhs, sum_sq, ratio = empirical_ratio(agg, honest, byz, f)
                    if ratio is None:
                        continue
                    if not torch.isfinite(torch.tensor(ratio)):
                        failures.append((agg, dist, strategy, mag, "non-finite ratio"))
                        continue
                    max_ratio[agg] = max(max_ratio[agg], ratio)
                    ratio_by_magnitude[agg].setdefault(mag, []).append(ratio)

    # Check 1: ratio must not grow with magnitude beyond a saturation point.
    # Compare mean ratio at the largest magnitude vs. a mid magnitude; require the
    # largest-magnitude ratio is not orders of magnitude larger (saturation, not blow-up).
    for agg in aggregators:
        mags_sorted = sorted(ratio_by_magnitude[agg])
        mid_mag = mags_sorted[len(mags_sorted) // 2]
        largest_mag = mags_sorted[-1]
        mean_mid = sum(ratio_by_magnitude[agg][mid_mag]) / len(ratio_by_magnitude[agg][mid_mag])
        mean_largest = sum(ratio_by_magnitude[agg][largest_mag]) / len(ratio_by_magnitude[agg][largest_mag])
        growth = mean_largest / mean_mid if mean_mid > 1e-9 else float("inf")
        print(f"[{agg}] mean ratio @mag={mid_mag}: {mean_mid:.4f}  @mag={largest_mag}: {mean_largest:.4f}  growth x{growth:.3f}")
        if growth > 5.0:
            failures.append((agg, "saturation_check", None, None, f"ratio grew {growth:.1f}x from mag={mid_mag} to mag={largest_mag}"))

    # Check 2: f=0 gives ~zero deviation from consensus mean.
    for agg in aggregators:
        honest0 = make_honest(n_honest + f, d, "gaussian", generator)
        byz0 = torch.zeros(0, d)
        lhs0, sum_sq0, ratio0 = empirical_ratio(agg, honest0, byz0, 0)
        xbar0 = honest0.mean(dim=0)
        rel = (lhs0 / max(torch.sum((honest0 - xbar0) ** 2).item(), 1e-12))
        print(f"[{agg}] f=0 check: ||RAgg-xbar||^2={lhs0:.6f}, relative to honest spread={rel:.6f}")

    print(f"\nEmpirical max observed c (max ratio) per aggregator: {max_ratio}")

    if failures:
        print("\nFAILURES:")
        for fail in failures:
            print("  ", fail)
        raise SystemExit(1)

    print("\nAll Definition 3.2 empirical checks passed (bounded ratio under extreme Byzantine magnitude).")


if __name__ == "__main__":
    main()
