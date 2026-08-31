"""Synthetic control for the H1 gamma-confound (paper Section 5.5): Hill alpha
and kurtosis move toward "lighter tail" as gamma shrinks from 0.1 to 0.02,
while extreme-quantile exceedance ratios move the OPPOSITE direction. This
script asks whether that opposite-direction pattern could be a pure
measurement-pipeline artifact of the per-(snapshot,client)-group
standardization step (src/subgaussian_analysis.py fit_and_report), rather
than a real gamma-dependent change in gradient-noise tail shape.

Method: draw synthetic noise from a FIXED, KNOWN heavy-tailed distribution
(Student-t, df=3, tail index alpha=3 exactly by construction) at several
different scales, mimicking how gradient/noise magnitude shrinks as gamma
decreases and training converges further. Feed these synthetic samples
through the IDENTICAL fit_and_report pipeline used on the real gradient data.
Because the Student-t family is scale-invariant (all statistics of interest
here are computed on STANDARDIZED scalars), a correct, artifact-free pipeline
should recover the same Hill alpha/kurtosis/exceedance-ratio values at every
scale -- any scale-dependent trend that appears here is by construction a
pipeline artifact, not a real effect, since the ground-truth tail shape is
held fixed.
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from subgaussian_analysis import fit_and_report

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "subgaussian")
os.makedirs(RESULTS_DIR, exist_ok=True)

DF_TRUE = 3  # ground-truth Student-t degrees of freedom -> fixed, known heavy tail
N_GROUPS = 60  # mimics 20 clients x 3 snapshots
DRAWS_PER_CLIENT = 100
D_SYNTH = 50  # synthetic "coordinate" dimension, small for speed; standardization is per-coordinate
SCALES = [1.0, 0.5, 0.2, 0.1, 0.05]  # mimics shrinking gradient-noise magnitude as gamma decreases
SEED = 0


def synth_group(rng, scale):
    """One (snapshot, client) group: draws_per_client x D_SYNTH iid Student-t(DF_TRUE) draws,
    scaled, mimicking one entry of collect_gradient_noise's per-group theta array."""
    draws = rng.standard_t(DF_TRUE, size=(DRAWS_PER_CLIENT, D_SYNTH)) * scale
    proxy_mean = draws.mean(axis=0)
    theta = draws - proxy_mean
    return theta


def run_for_scale(scale, seed=SEED):
    rng = np.random.default_rng(seed)
    theta_norms = []
    std_scalars = []
    for _ in range(N_GROUPS):
        theta = synth_group(rng, scale)
        norms = np.linalg.norm(theta, axis=1)
        theta_norms.extend(norms.tolist())
        coord_std = np.clip(theta.std(axis=0), 1e-12, None)
        standardized = (theta / coord_std).flatten()
        idx = rng.choice(standardized.size, size=min(2000, standardized.size), replace=False)
        std_scalars.extend(standardized[idx].tolist())

    result = {
        "theta_norms": np.array(theta_norms),
        "std_scalars": np.array(std_scalars),
        "d": D_SYNTH,
        "draws_per_client": DRAWS_PER_CLIENT,
    }
    return fit_and_report(result, label=f"synthetic_t{DF_TRUE}_scale{scale}")


def main():
    reports = []
    for scale in SCALES:
        r = run_for_scale(scale)
        print(f"scale={scale}: Hill_alpha={r['hill_tail_index_alpha']:.3f} "
              f"kurtosis={r['excess_kurtosis']:.3f} "
              f"exceedance@4sigma_vs_t={r['excess_ratio_empirical_over_t_dist_bound'][-1]:.3f}",
              flush=True)
        reports.append({"scale": scale, **r})

    out_path = os.path.join(RESULTS_DIR, "h1_gamma_confound_synthetic_control.json")
    with open(out_path, "w") as f:
        json.dump(reports, f, indent=2)

    # Verdict: ground truth is scale-invariant (Student-t family), so any monotonic
    # trend across scale in the synthetic control indicates a pipeline artifact.
    hill_vals = [r["hill_tail_index_alpha"] for r in reports]
    kurt_vals = [r["excess_kurtosis"] for r in reports]
    exceed_vals = [r["excess_ratio_empirical_over_t_dist_bound"][-1] for r in reports]

    def is_monotonic(vals):
        diffs = np.diff(vals)
        return np.all(diffs > 0) or np.all(diffs < 0)

    print("\n=== Verdict ===")
    print(f"Hill alpha monotonic across scale (ground truth alpha=3 fixed): {is_monotonic(hill_vals)}")
    print(f"Kurtosis monotonic across scale (ground truth fixed): {is_monotonic(kurt_vals)}")
    print(f"Exceedance ratio monotonic across scale (ground truth fixed): {is_monotonic(exceed_vals)}")
    print("If none are monotonic (values fluctuate without trend), the pipeline correctly recovers "
          "scale-invariance for a known-fixed tail, supporting a REAL (not artifact) gamma effect "
          "in the actual CIFAR-10 measurement. If any is cleanly monotonic here too, that specific "
          "statistic's real-data gamma-trend is at least partly a measurement artifact.")

    with open(out_path, "w") as f:
        json.dump({"reports": reports,
                    "hill_monotonic": bool(is_monotonic(hill_vals)),
                    "kurtosis_monotonic": bool(is_monotonic(kurt_vals)),
                    "exceedance_monotonic": bool(is_monotonic(exceed_vals))}, f, indent=2)


if __name__ == "__main__":
    main()
