"""Paired Wilcoxon signed-rank significance testing for cross-condition comparisons.

Compares two conditions run at matching seeds (same seed -> same data partition /
model init / minibatch order, differing only in the condition under test), which is
exactly the pairing structure this project's scale-up scripts already produce.

Uses scipy.stats.wilcoxon with method="auto" (exact distribution for small n,
normal approximation for larger n -- scipy picks based on n and the presence of
ties/zeros). Reports the exact n used, since half this project's comparisons are
underpowered by design (n=5) and one MNIST secondary check would be n=5 as well;
a test run at n<8 is flagged as likely underpowered rather than hidden.
"""

import json
import os

from scipy.stats import wilcoxon

MIN_RELIABLE_N = 8  # below this, flag the result as likely underpowered rather than suppress it


def load_final_acc(results_dir, stage, name):
    path = os.path.join(results_dir, f"{stage}__{name}.json")
    with open(path) as f:
        return json.load(f)["final_test_acc"]


def paired_wilcoxon(results_dir, stage, name_template_a, name_template_b, seeds, label_a, label_b):
    """Run a paired Wilcoxon signed-rank test between two conditions across `seeds`.

    name_template_a / name_template_b are format strings taking `seed=`.
    Returns a dict with n, values, statistic, p-value, and an underpowered flag.
    Raises FileNotFoundError if any required run JSON is missing (fail loudly --
    a silently-skipped seed would misreport n).
    """
    a_vals = [load_final_acc(results_dir, stage, name_template_a.format(seed=s)) for s in seeds]
    b_vals = [load_final_acc(results_dir, stage, name_template_b.format(seed=s)) for s in seeds]
    n = len(seeds)

    diffs = [a - b for a, b in zip(a_vals, b_vals)]
    all_equal = all(d == 0 for d in diffs)

    if all_equal:
        statistic, pvalue = 0.0, 1.0
    else:
        statistic, pvalue = wilcoxon(a_vals, b_vals, method="auto")

    return {
        "comparison": f"{label_a} vs {label_b}",
        "stage": stage,
        "n": n,
        "seeds": list(seeds),
        f"{label_a}_values": a_vals,
        f"{label_b}_values": b_vals,
        f"{label_a}_mean": sum(a_vals) / n,
        f"{label_b}_mean": sum(b_vals) / n,
        "statistic": float(statistic),
        "p_value": float(pvalue),
        "significant_at_0.05": bool(pvalue < 0.05),
        "underpowered": n < MIN_RELIABLE_N,
    }
