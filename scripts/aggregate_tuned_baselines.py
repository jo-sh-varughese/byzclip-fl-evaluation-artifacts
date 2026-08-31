"""Aggregates results/mnist/external_baselines/BASE2_tuned__*.json (each
baseline run at its OWN independently-tuned gamma/tau, per
scripts/baseline_independent_tuning.py winners) into mean/std/n cells,
alongside paired Wilcoxon tests against Byz-Clip21-SGD2M's existing MNIST
B2_main numbers, matching the structure of aggregate_external_baselines.py.
"""
import os
import json
from scipy.stats import wilcoxon

ROOT = os.path.join(os.path.dirname(__file__), "..")
MNIST_DIR = os.path.join(ROOT, "results", "mnist", "external_baselines")
MNIST_MAIN_DIR = os.path.join(ROOT, "results", "mnist", "partB_scaleup")

ALGOS = ["byz_clip_sgd", "safe_dshb"]


def load_cell(stage, algo, template, seeds):
    vals = []
    for s in seeds:
        path = os.path.join(MNIST_DIR, f"{stage}__{algo}__{template.format(seed=s)}.json")
        if not os.path.exists(path):
            break
        with open(path) as f:
            vals.append(json.load(f)["final_test_acc"])
    return vals


def mean_std(vals):
    n = len(vals)
    if n == 0:
        return (0.0, 0.0, 0)
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / max(n - 1, 1)
    return m, var ** 0.5, n


if __name__ == "__main__":
    print("=" * 70)
    print("MNIST BASE2_tuned (independently-tuned gamma/tau per baseline), n=10/cell")
    print("=" * 70)
    cells = {}
    for algo in ALGOS:
        for cond in ["clean", "ipm", "label_flip"]:
            for eps in [8, 18]:
                vals = load_cell("BASE2_tuned", algo, f"{cond}_eps{eps}_seed{{seed}}", range(10))
                m, s, n = mean_std(vals)
                cells[(algo, cond, eps)] = (m, s, n, vals)
                print(f"  {algo:<14} {cond:<12} eps={eps} n={n:<3} mean={m:.4f} std={s:.4f}")

    print("\n" + "=" * 70)
    print("PAIRED WILCOXON: Byz-Clip21-SGD2M vs tuned external baselines (matched seed)")
    print("=" * 70)
    for algo in ALGOS:
        for cond in ["clean", "ipm", "label_flip"]:
            for eps in [8, 18]:
                a_vals = []
                ok = True
                for s in range(10):
                    pa = os.path.join(MNIST_MAIN_DIR, f"B2_main__{cond}_eps{eps}_seed{s}.json")
                    if not os.path.exists(pa):
                        ok = False
                        break
                    with open(pa) as f:
                        a_vals.append(json.load(f)["final_test_acc"])
                b_vals = cells[(algo, cond, eps)][3]
                if not ok or len(b_vals) != 10:
                    continue
                try:
                    stat, p = wilcoxon(a_vals, b_vals)
                except ValueError:
                    p = float("nan")
                m_a, _, _ = mean_std(a_vals)
                m_b = cells[(algo, cond, eps)][0]
                sign = "SGD2M higher" if m_a > m_b else "baseline higher"
                print(f"  {cond:<12} eps={eps} vs {algo:<14}: p={p:.4f} ({sign}, {m_a:.4f} vs {m_b:.4f})")

    print("\nDone.")
