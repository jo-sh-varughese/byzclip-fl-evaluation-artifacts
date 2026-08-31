"""Aggregates results/{mnist,cifar10}/external_baselines/*.json into mean/std/n
cells matching the structure of the existing Byz-Clip21-SGD2M tables (B2/B5/MLP
on MNIST; C1-equivalent/C3-equivalent/C2-equivalent on CIFAR-10), for both
external baselines (byz_clip_sgd, safe_dshb). Also runs paired Wilcoxon tests
against the corresponding Byz-Clip21-SGD2M numbers where a matched-seed
comparison is possible.
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from significance import paired_wilcoxon

ROOT = os.path.join(os.path.dirname(__file__), "..")
MNIST_DIR = os.path.join(ROOT, "results", "mnist", "external_baselines")
CIFAR_DIR = os.path.join(ROOT, "results", "cifar10", "external_baselines")
MNIST_MAIN_DIR = os.path.join(ROOT, "results", "mnist", "partB_scaleup")
CIFAR_MAIN_DIR = os.path.join(ROOT, "results", "cifar10")

ALGOS = ["byz_clip_sgd", "safe_dshb"]


def load_cell(results_dir, stage, algo, template, seeds):
    vals = []
    for s in seeds:
        path = os.path.join(results_dir, f"{stage}__{algo}__{template.format(seed=s)}.json")
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


def report(title, results_dir, stage, template, seeds):
    print(f"\n-- {title} --")
    for algo in ALGOS:
        vals = load_cell(results_dir, stage, algo, template, seeds)
        m, s, n = mean_std(vals)
        print(f"  {algo:<14} n={n:<3} mean={m:.4f} std={s:.4f}")


if __name__ == "__main__":
    print("=" * 70)
    print("MNIST BASE2_main (matches Table tab:mnist-b2), n=10/cell, CNN")
    print("=" * 70)
    for cond in ["clean", "ipm", "label_flip"]:
        for eps in [8, 18]:
            report(f"{cond} eps={eps}", MNIST_DIR, "BASE2_main", f"{cond}_eps{eps}_seed{{seed}}", range(10))

    print("\n" + "=" * 70)
    print("MNIST BASE5_repl (matches Table tab:mnist-b5 condition: ipm, eps=8), n=10, CNN")
    print("=" * 70)
    report("ipm eps=8", MNIST_DIR, "BASE5_repl", "ipm_eps8_seed{seed}", range(10))

    print("\n" + "=" * 70)
    print("MNIST BASE3_mlp (matches MLP secondary check), n=5, MLP")
    print("=" * 70)
    for cond in ["clean", "ipm"]:
        report(f"{cond} eps=8", MNIST_DIR, "BASE3_mlp", f"{cond}_eps8_seed{{seed}}", range(5))

    print("\n" + "=" * 70)
    print("CIFAR-10 CBASE1_main (matches Table tab:cifar-c1), n=10/cell, SmallCNN")
    print("=" * 70)
    for cond in ["clean", "ipm", "label_flip"]:
        for eps in [8, 18]:
            report(f"{cond} eps={eps}", CIFAR_DIR, "CBASE1_main", f"{cond}_eps{eps}_seed{{seed}}", range(10))

    print("\n" + "=" * 70)
    print("CIFAR-10 CBASE3_repl (matches C1 ipm eps=8 cell / C3 ablation IPM condition), n=10")
    print("=" * 70)
    report("ipm eps=8", CIFAR_DIR, "CBASE3_repl", "ipm_eps8_seed{seed}", range(10))

    print("\n" + "=" * 70)
    print("CIFAR-10 CBASE2_dir (matches C2 Dirichlet extension), n=5, alpha=0.5")
    print("=" * 70)
    for cond in ["clean", "ipm"]:
        report(f"{cond} eps=8", CIFAR_DIR, "CBASE2_dir", f"{cond}_eps8_seed{{seed}}", range(5))

    # --- Reference Byz-Clip21-SGD2M numbers for comparison ---
    print("\n" + "=" * 70)
    print("REFERENCE: Byz-Clip21-SGD2M (this paper's own algorithm)")
    print("=" * 70)
    for cond in ["clean", "ipm", "label_flip"]:
        for eps in [8, 18]:
            vals = []
            for s in range(10):
                p = os.path.join(MNIST_MAIN_DIR, f"B2_main__{cond}_eps{eps}_seed{s}.json")
                with open(p) as f:
                    vals.append(json.load(f)["final_test_acc"])
            m, s_, n = mean_std(vals)
            print(f"  MNIST B2 {cond} eps={eps}: n={n} mean={m:.4f} std={s_:.4f}")
    for cond in ["clean", "ipm", "label_flip"]:
        for eps in [8, 18]:
            vals = []
            for s in range(10):
                p = os.path.join(CIFAR_MAIN_DIR, f"C1_main_iid__{cond}_eps{eps}_seed{s}.json")
                with open(p) as f:
                    vals.append(json.load(f)["final_test_acc"])
            m, s_, n = mean_std(vals)
            print(f"  CIFAR C1 {cond} eps={eps}: n={n} mean={m:.4f} std={s_:.4f}")

    # --- Paired Wilcoxon: Byz-Clip21-SGD2M vs each baseline, matched by seed ---
    print("\n" + "=" * 70)
    print("PAIRED WILCOXON: Byz-Clip21-SGD2M vs external baselines (matched seed)")
    print("=" * 70)
    results = []
    for algo in ALGOS:
        for eps in [8, 18]:
            for cond in ["clean", "ipm", "label_flip"]:
                a_vals = []
                b_vals = []
                ok = True
                for s in range(10):
                    pa = os.path.join(MNIST_MAIN_DIR, f"B2_main__{cond}_eps{eps}_seed{s}.json")
                    pb = os.path.join(MNIST_DIR, f"BASE2_main__{algo}__{cond}_eps{eps}_seed{s}.json")
                    if not (os.path.exists(pa) and os.path.exists(pb)):
                        ok = False
                        break
                    with open(pa) as f:
                        a_vals.append(json.load(f)["final_test_acc"])
                    with open(pb) as f:
                        b_vals.append(json.load(f)["final_test_acc"])
                if not ok:
                    continue
                from scipy.stats import wilcoxon
                try:
                    stat, p = wilcoxon(a_vals, b_vals)
                except ValueError:
                    p = float("nan")
                print(f"  MNIST cond={cond:<12} eps={eps} Byz-Clip21-SGD2M vs {algo}: n=10 p={p:.4f}")

    print("\nDone. Use these numbers to populate paper/main.tex.")
