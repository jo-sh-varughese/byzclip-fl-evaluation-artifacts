"""Runs the paired Wilcoxon signed-rank tests for this project's key comparative
claims (MNIST, already at full n; CIFAR-10, at whatever n is currently on disk --
re-run this script after the CIFAR-10 seed scale-up finishes to get final n=10/n=5
CIFAR-10 numbers).

Each comparison pairs two conditions run at the SAME seeds (same data partition/
model init/minibatch order), differing only in the condition under test -- exactly
the structure needed for a paired test.
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from significance import paired_wilcoxon

ROOT = os.path.join(os.path.dirname(__file__), "..")
MNIST_DIR = os.path.join(ROOT, "results", "mnist", "partB_scaleup")
MNIST_B5_DIR = os.path.join(ROOT, "results", "mnist", "partB_scaleup")
CIFAR_DIR = os.path.join(ROOT, "results", "cifar10")

OUT_PATH = os.path.join(ROOT, "results", "significance_tests.json")


def run_all():
    results = []

    # --- MNIST B2 main sweep (n=10/cell) ---
    for eps in [8, 18]:
        for other in ["ipm", "label_flip"]:
            results.append(paired_wilcoxon(
                MNIST_DIR, "B2_main",
                "clean_eps%d_seed{seed}" % eps,
                f"{other}_eps%d_seed{{seed}}" % eps,
                seeds=range(10), label_a="clean", label_b=other,
            ) | {"dataset": "MNIST", "table": f"B2 (eps={eps})"})

    # --- MNIST B5 isolating ablation (n=10), IPM active, eps=8 ---
    results.append(paired_wilcoxon(
        MNIST_B5_DIR, "B5_ablation",
        "no_clip_no_dp_seed{seed}", "no_momentum_seed{seed}",
        seeds=range(10), label_a="no_clip_no_dp", label_b="no_momentum",
    ) | {"dataset": "MNIST", "table": "B5 (ipm, eps=8)"})

    # --- CIFAR-10 C1 main sweep ---
    n_c1 = len([f for f in os.listdir(CIFAR_DIR) if f.startswith("C1_main_iid__clean_eps8_seed")])
    for eps in [8, 18]:
        for other in ["ipm", "label_flip"]:
            try:
                results.append(paired_wilcoxon(
                    CIFAR_DIR, "C1_main_iid",
                    "clean_eps%d_seed{seed}" % eps,
                    f"{other}_eps%d_seed{{seed}}" % eps,
                    seeds=range(n_c1), label_a="clean", label_b=other,
                ) | {"dataset": "CIFAR-10", "table": f"C1 (eps={eps})"})
            except FileNotFoundError as e:
                results.append({"dataset": "CIFAR-10", "table": f"C1 (eps={eps}) clean vs {other}",
                                 "error": str(e)})

    # --- CIFAR-10 C2 Dirichlet extension ---
    n_c2 = len([f for f in os.listdir(CIFAR_DIR) if f.startswith("C2_dirichlet__clean_eps8_seed")])
    try:
        results.append(paired_wilcoxon(
            CIFAR_DIR, "C2_dirichlet",
            "clean_eps8_seed{seed}", "ipm_eps8_seed{seed}",
            seeds=range(n_c2), label_a="clean", label_b="ipm",
        ) | {"dataset": "CIFAR-10", "table": "C2 (Dirichlet, eps=8)"})
    except FileNotFoundError as e:
        results.append({"dataset": "CIFAR-10", "table": "C2 (Dirichlet, eps=8)", "error": str(e)})

    # --- CIFAR-10 C3 isolating ablation ---
    n_c3 = len([f for f in os.listdir(CIFAR_DIR) if f.startswith("C3_ablation__clean_no_dp_T80_control_seed")])
    try:
        results.append(paired_wilcoxon(
            CIFAR_DIR, "C3_ablation",
            "clean_no_dp_T80_control_seed{seed}", "no_clip_no_dp_ipm_seed{seed}",
            seeds=range(n_c3), label_a="clean_control", label_b="no_clip_no_dp_ipm",
        ) | {"dataset": "CIFAR-10", "table": "C3 (ablation)"})
    except FileNotFoundError as e:
        results.append({"dataset": "CIFAR-10", "table": "C3 (ablation)", "error": str(e)})

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"{'dataset':<10} {'table':<22} {'comparison':<28} {'n':>3} {'p':>10} {'sig@.05':>8} {'underpowered':>13}")
    for r in results:
        if "error" in r:
            print(f"{r['dataset']:<10} {r['table']:<22} SKIPPED (missing data): {r['error']}")
            continue
        print(f"{r['dataset']:<10} {r['table']:<22} {r['comparison']:<28} {r['n']:>3} "
              f"{r['p_value']:>10.4f} {str(r['significant_at_0.05']):>8} {str(r['underpowered']):>13}")

    print(f"\nWritten to {OUT_PATH}")


if __name__ == "__main__":
    run_all()
