"""Part B: seed count scale-up for MNIST tables (AAAI scale-up).

Targets per the scale-up spec: B1 n=10, B2 n=10/cell, B3 n=5, B4 n=5, B5 n=10.
Also corrects a parameterization bug in the original pilot's Stage B1: the source
paper (Islamov et al. 2026, Section 6) fixes beta=0.1 (client momentum) identically
for Byz-Clip21-SGD2M and the Safe-DSHB baseline, and uses beta_hat=0.01 (server EF21
momentum) specifically for Byz-Clip21-SGD2M -- these are two distinct symbols, not
a contradiction. The original pilot swept `beta` at fixed beta_hat=0.1; this script
sweeps `beta_hat` (the symbol the paper's "0.01" value actually refers to) at fixed
beta=0.1, and keeps the original (mislabeled but still real) beta sweep as a
separate, clearly-labeled table rather than silently discarding it.

Each stage appends to its own manifest file incrementally so progress survives a
crash/interruption; re-running skips seeds already completed.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import load_mnist
from models import MNIST_CNN, MNIST_MLP
from federated_experiment import run_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "mnist", "partB_scaleup")
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_GAMMA = 0.1
BEST_TAU = 1.0
BEST_BETA_HAT = 0.1
N_REGULAR = 20
N_BYZANTINE = 4
T_MAIN = 80
BATCH_SIZE_CNN = 32
BATCH_SIZE_MLP = 64

N_B1 = 10
N_B2 = 10
N_B3 = 5
N_B4 = 5
N_B5 = 10


def save_run(stage, name, result):
    path = os.path.join(RESULTS_DIR, f"{stage}__{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def already_done(stage, name):
    return os.path.exists(os.path.join(RESULTS_DIR, f"{stage}__{name}.json"))


def run_stage_b1_corrected_beta_hat_sweep(train, test):
    print("== Stage B1-corrected: beta_hat 3-way comparison (clean, CNN), n=%d ==" % N_B1)
    out = []
    for beta_hat in [0.01, 0.05, 0.1]:
        for seed in range(N_B1):
            name = f"betahat{beta_hat}_seed{seed}"
            if already_done("B1c_betahat", name):
                continue
            t0 = time.time()
            res = run_experiment(
                model_ctor=MNIST_CNN, train_dataset=train, test_dataset=test,
                n_regular=N_REGULAR, n_byzantine=0, attack_type=None,
                beta=0.1, beta_hat=beta_hat, gamma=BEST_GAMMA, tau=BEST_TAU, epsilon=None,
                ragg_name="trimmed_mean", T=T_MAIN, batch_size=BATCH_SIZE_CNN, seed=seed,
            )
            elapsed = time.time() - t0
            path = save_run("B1c_betahat", name, res)
            print(f"  {name}: final_acc={res['final_test_acc']:.4f} ({elapsed:.1f}s)")
            out.append({"beta_hat": beta_hat, "seed": seed, "final_acc": res["final_test_acc"],
                        "diverged": res["diverged"], "path": path})
    return out


def run_stage_b1_original_beta_sweep(train, test):
    print("== Stage B1-original: beta 3-way comparison (clean, CNN), n=%d (kept for continuity) ==" % N_B1)
    out = []
    for beta in [0.01, 0.05, 0.1]:
        for seed in range(N_B1):
            name = f"beta{beta}_seed{seed}"
            if already_done("B1_beta", name):
                continue
            t0 = time.time()
            res = run_experiment(
                model_ctor=MNIST_CNN, train_dataset=train, test_dataset=test,
                n_regular=N_REGULAR, n_byzantine=0, attack_type=None,
                beta=beta, beta_hat=BEST_BETA_HAT, gamma=BEST_GAMMA, tau=BEST_TAU, epsilon=None,
                ragg_name="trimmed_mean", T=T_MAIN, batch_size=BATCH_SIZE_CNN, seed=seed,
            )
            elapsed = time.time() - t0
            path = save_run("B1_beta", name, res)
            print(f"  {name}: final_acc={res['final_test_acc']:.4f} ({elapsed:.1f}s)")
            out.append({"beta": beta, "seed": seed, "final_acc": res["final_test_acc"],
                        "diverged": res["diverged"], "path": path})
    return out


def run_stage_b2_main_sweep(train, test):
    print("== Stage B2: main sweep on CNN, n=%d/cell ==" % N_B2)
    out = []
    conditions = [("clean", None, 0), ("ipm", "ipm", N_BYZANTINE), ("label_flip", "label_flip", N_BYZANTINE)]
    for cond_name, attack_type, n_byz in conditions:
        for epsilon in [8, 18]:
            for seed in range(N_B2):
                name = f"{cond_name}_eps{epsilon}_seed{seed}"
                if already_done("B2_main", name):
                    continue
                t0 = time.time()
                res = run_experiment(
                    model_ctor=MNIST_CNN, train_dataset=train, test_dataset=test,
                    n_regular=N_REGULAR, n_byzantine=n_byz, attack_type=attack_type,
                    beta=0.1, beta_hat=BEST_BETA_HAT, gamma=BEST_GAMMA, tau=BEST_TAU, epsilon=epsilon,
                    ragg_name="trimmed_mean", T=T_MAIN, batch_size=BATCH_SIZE_CNN, seed=seed,
                )
                elapsed = time.time() - t0
                path = save_run("B2_main", name, res)
                print(f"  {name}: final_acc={res['final_test_acc']:.4f} ({elapsed:.1f}s)")
                out.append({"condition": cond_name, "epsilon": epsilon, "seed": seed,
                            "final_acc": res["final_test_acc"], "diverged": res["diverged"],
                            "sigma_omega": res["sigma_omega"], "path": path})
    return out


def run_stage_b3_mlp(train, test):
    print("== Stage B3: MLP secondary check, n=%d ==" % N_B3)
    out = []
    for cond_name, attack_type, n_byz in [("clean", None, 0), ("ipm", "ipm", N_BYZANTINE)]:
        for seed in range(N_B3):
            name = f"{cond_name}_eps8_seed{seed}"
            if already_done("B3_mlp", name):
                continue
            t0 = time.time()
            res = run_experiment(
                model_ctor=MNIST_MLP, train_dataset=train, test_dataset=test,
                n_regular=N_REGULAR, n_byzantine=n_byz, attack_type=attack_type,
                beta=0.1, beta_hat=BEST_BETA_HAT, gamma=BEST_GAMMA, tau=BEST_TAU, epsilon=8,
                ragg_name="trimmed_mean", T=T_MAIN, batch_size=BATCH_SIZE_MLP, seed=seed,
            )
            elapsed = time.time() - t0
            path = save_run("B3_mlp", name, res)
            print(f"  {name}: final_acc={res['final_test_acc']:.4f} ({elapsed:.1f}s)")
            out.append({"condition": cond_name, "seed": seed, "final_acc": res["final_test_acc"],
                        "diverged": res["diverged"], "path": path})
    return out


def run_stage_b4_ragg(train, test):
    print("== Stage B4: RAgg comparison, n=%d ==" % N_B4)
    out = []
    for ragg_name in ["trimmed_mean", "median"]:
        for seed in range(N_B4):
            name = f"{ragg_name}_seed{seed}"
            if already_done("B4_ragg", name):
                continue
            t0 = time.time()
            res = run_experiment(
                model_ctor=MNIST_CNN, train_dataset=train, test_dataset=test,
                n_regular=N_REGULAR, n_byzantine=N_BYZANTINE, attack_type="ipm",
                beta=0.1, beta_hat=BEST_BETA_HAT, gamma=BEST_GAMMA, tau=BEST_TAU, epsilon=8,
                ragg_name=ragg_name, T=T_MAIN, batch_size=BATCH_SIZE_CNN, seed=seed,
            )
            elapsed = time.time() - t0
            path = save_run("B4_ragg", name, res)
            print(f"  {name}: final_acc={res['final_test_acc']:.4f} ({elapsed:.1f}s)")
            out.append({"ragg_name": ragg_name, "seed": seed, "final_acc": res["final_test_acc"],
                        "diverged": res["diverged"], "path": path})
    return out


def run_stage_b5_ablations(train, test):
    print("== Stage B5: ablations (key isolating result), n=%d ==" % N_B5)
    out = []
    for ablation in ["no_momentum", "no_clip_no_dp"]:
        for seed in range(N_B5):
            name = f"{ablation}_seed{seed}"
            if already_done("B5_ablation", name):
                continue
            t0 = time.time()
            res = run_experiment(
                model_ctor=MNIST_CNN, train_dataset=train, test_dataset=test,
                n_regular=N_REGULAR, n_byzantine=N_BYZANTINE, attack_type="ipm",
                beta=0.1, beta_hat=BEST_BETA_HAT, gamma=BEST_GAMMA, tau=BEST_TAU, epsilon=8,
                ragg_name="trimmed_mean", T=T_MAIN, batch_size=BATCH_SIZE_CNN, seed=seed,
                ablation=ablation,
            )
            elapsed = time.time() - t0
            path = save_run("B5_ablation", name, res)
            print(f"  {name}: final_acc={res['final_test_acc']:.4f} ({elapsed:.1f}s)")
            out.append({"ablation": ablation, "seed": seed, "final_acc": res["final_test_acc"],
                        "diverged": res["diverged"], "path": path})
    return out


def main():
    train, test = load_mnist(DATA_ROOT)
    manifest_path = os.path.join(RESULTS_DIR, "run_manifest.json")

    manifest = {}
    # Priority order: B5 (key isolating result) and B2 (central DP-dominance table) first,
    # then B1-corrected, then B3/B4, then the original (mislabeled) B1 sweep for continuity.
    manifest["stage_b5_ablations"] = run_stage_b5_ablations(train, test)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    manifest["stage_b2_main_sweep"] = run_stage_b2_main_sweep(train, test)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    manifest["stage_b1_corrected_betahat_sweep"] = run_stage_b1_corrected_beta_hat_sweep(train, test)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    manifest["stage_b3_mlp"] = run_stage_b3_mlp(train, test)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    manifest["stage_b4_ragg"] = run_stage_b4_ragg(train, test)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    manifest["stage_b1_original_beta_sweep"] = run_stage_b1_original_beta_sweep(train, test)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nDone. Manifest written to", manifest_path)


if __name__ == "__main__":
    main()
