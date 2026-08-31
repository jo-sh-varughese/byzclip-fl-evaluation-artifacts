"""Extends CIFAR-10 C2 (Dirichlet non-IID, Byz-Clip21-SGD2M) and CBASE2_dir
(external baselines, Dirichlet non-IID) from n=5 to n=8, clearing this
paper's own n>=8 reliability threshold. Same protocol/hyperparameters as the
existing C2_dirichlet / CBASE2_dir runs -- only extends seed range.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import load_cifar10, partition_dirichlet
from models import SmallCNN
from federated_experiment import run_experiment, run_baseline_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
CIFAR_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cifar10")
BASELINE_DIR = os.path.join(CIFAR_DIR, "external_baselines")

BEST_GAMMA, BEST_TAU, BEST_BETA_HAT = 0.1, 1.0, 0.1
N_REGULAR, N_BYZANTINE = 20, 4
T_MAIN, BATCH_SIZE, DIRICHLET_ALPHA = 80, 32, 0.5
N_C2_NEW = 8
ALGOS = ["byz_clip_sgd", "safe_dshb"]


def run_main(train, test):
    for cond_name, attack_type, n_byz in [("clean", None, 0), ("ipm", "ipm", N_BYZANTINE)]:
        for seed in range(N_C2_NEW):
            name = f"{cond_name}_eps8_seed{seed}"
            path = os.path.join(CIFAR_DIR, f"C2_dirichlet__{name}.json")
            if os.path.exists(path):
                continue
            t0 = time.time()
            res = run_experiment(
                model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                n_regular=N_REGULAR, n_byzantine=n_byz, attack_type=attack_type,
                beta=0.1, beta_hat=BEST_BETA_HAT, gamma=BEST_GAMMA, tau=BEST_TAU, epsilon=8,
                ragg_name="trimmed_mean", T=T_MAIN, batch_size=BATCH_SIZE, seed=seed,
                partition_fn=partition_dirichlet, partition_kwargs={"alpha": DIRICHLET_ALPHA},
            )
            elapsed = time.time() - t0
            with open(path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  [C2/main] {name}: final_acc={res['final_test_acc']:.4f} "
                  f"diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)


def run_baselines(train, test):
    for algo in ALGOS:
        for cond_name, attack_type, n_byz in [("clean", None, 0), ("ipm", "ipm", N_BYZANTINE)]:
            for seed in range(N_C2_NEW):
                name = f"{cond_name}_eps8_seed{seed}"
                path = os.path.join(BASELINE_DIR, f"CBASE2_dir__{algo}__{name}.json")
                if os.path.exists(path):
                    continue
                t0 = time.time()
                res = run_baseline_experiment(
                    algo_name=algo, model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                    n_regular=N_REGULAR, n_byzantine=n_byz, attack_type=attack_type,
                    gamma=BEST_GAMMA, tau=BEST_TAU, epsilon=8, ragg_name="trimmed_mean",
                    T=T_MAIN, batch_size=BATCH_SIZE, seed=seed, beta=0.1,
                    partition_fn=partition_dirichlet, partition_kwargs={"alpha": DIRICHLET_ALPHA},
                )
                elapsed = time.time() - t0
                with open(path, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"  [CBASE2_dir/{algo}] {name}: final_acc={res['final_test_acc']:.4f} "
                      f"diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)


def main():
    print("Loading CIFAR-10...", flush=True)
    train, test = load_cifar10(DATA_ROOT)
    run_main(train, test)
    run_baselines(train, test)
    print("\nC2 seed extension done.", flush=True)


if __name__ == "__main__":
    main()
