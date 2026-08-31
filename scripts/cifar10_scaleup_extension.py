"""CIFAR-10 seed-count scale-up: C1 (n=3->10), C2 (n=2->5), C3 (n=1-2->10).

Extends the existing pilot's C1/C2/C3 CIFAR-10 tables (results/cifar10/C1_main_iid__*,
C2_dirichlet__*, C3_ablation__*) to their full target seed counts, matching the
convention already used for the MNIST Part B scale-up (scripts/mnist_partB_scaleup.py):
same hyperparameters/protocol, seed in {0, ..., n-1}, incremental per-run JSON so
progress survives a crash/interruption, re-running skips seeds already completed.

No re-tuning: BEST_GAMMA/BEST_TAU/BEST_BETA_HAT and all other config are copied
unchanged from run_cifar10_extension.py and cifar10_ablation_check.py.

C3 note: the original pilot's "clean_no_dp_T80_control" run only has seed=0. To
support a paired Wilcoxon comparison against the no_clip_no_dp+ipm ablation arm at
n=10 (Part B of this scale-up), this script also extends the control arm to n=10,
not just the ablation arm -- both arms need matching seed counts for a paired test.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import load_cifar10, partition_dirichlet
from models import SmallCNN
from federated_experiment import run_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cifar10")
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_GAMMA = 0.1
BEST_TAU = 1.0
BEST_BETA_HAT = 0.1
N_REGULAR = 20
N_BYZANTINE = 4
T_MAIN = 80
BATCH_SIZE = 32
DIRICHLET_ALPHA = 0.5

N_C1 = 10
N_C2 = 5
N_C3 = 10


def save_run(stage, name, result):
    path = os.path.join(RESULTS_DIR, f"{stage}__{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def already_done(stage, name):
    return os.path.exists(os.path.join(RESULTS_DIR, f"{stage}__{name}.json"))


def run_stage_c1(train, test):
    print(f"== Stage C1: main sweep, IID, target n={N_C1}/cell ==")
    conditions = [
        ("clean", None, 0),
        ("ipm", "ipm", N_BYZANTINE),
        ("label_flip", "label_flip", N_BYZANTINE),
    ]
    for cond_name, attack_type, n_byz in conditions:
        for epsilon in [8, 18]:
            for seed in range(N_C1):
                name = f"{cond_name}_eps{epsilon}_seed{seed}"
                if already_done("C1_main_iid", name):
                    continue
                t0 = time.time()
                res = run_experiment(
                    model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                    n_regular=N_REGULAR, n_byzantine=n_byz, attack_type=attack_type,
                    beta=0.1, beta_hat=BEST_BETA_HAT, gamma=BEST_GAMMA, tau=BEST_TAU,
                    epsilon=epsilon, ragg_name="trimmed_mean", T=T_MAIN, batch_size=BATCH_SIZE,
                    seed=seed,
                )
                elapsed = time.time() - t0
                save_run("C1_main_iid", name, res)
                print(f"  [C1] {name}: final_acc={res['final_test_acc']:.4f} "
                      f"diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)


def run_stage_c2(train, test):
    print(f"== Stage C2: Dirichlet non-IID extension (alpha={DIRICHLET_ALPHA}), target n={N_C2}/cell ==")
    conditions = [("clean", None, 0), ("ipm", "ipm", N_BYZANTINE)]
    for cond_name, attack_type, n_byz in conditions:
        for seed in range(N_C2):
            name = f"{cond_name}_eps8_seed{seed}"
            if already_done("C2_dirichlet", name):
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
            save_run("C2_dirichlet", name, res)
            print(f"  [C2] {name}: final_acc={res['final_test_acc']:.4f} "
                  f"diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)


def run_stage_c3(train, test):
    print(f"== Stage C3: isolating ablation check, target n={N_C3} per arm ==")
    for seed in range(N_C3):
        name = f"clean_no_dp_T80_control_seed{seed}"
        if already_done("C3_ablation", name):
            continue
        t0 = time.time()
        res = run_experiment(
            model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
            n_regular=N_REGULAR, n_byzantine=0, attack_type=None, epsilon=None,
            beta=0.1, beta_hat=BEST_BETA_HAT, gamma=BEST_GAMMA, tau=BEST_TAU,
            ragg_name="trimmed_mean", T=T_MAIN, batch_size=BATCH_SIZE, seed=seed,
            ablation=None,
        )
        elapsed = time.time() - t0
        save_run("C3_ablation", name, res)
        print(f"  [C3] {name}: final_acc={res['final_test_acc']:.4f} "
              f"diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)

    for seed in range(N_C3):
        name = f"no_clip_no_dp_ipm_seed{seed}"
        if already_done("C3_ablation", name):
            continue
        t0 = time.time()
        res = run_experiment(
            model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
            n_regular=N_REGULAR, n_byzantine=N_BYZANTINE, attack_type="ipm", epsilon=8,
            beta=0.1, beta_hat=BEST_BETA_HAT, gamma=BEST_GAMMA, tau=BEST_TAU,
            ragg_name="trimmed_mean", T=T_MAIN, batch_size=BATCH_SIZE, seed=seed,
            ablation="no_clip_no_dp",
        )
        elapsed = time.time() - t0
        save_run("C3_ablation", name, res)
        print(f"  [C3] {name}: final_acc={res['final_test_acc']:.4f} "
              f"diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)


def main():
    print("Loading CIFAR-10...", flush=True)
    train, test = load_cifar10(DATA_ROOT)
    t_start = time.time()
    run_stage_c1(train, test)
    run_stage_c2(train, test)
    run_stage_c3(train, test)
    print(f"\nAll stages done in {(time.time()-t_start)/3600:.2f} hours.", flush=True)


if __name__ == "__main__":
    main()
