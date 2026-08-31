"""External-baseline replication: Byz-Clip-SGD (Algorithm 3) and Safe-DSHB (Algorithm 4)
of Islamov, Malinovsky, Gaponov, Lucchi, Richtarik, Gorbunov, arXiv:2603.23472, Appendix
F.1 (p.67-68), run under the identical MNIST/CIFAR-10 protocol already used for
Byz-Clip21-SGD2M in this project (scripts/mnist_partB_scaleup.py,
scripts/cifar10_scaleup_extension.py): same clients, attacks, epsilon grid, seed counts,
gamma=0.1, tau=1.0.

Hyperparameter-grid note: the source paper independently tunes learning rate and
clipping threshold per algorithm over its own grid (Sec. 6). Re-deriving a fresh
tuning grid for two more algorithms was out of scope for this replication; instead we
reuse this project's already-established gamma=0.1, tau=1.0 (tuned for
Byz-Clip21-SGD2M) for both baselines too, giving an apples-to-apples comparison under
one fixed hyperparameter setting rather than three independently-tuned ones. This is a
disclosed deviation from the source paper's per-algorithm tuning protocol, not a hidden
one -- see the paper's Limitations section.

For Safe-DSHB, beta=0.1, matching the source paper's Sec. 6 ("For Byz-Clip21-SGD2M and
Safe-DSHB, we fix the local momentum parameter beta=0.1").

Stages (mirroring the existing Byz-Clip21-SGD2M tables so results are directly
comparable row-for-row):
  MNIST:
    BASE2_main   -- mirrors B2_main: CNN, clean/ipm/label_flip x eps{8,18}, n=10
    BASE5_repl   -- mirrors B5_ablation's *slot* (this is what replaces the
                    "no_momentum"/"no_clip_no_dp" placeholder-ablation comparison):
                    CNN, ipm attack, eps=8, n=10
    BASE3_mlp    -- mirrors B3_mlp: MLP, clean/ipm, eps=8, n=5
  CIFAR-10:
    CBASE1_main  -- mirrors C1_main_iid: clean/ipm/label_flip x eps{8,18}, n=10
    CBASE2_dir   -- mirrors C2_dirichlet: clean/ipm, eps=8, n=5
    CBASE3_repl  -- mirrors C3_ablation's *slot*: ipm attack, eps=8, n=10

Each run is saved incrementally per-seed so progress survives interruption; reruns
skip seeds already completed.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import load_mnist, load_cifar10, partition_dirichlet
from models import MNIST_CNN, MNIST_MLP, SmallCNN
from federated_experiment import run_baseline_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
MNIST_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "mnist", "external_baselines")
CIFAR_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cifar10", "external_baselines")
os.makedirs(MNIST_RESULTS_DIR, exist_ok=True)
os.makedirs(CIFAR_RESULTS_DIR, exist_ok=True)

GAMMA = 0.1
TAU = 1.0
BETA = 0.1
N_REGULAR = 20
N_BYZANTINE = 4
T_MAIN = 80
BATCH_SIZE_CNN = 32
BATCH_SIZE_MLP = 64
DIRICHLET_ALPHA = 0.5

ALGOS = ["byz_clip_sgd", "safe_dshb"]

N_B2, N_B5, N_B3 = 10, 10, 5
N_C1, N_C2, N_C3 = 10, 5, 10


def save_run(results_dir, stage, algo_name, name, result):
    path = os.path.join(results_dir, f"{stage}__{algo_name}__{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def already_done(results_dir, stage, algo_name, name):
    return os.path.exists(os.path.join(results_dir, f"{stage}__{algo_name}__{name}.json"))


def run_stage_base2_main(train, test):
    print(f"== MNIST BASE2_main: CNN, n={N_B2}/cell, both baselines ==", flush=True)
    conditions = [("clean", None, 0), ("ipm", "ipm", N_BYZANTINE), ("label_flip", "label_flip", N_BYZANTINE)]
    for algo_name in ALGOS:
        for cond_name, attack_type, n_byz in conditions:
            for epsilon in [8, 18]:
                for seed in range(N_B2):
                    name = f"{cond_name}_eps{epsilon}_seed{seed}"
                    if already_done(MNIST_RESULTS_DIR, "BASE2_main", algo_name, name):
                        continue
                    t0 = time.time()
                    res = run_baseline_experiment(
                        algo_name=algo_name, model_ctor=MNIST_CNN, train_dataset=train, test_dataset=test,
                        n_regular=N_REGULAR, n_byzantine=n_byz, attack_type=attack_type,
                        gamma=GAMMA, tau=TAU, epsilon=epsilon, ragg_name="trimmed_mean",
                        T=T_MAIN, batch_size=BATCH_SIZE_CNN, seed=seed, beta=BETA,
                    )
                    elapsed = time.time() - t0
                    save_run(MNIST_RESULTS_DIR, "BASE2_main", algo_name, name, res)
                    print(f"  [BASE2/{algo_name}] {name}: final_acc={res['final_test_acc']:.4f} "
                          f"diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)


def run_stage_base5_replacement(train, test):
    print(f"== MNIST BASE5_repl: CNN, ipm, eps=8, n={N_B5}, both baselines ==", flush=True)
    for algo_name in ALGOS:
        for seed in range(N_B5):
            name = f"ipm_eps8_seed{seed}"
            if already_done(MNIST_RESULTS_DIR, "BASE5_repl", algo_name, name):
                continue
            t0 = time.time()
            res = run_baseline_experiment(
                algo_name=algo_name, model_ctor=MNIST_CNN, train_dataset=train, test_dataset=test,
                n_regular=N_REGULAR, n_byzantine=N_BYZANTINE, attack_type="ipm",
                gamma=GAMMA, tau=TAU, epsilon=8, ragg_name="trimmed_mean",
                T=T_MAIN, batch_size=BATCH_SIZE_CNN, seed=seed, beta=BETA,
            )
            elapsed = time.time() - t0
            save_run(MNIST_RESULTS_DIR, "BASE5_repl", algo_name, name, res)
            print(f"  [BASE5/{algo_name}] {name}: final_acc={res['final_test_acc']:.4f} "
                  f"diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)


def run_stage_base3_mlp(train, test):
    print(f"== MNIST BASE3_mlp: MLP, n={N_B3}, both baselines ==", flush=True)
    for algo_name in ALGOS:
        for cond_name, attack_type, n_byz in [("clean", None, 0), ("ipm", "ipm", N_BYZANTINE)]:
            for seed in range(N_B3):
                name = f"{cond_name}_eps8_seed{seed}"
                if already_done(MNIST_RESULTS_DIR, "BASE3_mlp", algo_name, name):
                    continue
                t0 = time.time()
                res = run_baseline_experiment(
                    algo_name=algo_name, model_ctor=MNIST_MLP, train_dataset=train, test_dataset=test,
                    n_regular=N_REGULAR, n_byzantine=n_byz, attack_type=attack_type,
                    gamma=GAMMA, tau=TAU, epsilon=8, ragg_name="trimmed_mean",
                    T=T_MAIN, batch_size=BATCH_SIZE_MLP, seed=seed, beta=BETA,
                )
                elapsed = time.time() - t0
                save_run(MNIST_RESULTS_DIR, "BASE3_mlp", algo_name, name, res)
                print(f"  [BASE3/{algo_name}] {name}: final_acc={res['final_test_acc']:.4f} "
                      f"diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)


def run_stage_cbase1_main(train, test):
    print(f"== CIFAR CBASE1_main: n={N_C1}/cell, both baselines ==", flush=True)
    conditions = [("clean", None, 0), ("ipm", "ipm", N_BYZANTINE), ("label_flip", "label_flip", N_BYZANTINE)]
    for algo_name in ALGOS:
        for cond_name, attack_type, n_byz in conditions:
            for epsilon in [8, 18]:
                for seed in range(N_C1):
                    name = f"{cond_name}_eps{epsilon}_seed{seed}"
                    if already_done(CIFAR_RESULTS_DIR, "CBASE1_main", algo_name, name):
                        continue
                    t0 = time.time()
                    res = run_baseline_experiment(
                        algo_name=algo_name, model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                        n_regular=N_REGULAR, n_byzantine=n_byz, attack_type=attack_type,
                        gamma=GAMMA, tau=TAU, epsilon=epsilon, ragg_name="trimmed_mean",
                        T=T_MAIN, batch_size=BATCH_SIZE_CNN, seed=seed, beta=BETA,
                    )
                    elapsed = time.time() - t0
                    save_run(CIFAR_RESULTS_DIR, "CBASE1_main", algo_name, name, res)
                    print(f"  [CBASE1/{algo_name}] {name}: final_acc={res['final_test_acc']:.4f} "
                          f"diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)


def run_stage_cbase2_dirichlet(train, test):
    print(f"== CIFAR CBASE2_dir: n={N_C2}, both baselines ==", flush=True)
    for algo_name in ALGOS:
        for cond_name, attack_type, n_byz in [("clean", None, 0), ("ipm", "ipm", N_BYZANTINE)]:
            for seed in range(N_C2):
                name = f"{cond_name}_eps8_seed{seed}"
                if already_done(CIFAR_RESULTS_DIR, "CBASE2_dir", algo_name, name):
                    continue
                t0 = time.time()
                res = run_baseline_experiment(
                    algo_name=algo_name, model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                    n_regular=N_REGULAR, n_byzantine=n_byz, attack_type=attack_type,
                    gamma=GAMMA, tau=TAU, epsilon=8, ragg_name="trimmed_mean",
                    T=T_MAIN, batch_size=BATCH_SIZE_CNN, seed=seed, beta=BETA,
                    partition_fn=partition_dirichlet, partition_kwargs={"alpha": DIRICHLET_ALPHA},
                )
                elapsed = time.time() - t0
                save_run(CIFAR_RESULTS_DIR, "CBASE2_dir", algo_name, name, res)
                print(f"  [CBASE2/{algo_name}] {name}: final_acc={res['final_test_acc']:.4f} "
                      f"diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)


def run_stage_cbase3_replacement(train, test):
    print(f"== CIFAR CBASE3_repl: ipm, eps=8, n={N_C3}, both baselines ==", flush=True)
    for algo_name in ALGOS:
        for seed in range(N_C3):
            name = f"ipm_eps8_seed{seed}"
            if already_done(CIFAR_RESULTS_DIR, "CBASE3_repl", algo_name, name):
                continue
            t0 = time.time()
            res = run_baseline_experiment(
                algo_name=algo_name, model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                n_regular=N_REGULAR, n_byzantine=N_BYZANTINE, attack_type="ipm",
                gamma=GAMMA, tau=TAU, epsilon=8, ragg_name="trimmed_mean",
                T=T_MAIN, batch_size=BATCH_SIZE_CNN, seed=seed, beta=BETA,
            )
            elapsed = time.time() - t0
            save_run(CIFAR_RESULTS_DIR, "CBASE3_repl", algo_name, name, res)
            print(f"  [CBASE3/{algo_name}] {name}: final_acc={res['final_test_acc']:.4f} "
                  f"diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)


def main():
    t_start = time.time()
    print("Loading MNIST...", flush=True)
    mnist_train, mnist_test = load_mnist(DATA_ROOT)
    run_stage_base5_replacement(mnist_train, mnist_test)
    run_stage_base2_main(mnist_train, mnist_test)
    run_stage_base3_mlp(mnist_train, mnist_test)

    print("Loading CIFAR-10...", flush=True)
    cifar_train, cifar_test = load_cifar10(DATA_ROOT)
    run_stage_cbase3_replacement(cifar_train, cifar_test)
    run_stage_cbase1_main(cifar_train, cifar_test)
    run_stage_cbase2_dirichlet(cifar_train, cifar_test)

    print(f"\nAll external-baseline stages done in {(time.time()-t_start)/3600:.2f} hours.", flush=True)


if __name__ == "__main__":
    main()
