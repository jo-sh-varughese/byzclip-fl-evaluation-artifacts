"""Full MNIST replication (clean/ipm/label_flip x eps8/eps18, n=10/cell) for
each external baseline using its OWN independently-tuned (gamma, tau) from
baseline_independent_tuning.py's TUNE_winners.json, instead of Byz-Clip21-SGD2M's
reused (gamma=0.1, tau=1.0). Saved under stage "BASE2_tuned" so it does not
overwrite the original shared-hyperparameter "BASE2_main" results -- both are
kept and reported side by side in the paper.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import load_mnist
from models import MNIST_CNN
from federated_experiment import run_baseline_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "mnist", "external_baselines")

BETA = 0.1
N_REGULAR = 20
N_BYZANTINE = 4
T_MAIN = 80
BATCH_SIZE = 32
N_SEEDS = 10


def save_run(stage, algo, name, result):
    path = os.path.join(RESULTS_DIR, f"{stage}__{algo}__{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def already_done(stage, algo, name):
    return os.path.exists(os.path.join(RESULTS_DIR, f"{stage}__{algo}__{name}.json"))


def run(train, test, winners):
    conditions = [
        ("clean", None, 0),
        ("ipm", "ipm", N_BYZANTINE),
        ("label_flip", "label_flip", N_BYZANTINE),
    ]
    for algo, cfg in winners.items():
        gamma, tau = cfg["gamma"], cfg["tau"]
        print(f"== BASE2_tuned: {algo}, gamma={gamma}, tau={tau} ==", flush=True)
        for cond_name, attack_type, n_byz in conditions:
            for epsilon in [8, 18]:
                for seed in range(N_SEEDS):
                    name = f"{cond_name}_eps{epsilon}_seed{seed}"
                    if already_done("BASE2_tuned", algo, name):
                        continue
                    t0 = time.time()
                    res = run_baseline_experiment(
                        algo_name=algo, model_ctor=MNIST_CNN, train_dataset=train, test_dataset=test,
                        n_regular=N_REGULAR, n_byzantine=n_byz, attack_type=attack_type,
                        gamma=gamma, tau=tau, epsilon=epsilon, ragg_name="trimmed_mean",
                        T=T_MAIN, batch_size=BATCH_SIZE, seed=seed, beta=BETA,
                    )
                    elapsed = time.time() - t0
                    save_run("BASE2_tuned", algo, name, res)
                    print(f"  [BASE2_tuned/{algo}] {name}: final_acc={res['final_test_acc']:.4f} "
                          f"diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)


if __name__ == "__main__":
    winners_path = os.path.join(RESULTS_DIR, "TUNE_winners.json")
    with open(winners_path) as f:
        winners = json.load(f)
    print("Loading MNIST...", flush=True)
    train, test = load_mnist(DATA_ROOT)
    t0 = time.time()
    run(train, test, winners)
    print(f"\nBASE2_tuned replication done in {(time.time()-t0)/3600:.2f} hours.", flush=True)
