"""Extends the Stage-A hyperparameter probe (hp_probe.py / hp_probe_cifar10.py,
originally single-seed) to n=3 seeds per config, both datasets, same grid,
same T=60, clean/no-DP/no-Byzantine. Closes the "Stage-A probe remains at
n=1" disclosed gap. Results saved incrementally per (dataset, config, seed).
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import load_mnist, load_cifar10
from models import MNIST_CNN, SmallCNN
from federated_experiment import run_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "stageA_seed_extend")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIGS = [
    dict(gamma=1.0, tau=1.0, beta=0.1),
    dict(gamma=0.1, tau=1.0, beta=0.1),
    dict(gamma=0.1, tau=0.1, beta=0.1),
    dict(gamma=0.01, tau=0.1, beta=0.1),
    dict(gamma=0.1, tau=0.01, beta=0.1),
    dict(gamma=1.0, tau=0.1, beta=0.1),
]
SEEDS = [1, 2]  # seed 0 already exists in results/{mnist,cifar10}/hp_probe.json


def run_dataset(name, model_ctor, train, test):
    for cfg in CONFIGS:
        for seed in SEEDS:
            tag = f"{name}_gamma{cfg['gamma']}_tau{cfg['tau']}_seed{seed}"
            path = os.path.join(RESULTS_DIR, f"{tag}.json")
            if os.path.exists(path):
                continue
            t0 = time.time()
            res = run_experiment(
                model_ctor=model_ctor, train_dataset=train, test_dataset=test,
                n_regular=20, n_byzantine=0, attack_type=None,
                beta=cfg["beta"], beta_hat=0.1, gamma=cfg["gamma"], tau=cfg["tau"], epsilon=None,
                ragg_name="trimmed_mean", T=60, batch_size=32, seed=seed, eval_every=20,
            )
            elapsed = time.time() - t0
            with open(path, "w") as f:
                json.dump({"config": cfg, "seed": seed, "final_acc": res["final_test_acc"],
                           "diverged": res["diverged"]}, f, indent=2)
            print(f"  [{tag}] final_acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)",
                  flush=True)


def main():
    print("Loading MNIST...", flush=True)
    mtrain, mtest = load_mnist(DATA_ROOT)
    run_dataset("mnist", MNIST_CNN, mtrain, mtest)

    print("Loading CIFAR-10...", flush=True)
    ctrain, ctest = load_cifar10(DATA_ROOT)
    run_dataset("cifar10", SmallCNN, ctrain, ctest)

    print("\nStage-A seed extension done.", flush=True)


if __name__ == "__main__":
    main()
