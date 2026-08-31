"""Pilot: does WarmStartClip21SGD2M (src/byz_clip21_sgd2m_warmstart.py -- fast-start
beta/beta_hat schedule decaying to the paper's target 0.1/0.1) recover CIFAR-10 accuracy
over the fixed-rate baseline, holding architecture (plain SmallCNN), gamma (0.1), tau
(1.0), T (80), and seed count (3) all identical to results/adaptive_clip_check/?

Reuses that run's baseline__clean/ipm__seed*.json files as the fixed-rate reference
instead of re-running them (same code path, same seeds, same everything -- re-running
would only burn compute for numbers already on disk).
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import load_cifar10
from models import SmallCNN
from federated_experiment import run_warmstart_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "warmstart_check")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMA = 0.1
T = 80
N_SEEDS = 3
CONDITIONS = [
    ("clean", dict(n_byzantine=0, attack_type=None)),
    ("ipm", dict(n_byzantine=4, attack_type="ipm")),
]

train, test = load_cifar10(DATA_ROOT)


def main():
    for cond_name, cond_kwargs in CONDITIONS:
        for seed in range(N_SEEDS):
            tag = f"warmstart__{cond_name}__seed{seed}"
            path = os.path.join(RESULTS_DIR, f"{tag}.json")
            if os.path.exists(path):
                continue
            t0 = time.time()
            res = run_warmstart_experiment(
                model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                beta=0.1, beta_hat=0.1, gamma=GAMMA, tau=1.0, epsilon=None,
                ragg_name="trimmed_mean", T=T, batch_size=32, seed=seed,
            )
            elapsed = time.time() - t0
            res.update({"cond": cond_name, "seed": seed})
            with open(path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  [{tag}] acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)

    print("\nWarm-start pilot done.", flush=True)


if __name__ == "__main__":
    main()
