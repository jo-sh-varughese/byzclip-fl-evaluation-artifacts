"""n=10 scale-up of the momentum warm-start pilot (results/warmstart_check, n=3: clean
18.75%->19.12%, ipm 16.52%->17.96%, neither formally tested). Reuses seeds 0-2 already
on disk; computes seeds 3-9 fresh. Adds a proper paired Wilcoxon test, addressing the
reviewer's critical request to either soften n=3 "ruled out" language or scale to n=10
for this specific mechanism.
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy.stats import wilcoxon

from data import load_cifar10
from models import SmallCNN
from federated_experiment import run_experiment, run_warmstart_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "warmstart_scaleup")
os.makedirs(RESULTS_DIR, exist_ok=True)

OLD_BASELINE_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "adaptive_clip_check")
OLD_WARMSTART_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "warmstart_check")

GAMMA = 0.1
TAU = 1.0
T = 80
N_SEEDS = 10
CONDITIONS = [
    ("clean", dict(n_byzantine=0, attack_type=None)),
    ("ipm", dict(n_byzantine=4, attack_type="ipm")),
]

train, test = load_cifar10(DATA_ROOT)


def get_or_run(tag, old_dir, run_fn):
    new_path = os.path.join(RESULTS_DIR, f"{tag}.json")
    if os.path.exists(new_path):
        with open(new_path) as f:
            return json.load(f)["final_test_acc"]
    old_path = os.path.join(old_dir, f"{tag}.json") if old_dir else None
    if old_path and os.path.exists(old_path):
        with open(old_path) as f:
            res = json.load(f)
        with open(new_path, "w") as f:
            json.dump(res, f, indent=2)
        return res["final_test_acc"]
    t0 = time.time()
    res = run_fn()
    elapsed = time.time() - t0
    with open(new_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  [{tag}] acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)
    return res["final_test_acc"]


def main():
    for cond_name, cond_kwargs in CONDITIONS:
        base_accs, warm_accs = [], []
        for seed in range(N_SEEDS):
            b_tag = f"baseline__{cond_name}__seed{seed}"
            b_old = OLD_BASELINE_DIR if seed < 3 else None
            b_acc = get_or_run(b_tag, b_old, lambda: run_experiment(
                model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                beta=0.1, beta_hat=0.1, gamma=GAMMA, tau=TAU, epsilon=None,
                ragg_name="trimmed_mean", T=T, batch_size=32, seed=seed, ablation=None,
            ))
            base_accs.append(b_acc)

            w_tag = f"warmstart__{cond_name}__seed{seed}"
            w_old = OLD_WARMSTART_DIR if seed < 3 else None
            w_acc = get_or_run(w_tag, w_old, lambda: run_warmstart_experiment(
                model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                beta=0.1, beta_hat=0.1, gamma=GAMMA, tau=TAU, epsilon=None,
                ragg_name="trimmed_mean", T=T, batch_size=32, seed=seed,
            ))
            warm_accs.append(w_acc)

        base_accs, warm_accs = np.array(base_accs), np.array(warm_accs)
        stat, p = wilcoxon(warm_accs, base_accs)
        print(f"[{cond_name}] n={N_SEEDS} baseline={base_accs.mean():.4f}+-{base_accs.std():.4f}  "
              f"warmstart={warm_accs.mean():.4f}+-{warm_accs.std():.4f}  wilcoxon p={p:.6f}", flush=True)

    print("\nWarm-start scale-up done.", flush=True)


if __name__ == "__main__":
    main()
