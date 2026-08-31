"""Extends results/adaptive_clip_check's fixed-tau BASELINE only (seeds 3-9) to n=10,
in the same directory and tag format as cifar10_adaptive_clip_check.py's own baseline
loop, so cifar10_evt_quantile_pilot.py's now-n=10 EVT-quantile ceiling cells have a
matched-n baseline to compare against.

Deliberately does NOT extend the ac21__* (adaptive-quantile ceiling) cells themselves:
that mechanism is a mechanistically-proven no-op (tau provably converges to a value
that never clips anything, docs/TAIL_ADAPTIVE_CLIPPING_THEORY.md), so its claim does
not rest on sample size and additional seeds would not change its evidentiary status --
see paper/main_v2.tex Section 4.3. Scaling it up would cost ~2 more hours of compute for
zero additional epistemic value, so we deliberately don't.
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import load_cifar10
from models import SmallCNN
from federated_experiment import run_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "adaptive_clip_check")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMA = 0.1
T = 80
N_SEEDS = 10
CONDITIONS = [
    ("clean", dict(n_byzantine=0, attack_type=None)),
    ("ipm", dict(n_byzantine=4, attack_type="ipm")),
]

train, test = load_cifar10(DATA_ROOT)


def main():
    for cond_name, cond_kwargs in CONDITIONS:
        for seed in range(N_SEEDS):
            tag = f"baseline__{cond_name}__seed{seed}"
            path = os.path.join(RESULTS_DIR, f"{tag}.json")
            if os.path.exists(path):
                continue
            t0 = time.time()
            res = run_experiment(
                model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                beta=0.1, beta_hat=0.1, gamma=GAMMA, tau=1.0, epsilon=None,
                ragg_name="trimmed_mean", T=T, batch_size=32, seed=seed, ablation=None,
            )
            elapsed = time.time() - t0
            res.update({"cond": cond_name, "seed": seed})
            with open(path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  [{tag}] acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)

    print("\nAdaptive-clip baseline scale-up done.", flush=True)


if __name__ == "__main__":
    main()
