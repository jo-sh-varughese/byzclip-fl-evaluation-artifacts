"""CIFAR-10 counterpart of the MNIST Stage B5 isolating ablation check: does removing
DP noise and clipping (while the IPM attack stays active) recover accuracy toward the
clean baseline, as it did dramatically on MNIST (0.10 -> 0.69)? Also includes a clean,
no-DP, no-Byzantine control at the SAME T=80 budget as the main sweep (the earlier
Stage-A/extended-check clean numbers used T=60/T=300, not the main sweep's T=80, so
this control fills that gap directly).
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
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cifar10")
os.makedirs(RESULTS_DIR, exist_ok=True)

train, test = load_cifar10(DATA_ROOT)

configs = [
    ("clean_no_dp_T80_control", dict(n_regular=20, n_byzantine=0, attack_type=None, epsilon=None, ablation=None)),
    ("no_clip_no_dp_ipm_seed0", dict(n_regular=20, n_byzantine=4, attack_type="ipm", epsilon=8, ablation="no_clip_no_dp")),
    ("no_clip_no_dp_ipm_seed1", dict(n_regular=20, n_byzantine=4, attack_type="ipm", epsilon=8, ablation="no_clip_no_dp")),
]

results = {}
for name, cfg in configs:
    seed = 1 if name.endswith("seed1") else 0
    t0 = time.time()
    res = run_experiment(
        model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
        beta=0.1, beta_hat=0.1, gamma=0.1, tau=1.0,
        ragg_name="trimmed_mean", T=80, batch_size=32, seed=seed,
        **cfg,
    )
    elapsed = time.time() - t0
    print(f"{name}: final_acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)")
    results[name] = res
    with open(os.path.join(RESULTS_DIR, f"C3_ablation__{name}.json"), "w") as f:
        json.dump(res, f, indent=2)

print("Done.")
