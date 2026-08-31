"""Diagnostic (not part of the formal pilot grid): does CIFAR-10 SmallCNN reach
non-trivial accuracy given a longer round budget than the T=60-80 used elsewhere,
or does it stay near chance regardless? Disentangles "task/budget mismatch" from
"algorithm/assumption failure" before interpreting the main CIFAR-10 sweep.
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
train, test = load_cifar10(DATA_ROOT)

t0 = time.time()
res = run_experiment(
    model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
    n_regular=20, n_byzantine=0, attack_type=None,
    beta=0.1, beta_hat=0.1, gamma=0.1, tau=1.0, epsilon=None,
    ragg_name="trimmed_mean", T=300, batch_size=32, seed=0, eval_every=30,
)
elapsed = time.time() - t0
print("T=300 clean CIFAR-10 SmallCNN, gamma=0.1, tau=1.0, no DP, no Byzantine")
print("final_acc:", res["final_test_acc"], "diverged:", res["diverged"], f"({elapsed:.1f}s)")
print("trace:", res["accuracy_trace"])

out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "cifar10")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "extended_clean_check_T300.json"), "w") as f:
    json.dump(res, f, indent=2)
