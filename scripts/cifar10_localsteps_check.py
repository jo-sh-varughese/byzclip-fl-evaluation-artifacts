"""Pilot: does adding local computation (multiple local SGD steps per communication
round, the actual FedAvg mechanism -- see run_localsteps_experiment's docstring) help
CIFAR-10 beyond what floor-and-ceiling clipping alone achieved
(results/bandclip_scaleup/: clean 17.1%->19.3% p=0.105, ipm 16.3%->18.0% p=0.064)?

Holds tau_floor=0.8 fixed (the best-performing floor from the earlier pilot) and varies
local_steps in {1, 5} -- both computed fresh through run_localsteps_experiment (not
reused from earlier scripts) so the comparison is apples-to-apples on identical code,
not conflated with the small floating-point drift documented in that function's
validation. local_lr=0.01, the one rate this session's centralized_sanity_check.py
already found stable for this exact architecture.
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import load_cifar10
from models import SmallCNN
from federated_experiment import run_localsteps_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "localsteps_check")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMA = 0.1
TAU = 1.0
TAU_FLOOR = 0.8
LOCAL_LR = 0.01
T = 80
N_SEEDS = 3
LOCAL_STEPS_GRID = [1, 5]
CONDITIONS = [
    ("clean", dict(n_byzantine=0, attack_type=None)),
    ("ipm", dict(n_byzantine=4, attack_type="ipm")),
]

train, test = load_cifar10(DATA_ROOT)


def main():
    for cond_name, cond_kwargs in CONDITIONS:
        for local_steps in LOCAL_STEPS_GRID:
            for seed in range(N_SEEDS):
                tag = f"localsteps__{cond_name}__E{local_steps}__seed{seed}"
                path = os.path.join(RESULTS_DIR, f"{tag}.json")
                if os.path.exists(path):
                    continue
                t0 = time.time()
                res = run_localsteps_experiment(
                    model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                    n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                    beta=0.1, beta_hat=0.1, gamma=GAMMA, tau=TAU, tau_floor=TAU_FLOOR, epsilon=None,
                    ragg_name="trimmed_mean", T=T, batch_size=32, seed=seed,
                    local_steps=local_steps, local_lr=LOCAL_LR,
                )
                elapsed = time.time() - t0
                res.update({"cond": cond_name, "local_steps": local_steps, "seed": seed})
                with open(path, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"  [{tag}] acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)

    print("\nLocal-steps pilot done.", flush=True)


if __name__ == "__main__":
    main()
