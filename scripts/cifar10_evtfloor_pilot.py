"""Pilot (n=3) for EVTFloorClip21SGD2M (src/byz_clip21_sgd2m_evtfloor.py): EVT-quantile-
derived floor, fixed ceiling tau=1.0. Smoke check (T=30, q_target=0.05) settled the floor
at ~0.17 with alpha_hat~5.0 -- smaller than the hand-tuned floor=0.8 that gave the best
n=3 signal in results/bandclip_check (clean 18.8%->23.5%). Sweeping toward smaller
q_target (which the formula's own math implies gives a LARGER floor) to see whether the
EVT-derived floor can reach a comparable range, or whether the earlier hand-tuned value
was simply larger than what this measurement-driven approach naturally produces.
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import load_cifar10
from models import SmallCNN
from federated_experiment import run_evtfloor_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "evtfloor_pilot")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMA = 0.1
TAU = 1.0
T = 80
N_SEEDS = 3
Q_TARGETS = [0.001, 0.01, 0.05]
CONDITIONS = [
    ("clean", dict(n_byzantine=0, attack_type=None)),
    ("ipm", dict(n_byzantine=4, attack_type="ipm")),
]

train, test = load_cifar10(DATA_ROOT)


def main():
    for cond_name, cond_kwargs in CONDITIONS:
        for q in Q_TARGETS:
            for seed in range(N_SEEDS):
                tag = f"evtfloor__{cond_name}__q{q}__seed{seed}"
                path = os.path.join(RESULTS_DIR, f"{tag}.json")
                if os.path.exists(path):
                    continue
                t0 = time.time()
                res = run_evtfloor_experiment(
                    model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                    n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                    beta=0.1, beta_hat=0.1, gamma=GAMMA, tau=TAU, epsilon=None,
                    ragg_name="trimmed_mean", T=T, batch_size=32, seed=seed, q_target=q,
                )
                elapsed = time.time() - t0
                res.update({"cond": cond_name, "q_target": q, "seed": seed})
                with open(path, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"  [{tag}] acc={res['final_test_acc']:.4f} diverged={res['diverged']} "
                      f"final_floor={res['final_floor']:.4f} final_alpha={res['final_alpha_hat']} ({elapsed:.1f}s)", flush=True)

    print("\nEVT-floor pilot done.", flush=True)


if __name__ == "__main__":
    main()
