"""Pilot: does BandClip21SGD2M (src/byz_clip21_sgd2m_bandclip.py -- floor-and-ceiling
clip_band, tau_max=1.0 unchanged from the base algorithm) recover CIFAR-10 accuracy over
the fixed-tau, ceiling-only baseline?

Motivated by a quick direct measurement (this session): honest diff norms under this
exact config sit around 0.04-0.16 throughout training -- 6-25x below the tau=1.0
ceiling, which is why AC21 (adaptive ceiling) found the ceiling was never binding. A
floor in {0.3, 0.5, 0.8} sits comfortably above that natural range (so it should engage
on most rounds) and comfortably below tau_max=1.0 (so it never conflicts with the
existing ceiling / Byzantine protection).

Holds architecture (plain SmallCNN), gamma (0.1), tau (1.0), T (80), and seed count (3)
identical to results/adaptive_clip_check/ and results/warmstart_check/ for direct
comparability; reuses that run's baseline__clean/ipm__seed*.json as the reference
instead of re-running them.
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import load_cifar10
from models import SmallCNN
from federated_experiment import run_bandclip_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "bandclip_check")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMA = 0.1
TAU = 1.0
T = 80
N_SEEDS = 3
TAU_FLOORS = [0.3, 0.5, 0.8]
CONDITIONS = [
    ("clean", dict(n_byzantine=0, attack_type=None)),
    ("ipm", dict(n_byzantine=4, attack_type="ipm")),
]

train, test = load_cifar10(DATA_ROOT)


def main():
    for cond_name, cond_kwargs in CONDITIONS:
        for tau_floor in TAU_FLOORS:
            for seed in range(N_SEEDS):
                tag = f"bandclip__{cond_name}__floor{tau_floor}__seed{seed}"
                path = os.path.join(RESULTS_DIR, f"{tag}.json")
                if os.path.exists(path):
                    continue
                t0 = time.time()
                res = run_bandclip_experiment(
                    model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                    n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                    beta=0.1, beta_hat=0.1, gamma=GAMMA, tau=TAU, tau_floor=tau_floor, epsilon=None,
                    ragg_name="trimmed_mean", T=T, batch_size=32, seed=seed,
                )
                elapsed = time.time() - t0
                res.update({"cond": cond_name, "tau_floor": tau_floor, "seed": seed})
                with open(path, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"  [{tag}] acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)

    print("\nBand-clip pilot done.", flush=True)


if __name__ == "__main__":
    main()
