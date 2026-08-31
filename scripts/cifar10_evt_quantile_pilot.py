"""Pilot (n=3) for TailAdaptiveClip21SGD2M's corrected evt_quantile mode (see
docs/TAIL_ADAPTIVE_CLIPPING_THEORY.md -- Iteration 2). The first design
(mode="moment_schedule") was measured to be a provable no-op (tau grows past the natural
diff-norm ceiling and never returns; results/tailadaptive_check, Wilcoxon p=1.000 on
every condition). A quick smoke check (T=30) showed evt_quantile settles tau ~0.13,
inside the actually-relevant range (natural diff norms ~0.04-0.16), so this mode has a
real chance of engaging. Small q_target grid before committing to a full n=10 run.
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import load_cifar10
from models import SmallCNN
from federated_experiment import run_tailadaptive_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "evt_quantile_pilot")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMA = 0.1
T = 80
N_SEEDS = 3
Q_TARGETS = [0.01, 0.05, 0.2]
CONDITIONS = [
    ("clean", dict(n_byzantine=0, attack_type=None)),
    ("ipm", dict(n_byzantine=4, attack_type="ipm")),
]

train, test = load_cifar10(DATA_ROOT)


def main():
    for cond_name, cond_kwargs in CONDITIONS:
        for q in Q_TARGETS:
            for seed in range(N_SEEDS):
                tag = f"evt__{cond_name}__q{q}__seed{seed}"
                path = os.path.join(RESULTS_DIR, f"{tag}.json")
                if os.path.exists(path):
                    continue
                t0 = time.time()
                res = run_tailadaptive_experiment(
                    model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                    n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                    beta=0.1, beta_hat=0.1, gamma=GAMMA, T=T, batch_size=32, seed=seed,
                    mode="evt_quantile", q_target=q, tau_init=1.0,
                )
                elapsed = time.time() - t0
                res.update({"cond": cond_name, "q_target": q, "seed": seed})
                with open(path, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"  [{tag}] acc={res['final_test_acc']:.4f} diverged={res['diverged']} "
                      f"final_tau={res['final_tau']:.4f} final_alpha={res['final_alpha_hat']} ({elapsed:.1f}s)", flush=True)

    print("\nEVT-quantile pilot done.", flush=True)


if __name__ == "__main__":
    main()
