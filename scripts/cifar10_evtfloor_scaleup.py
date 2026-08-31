"""n=10 confirmatory test for EVTFloorClip21SGD2M at q_target=0.001, the config that
showed the most consistent (if modest) positive signal in the n=3 pilot
(results/evtfloor_pilot: clean +2.5pp, ipm +1.5pp over the fixed-tau baseline, floor
saturating at the ceiling -- i.e. full normalization of every honest diff to norm tau).
Reuses the 3 seeds already on disk from that pilot; only computes seeds 3-9. Baseline
reused from results/tailadaptive_check (identical config: SmallCNN, gamma=0.1, tau=1.0,
T=80, clean/ipm, n=10) instead of re-running.
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
from federated_experiment import run_evtfloor_experiment, run_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "evtfloor_scaleup")
os.makedirs(RESULTS_DIR, exist_ok=True)

OLD_PILOT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "evtfloor_pilot")
BASELINE_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "tailadaptive_check")

GAMMA = 0.1
TAU = 1.0
Q_TARGET = 0.001
T = 80
N_SEEDS = 10
CONDITIONS = [
    ("clean", dict(n_byzantine=0, attack_type=None)),
    ("ipm", dict(n_byzantine=4, attack_type="ipm")),
]

train, test = load_cifar10(DATA_ROOT)


def main():
    for cond_name, cond_kwargs in CONDITIONS:
        base_accs, evt_accs = [], []
        for seed in range(N_SEEDS):
            b_old_path = os.path.join(BASELINE_DIR, f"baseline__{cond_name}__seed{seed}.json")
            b_new_path = os.path.join(RESULTS_DIR, f"baseline__{cond_name}__seed{seed}.json")
            if os.path.exists(b_old_path):
                with open(b_old_path) as f:
                    base_accs.append(json.load(f)["final_test_acc"])
            elif os.path.exists(b_new_path):
                with open(b_new_path) as f:
                    base_accs.append(json.load(f)["final_test_acc"])
            else:
                t0 = time.time()
                bres = run_experiment(
                    model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                    n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                    beta=0.1, beta_hat=0.1, gamma=GAMMA, tau=TAU, epsilon=None,
                    ragg_name="trimmed_mean", T=T, batch_size=32, seed=seed, ablation=None,
                )
                elapsed = time.time() - t0
                with open(b_new_path, "w") as f:
                    json.dump(bres, f, indent=2)
                print(f"  [baseline__{cond_name}__seed{seed}] acc={bres['final_test_acc']:.4f} "
                      f"diverged={bres['diverged']} ({elapsed:.1f}s)", flush=True)
                base_accs.append(bres["final_test_acc"])

            tag = f"evtfloor__{cond_name}__q{Q_TARGET}__seed{seed}"
            new_path = os.path.join(RESULTS_DIR, f"{tag}.json")
            old_path = os.path.join(OLD_PILOT_DIR, f"{tag}.json")
            if os.path.exists(new_path):
                with open(new_path) as f:
                    evt_accs.append(json.load(f)["final_test_acc"])
                continue
            if os.path.exists(old_path):
                with open(old_path) as f:
                    res = json.load(f)
                with open(new_path, "w") as f:
                    json.dump(res, f, indent=2)
                evt_accs.append(res["final_test_acc"])
                continue

            t0 = time.time()
            res = run_evtfloor_experiment(
                model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                beta=0.1, beta_hat=0.1, gamma=GAMMA, tau=TAU, epsilon=None,
                ragg_name="trimmed_mean", T=T, batch_size=32, seed=seed, q_target=Q_TARGET,
            )
            elapsed = time.time() - t0
            res.update({"cond": cond_name, "q_target": Q_TARGET, "seed": seed})
            with open(new_path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  [{tag}] acc={res['final_test_acc']:.4f} diverged={res['diverged']} "
                  f"final_floor={res['final_floor']:.4f} ({elapsed:.1f}s)", flush=True)
            evt_accs.append(res["final_test_acc"])

        base_accs, evt_accs = np.array(base_accs), np.array(evt_accs)
        stat, p = wilcoxon(evt_accs, base_accs)
        print(f"[{cond_name}] n={N_SEEDS} baseline={base_accs.mean():.4f}+-{base_accs.std():.4f}  "
              f"evtfloor(q={Q_TARGET})={evt_accs.mean():.4f}+-{evt_accs.std():.4f}  wilcoxon p={p:.6f}", flush=True)

    print("\nEVT-floor scale-up done.", flush=True)


if __name__ == "__main__":
    main()
