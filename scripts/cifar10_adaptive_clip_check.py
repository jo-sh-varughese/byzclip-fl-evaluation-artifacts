"""Pilot: does AdaptiveClip21SGD2M (src/byz_clip21_sgd2m_adaptive.py) recover CIFAR-10
accuracy over the fixed-tau baseline, holding architecture (plain SmallCNN, so any effect
is attributable to the algorithm, not to normalization), gamma (0.1, the paper's
headline), T (80, matches C3_ablation's clean_no_dp_T80_control), and seed count (3,
pilot scale, matches results/confound_check/) all fixed?

Two conditions per tau_k:
  - clean: n_byzantine=0, attack_type=None (compares directly against the 17.1%+-3.3%
    fixed-tau control already in the paper, and against results/confound_check's own
    18.7% mean at gamma=0.1 -- this run's own fixed-tau baseline below should reproduce
    that number as an internal consistency check).
  - ipm: n_byzantine=4, attack_type="ipm" (the paper's actual threat model -- a fix that
    only helps the clean case is not yet evidence of a useful algorithm for this setting).

target_quantile in {0.3, 0.5, 0.7}: small grid over AC21's main free hyperparameter (what
fraction of honest clients' diffs it aims to keep unclipped each round). No DP (epsilon
not exposed by run_adaptive_experiment -- see module docstring: AC21 is not yet privacy-
accounted for a time-varying tau).
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import load_cifar10
from models import SmallCNN
from federated_experiment import run_experiment, run_adaptive_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "adaptive_clip_check")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMA = 0.1
T = 80
N_SEEDS = 3
TARGET_QUANTILES = [0.3, 0.5, 0.7]
CONDITIONS = [
    ("clean", dict(n_byzantine=0, attack_type=None)),
    ("ipm", dict(n_byzantine=4, attack_type="ipm")),
]

train, test = load_cifar10(DATA_ROOT)


def save(name, res):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)


def main():
    # Fixed-tau baseline (vanilla ByzClip21SGD2M, tau=1.0), same conditions, for a direct
    # apples-to-apples comparison run in this exact script (not just cited from elsewhere).
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
            save(tag, res)
            print(f"  [{tag}] acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)

    # AdaptiveClip21SGD2M, same conditions, small target_quantile grid.
    for cond_name, cond_kwargs in CONDITIONS:
        for q in TARGET_QUANTILES:
            for seed in range(N_SEEDS):
                tag = f"ac21__{cond_name}__q{q}__seed{seed}"
                path = os.path.join(RESULTS_DIR, f"{tag}.json")
                if os.path.exists(path):
                    continue
                t0 = time.time()
                res = run_adaptive_experiment(
                    model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                    n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                    beta=0.1, beta_hat=0.1, gamma=GAMMA, T=T, batch_size=32, seed=seed,
                    tau_init=1.0, target_quantile=q,
                )
                elapsed = time.time() - t0
                res.update({"cond": cond_name, "target_quantile": q, "seed": seed})
                save(tag, res)
                print(f"  [{tag}] acc={res['final_test_acc']:.4f} diverged={res['diverged']} "
                      f"final_tau={res['final_tau']:.3f} ({elapsed:.1f}s)", flush=True)

    print("\nAdaptive-clip pilot done.", flush=True)


if __name__ == "__main__":
    main()
