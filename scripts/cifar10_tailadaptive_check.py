"""Does TailAdaptiveClip21SGD2M (src/byz_clip21_sgd2m_tailadaptive.py -- the tail-index-
calibrated clipping schedule, see docs/TAIL_ADAPTIVE_CLIPPING_THEORY.md for exactly what
is proven vs. conjectured) improve CIFAR-10 accuracy over the fixed-tau baseline, on RAW
PIXELS (the algorithm's actual intended setting -- not the pretrained-backbone shortcut
used in cifar10_pretrained_head_check*.py, which is a separate, non-novel finding)?

Same architecture (SmallCNN), gamma (0.1), T (80), n_regular (20) as every other raw-
pixel pilot this session, so results are directly comparable to results/adaptive_clip_check
(baseline: clean 17.1%+-3.2%, ipm 16.3%+-2.3%, both n=10) and results/bandclip_scaleup
(floor-clip alone: clean 19.3%+-3.7% p=0.105, ipm 18.0%+-3.4% p=0.064, both n=10, neither
significant). Going straight to n=10 with a paired Wilcoxon test.
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
from federated_experiment import run_experiment, run_tailadaptive_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "tailadaptive_check")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMA = 0.1
TAU = 1.0
T = 80
N_SEEDS = 10
CONDITIONS = [
    ("clean", dict(n_byzantine=0, attack_type=None)),
    ("ipm", dict(n_byzantine=4, attack_type="ipm")),
]

train, test = load_cifar10(DATA_ROOT)


def main():
    for cond_name, cond_kwargs in CONDITIONS:
        base_accs, tail_accs = [], []
        for seed in range(N_SEEDS):
            b_tag = f"baseline__{cond_name}__seed{seed}"
            b_path = os.path.join(RESULTS_DIR, f"{b_tag}.json")
            if not os.path.exists(b_path):
                t0 = time.time()
                res = run_experiment(
                    model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                    n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                    beta=0.1, beta_hat=0.1, gamma=GAMMA, tau=TAU, epsilon=None,
                    ragg_name="trimmed_mean", T=T, batch_size=32, seed=seed, ablation=None,
                )
                elapsed = time.time() - t0
                res.update({"cond": cond_name, "seed": seed})
                with open(b_path, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"  [{b_tag}] acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)
            with open(b_path) as f:
                base_accs.append(json.load(f)["final_test_acc"])

            t_tag = f"tailadaptive__{cond_name}__seed{seed}"
            t_path = os.path.join(RESULTS_DIR, f"{t_tag}.json")
            if not os.path.exists(t_path):
                t0 = time.time()
                res2 = run_tailadaptive_experiment(
                    model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                    n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                    beta=0.1, beta_hat=0.1, gamma=GAMMA, T=T, batch_size=32, seed=seed,
                    tau_init=TAU, n_byz_assumed=(cond_kwargs["n_byzantine"] if cond_kwargs["attack_type"] == "label_flip" else 0),
                )
                elapsed = time.time() - t0
                res2.update({"cond": cond_name, "seed": seed})
                with open(t_path, "w") as f:
                    json.dump(res2, f, indent=2)
                print(f"  [{t_tag}] acc={res2['final_test_acc']:.4f} diverged={res2['diverged']} "
                      f"final_tau={res2['final_tau']:.4f} final_alpha={res2['final_alpha_hat']} ({elapsed:.1f}s)", flush=True)
            with open(t_path) as f:
                tail_accs.append(json.load(f)["final_test_acc"])

        base_accs, tail_accs = np.array(base_accs), np.array(tail_accs)
        stat, p = wilcoxon(tail_accs, base_accs)
        print(f"[{cond_name}] n={N_SEEDS} baseline={base_accs.mean():.4f}+-{base_accs.std():.4f}  "
              f"tailadaptive={tail_accs.mean():.4f}+-{tail_accs.std():.4f}  wilcoxon p={p:.6f}", flush=True)

    print("\nTail-adaptive pilot done.", flush=True)


if __name__ == "__main__":
    main()
