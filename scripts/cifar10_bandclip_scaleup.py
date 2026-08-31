"""Seed scale-up (n=10) + significance test for BandClip21SGD2M at tau_floor=0.8, the
best- or tied-best-performing floor in the n=3 pilot (results/bandclip_check/): clean
18.8%->23.5%, ipm 16.5%->19.6%. Reuses seeds 0-2 already on disk from
results/adaptive_clip_check (baseline) and results/bandclip_check (floor=0.8) instead of
recomputing; only runs the new seeds 3-9.

Paired Wilcoxon signed-rank test (scipy.stats.wilcoxon, two-sided), matching this
project's own convention (paper/main_preprint.tex, Table sigtests) for the same-seed
baseline-vs-treatment comparison.
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
from federated_experiment import run_experiment, run_bandclip_experiment

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "bandclip_scaleup")
os.makedirs(RESULTS_DIR, exist_ok=True)

OLD_BASELINE_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "adaptive_clip_check")
OLD_BANDCLIP_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "bandclip_check")

GAMMA = 0.1
TAU = 1.0
TAU_FLOOR = 0.8
T = 80
N_SEEDS = 10
CONDITIONS = [
    ("clean", dict(n_byzantine=0, attack_type=None)),
    ("ipm", dict(n_byzantine=4, attack_type="ipm")),
]

train, test = load_cifar10(DATA_ROOT)


def get_or_run(tag, old_path, run_fn):
    new_path = os.path.join(RESULTS_DIR, f"{tag}.json")
    if os.path.exists(new_path):
        with open(new_path) as f:
            return json.load(f)["final_test_acc"]
    if old_path and os.path.exists(old_path):
        with open(old_path) as f:
            res = json.load(f)
        with open(new_path, "w") as f:
            json.dump(res, f, indent=2)
        return res["final_test_acc"]
    t0 = time.time()
    res = run_fn()
    elapsed = time.time() - t0
    with open(new_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  [{tag}] acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)
    return res["final_test_acc"]


def main():
    results = {}
    for cond_name, cond_kwargs in CONDITIONS:
        baseline_accs, bandclip_accs = [], []
        for seed in range(N_SEEDS):
            b_tag = f"baseline__{cond_name}__seed{seed}"
            b_old = os.path.join(OLD_BASELINE_DIR, f"{b_tag}.json") if seed < 3 else None
            b_acc = get_or_run(b_tag, b_old, lambda: run_experiment(
                model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                beta=0.1, beta_hat=0.1, gamma=GAMMA, tau=TAU, epsilon=None,
                ragg_name="trimmed_mean", T=T, batch_size=32, seed=seed, ablation=None,
            ))
            baseline_accs.append(b_acc)

            f_tag = f"bandclip__{cond_name}__floor{TAU_FLOOR}__seed{seed}"
            f_old = os.path.join(OLD_BANDCLIP_DIR, f"{f_tag}.json") if seed < 3 else None
            f_acc = get_or_run(f_tag, f_old, lambda: run_bandclip_experiment(
                model_ctor=SmallCNN, train_dataset=train, test_dataset=test,
                n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                beta=0.1, beta_hat=0.1, gamma=GAMMA, tau=TAU, tau_floor=TAU_FLOOR, epsilon=None,
                ragg_name="trimmed_mean", T=T, batch_size=32, seed=seed,
            ))
            bandclip_accs.append(f_acc)

        baseline_accs = np.array(baseline_accs)
        bandclip_accs = np.array(bandclip_accs)
        stat, p = wilcoxon(bandclip_accs, baseline_accs)
        results[cond_name] = {
            "baseline_mean": float(baseline_accs.mean()), "baseline_std": float(baseline_accs.std()),
            "bandclip_mean": float(bandclip_accs.mean()), "bandclip_std": float(bandclip_accs.std()),
            "wilcoxon_stat": float(stat), "wilcoxon_p": float(p),
            "baseline_accs": baseline_accs.tolist(), "bandclip_accs": bandclip_accs.tolist(),
        }
        print(f"[{cond_name}] baseline={baseline_accs.mean():.4f}+-{baseline_accs.std():.4f}  "
              f"bandclip(floor={TAU_FLOOR})={bandclip_accs.mean():.4f}+-{bandclip_accs.std():.4f}  "
              f"wilcoxon p={p:.5f}", flush=True)

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nBand-clip scale-up done.", flush=True)


if __name__ == "__main__":
    main()
