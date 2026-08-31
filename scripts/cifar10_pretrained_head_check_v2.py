"""Stage 2: same baseline/combo comparison as cifar10_pretrained_head_check.py, but on
the upgraded backbone's features (results/pretrained_features_v2/ -- DeepBackboneCNN,
BatchNorm, augmentation, 40 epochs; see pretrain_and_extract_features_v2.py). Stage 1
result (2-conv backbone, no augmentation, 15 epochs): clean 54.3%+-0.3%, ipm 52.6%+-0.4%,
both p=0.00195 at n=10 vs. their own pretrained-but-untricked baselines. Going straight
to n=10 seeds here since Stage 1 already established the pattern is not a small-sample
artifact.
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from scipy.stats import wilcoxon

from data import FeatureDataset
from models import SmallCNNHead
from federated_experiment import run_experiment, run_localsteps_experiment

FEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "pretrained_features_v2")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "pretrained_head_check_v2")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMA = 0.1
TAU = 1.0
T = 80
N_SEEDS = 10
CONDITIONS = [
    ("clean", dict(n_byzantine=0, attack_type=None)),
    ("ipm", dict(n_byzantine=4, attack_type="ipm")),
]

train_feats = torch.load(os.path.join(FEAT_DIR, "client_pool_features.pt"))
train_labels = torch.load(os.path.join(FEAT_DIR, "client_pool_labels.pt"))
test_feats = torch.load(os.path.join(FEAT_DIR, "test_features.pt"))
test_labels = torch.load(os.path.join(FEAT_DIR, "test_labels.pt"))
train_ds = FeatureDataset(train_feats, train_labels)
test_ds = FeatureDataset(test_feats, test_labels)
feat_dim = train_feats.shape[1]
print(f"feat_dim={feat_dim}", flush=True)


def model_ctor(num_classes=10):
    return SmallCNNHead(num_classes=num_classes, feat_dim=feat_dim)


def main():
    for cond_name, cond_kwargs in CONDITIONS:
        for seed in range(N_SEEDS):
            tag = f"baseline__{cond_name}__seed{seed}"
            path = os.path.join(RESULTS_DIR, f"{tag}.json")
            if not os.path.exists(path):
                t0 = time.time()
                res = run_experiment(
                    model_ctor=model_ctor, train_dataset=train_ds, test_dataset=test_ds,
                    n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                    beta=0.1, beta_hat=0.1, gamma=GAMMA, tau=TAU, epsilon=None,
                    ragg_name="trimmed_mean", T=T, batch_size=32, seed=seed, ablation=None,
                )
                elapsed = time.time() - t0
                res.update({"cond": cond_name, "seed": seed})
                with open(path, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"  [{tag}] acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)

            tag2 = f"combo__{cond_name}__seed{seed}"
            path2 = os.path.join(RESULTS_DIR, f"{tag2}.json")
            if not os.path.exists(path2):
                t0 = time.time()
                res2 = run_localsteps_experiment(
                    model_ctor=model_ctor, train_dataset=train_ds, test_dataset=test_ds,
                    n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                    beta=0.1, beta_hat=0.1, gamma=GAMMA, tau=TAU, tau_floor=0.8, epsilon=None,
                    ragg_name="trimmed_mean", T=T, batch_size=32, seed=seed,
                    local_steps=5, local_lr=0.01,
                )
                elapsed = time.time() - t0
                res2.update({"cond": cond_name, "seed": seed})
                with open(path2, "w") as f:
                    json.dump(res2, f, indent=2)
                print(f"  [{tag2}] acc={res2['final_test_acc']:.4f} diverged={res2['diverged']} ({elapsed:.1f}s)", flush=True)

    for cond_name, _ in CONDITIONS:
        base_accs, combo_accs = [], []
        for seed in range(N_SEEDS):
            with open(os.path.join(RESULTS_DIR, f"baseline__{cond_name}__seed{seed}.json")) as f:
                base_accs.append(json.load(f)["final_test_acc"])
            with open(os.path.join(RESULTS_DIR, f"combo__{cond_name}__seed{seed}.json")) as f:
                combo_accs.append(json.load(f)["final_test_acc"])
        base_accs, combo_accs = np.array(base_accs), np.array(combo_accs)
        stat, p = wilcoxon(combo_accs, base_accs)
        print(f"[{cond_name}] n={N_SEEDS} baseline={base_accs.mean():.4f}+-{base_accs.std():.4f}  "
              f"combo={combo_accs.mean():.4f}+-{combo_accs.std():.4f}  wilcoxon p={p:.6f}", flush=True)

    print("\nStage-2 pretrained-head pilot done.", flush=True)


if __name__ == "__main__":
    main()
