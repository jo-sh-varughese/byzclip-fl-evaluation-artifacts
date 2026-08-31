"""n=10 scale-up of the T=300 combo result (results/extended_rounds_check, n=5: clean
72.8%+-0.1%, ipm 70.2%+-0.2%). Reuses seeds 0-4 already on disk; computes seeds 5-9.
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy.stats import wilcoxon

import torch

from data import FeatureDataset
from models import SmallCNNHead
from federated_experiment import run_localsteps_experiment

FEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "pretrained_features_v2")
OLD_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "extended_rounds_check")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "t300_scaleup")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMA = 0.1
TAU = 1.0
T = 300
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


def model_ctor(num_classes=10):
    return SmallCNNHead(num_classes=num_classes, feat_dim=feat_dim)


def main():
    for cond_name, cond_kwargs in CONDITIONS:
        accs = []
        for seed in range(N_SEEDS):
            tag = f"combo__{cond_name}__T{T}__seed{seed}"
            new_path = os.path.join(RESULTS_DIR, f"{tag}.json")
            old_path = os.path.join(OLD_DIR, f"{tag}.json")
            if os.path.exists(new_path):
                with open(new_path) as f:
                    accs.append(json.load(f)["final_test_acc"])
                continue
            if os.path.exists(old_path):
                with open(old_path) as f:
                    res = json.load(f)
                with open(new_path, "w") as f:
                    json.dump(res, f, indent=2)
                accs.append(res["final_test_acc"])
                continue
            t0 = time.time()
            res = run_localsteps_experiment(
                model_ctor=model_ctor, train_dataset=train_ds, test_dataset=test_ds,
                n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                beta=0.1, beta_hat=0.1, gamma=GAMMA, tau=TAU, tau_floor=0.8, epsilon=None,
                ragg_name="trimmed_mean", T=T, batch_size=32, seed=seed,
                local_steps=5, local_lr=0.01,
            )
            elapsed = time.time() - t0
            res.update({"cond": cond_name, "T": T, "seed": seed})
            with open(new_path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  [{tag}] acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)
            accs.append(res["final_test_acc"])

        accs = np.array(accs)
        print(f"[{cond_name}] n={N_SEEDS} T={T} combo={accs.mean():.4f}+-{accs.std():.4f}", flush=True)

    print("\nT=300 scale-up done.", flush=True)


if __name__ == "__main__":
    main()
