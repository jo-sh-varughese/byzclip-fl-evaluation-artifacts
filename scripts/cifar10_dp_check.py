"""Does the recommended resolution configuration (pretrained backbone Stage 2 + floor-
clip 0.8 + 5 local steps + gamma=0.1) still recover accuracy once real DP noise is
active? Every result in Section 5 of the paper uses epsilon=None (sigma_omega=0);
Byz-Clip21-SGD2M's whole point is JOINT Byzantine-robustness and DP, so this is a real,
previously-untested gap. epsilon in {8, 18} matches the source paper's own privacy-
budget grid (results/cifar10/C1_main_iid__*_eps8_*.json, ..._eps18_*.json).

T=80 for compute reasons (not the recommended T=300 -- DP noise scales with sqrt(T) for
fixed epsilon, so this is also a harder setting for DP than T=300 would be per-round,
partially compensating). n=3 pilot given time constraints; disclosed as such.
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from data import FeatureDataset
from models import SmallCNNHead
from federated_experiment import run_localsteps_experiment

FEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "pretrained_features_v2")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "dp_check")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMA = 0.1
TAU = 1.0
T = 80
N_SEEDS = 3
EPSILONS = [18, 8]
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
        for eps in EPSILONS:
            for seed in range(N_SEEDS):
                tag = f"dp__{cond_name}__eps{eps}__seed{seed}"
                path = os.path.join(RESULTS_DIR, f"{tag}.json")
                if os.path.exists(path):
                    continue
                t0 = time.time()
                res = run_localsteps_experiment(
                    model_ctor=model_ctor, train_dataset=train_ds, test_dataset=test_ds,
                    n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                    beta=0.1, beta_hat=0.1, gamma=GAMMA, tau=TAU, tau_floor=0.8, epsilon=eps,
                    ragg_name="trimmed_mean", T=T, batch_size=32, seed=seed,
                    local_steps=5, local_lr=0.01,
                )
                elapsed = time.time() - t0
                res.update({"cond": cond_name, "eps": eps, "seed": seed})
                with open(path, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"  [{tag}] acc={res['final_test_acc']:.4f} diverged={res['diverged']} "
                      f"sigma_omega={res['sigma_omega']:.4f} ({elapsed:.1f}s)", flush=True)

    print("\nDP check done.", flush=True)


if __name__ == "__main__":
    main()
