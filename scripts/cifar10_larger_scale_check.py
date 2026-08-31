"""Reviewer-requested check: does the headline resolution (pretrained backbone +
floor-clip + local steps, gamma=0.1, T=80) hold at double the client count? 40 clients
(n_regular=32, n_byzantine=8, keeping the same ~16.7% Byzantine fraction as the
headline's 20/4 split), same cached Stage-2 backbone features, n=3 pilot given CPU
budget -- this bounds, but does not eliminate, the CPU-pilot-scale caveat in
Limitations; a single doubled-scale check is not a full scale-generalization study.
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
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "larger_scale_check")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMA = 0.1
TAU = 1.0
T = 80
N_SEEDS = 3
CONDITIONS = [
    ("clean", dict(n_regular=40, n_byzantine=0, attack_type=None)),
    ("ipm", dict(n_regular=32, n_byzantine=8, attack_type="ipm")),
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
        for seed in range(N_SEEDS):
            tag = f"combo40__{cond_name}__seed{seed}"
            path = os.path.join(RESULTS_DIR, f"{tag}.json")
            if os.path.exists(path):
                continue
            t0 = time.time()
            res = run_localsteps_experiment(
                model_ctor=model_ctor, train_dataset=train_ds, test_dataset=test_ds,
                n_regular=cond_kwargs["n_regular"], attack_type=cond_kwargs["attack_type"],
                n_byzantine=cond_kwargs["n_byzantine"],
                beta=0.1, beta_hat=0.1, gamma=GAMMA, tau=TAU, tau_floor=0.8, epsilon=None,
                ragg_name="trimmed_mean", T=T, batch_size=32, seed=seed,
                local_steps=5, local_lr=0.01,
            )
            elapsed = time.time() - t0
            res.update({"cond": cond_name, "seed": seed, "n_clients": cond_kwargs["n_regular"] + cond_kwargs["n_byzantine"]})
            with open(path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  [{tag}] acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)

    print("\nLarger-scale (40-client) check done.", flush=True)


if __name__ == "__main__":
    main()
