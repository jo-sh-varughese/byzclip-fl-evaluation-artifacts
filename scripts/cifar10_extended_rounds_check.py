"""Does the Stage-2 pretrained-head combo (floor-clip + 5 local steps) keep improving
past T=80? Its own accuracy_trace (results/pretrained_head_check_v2/combo__clean__seed0)
shows steady gains through round 80 with no sign of flattening (0.5865 -> 0.5994 -> 0.6140
-> 0.6275 across the last four eval points) -- unlike every raw-pixel experiment this
session, head-only training on cached features is cheap (~35-40s for T=80), so T=300 is
close to free to test. n=5 seeds (pilot scale, cheap enough to go straight past n=3),
clean + ipm, comparing T=80 vs T=300 at the same combo config.
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
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "extended_rounds_check")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMA = 0.1
TAU = 1.0
N_SEEDS = 5
T_GRID = [80, 300]
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
        for T in T_GRID:
            for seed in range(N_SEEDS):
                tag = f"combo__{cond_name}__T{T}__seed{seed}"
                path = os.path.join(RESULTS_DIR, f"{tag}.json")
                if os.path.exists(path):
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
                with open(path, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"  [{tag}] acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)

    print("\nExtended-rounds check done.", flush=True)


if __name__ == "__main__":
    main()
