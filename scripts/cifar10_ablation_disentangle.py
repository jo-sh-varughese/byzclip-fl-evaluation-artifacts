"""Reviewer-requested ablation: Section 5's recommended configuration stacks floor-clip
(0.8) and 5 local steps together on top of the pretrained backbone, jumping from the
untricked baseline (54.23%) to 63.12%. Floor-clip ALONE was shown (Section 3.5, raw
pixels) to have a small, non-significant effect (+1.7-2.1pp) -- so how much of this 9pp
jump is really local-steps, and how much is floor? This isolates each: backbone +
local-steps-only (floor=0, i.e. plain clip_tau ceiling, no floor) vs. backbone +
floor-only (local_steps=1, i.e. no extra local computation) vs. the full combo (already
in results/pretrained_head_check_v2), all on the same Stage-2 backbone features, T=80,
n=3 pilot.
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
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "ablation_disentangle")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMA = 0.1
TAU = 1.0
T = 80
N_SEEDS = 3
CONDITIONS = [
    ("clean", dict(n_byzantine=0, attack_type=None)),
    ("ipm", dict(n_byzantine=4, attack_type="ipm")),
]
# (name, tau_floor, local_steps)
ARMS = [
    ("localsteps_only", 0.0, 5),
    ("floor_only", 0.8, 1),
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
        for arm_name, tau_floor, local_steps in ARMS:
            for seed in range(N_SEEDS):
                tag = f"{arm_name}__{cond_name}__seed{seed}"
                path = os.path.join(RESULTS_DIR, f"{tag}.json")
                if os.path.exists(path):
                    continue
                t0 = time.time()
                res = run_localsteps_experiment(
                    model_ctor=model_ctor, train_dataset=train_ds, test_dataset=test_ds,
                    n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                    beta=0.1, beta_hat=0.1, gamma=GAMMA, tau=TAU, tau_floor=tau_floor, epsilon=None,
                    ragg_name="trimmed_mean", T=T, batch_size=32, seed=seed,
                    local_steps=local_steps, local_lr=0.01,
                )
                elapsed = time.time() - t0
                res.update({"cond": cond_name, "arm": arm_name, "seed": seed})
                with open(path, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"  [{tag}] acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)

    print("\nAblation disentangle done.", flush=True)


if __name__ == "__main__":
    main()
