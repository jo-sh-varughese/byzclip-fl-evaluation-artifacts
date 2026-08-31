"""Does the recommended resolution configuration hold under non-IID (Dirichlet label-
skew) partitioning? Every result in the paper so far uses IID partitioning
(partition_iid); Byzantine-robust FL is a heterogeneity-sensitive literature, so this
is a real, previously-untested dimension. alpha=0.5 (moderate skew, a level used
elsewhere in this project's own earlier work) applied to the same 45,000-image client
pool used throughout Section 5. T=80 for compute reasons, n=3 pilot given time
constraints; disclosed as such. Also runs the pretrained-but-untricked baseline under
the same non-IID partition, so the comparison is a fair like-for-like (both IID-vs-non-
IID delta and combo-vs-baseline delta are visible, not conflated).
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from data import FeatureDataset, partition_dirichlet
from models import SmallCNNHead
from federated_experiment import run_experiment, run_localsteps_experiment

FEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "pretrained_features_v2")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "noniid_check")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMA = 0.1
TAU = 1.0
T = 80
N_SEEDS = 3
ALPHA = 0.5
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
        for seed in range(N_SEEDS):
            b_tag = f"noniid_baseline__{cond_name}__seed{seed}"
            b_path = os.path.join(RESULTS_DIR, f"{b_tag}.json")
            if not os.path.exists(b_path):
                t0 = time.time()
                res = run_experiment(
                    model_ctor=model_ctor, train_dataset=train_ds, test_dataset=test_ds,
                    n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                    beta=0.1, beta_hat=0.1, gamma=GAMMA, tau=TAU, epsilon=None,
                    ragg_name="trimmed_mean", T=T, batch_size=32, seed=seed, ablation=None,
                    partition_fn=partition_dirichlet, partition_kwargs={"alpha": ALPHA},
                )
                elapsed = time.time() - t0
                res.update({"cond": cond_name, "seed": seed})
                with open(b_path, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"  [{b_tag}] acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)

            c_tag = f"noniid_combo__{cond_name}__seed{seed}"
            c_path = os.path.join(RESULTS_DIR, f"{c_tag}.json")
            if not os.path.exists(c_path):
                t0 = time.time()
                res2 = run_localsteps_experiment(
                    model_ctor=model_ctor, train_dataset=train_ds, test_dataset=test_ds,
                    n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                    beta=0.1, beta_hat=0.1, gamma=GAMMA, tau=TAU, tau_floor=0.8, epsilon=None,
                    ragg_name="trimmed_mean", T=T, batch_size=32, seed=seed,
                    local_steps=5, local_lr=0.01,
                    partition_fn=partition_dirichlet, partition_kwargs={"alpha": ALPHA},
                )
                elapsed = time.time() - t0
                res2.update({"cond": cond_name, "seed": seed})
                with open(c_path, "w") as f:
                    json.dump(res2, f, indent=2)
                print(f"  [{c_tag}] acc={res2['final_test_acc']:.4f} diverged={res2['diverged']} ({elapsed:.1f}s)", flush=True)

    print("\nNon-IID check done.", flush=True)


if __name__ == "__main__":
    main()
