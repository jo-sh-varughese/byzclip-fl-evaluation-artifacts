"""Final check before locking headline numbers: combine the two best-found levers from
this session's pretrained-head track -- gamma=0.3 (peak of the T=80 gamma sweep;
results/gamma_retune_head, gamma_retune_head_ext show 0.4+ destabilizes) and T=300
(results/t300_scaleup, n=10: gamma=0.1 baseline reaches 72.9%/70.2%). Do they compound,
or does T=300 already capture most of what gamma=0.3 buys at T=80? n=10 directly (no
further pilot-then-scale needed; the pattern has been consistent all session).
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from data import FeatureDataset
from models import SmallCNNHead
from federated_experiment import run_localsteps_experiment

FEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "pretrained_features_v2")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "final_config_check")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMA = 0.3
TAU = 1.0
T = 300
N_SEEDS = 10
CONDITIONS = [
    ("clean", dict(n_byzantine=0, attack_type=None)),
    ("ipm", dict(n_byzantine=4, attack_type="ipm")),
]

import torch
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
            tag = f"combo__{cond_name}__gamma{GAMMA}__T{T}__seed{seed}"
            path = os.path.join(RESULTS_DIR, f"{tag}.json")
            if not os.path.exists(path):
                t0 = time.time()
                res = run_localsteps_experiment(
                    model_ctor=model_ctor, train_dataset=train_ds, test_dataset=test_ds,
                    n_regular=20, attack_type=cond_kwargs["attack_type"], n_byzantine=cond_kwargs["n_byzantine"],
                    beta=0.1, beta_hat=0.1, gamma=GAMMA, tau=TAU, tau_floor=0.8, epsilon=None,
                    ragg_name="trimmed_mean", T=T, batch_size=32, seed=seed,
                    local_steps=5, local_lr=0.01,
                )
                elapsed = time.time() - t0
                res.update({"cond": cond_name, "gamma": GAMMA, "T": T, "seed": seed})
                with open(path, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"  [{tag}] acc={res['final_test_acc']:.4f} diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)
            with open(path) as f:
                accs.append(json.load(f)["final_test_acc"])
        accs = np.array(accs)
        print(f"[{cond_name}] n={N_SEEDS} gamma={GAMMA} T={T} combo={accs.mean():.4f}+-{accs.std():.4f}", flush=True)

    print("\nFinal config check done.", flush=True)


if __name__ == "__main__":
    main()
