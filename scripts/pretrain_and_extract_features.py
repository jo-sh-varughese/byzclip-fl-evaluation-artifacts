"""Pretrain SmallCNN's conv backbone centrally on a small, DISJOINT public slice of
CIFAR-10's training set, freeze it, and extract features for everything else -- so the
federated phase only ever trains a lightweight FC head on top of already-discriminative
features, instead of learning heavy-tailed-noise-plagued conv features from scratch
under a Byzantine-robust, DP-aware, 80-round budget.

Motivation: every prior mechanism tried this session (adaptive clipping, momentum warm-
start, floor clipping, multiple local steps) worked WITHIN the constraint of learning
conv features from raw pixels under this harness's round/compute budget. None produced a
large, unambiguous effect. Pretrained-backbone FL is a different kind of lever entirely:
it removes the hardest part of the optimization problem (feature learning) from the
federated loop altogether. This is a live, current line of work (federated learning with
frozen/pretrained backbones and lightweight trainable heads), not something invented for
this project.

Split: 5000 of CIFAR-10's 50000 training images (10%, fixed seed) held out as PUBLIC
pretraining data; the remaining 45000 form the client pool used for the federated phase
(replacing the datasets used everywhere else in this repo -- a strict subset, so no
client-pool image is ever seen during backbone pretraining). The standard 10000-image
test set is untouched and unused for pretraining, exactly as everywhere else in this
project.

Outputs (results/pretrained_features/): client_pool_features.pt, client_pool_labels.pt,
test_features.pt, test_labels.pt, backbone_conv_state.pt (for provenance).
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import torch.nn as nn

from data import load_cifar10
from models import SmallCNN

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "pretrained_features")
os.makedirs(OUT_DIR, exist_ok=True)

N_PUBLIC = 5000
PRETRAIN_EPOCHS = 15
PRETRAIN_LR = 0.01  # this session's centralized_sanity_check.py validated this rate as
                     # stable for this exact architecture; lr=0.1 diverged to chance.
BATCH_SIZE = 32
SEED = 0


def main():
    train_ds, test_ds = load_cifar10(DATA_ROOT)
    n_total = len(train_ds)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n_total)
    public_idx = perm[:N_PUBLIC]
    client_pool_idx = perm[N_PUBLIC:]
    print(f"public pretrain set: {len(public_idx)}  client pool: {len(client_pool_idx)}", flush=True)

    public_subset = torch.utils.data.Subset(train_ds, public_idx.tolist())
    public_loader = torch.utils.data.DataLoader(public_subset, batch_size=BATCH_SIZE, shuffle=True,
                                                 generator=torch.Generator().manual_seed(SEED))

    torch.manual_seed(SEED)
    model = SmallCNN(num_classes=10)
    opt = torch.optim.SGD(model.parameters(), lr=PRETRAIN_LR, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.time()
    model.train()
    for epoch in range(PRETRAIN_EPOCHS):
        total_loss, n_batches = 0.0, 0
        for xb, yb in public_loader:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"  pretrain epoch {epoch+1}/{PRETRAIN_EPOCHS} avg_loss={total_loss/n_batches:.4f}", flush=True)
    print(f"Pretraining done in {time.time()-t0:.1f}s", flush=True)

    torch.save(model.conv.state_dict(), os.path.join(OUT_DIR, "backbone_conv_state.pt"))

    model.eval()

    @torch.no_grad()
    def extract(dataset, indices=None):
        if indices is not None:
            dataset = torch.utils.data.Subset(dataset, indices.tolist())
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
        feats, labels = [], []
        for xb, yb in loader:
            f = model.conv(xb)
            f = torch.flatten(f, 1)
            feats.append(f)
            labels.append(yb)
        return torch.cat(feats), torch.cat(labels)

    t0 = time.time()
    client_pool_features, client_pool_labels = extract(train_ds, client_pool_idx)
    test_features, test_labels = extract(test_ds)
    print(f"Feature extraction done in {time.time()-t0:.1f}s "
          f"(client_pool={client_pool_features.shape}, test={test_features.shape})", flush=True)

    torch.save(client_pool_features, os.path.join(OUT_DIR, "client_pool_features.pt"))
    torch.save(client_pool_labels, os.path.join(OUT_DIR, "client_pool_labels.pt"))
    torch.save(test_features, os.path.join(OUT_DIR, "test_features.pt"))
    torch.save(test_labels, os.path.join(OUT_DIR, "test_labels.pt"))
    print("Saved features to", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
