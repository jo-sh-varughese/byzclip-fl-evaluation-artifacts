"""Stage 1 of the Byzantine-bias-floor study: measure empirical gradient heterogeneity
zeta(alpha) for each (dataset, Dirichlet-alpha) cell used in the main sweep.

For each dataset, we run a short, Byzantine-free, no-DP warmup under IID partitioning
(alpha irrelevant to warmup -- only the MODEL TRAJECTORY needs to be representative) to
get two parameter snapshots (early = round 5, mid = round 20), then re-partition data at
each target alpha and measure zeta at BOTH snapshots under that partition, per
src/heterogeneity.py. Using the same warmup trajectory for every alpha isolates the
partitioning's effect on zeta from any effect of alpha on the trajectory itself (alpha
only affects which client sees which examples, not what "x" values we probe zeta at).
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn

from data import load_mnist, load_cifar10, partition_iid, partition_dirichlet, make_client_loaders, InfiniteLoaderIter
from models import MNIST_CNN, SmallCNN
from heterogeneity import measure_zeta_at_snapshots

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "bias_floor")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHAS = [100.0, 0.5, 0.1]
N_CLIENTS = 20
BATCH_SIZE = 32
K_GRAD = 10
SEED = 0
WARMUP_SNAPSHOT_ROUNDS = [5, 20]


def warmup_snapshots(model_ctor, train_dataset, num_classes, device="cpu"):
    """Short IID, Byzantine-free, no-DP SGD warmup; return x at the target rounds."""
    torch.manual_seed(SEED)
    model = model_ctor(num_classes=num_classes)
    loss_fn = nn.CrossEntropyLoss()
    shards = partition_iid(train_dataset, N_CLIENTS, seed=SEED)
    loaders = make_client_loaders(train_dataset, shards, BATCH_SIZE, seed=SEED)
    iters = [InfiniteLoaderIter(loader) for loader in loaders]

    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    snapshots = []
    max_round = max(WARMUP_SNAPSHOT_ROUNDS)
    for t in range(1, max_round + 1):
        ci = t % N_CLIENTS
        xb, yb = iters[ci].next_batch()
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        if t in WARMUP_SNAPSHOT_ROUNDS:
            snapshots.append(torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone())
    return snapshots


def run_dataset(name, model_ctor, train_dataset, num_classes):
    print(f"[{name}] warmup...", flush=True)
    x_snapshots = warmup_snapshots(model_ctor, train_dataset, num_classes)

    for alpha in ALPHAS:
        path = os.path.join(RESULTS_DIR, f"zeta__{name}__alpha{alpha}.json")
        if os.path.exists(path):
            continue
        shards = partition_dirichlet(train_dataset, N_CLIENTS, alpha=alpha, seed=SEED)
        result = measure_zeta_at_snapshots(
            model_ctor, train_dataset, shards, x_snapshots,
            BATCH_SIZE, num_classes, K=K_GRAD, seed=SEED,
        )
        result["alpha"] = alpha
        result["dataset"] = name
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  [{name}] alpha={alpha}: zeta_mean={result['zeta_mean']:.4f} "
              f"(per-snapshot zeta^2={[round(z,4) for z in result['zeta_sq_per_snapshot']]})", flush=True)


def main():
    print("Loading MNIST...", flush=True)
    mtrain, _ = load_mnist(DATA_ROOT)
    run_dataset("mnist", MNIST_CNN, mtrain, 10)

    print("Loading CIFAR-10...", flush=True)
    ctrain, _ = load_cifar10(DATA_ROOT)
    run_dataset("cifar10", SmallCNN, ctrain, 10)

    print("\nHeterogeneity measurement done.", flush=True)


if __name__ == "__main__":
    main()
