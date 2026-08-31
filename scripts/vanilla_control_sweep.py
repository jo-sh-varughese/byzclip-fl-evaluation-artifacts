"""Control sweep for reviewer concern #1 (decoupling intrinsic dataset difficulty
from algorithmic failure): does CIFAR-10 collapse to chance under a completely
standard federated-averaging SGD loop -- no ByzClip21SGD2M double-momentum
mechanism, no robust aggregator, no Byzantine workers, no DP noise -- at the same
architecture, heterogeneity levels, batch size, and round budget used elsewhere in
this project? If CIFAR-10 still fails to rise above chance here, the collapse
cannot be attributed to the Byzantine-clipping/robust-aggregation machinery or to
DP noise calibration, since neither is present; that isolates dataset/architecture
optimization difficulty (consistent with the heavier measured tail-index) as the
remaining explanation. This is a self-contained loop (does not import
byz_clip21_sgd2m.py or robust_aggregators.py) so it cannot alter any existing
experiment's behavior.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

from data import load_mnist, load_cifar10, partition_dirichlet, make_client_loaders, InfiniteLoaderIter
from models import MNIST_CNN, SmallCNN

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "vanilla_control")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHAS = [100.0, 0.5, 0.1]
N_HONEST = 20
BATCH_SIZE = 32
GAMMA, TAU, MOMENTUM = 0.1, 1.0, 0.9   # gamma/tau match the paper's main sweep;
                                        # plain SGD momentum, not double-momentum
N_SEEDS = 5   # pilot scale (compute-budget-limited); see Limitations note
T_MNIST, T_CIFAR = 100, 100  # match bias_floor_sweep.py's round budget exactly


def run_one(model_ctor, train_dataset, test_dataset, num_classes, alpha, seed, T, device="cpu"):
    torch.manual_seed(seed)

    shards = partition_dirichlet(train_dataset, N_HONEST, alpha=alpha, seed=seed)
    loaders = make_client_loaders(train_dataset, shards, BATCH_SIZE, seed=seed)
    iters = [InfiniteLoaderIter(loader) for loader in loaders]

    model = model_ctor(num_classes=num_classes)
    loss_fn = nn.CrossEntropyLoss()
    momentum_buf = torch.zeros(sum(p.numel() for p in model.parameters()))

    def get_flat():
        return nn_utils.parameters_to_vector(model.parameters()).detach().clone()

    def set_flat(x):
        nn_utils.vector_to_parameters(x, model.parameters())

    diverged = False
    x = get_flat()
    for t in range(T):
        set_flat(x)
        model.zero_grad()
        grads = torch.zeros_like(x)
        for i in range(N_HONEST):
            xb, yb = iters[i].next_batch()
            model.zero_grad()
            out = model(xb.to(device))
            loss = loss_fn(out, yb.to(device))
            loss.backward()
            g = nn_utils.parameters_to_vector([p.grad for p in model.parameters()]).detach().clone()
            # standard per-client gradient clipping (matches tau used elsewhere),
            # then plain FedAvg (unweighted mean) -- no robust aggregator.
            gnorm = g.norm()
            if gnorm > TAU:
                g = g * (TAU / gnorm)
            grads += g
        grad_mean = grads / N_HONEST
        momentum_buf = MOMENTUM * momentum_buf + grad_mean
        x = x - GAMMA * momentum_buf
        if not torch.isfinite(x).all():
            diverged = True
            break

    if diverged:
        return {"final_test_acc": 0.0, "final_test_loss": float("nan"), "diverged": True}

    set_flat(x)
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    correct, total, loss_sum, n_batches = 0, 0, 0.0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(device))
            pred = out.argmax(dim=1).cpu()
            correct += (pred == yb).sum().item()
            total += yb.shape[0]
            loss_sum += loss_fn(out, yb.to(device)).item()
            n_batches += 1
    return {
        "final_test_acc": correct / total,
        "final_test_loss": loss_sum / n_batches,
        "diverged": False,
    }


def run_dataset(name, model_ctor, train_dataset, test_dataset, num_classes, T):
    for alpha in ALPHAS:
        for seed in range(N_SEEDS):
            tag = f"{name}__alpha{alpha}__seed{seed}"
            path = os.path.join(RESULTS_DIR, f"vanilla__{tag}.json")
            if os.path.exists(path):
                continue
            t0 = time.time()
            res = run_one(model_ctor, train_dataset, test_dataset, num_classes, alpha, seed, T)
            elapsed = time.time() - t0
            res.update({"dataset": name, "alpha": alpha, "seed": seed, "T": T})
            with open(path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  [{tag}] acc={res['final_test_acc']:.4f} loss={res['final_test_loss']:.4f} "
                  f"diverged={res['diverged']} ({elapsed:.1f}s)", flush=True)


def main():
    print("Loading MNIST...", flush=True)
    mtrain, mtest = load_mnist(DATA_ROOT)
    run_dataset("mnist", MNIST_CNN, mtrain, mtest, 10, T_MNIST)

    print("Loading CIFAR-10...", flush=True)
    ctrain, ctest = load_cifar10(DATA_ROOT)
    run_dataset("cifar10", SmallCNN, ctrain, ctest, 10, T_CIFAR)

    print("\nVanilla control sweep done.", flush=True)


if __name__ == "__main__":
    main()
