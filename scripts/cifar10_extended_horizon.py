"""Reviewer concern #2 (compute-budget confound): does CIFAR-10 accuracy stay
flat-lined past the paper's main T=300 probe, or does it begin a delayed
convergence slope given more rounds? Compromise horizon T=700 (vs the main
sweep's T=100-300) on a small seed subset -- a full T=1500 run is
compute-infeasible in this environment (~90s/round -> ~37h/seed) -- run at
alpha=0.5 (moderate heterogeneity) for n_byz in {0, 2}, logging accuracy every
EVAL_EVERY rounds so we can see the trajectory shape, not just the endpoint.
Uses the same ByzClip21SGD2M + trimmed-mean RAgg harness as bias_floor_sweep.py
(self-contained, does not alter that file).
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

from data import load_cifar10, partition_dirichlet, make_client_loaders, InfiniteLoaderIter
from models import SmallCNN
from byz_clip21_sgd2m import ByzClip21SGD2M
from robust_aggregators import apply_ragg
from attacks import ipm_byzantine_batch

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cifar10_extended_horizon")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA = 0.5
BYZ_COUNTS = [0, 2]
N_HONEST = 20
BATCH_SIZE = 32
GAMMA, TAU, BETA, BETA_HAT = 0.1, 1.0, 0.1, 0.1
RAGG_NAME = "trimmed_mean"
N_SEEDS = 3
T = 700
EVAL_EVERY = 50


def evaluate(model, set_flat, x, test_loader, loss_fn, device="cpu"):
    set_flat(x)
    model.eval()
    correct, total, loss_sum, n_batches = 0, 0, 0.0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(device))
            pred = out.argmax(dim=1).cpu()
            correct += (pred == yb).sum().item()
            total += yb.shape[0]
            loss_sum += loss_fn(out, yb.to(device)).item()
            n_batches += 1
    model.train()
    return correct / total, loss_sum / n_batches


def run_one(train_dataset, test_dataset, n_byz, seed, device="cpu"):
    torch.manual_seed(seed)
    n_total = N_HONEST + n_byz
    shards = partition_dirichlet(train_dataset, n_total, alpha=ALPHA, seed=seed)
    loaders = make_client_loaders(train_dataset, shards, BATCH_SIZE, seed=seed)
    iters = [InfiniteLoaderIter(loader) for loader in loaders]

    model = SmallCNN(num_classes=10)
    d = sum(p.numel() for p in model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    def set_flat(x):
        nn_utils.vector_to_parameters(x, model.parameters())

    def get_flat():
        return nn_utils.parameters_to_vector(model.parameters()).detach().clone()

    def flat_grad(x, xb, yb):
        set_flat(x)
        model.zero_grad()
        out = model(xb.to(device))
        loss = loss_fn(out, yb.to(device))
        loss.backward()
        return nn_utils.parameters_to_vector([p.grad for p in model.parameters()]).detach().clone()

    def grad_fn(x):
        grads = torch.zeros(N_HONEST, d)
        for i in range(N_HONEST):
            xb, yb = iters[i].next_batch()
            grads[i] = flat_grad(x, xb, yb)
        return grads

    byzantine_fn = None
    if n_byz > 0:
        byzantine_fn = lambda honest_c: ipm_byzantine_batch(honest_c, n_byz, scale=-10.0)
    ragg_fn = lambda vectors, num_byz: apply_ragg(RAGG_NAME, vectors, num_byz)

    algo = ByzClip21SGD2M(
        d=d, n_honest=N_HONEST, n_byzantine=n_byz,
        beta=BETA, beta_hat=BETA_HAT, gamma=GAMMA, tau=TAU, sigma_omega=0.0,
        ragg_fn=ragg_fn, device=device,
    )
    algo.set_x(get_flat())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    trajectory = []
    x = algo.x
    diverged = False
    for t in range(1, T + 1):
        x = algo.step(grad_fn, byzantine_fn)
        if not torch.isfinite(x).all():
            diverged = True
            break
        if t % EVAL_EVERY == 0 or t == T:
            acc, loss = evaluate(model, set_flat, x, test_loader, loss_fn, device)
            trajectory.append({"round": t, "acc": acc, "loss": loss})
            print(f"    round {t}: acc={acc:.4f} loss={loss:.4f}", flush=True)

    return {"diverged": diverged, "trajectory": trajectory}


def main():
    print("Loading CIFAR-10...", flush=True)
    ctrain, ctest = load_cifar10(DATA_ROOT)

    for n_byz in BYZ_COUNTS:
        for seed in range(N_SEEDS):
            tag = f"cifar10__alpha{ALPHA}__byz{n_byz}__seed{seed}__T{T}"
            path = os.path.join(RESULTS_DIR, f"extended__{tag}.json")
            if os.path.exists(path):
                continue
            print(f"[{tag}]", flush=True)
            t0 = time.time()
            res = run_one(ctrain, ctest, n_byz, seed)
            res.update({"alpha": ALPHA, "n_byz": n_byz, "seed": seed, "T": T,
                        "elapsed_s": time.time() - t0})
            with open(path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  done in {res['elapsed_s']:.1f}s, diverged={res['diverged']}", flush=True)

    print("\nExtended-horizon sweep done.", flush=True)


if __name__ == "__main__":
    main()
