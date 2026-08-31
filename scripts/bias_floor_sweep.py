"""Stage 2 (main sweep) of the Byzantine-bias-floor study.

Measures the empirical Byzantine bias Delta(alpha, delta) := metric(alpha, delta=0) -
metric(alpha, delta) at matched heterogeneity (Dirichlet alpha) and Byzantine fraction
delta, isolating the BYZANTINE-INDUCED component of the accuracy/loss plateau from the
heterogeneity-only optimization difficulty captured by the delta=0 baseline at that same
alpha. This is a deliberately self-contained training loop (NOT a call into
federated_experiment.run_experiment) so this new study cannot alter that file's behavior
or risk the existing TMLR paper's reproducibility -- it duplicates the relevant slice of
that harness's logic (data partition -> ByzClip21SGD2M -> IPM attack -> trimmed-mean
RAgg) plus one addition run_experiment does not provide: final TEST LOSS (cross-entropy),
tracked because the theoretical floors we compare against
(Allouah/Gaucher-family, Shi et al.) are stated in loss/gradient-norm terms, not
accuracy; accuracy is reported alongside as the paper's existing, more interpretable
secondary metric.

DP noise is disabled (epsilon=None) throughout: this isolates the Byzantine/heterogeneity
floor from the DP-noise effect already characterized in the TMLR paper's own H2 tau-search
section, rather than conflating the two.
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
from byz_clip21_sgd2m import ByzClip21SGD2M
from robust_aggregators import apply_ragg
from attacks import ipm_byzantine_batch

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "bias_floor")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHAS = [100.0, 0.5, 0.1]
BYZ_COUNTS = [0, 2, 4]     # delta_byz = f/(n_honest) convention, n_honest=20 -> 0, 0.1, 0.2
N_HONEST = 20
BATCH_SIZE = 32
GAMMA, TAU, BETA, BETA_HAT = 0.1, 1.0, 0.1, 0.1
RAGG_NAME = "trimmed_mean"
N_SEEDS = 10  # scaled up from 5 per the code-review's power critique
T_MNIST, T_CIFAR = 100, 100
EVAL_EVERY_FRAC = 10


def run_one(model_ctor, train_dataset, test_dataset, num_classes, alpha, n_byz, seed, T, device="cpu"):
    torch.manual_seed(seed)

    n_total = N_HONEST + n_byz
    shards = partition_dirichlet(train_dataset, n_total, alpha=alpha, seed=seed)
    loaders = make_client_loaders(train_dataset, shards, BATCH_SIZE, seed=seed)
    iters = [InfiniteLoaderIter(loader) for loader in loaders]

    model = model_ctor(num_classes=num_classes)
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
    diverged = False
    x = algo.x
    for t in range(T):
        x = algo.step(grad_fn, byzantine_fn)
        if not torch.isfinite(x).all():
            diverged = True
            break

    if diverged:
        return {"final_test_acc": 0.0, "final_test_loss": float("nan"), "diverged": True}

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
    return {
        "final_test_acc": correct / total,
        "final_test_loss": loss_sum / n_batches,
        "diverged": False,
    }


def run_dataset(name, model_ctor, train_dataset, test_dataset, num_classes, T):
    for alpha in ALPHAS:
        for n_byz in BYZ_COUNTS:
            for seed in range(N_SEEDS):
                tag = f"{name}__alpha{alpha}__byz{n_byz}__seed{seed}"
                path = os.path.join(RESULTS_DIR, f"sweep__{tag}.json")
                if os.path.exists(path):
                    continue
                t0 = time.time()
                res = run_one(model_ctor, train_dataset, test_dataset, num_classes, alpha, n_byz, seed, T)
                elapsed = time.time() - t0
                res.update({"dataset": name, "alpha": alpha, "n_byz": n_byz, "seed": seed, "T": T})
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

    print("\nBias-floor sweep done.", flush=True)


if __name__ == "__main__":
    main()
