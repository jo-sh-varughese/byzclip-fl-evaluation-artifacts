import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.utils as nn_utils

from data import load_cifar10, partition_iid, make_client_loaders, InfiniteLoaderIter
from models import SmallCNN
from byz_clip21_sgd2m import ByzClip21SGD2M
from robust_aggregators import apply_ragg
from subgaussian_analysis import collect_gradient_noise, fit_and_report, qq_plot

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "subgaussian")
os.makedirs(RESULTS_DIR, exist_ok=True)

# From results/cifar10/hp_probe.json (best of the reduced probe grid; see run_cifar10_extension.py).
# GAMMA can be overridden via sys.argv[2] to re-run under a Gap-2 "stabilized" config if one is found.
BEST_TAU, BEST_BETA, BEST_BETA_HAT = 1.0, 0.1, 0.1
N_CLIENTS = 20
BATCH_SIZE = 32
SNAPSHOT_ROUNDS = [10, 30, 60]

DRAWS_PER_CLIENT = int(sys.argv[1]) if len(sys.argv) > 1 else 20
GAMMA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
GAMMA_TAG = "" if GAMMA == 0.1 else f"_gamma{GAMMA}"
SUFFIX = (f"_n{DRAWS_PER_CLIENT}" if DRAWS_PER_CLIENT != 20 else "") + GAMMA_TAG


def get_snapshots(train, test):
    """Run clean federated training and capture x AND test accuracy at SNAPSHOT_ROUNDS.
    Accuracy trace added for Gap 3 (results_summary.md) -- confirms whether these
    snapshot rounds fell during a stable/improving stretch or a spike/crash transient,
    per the §4 T=300 oscillation finding."""
    torch.manual_seed(0)
    model = SmallCNN()
    d = sum(p.numel() for p in model.parameters())
    shards = partition_iid(train, N_CLIENTS, seed=0)
    loaders = make_client_loaders(train, shards, BATCH_SIZE, seed=0)
    iters = [InfiniteLoaderIter(l) for l in loaders]
    test_loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False)

    def grad_fn(x):
        grads = torch.zeros(N_CLIENTS, d)
        nn_utils.vector_to_parameters(x, model.parameters())
        for i in range(N_CLIENTS):
            xb, yb = iters[i].next_batch()
            model.zero_grad()
            out = model(xb)
            loss = torch.nn.functional.cross_entropy(out, yb)
            loss.backward()
            grads[i] = nn_utils.parameters_to_vector([p.grad for p in model.parameters()]).detach().clone()
            nn_utils.vector_to_parameters(x, model.parameters())
        return grads

    @torch.no_grad()
    def evaluate(x):
        nn_utils.vector_to_parameters(x, model.parameters())
        model.eval()
        correct, total = 0, 0
        for xb, yb in test_loader:
            out = model(xb)
            correct += (out.argmax(dim=1) == yb).sum().item()
            total += yb.shape[0]
        model.train()
        return correct / total

    ragg_fn = lambda vectors, num_byz: apply_ragg("trimmed_mean", vectors, num_byz)
    algo = ByzClip21SGD2M(d=d, n_honest=N_CLIENTS, n_byzantine=0, beta=BEST_BETA, beta_hat=BEST_BETA_HAT,
                          gamma=GAMMA, tau=BEST_TAU, sigma_omega=0.0, ragg_fn=ragg_fn)
    x0 = nn_utils.parameters_to_vector(model.parameters()).detach().clone()
    algo.set_x(x0)

    snapshots = {}
    accuracy_at_snapshot = {}
    max_round = max(SNAPSHOT_ROUNDS)
    for t in range(max_round):
        x = algo.step(grad_fn)
        if (t + 1) in SNAPSHOT_ROUNDS:
            snapshots[t + 1] = x.clone()
            acc = evaluate(x)
            accuracy_at_snapshot[t + 1] = acc
            print(f"  captured snapshot at round {t+1}, test_acc={acc:.4f}")
    return snapshots, accuracy_at_snapshot


def main():
    print(f"gamma={GAMMA}, draws_per_client={DRAWS_PER_CLIENT}")
    train, test = load_cifar10(DATA_ROOT)
    print("Capturing training snapshots (clean, no-DP, no-Byzantine, CIFAR-10 SmallCNN)...")
    snapshots, accuracy_at_snapshot = get_snapshots(train, test)

    print("Collecting gradient-noise samples across snapshots/clients...")
    result = collect_gradient_noise(
        model_ctor=SmallCNN, train_dataset=train, n_clients=N_CLIENTS, batch_size=BATCH_SIZE,
        num_classes=10, x_snapshots=list(snapshots.values()), draws_per_client=DRAWS_PER_CLIENT, seed=0,
    )
    report = fit_and_report(result, label=f"CIFAR10_SmallCNN{SUFFIX}")
    report["accuracy_at_snapshot_round"] = accuracy_at_snapshot
    report["gamma_used"] = GAMMA
    print(json.dumps(report, indent=2))

    with open(os.path.join(RESULTS_DIR, f"cifar10_subgaussian_report{SUFFIX}.json"), "w") as f:
        json.dump(report, f, indent=2)

    qq_plot(result["std_scalars"], f"CIFAR-10 SmallCNN: standardized gradient-noise QQ-plot vs N(0,1) (n={DRAWS_PER_CLIENT}, gamma={GAMMA})",
            os.path.join(RESULTS_DIR, f"cifar10_qq_plot{SUFFIX}.png"))
    print("Saved report and QQ-plot to", RESULTS_DIR)


if __name__ == "__main__":
    main()
