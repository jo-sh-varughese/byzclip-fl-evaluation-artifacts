"""FedProto (src/fedproto.py) comparison, run under the closest fair protocol to
FedProto's own paper (T=110 CIFAR-10 / T=100 MNIST, 20 clients, batch size 8,
lambda=0.1 CIFAR-10 / 1.0 MNIST, IID partition) that this project's CPU compute budget
can support.

ONE disclosed deviation from the paper's own protocol: FedProto specifies one full
local EPOCH per round (~312 steps/client/round at CIFAR-10's ~2500 images/client,
batch 8) -- at 20 clients x T=110 rounds, that is ~686,000 forward/backward passes,
far beyond this project's CPU pilot-scale budget. We use LOCAL_STEPS=25/round
instead and state this explicitly rather than silently matching the paper's own
number while not actually running it -- see paper/main_v2.tex's FedProto section for
how this is reported.

LOCAL_STEPS=25 is not an arbitrary compute-driven guess: an initial attempt at
LOCAL_STEPS=5 produced a global mean-client-accuracy stuck at EXACTLY 10.0% (chance)
for the entire run. Diagnosing this (not simply reporting it) found the cause: the
prototype-regularization loss has a trivial degenerate minimizer -- collapsing every
class's embedding toward the same point drives the L2 regularization term to zero
without requiring correct classification -- and at LOCAL_STEPS=5 there is not enough
classification-loss gradient per round to outweigh that collapse. Measured directly
(mean global-prototype norm over 15 rounds, 3 clients): LOCAL_STEPS=5 collapses from
~0.5 to ~0.02; LOCAL_STEPS=15 partially collapses to ~0.20; LOCAL_STEPS=25 and 40
both stabilize around ~0.40-0.42 (matching LOCAL_STEPS=100's ~0.34-0.35 stable
plateau); LOCAL_STEPS=5's collapse is therefore attributable to this project's own
compute-reduction choice, not to a defect in FedProto's own algorithm or its
originally specified full-local-epoch protocol. LOCAL_STEPS=25 is the smallest value
tested that avoids collapse, chosen to stay within this project's CPU budget while
not confusing an artifact of under-provisioned local computation with a property of
the algorithm being evaluated -- exactly the failure mode Section~4/5 of the paper
audits Byz-Clip21-SGD2M for, applied here to a baseline instead.

Evaluation metric: FedProto keeps NO single shared model (only prototypes are
aggregated), so "global test accuracy" is not directly defined for it the way it is
for Byz-Clip21-SGD2M. We report the mean, over clients, of each client's OWN local
model's accuracy on the shared CIFAR-10/MNIST test set -- the natural metric on a
shared test set when there is no single global model, though it is not FedProto's own
preferred metric (personalized accuracy on each client's own local distribution).

Byzantine condition: n_byzantine clients submit an IPM-crafted prototype (see
src/fedproto.py:byzantine_prototype_attack) for every class every round, instead of a
genuine one -- FedProto's plain weighted-mean prototype aggregation has no defense
against this by design; this is a stress test of an algorithm that was never built
for this threat model, not a claim that FedProto is a bad algorithm.
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from data import load_cifar10, load_mnist, partition_iid, make_client_loaders, InfiniteLoaderIter
from models import SmallCNN, MNIST_CNN
from fedproto import fedproto_local_update, aggregate_prototypes, byzantine_prototype_attack

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "fedproto")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_CLIENTS = 20
BATCH_SIZE = 8
LOCAL_STEPS = 25  # see module docstring: smallest value measured to avoid the
                  # prototype-collapse failure mode found at LOCAL_STEPS=5
LR = 0.01
EVAL_EVERY = 20


def run_one(model_ctor, train_dataset, test_dataset, num_classes, lam, n_byzantine,
            T, seed, device="cpu"):
    torch.manual_seed(seed)

    shards = partition_iid(train_dataset, N_CLIENTS, seed=seed)
    loaders = make_client_loaders(train_dataset, shards, BATCH_SIZE, seed=seed)
    iters = [InfiniteLoaderIter(loader) for loader in loaders]
    client_next_batch = [it.next_batch for it in iters]

    models = [model_ctor(num_classes=num_classes) for _ in range(N_CLIENTS)]
    global_protos = {}

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    def eval_mean_client_acc():
        accs = []
        with torch.no_grad():
            for m in models:
                m.eval()
                correct, total = 0, 0
                for xb, yb in test_loader:
                    logits = m.classify(m.embed(xb.to(device)))
                    pred = logits.argmax(dim=1).cpu()
                    correct += (pred == yb).sum().item()
                    total += yb.shape[0]
                accs.append(correct / total)
                m.train()
        return sum(accs) / len(accs)

    accuracy_trace = []
    for t in range(T):
        all_sums, all_counts = [], []
        for i in range(N_CLIENTS):
            sums, counts = fedproto_local_update(
                models[i], client_next_batch[i], global_protos, num_classes,
                lam=lam, lr=LR, local_steps=LOCAL_STEPS, device=device,
            )
            all_sums.append(sums)
            all_counts.append(counts)

        if n_byzantine > 0:
            byz_sums, byz_counts = byzantine_prototype_attack(
                all_sums, all_counts, num_classes, n_byzantine, scale=-10.0,
            )
            all_sums += byz_sums
            all_counts += byz_counts

        global_protos = aggregate_prototypes(all_sums, all_counts, num_classes)

        if (t + 1) % EVAL_EVERY == 0 or t == T - 1:
            acc = eval_mean_client_acc()
            accuracy_trace.append({"round": t + 1, "test_acc": acc})
            print(f"    round {t+1}: mean_client_acc={acc:.4f}", flush=True)

    return {
        "final_test_acc": accuracy_trace[-1]["test_acc"],
        "accuracy_trace": accuracy_trace,
        "config": {"n_byzantine": n_byzantine, "lam": lam, "T": T, "seed": seed,
                   "local_steps": LOCAL_STEPS, "batch_size": BATCH_SIZE},
    }


def main():
    N_SEEDS = 3
    print("Loading CIFAR-10...", flush=True)
    ctrain, ctest = load_cifar10(DATA_ROOT)
    for cond_name, n_byz in [("clean", 0), ("ipm", 4)]:
        for seed in range(N_SEEDS):
            tag = f"cifar10__{cond_name}__seed{seed}"
            path = os.path.join(RESULTS_DIR, f"{tag}.json")
            if os.path.exists(path):
                continue
            t0 = time.time()
            res = run_one(SmallCNN, ctrain, ctest, 10, lam=0.1, n_byzantine=n_byz, T=110, seed=seed)
            elapsed = time.time() - t0
            res.update({"cond": cond_name, "seed": seed})
            with open(path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  [{tag}] final_acc={res['final_test_acc']:.4f} ({elapsed:.1f}s)", flush=True)

    print("\nFedProto CIFAR-10 done.", flush=True)


if __name__ == "__main__":
    main()
