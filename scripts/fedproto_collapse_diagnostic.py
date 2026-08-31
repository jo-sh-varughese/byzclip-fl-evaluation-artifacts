"""Reproduces Table 2 of paper/main_v2.tex (Section "Attempting a Comparison with
FedProto"): mean global-prototype norm after 15-20 rounds, as a function of local
steps/round and client count. Consolidates the interactive diagnostic commands run
during debugging into a single, rerunnable script -- every number quoted in that
section should be reproducible from this file, not from a one-off shell command.

Context: an initial FedProto run (scripts/run_fedproto.py) at LOCAL_STEPS=5 produced
mean client accuracy pinned at exactly 10.0% (chance) for its entire T=110 run.
Tracking the global prototype norm (rather than only downstream accuracy) diagnosed
why: the prototype-alignment loss has a trivial degenerate minimizer (collapse every
class's embedding to the same point), and with too little classification-loss signal
per round to counteract it, training collapses into exactly that minimizer.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from data import load_cifar10, partition_iid, make_client_loaders, InfiniteLoaderIter
from models import SmallCNN
from fedproto import fedproto_local_update, aggregate_prototypes

LR = 0.01
LAM = 0.1


def run(n_clients, local_steps, n_rounds, seed=0):
    torch.manual_seed(seed)
    train, _ = load_cifar10(os.path.join(os.path.dirname(__file__), "..", "data"))
    shards = partition_iid(train, n_clients, seed=seed)
    loaders = make_client_loaders(train, shards, 8, seed=seed)
    iters = [InfiniteLoaderIter(l) for l in loaders]
    models = [SmallCNN(num_classes=10) for _ in range(n_clients)]
    global_protos = {}
    for _ in range(n_rounds):
        all_sums, all_counts = [], []
        for i in range(n_clients):
            sums, counts = fedproto_local_update(
                models[i], iters[i].next_batch, global_protos, 10,
                lam=LAM, lr=LR, local_steps=local_steps,
            )
            all_sums.append(sums)
            all_counts.append(counts)
        global_protos = aggregate_prototypes(all_sums, all_counts, 10)
    norms = [v.norm().item() for v in global_protos.values()]
    return sum(norms) / len(norms)


def main():
    configs = [
        (3, 5, 15), (3, 15, 15), (3, 25, 15), (3, 40, 15), (3, 100, 15),
        (20, 25, 15), (20, 50, 15), (20, 100, 15),
    ]
    for n_clients, local_steps, n_rounds in configs:
        norm = run(n_clients, local_steps, n_rounds)
        print(f"clients={n_clients} local_steps={local_steps} rounds={n_rounds} "
              f"final_mean_proto_norm={norm:.4f}", flush=True)


if __name__ == "__main__":
    main()
