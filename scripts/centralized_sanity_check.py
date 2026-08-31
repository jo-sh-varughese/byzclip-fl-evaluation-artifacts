"""Reviewer sanity check: is Byz-Clip21-SGD2M's/the vanilla control's low CIFAR-10
accuracy (10-21% / 29-41% at T=100) evidence of a bug in the FL loop or
architecture, rather than a real finding about the algorithm/dataset?

Rules this out by training the IDENTICAL SmallCNN architecture (src/models.py)
on CIFAR-10 with a completely standard, non-federated, single-worker training
loop -- no data partitioning, no clipping, no momentum tricks, no aggregation,
just plain single-model SGD/Adam over the full training set -- at two budgets:
(a) matched compute: the same number of gradient steps as one T=100 federated
run (100 steps, batch 32) would use per "logical" worker, to see whether the
architecture is capable of much more than chance in that little training; and
(b) a generous budget (multiple full epochs) to establish the architecture's
achievable ceiling on CIFAR-10 at all, decoupled from any FL/algorithm effect.
If (b) reaches typical small-CNN CIFAR-10 accuracy (55-70%), that confirms
the architecture and data pipeline are not buggy, and the federated numbers
elsewhere in this paper reflect the FL/algorithm setup specifically.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn

from data import load_cifar10
from models import SmallCNN

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
BATCH_SIZE = 32
SEED = 0


def evaluate(model, loader, device="cpu"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device))
            pred = out.argmax(dim=1).cpu()
            correct += (pred == yb).sum().item()
            total += yb.shape[0]
    model.train()
    return correct / total


def run(n_steps=None, n_epochs=None, lr=0.1, momentum=0.9, tag=""):
    torch.manual_seed(SEED)
    train_ds, test_ds = load_cifar10(DATA_ROOT)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

    model = SmallCNN(num_classes=10)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.time()
    step = 0
    done = False
    epoch = 0
    while not done:
        epoch += 1
        for xb, yb in train_loader:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            step += 1
            if n_steps is not None and step >= n_steps:
                done = True
                break
        if n_epochs is not None and epoch >= n_epochs:
            done = True

    acc = evaluate(model, test_loader)
    elapsed = time.time() - t0
    print(f"[{tag}] steps={step} epochs={epoch} test_acc={acc:.4f} ({elapsed:.1f}s)", flush=True)
    return acc


def main():
    print("Loading CIFAR-10 / running centralized sanity checks...", flush=True)
    # (a) matched-compute: ~100 gradient steps, same as one T=100 federated run's
    # per-worker step count (batch 32, no averaging across 20 workers -- this run
    # sees 100 x 32 = 3200 examples total, comparable to one worker's exposure).
    run(n_steps=100, lr=0.1, momentum=0.9, tag="matched_compute_T100steps")
    # (b) generous budget: 10 full epochs (50000/32 ~= 1560 steps/epoch), to
    # establish the architecture's achievable ceiling, decoupled from any
    # FL/algorithm effect. lr=0.1 (this paper's headline gamma) diverged to
    # chance over 10 epochs on this un-batch-normed architecture -- itself a
    # relevant finding (this architecture is sensitive to lr at longer
    # horizons) -- so we additionally check a smaller, standard-for-plain-SGD
    # learning rate to separate "architecture can't learn CIFAR-10" from
    # "this specific lr is unstable at this horizon."
    run(n_epochs=10, lr=0.1, momentum=0.9, tag="generous_10epochs_lr0.1")   # diverges to chance (0.100)
    run(n_epochs=10, lr=0.01, momentum=0.9, tag="generous_10epochs_lr0.01")  # reaches 0.652
    print("\nCentralized sanity check done.", flush=True)


if __name__ == "__main__":
    main()
