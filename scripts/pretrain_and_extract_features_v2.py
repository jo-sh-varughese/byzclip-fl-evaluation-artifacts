"""Stage 2 of the pretrained-backbone approach (see pretrain_and_extract_features.py for
Stage 1, which got clean/no-DP/no-Byzantine CIFAR-10 from ~17% to 54.3%, ipm to 52.6%,
both p=0.00195 at n=10). That result was bottlenecked by feature quality, not by the
federated protocol -- the combo arm converged to a very tight band (std ~0.003) well
before the theoretical ceiling for this architecture family, meaning more communication
rounds or federated tricks would not have moved it much further. This script instead
improves the ONE thing Stage 1 left unimproved: the backbone itself.

Three additions, all standard, all applied ONLY to the central pretraining step (never
touching the federated phase, so none of them reopen the confounds ruled out earlier in
this session):
  1. DeepBackboneCNN (src/models.py) -- 3 conv blocks with BatchNorm instead of 2 without.
     BatchNorm is safe here specifically because this network is centrally trained, never
     federated-averaged (see DeepBackboneCNN's docstring).
  2. Standard CIFAR-10 augmentation (random crop + horizontal flip) during pretraining
     only -- this is the exact ingredient the original external critique flagged as
     missing from this whole project; it was never usable in the federated phase itself
     (client-side augmentation would be one more federated confound to isolate), but it
     is unambiguously safe and standard for a CENTRAL pretraining pass.
  3. Per-channel normalization (CIFAR-10's standard mean/std) and more epochs (40 vs 15),
     both enabled by the added capacity/regularization above no longer overfitting the
     same 5000-image public split as quickly.

Same disjoint 5000/45000 public/client-pool split (identical seed=0 permutation) as
Stage 1, so the client pool used for the federated phase is unchanged.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models import DeepBackboneCNN

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "pretrained_features_v2")
os.makedirs(OUT_DIR, exist_ok=True)

N_PUBLIC = 5000
PRETRAIN_EPOCHS = 40
PRETRAIN_LR = 0.01
BATCH_SIZE = 32
SEED = 0

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

train_tfm = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
eval_tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])


def main():
    rng = np.random.RandomState(SEED)
    n_total = len(datasets.CIFAR10(root=DATA_ROOT, train=True, download=False))
    perm = rng.permutation(n_total)
    public_idx = perm[:N_PUBLIC]
    client_pool_idx = perm[N_PUBLIC:]
    print(f"public pretrain set: {len(public_idx)}  client pool: {len(client_pool_idx)}", flush=True)

    train_full_aug = datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=train_tfm)
    train_full_eval = datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=eval_tfm)
    test_eval = datasets.CIFAR10(root=DATA_ROOT, train=False, download=False, transform=eval_tfm)

    public_subset = torch.utils.data.Subset(train_full_aug, public_idx.tolist())
    public_loader = torch.utils.data.DataLoader(public_subset, batch_size=BATCH_SIZE, shuffle=True,
                                                 generator=torch.Generator().manual_seed(SEED))

    torch.manual_seed(SEED)
    model = DeepBackboneCNN(num_classes=10)
    opt = torch.optim.SGD(model.parameters(), lr=PRETRAIN_LR, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.time()
    model.train()
    for epoch in range(PRETRAIN_EPOCHS):
        total_loss, n_batches, correct, total = 0.0, 0, 0, 0
        for xb, yb in public_loader:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.shape[0]
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  pretrain epoch {epoch+1}/{PRETRAIN_EPOCHS} avg_loss={total_loss/n_batches:.4f} "
                  f"train_acc={correct/total:.4f}", flush=True)
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
    client_pool_features, client_pool_labels = extract(train_full_eval, client_pool_idx)
    test_features, test_labels = extract(test_eval)
    print(f"Feature extraction done in {time.time()-t0:.1f}s "
          f"(client_pool={client_pool_features.shape}, test={test_features.shape})", flush=True)

    torch.save(client_pool_features, os.path.join(OUT_DIR, "client_pool_features.pt"))
    torch.save(client_pool_labels, os.path.join(OUT_DIR, "client_pool_labels.pt"))
    torch.save(test_features, os.path.join(OUT_DIR, "test_features.pt"))
    torch.save(test_labels, os.path.join(OUT_DIR, "test_labels.pt"))
    print("Saved features to", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
