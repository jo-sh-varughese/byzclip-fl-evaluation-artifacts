"""Model architectures.

IMPORTANT — documented gap (see results_summary.md "Known Gaps" section): the source
paper's exact MNIST CNN/MLP architectures are not fully specified in the text available
to us. The architectures below are reasonable, standard, small choices for MNIST/CIFAR-10
federated-learning benchmarks, NOT a verified reproduction of the paper's exact layer
sizes. Parameter counts are reported so any future comparison against the paper's real
numbers (once available) is straightforward.

The CIFAR-10 "SmallCNN" is likewise built fresh for this study -- the task prompt referred
to a prior "OTCD" implementation with 227,594 parameters, but no such prior code or cached
data exists anywhere on this machine (verified by search before starting). This SmallCNN
is a new, independently-sized small CNN; its actual parameter count is reported below and
is NOT expected to equal 227,594.
"""

import torch
import torch.nn as nn


class MNIST_MLP(nn.Module):
    """Simple 2-hidden-layer MLP: 784 -> 200 -> 200 -> 10."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class MNIST_CNN(nn.Module):
    """Simple 2-conv-layer CNN: conv(1->16)-conv(16->32)-fc(1568->128)-fc(128->10)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(self.conv(x))

    def embed(self, x):
        """Feature embedding for FedProto (Tan et al. 2022); see SmallCNN.embed."""
        feat = torch.flatten(self.conv(x), 1)
        return self.fc[2](self.fc[1](feat))

    def classify(self, z):
        return self.fc[3](z)


class SmallCNN(nn.Module):
    """Small CNN for CIFAR-10, built fresh for this study (see module docstring)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.fc(self.conv(x))

    def embed(self, x):
        """Feature embedding for FedProto (Tan et al. 2022): conv stack + first FC
        layer + ReLU, i.e. everything up to but not including the final classifier
        layer. Embedding dimension matches self.fc[1].out_features (64)."""
        feat = torch.flatten(self.conv(x), 1)
        return self.fc[2](self.fc[1](feat))

    def classify(self, z):
        """The final linear layer, applied to an embedding produced by embed()."""
        return self.fc[3](z)


class DeepBackboneCNN(nn.Module):
    """Three-conv-block backbone with BatchNorm, for CENTRAL pretraining only (see
    scripts/pretrain_and_extract_features_v2.py). BatchNorm is safe here specifically
    because this network is never federated-averaged -- it is trained once, centrally,
    on a small disjoint public split, then frozen; the whole reason BatchNorm was
    avoided everywhere else in this project (SmallCNNGN's docstring) is that its
    running-buffer statistics have no way to be aggregated across federated clients,
    which does not apply to a model that only ever sees one, centralized training run.
    More capacity (32->64->128 channels, one more block) than SmallCNN, motivated by
    plain CIFAR-10 benchmarks: a 2-conv-block net without augmentation or BatchNorm
    plateaus well below what a 3-block BN'd net reaches with augmentation.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 8 -> 4
        )
        self.feat_dim = 128 * 4 * 4
        self.pretrain_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.pretrain_head(self.conv(x))


class SmallCNNHead(nn.Module):
    """The FC half of SmallCNN in isolation, for use on precomputed, frozen conv
    features (see scripts/pretrain_and_extract_features.py). Architecturally identical
    to SmallCNN.fc -- same shapes, same init -- so any effect of using this head on
    pretrained features vs. the full SmallCNN trained end-to-end is attributable to the
    pretraining, not to a smaller or different head.
    """

    def __init__(self, num_classes=10, feat_dim=64 * 8 * 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.fc(x)


class SmallCNNGN(nn.Module):
    """SmallCNN + GroupNorm after each conv.

    GroupNorm, not BatchNorm: BatchNorm's running_mean/running_var are buffers, not
    parameters, so federated_experiment.FlatModel (which flattens only
    model.parameters()) never transmits or aggregates them -- under this harness's
    single-shared-model-object design, per-client forward passes would silently
    stomp on shared running stats in an order-dependent way, and the aggregated
    parameter vector `x` would carry no normalization statistics at all. GroupNorm's
    weight/bias are ordinary parameters (no buffers), so it composes correctly with
    the existing flatten/aggregate/unflatten loop with zero harness changes, and it
    is also the standard fix for BatchNorm's known instability under non-IID FedAvg
    (per-client batch statistics diverge under label skew) -- both problems this
    confound check needs to rule out at once.
    """

    def __init__(self, num_classes=10, num_groups=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.fc(self.conv(x))

    def embed(self, x):
        """Feature embedding for FedProto (Tan et al. 2022): conv stack + first FC
        layer + ReLU, i.e. everything up to but not including the final classifier
        layer. Embedding dimension matches self.fc[1].out_features (64)."""
        feat = torch.flatten(self.conv(x), 1)
        return self.fc[2](self.fc[1](feat))

    def classify(self, z):
        """The final linear layer, applied to an embedding produced by embed()."""
        return self.fc[3](z)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for name, cls in [("MNIST_MLP", MNIST_MLP), ("MNIST_CNN", MNIST_CNN), ("SmallCNN", SmallCNN),
                      ("SmallCNNGN", SmallCNNGN)]:
        m = cls()
        print(f"{name}: {count_params(m)} parameters")
