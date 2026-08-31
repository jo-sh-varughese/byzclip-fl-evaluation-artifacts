"""Dataset loading and client partitioning (IID + Dirichlet non-IID)."""

import numpy as np
import torch
from torchvision import datasets, transforms


def load_mnist(root):
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root=root, train=False, download=True, transform=tfm)
    return train, test


def load_cifar10(root):
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
    test = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)
    return train, test


class FeatureDataset(torch.utils.data.Dataset):
    """Wraps precomputed (feature, label) tensors with the minimal Dataset interface
    partition_iid/partition_dirichlet/make_client_loaders/get_labels expect (a .targets
    attribute, __len__, __getitem__), so scripts/pretrain_and_extract_features.py's
    cached features drop into the exact same harness as the raw-pixel datasets.
    """

    def __init__(self, features, labels):
        self.features = features
        self.targets = labels

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], int(self.targets[idx])


def get_labels(dataset):
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            return targets.numpy()
        return np.array(targets)
    raise ValueError("dataset has no .targets attribute")


def partition_iid(dataset, n_clients, seed=0):
    """Equal-size i.i.d. shards: shuffle indices, split into n_clients equal chunks."""
    n = len(dataset)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    return [idx[i::n_clients] for i in range(n_clients)]


def partition_dirichlet(dataset, n_clients, alpha, seed=0):
    """Label-skew non-IID partition via Dirichlet(alpha) per class (Hsu et al. 2019 style).

    For each class, draw a Dirichlet(alpha) vector over clients and split that class's
    indices proportionally. Smaller alpha => more label-skewed / non-IID.
    """
    labels = get_labels(dataset)
    n_classes = int(labels.max()) + 1
    rng = np.random.RandomState(seed)

    client_indices = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(alpha=[alpha] * n_clients)
        split_points = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        splits = np.split(class_idx, split_points)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    return [np.array(sorted(idx_list)) for idx_list in client_indices]


def make_client_loaders(dataset, client_index_lists, batch_size, seed=0):
    """Return a list of DataLoaders, one per client, each shuffling its own shard."""
    loaders = []
    for i, indices in enumerate(client_index_lists):
        subset = torch.utils.data.Subset(dataset, indices.tolist())
        gen = torch.Generator().manual_seed(seed + i)
        loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True, generator=gen)
        loaders.append(loader)
    return loaders


class InfiniteLoaderIter:
    """Wraps a DataLoader to provide an endless stream of (x, y) minibatches."""

    def __init__(self, loader):
        self.loader = loader
        self._iter = iter(loader)

    def next_batch(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)
