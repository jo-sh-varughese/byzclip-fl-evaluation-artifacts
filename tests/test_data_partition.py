import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from data import load_mnist, partition_iid, partition_dirichlet, get_labels

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


def test_iid_partition_covers_all_no_overlap():
    train, _ = load_mnist(DATA_ROOT)
    n_clients = 20
    shards = partition_iid(train, n_clients, seed=0)
    all_idx = np.concatenate(shards)
    assert len(all_idx) == len(train)
    assert len(set(all_idx.tolist())) == len(train), "overlap or duplicate indices"
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1, f"shards too unequal: {sizes}"
    print(f"IID partition OK: {n_clients} clients, sizes {min(sizes)}-{max(sizes)}")


def test_dirichlet_partition_covers_all_no_overlap_and_is_skewed():
    train, _ = load_mnist(DATA_ROOT)
    n_clients = 20
    shards_low_alpha = partition_dirichlet(train, n_clients, alpha=0.1, seed=0)
    shards_high_alpha = partition_dirichlet(train, n_clients, alpha=100.0, seed=0)

    for shards in (shards_low_alpha, shards_high_alpha):
        all_idx = np.concatenate(shards)
        assert len(set(all_idx.tolist())) == len(all_idx), "overlap within partition"
        assert len(all_idx) == len(train)

    labels = get_labels(train)

    def label_entropy(shard):
        counts = np.bincount(labels[shard], minlength=10)
        p = counts / counts.sum()
        p = p[p > 0]
        return -(p * np.log(p)).sum()

    low_alpha_entropy = np.mean([label_entropy(s) for s in shards_low_alpha])
    high_alpha_entropy = np.mean([label_entropy(s) for s in shards_high_alpha])
    print(f"Mean per-client label entropy: alpha=0.1 -> {low_alpha_entropy:.3f}, alpha=100 -> {high_alpha_entropy:.3f}")
    assert low_alpha_entropy < high_alpha_entropy, "low alpha should be MORE label-skewed (lower entropy)"
    print("Dirichlet partition OK: low alpha is more skewed than high alpha, as expected.")


if __name__ == "__main__":
    test_iid_partition_covers_all_no_overlap()
    test_dirichlet_partition_covers_all_no_overlap_and_is_skewed()
    print("All data partition tests passed.")
