import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from attacks import apply_label_flip, label_flip_permutation, ipm_attack, ipm_byzantine_batch


def test_label_flip_permutation():
    perm = label_flip_permutation(10)
    expected = torch.tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    assert torch.equal(perm, expected), perm
    labels = torch.tensor([0, 1, 2, 9, 5])
    flipped = apply_label_flip(labels, 10)
    assert torch.equal(flipped, torch.tensor([9, 8, 7, 0, 4])), flipped
    # Involution: flipping twice returns original.
    assert torch.equal(apply_label_flip(flipped, 10), labels)
    print("label flip permutation OK")


def test_ipm_attack():
    honest = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 0.0]])
    crafted = ipm_attack(honest, scale=-10.0)
    expected = -10.0 * honest.mean(dim=0)
    assert torch.allclose(crafted, expected), crafted
    batch = ipm_byzantine_batch(honest, num_byzantine=3, scale=-10.0)
    assert batch.shape == (3, 2)
    assert torch.allclose(batch[0], batch[1]) and torch.allclose(batch[1], batch[2])
    print("IPM attack OK")


if __name__ == "__main__":
    test_label_flip_permutation()
    test_ipm_attack()
    print("All attack tests passed.")
