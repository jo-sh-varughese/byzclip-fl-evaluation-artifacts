"""Byzantine attack implementations: IPM and label-flipping.

Both attacks assume an "omniscient" Byzantine coalition (standard in this literature --
Xie et al. 2019/2020 for IPM): Byzantine clients see the honest clients' vectors for the
current round before choosing their own response.
"""

import torch

# Fixed label-flip permutation for a 10-class problem (MNIST / CIFAR-10):
# label -> (num_classes - 1 - label), i.e. 0<->9, 1<->8, 2<->7, 3<->6, 4<->5.
# Documented explicitly per the task spec's requirement to record the exact permutation used.


def label_flip_permutation(num_classes):
    return torch.tensor([num_classes - 1 - c for c in range(num_classes)])


def apply_label_flip(labels, num_classes):
    """Apply the fixed reversal permutation label -> num_classes - 1 - label."""
    perm = label_flip_permutation(num_classes).to(labels.device)
    return perm[labels]


def ipm_attack(honest_vectors, scale=-10.0):
    """Inner Product Manipulation attack (Xie et al.).

    Each Byzantine client transmits `scale * mean(honest_vectors)` instead of an honest
    update. `honest_vectors` has shape (n_honest, d); returns a single (d,) vector to be
    replicated across all Byzantine clients (the omniscient-coalition assumption means
    every Byzantine client sends the identical crafted vector).
    """
    return scale * honest_vectors.mean(dim=0)


def ipm_byzantine_batch(honest_vectors, num_byzantine, scale=-10.0):
    """Return (num_byzantine, d) tensor of identical IPM-crafted vectors."""
    crafted = ipm_attack(honest_vectors, scale=scale)
    return crafted.unsqueeze(0).repeat(num_byzantine, 1)


ATTACK_REGISTRY = {
    "ipm": ipm_byzantine_batch,
}
