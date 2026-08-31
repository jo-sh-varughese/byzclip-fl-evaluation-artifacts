"""Robust aggregation rules (RAgg) satisfying Definition 3.2 of Islamov et al. (arXiv:2603.23472).

Definition 3.2 (paraphrased from the prompt spec): for a regular-client set S of size
n - |B| out of n total clients, with xbar = mean_{i in S}(x_i),

    ||RAgg(x_1,...,x_n) - xbar||^2 <= (c * delta_byz / (n - |B|)) * sum_{i in S} ||x_i - xbar||^2

Coordinate-wise trimmed mean is used here as the primary RAgg because it is a standard
aggregator known to satisfy this class of inequality (Allouah et al., "Fixing by Mixing:
A Recipe for Optimal Byzantine ML under Heterogeneity", AISTATS 2023 -- cited directly by
the paper for Definition 3.2). Coordinate-wise median is included as a secondary RAgg for
a robustness-to-aggregator-choice check.
"""

import torch


def coordinate_trimmed_mean(vectors, num_byzantine):
    """Coordinate-wise trimmed mean.

    For each coordinate, drop the `num_byzantine` largest and `num_byzantine` smallest
    values across clients, then average what remains. `num_byzantine` is the assumed
    (declared) Byzantine count `f` used to size the trim -- the standard usage of
    trimmed mean as a RAgg (Allouah et al. 2023a; Yin et al. 2018).

    Args:
        vectors: Tensor of shape (n, d) -- one row per client.
        num_byzantine: int, number of coordinates to trim from each side, per dimension.

    Returns:
        Tensor of shape (d,).
    """
    n = vectors.shape[0]
    f = int(num_byzantine)
    if 2 * f >= n:
        raise ValueError(f"trim count f={f} must satisfy 2f < n (n={n})")
    sorted_vals, _ = torch.sort(vectors, dim=0)
    if f == 0:
        trimmed = sorted_vals
    else:
        trimmed = sorted_vals[f: n - f]
    return trimmed.mean(dim=0)


def coordinate_median(vectors):
    """Coordinate-wise median.

    Args:
        vectors: Tensor of shape (n, d).

    Returns:
        Tensor of shape (d,).
    """
    return torch.median(vectors, dim=0).values


RAGG_REGISTRY = {
    "trimmed_mean": coordinate_trimmed_mean,
    "median": lambda vectors, num_byzantine: coordinate_median(vectors),
}


def apply_ragg(name, vectors, num_byzantine):
    if name not in RAGG_REGISTRY:
        raise ValueError(f"Unknown RAgg '{name}'. Options: {list(RAGG_REGISTRY)}")
    return RAGG_REGISTRY[name](vectors, num_byzantine)
