"""Empirical gradient-heterogeneity (zeta) estimator for the Byzantine-bias-floor study.

Both theoretical floors we compare against (Allouah/Gaucher-family (G,B)-heterogeneity,
Shi et al. gradient-dissimilarity zeta) are stated in terms of the SPREAD of honest
clients' true gradients around their mean, at a given point x:
    zeta(x)^2 := (1/n_honest) * sum_i || grad f_i(x) - grad f(x) ||^2
where grad f_i(x) is client i's TRUE (population) gradient, not a noisy minibatch
estimate. We approximate grad f_i(x) with a large-batch average over K minibatches
(the same large-batch-as-proxy-for-true-gradient device already used in
subgaussian_analysis.collect_gradient_noise), which trades a small, disclosed bias for
tractability under CPU-only compute -- consistent with this project's standing practice
of using large-but-finite batches as population-gradient proxies rather than claiming
an unbiased estimator.

We measure zeta at TWO snapshots (an early and a mid-training point along an honest,
Byzantine-free warmup trajectory) and report both plus their mean, so a caller can check
whether the cross-alpha comparison is sensitive to which point in training zeta is read
at (mirroring H1's own snapshot-sensitivity check for tail statistics).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

from data import make_client_loaders, InfiniteLoaderIter


def client_mean_gradients(model_ctor, train_dataset, client_index_lists, x, batch_size,
                           num_classes, K=10, seed=0, device="cpu"):
    """Large-batch (K-minibatch average) gradient per client at parameter vector x.

    Returns a (n_clients, d) tensor.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_ctor(num_classes=num_classes)
    loss_fn = nn.CrossEntropyLoss()
    loaders = make_client_loaders(train_dataset, client_index_lists, batch_size, seed=seed)
    iters = [InfiniteLoaderIter(loader) for loader in loaders]

    n_clients = len(client_index_lists)
    d = sum(p.numel() for p in model.parameters())
    out = torch.zeros(n_clients, d)

    for ci in range(n_clients):
        acc = torch.zeros(d)
        for _ in range(K):
            xb, yb = iters[ci].next_batch()
            nn_utils.vector_to_parameters(x, model.parameters())
            model.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            acc += nn_utils.parameters_to_vector([p.grad for p in model.parameters()]).detach()
        out[ci] = acc / K

    return out


def zeta_squared(client_grads):
    """zeta^2 = mean_i || g_i - mean_j g_j ||^2, given a (n_clients, d) gradient tensor."""
    mean_grad = client_grads.mean(dim=0)
    devs = client_grads - mean_grad.unsqueeze(0)
    return (devs.norm(dim=1) ** 2).mean().item()


def measure_zeta_at_snapshots(model_ctor, train_dataset, client_index_lists, x_snapshots,
                               batch_size, num_classes, K=10, seed=0, device="cpu"):
    """Returns dict: per-snapshot zeta^2, mean zeta^2, and mean zeta (sqrt)."""
    zetas_sq = []
    for x in x_snapshots:
        grads = client_mean_gradients(
            model_ctor, train_dataset, client_index_lists, x, batch_size,
            num_classes, K=K, seed=seed, device=device,
        )
        zetas_sq.append(zeta_squared(grads))
    zetas_sq = np.array(zetas_sq)
    return {
        "zeta_sq_per_snapshot": zetas_sq.tolist(),
        "zeta_sq_mean": float(zetas_sq.mean()),
        "zeta_mean": float(np.sqrt(zetas_sq.mean())),
    }
