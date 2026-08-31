"""Empirical sub-Gaussianity check for stochastic gradient noise (H1's mechanistic core).

Protocol (documented approximations relative to the idealized spec, given pilot-scale
CPU compute -- see results_summary.md "Known Gaps / Approximations" section):
  1. Run clean (no DP, no Byzantine) federated training for a short warmup, and take
     parameter snapshots x at a few points along that trajectory (early / mid / late),
     rather than collecting noise at every single round of a full run. We pool samples
     across these snapshots and across clients to get a reasonably sized sample under
     compute constraints; the full protocol would use every (client, round) pair.
  2. At each snapshot x, for a subset of clients, draw K minibatch gradients
     grad_f_i(x, xi) with fresh xi and treat their mean (a large-"batch" estimate) as a
     proxy for the true gradient grad_f_i(x). theta = grad_f_i(x, xi) - proxy_mean for
     each of the K-1 remaining draws (the draw used only to help estimate the mean is
     excluded from the noise pool to reduce estimation bias, though with K draws this
     bias is small already).
  3. Per-coordinate standardization: because gradient coordinates have very different
     natural scales (e.g. early conv filters vs. final-layer biases), we standardize
     each coordinate by its own empirical std before pooling scalars into one QQ-plot /
     Hill-estimator sample -- otherwise heterogeneous per-coordinate scale would itself
     look like "heavy tails" and confound the measurement. This is a standard treatment
     for checking a per-coordinate sub-Gaussian noise assumption.
  4. Fitted "sigma": report both (a) the overall gradient-noise norm scale
     sigma_norm = sqrt(mean_k ||theta_k||^2 / d) treating noise as isotropic-Gaussian-
     equivalent, matching the paper's sigma_omega-style scalar parameterization, and
     (b) the standardized-coordinate empirical tail behavior against a unit Gaussian.
  5. Tail-index: Hill estimator on the pooled ||theta_k|| sample (top 10% order
     statistics) as a practical heavier/lighter-tail comparator between MNIST and
     CIFAR-10, run through the identical measurement pipeline.

Gap 1 fix (see results_summary.md §5.1): standardizing each of the K draws in a
(snapshot, client) group by a sample std computed from those SAME K draws is exactly
the small-sample z-score-vs-t-statistic problem -- with K=20 draws (19 degrees of
freedom), the correct light-tailed reference for "how heavy would this look even under
exactly Gaussian data" is Student's-t(df=K-1), not N(0,1). `fit_and_report` now reports
tail-exceedance ratios against BOTH references side by side, and `draws_per_client` is
tracked through so the correct df is used automatically.
"""

import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
from scipy import stats

from data import make_client_loaders, InfiniteLoaderIter, partition_iid


def collect_gradient_noise(model_ctor, train_dataset, n_clients, batch_size, num_classes,
                            x_snapshots, draws_per_client=20, seed=0, device="cpu"):
    """Collect pooled theta = grad(x, xi) - mean_grad(x) samples across snapshots/clients.

    Returns a dict with:
      'theta_norms': np.array of ||theta_k|| (one per client per snapshot per draw-1)
      'theta_std_scalars': np.array of standardized scalar residuals pooled across dims
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_ctor(num_classes=num_classes)
    d = sum(p.numel() for p in model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    shards = partition_iid(train_dataset, n_clients, seed=seed)
    loaders = make_client_loaders(train_dataset, shards, batch_size, seed=seed)
    iters = [InfiniteLoaderIter(loader) for loader in loaders]

    def flat_grad(x, xb, yb):
        nn_utils.vector_to_parameters(x, model.parameters())
        model.zero_grad()
        out = model(xb.to(device))
        loss = loss_fn(out, yb.to(device))
        loss.backward()
        return nn_utils.parameters_to_vector([p.grad for p in model.parameters()]).detach().clone()

    theta_norms = []
    all_thetas = []  # kept in memory only per-snapshot to bound memory, standardized+pooled immediately
    std_scalars = []

    for x in x_snapshots:
        for ci in range(n_clients):
            draws = torch.zeros(draws_per_client, d)
            for k in range(draws_per_client):
                xb, yb = iters[ci].next_batch()
                draws[k] = flat_grad(x, xb, yb)
            proxy_mean = draws.mean(dim=0)
            theta = draws - proxy_mean.unsqueeze(0)  # (draws_per_client, d)
            norms = theta.norm(dim=1)
            theta_norms.extend(norms.tolist())

            coord_std = theta.std(dim=0).clamp_min(1e-12)
            standardized = (theta / coord_std.unsqueeze(0)).flatten()
            # Subsample standardized scalars per (snapshot, client) to bound memory/runtime.
            idx = np.random.choice(standardized.numel(), size=min(2000, standardized.numel()), replace=False)
            std_scalars.extend(standardized.flatten()[idx].tolist())

    return {
        "theta_norms": np.array(theta_norms),
        "std_scalars": np.array(std_scalars),
        "d": d,
        "draws_per_client": draws_per_client,
    }


def hill_estimator(values, tail_fraction=0.1):
    """Hill estimator for tail index alpha on positive values (larger alpha = lighter tail)."""
    x = np.sort(values)[::-1]
    n = len(x)
    k = max(int(n * tail_fraction), 5)
    x = x[x > 0]
    if len(x) <= k:
        return float("nan")
    top = x[:k]
    x_k1 = x[k] if k < len(x) else x[-1]
    if x_k1 <= 0:
        return float("nan")
    log_ratios = np.log(top / x_k1)
    return 1.0 / np.mean(log_ratios)


def fit_and_report(result, label):
    theta_norms = result["theta_norms"]
    std_scalars = result["std_scalars"]
    d = result["d"]
    draws_per_client = result.get("draws_per_client", 20)
    df = max(draws_per_client - 1, 1)  # degrees of freedom used in the per-coordinate standardization

    sigma_norm = np.sqrt(np.mean(theta_norms ** 2) / d)
    hill_alpha = hill_estimator(theta_norms, tail_fraction=0.1)

    # Empirical sub-Gaussian bound check on standardized scalars: P(|Z|>t) vs 2exp(-t^2/2)
    # (unit-variance reference since std_scalars are standardized to unit std per-coordinate).
    ts = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    empirical_tail = np.array([(np.abs(std_scalars) > t).mean() for t in ts])
    gaussian_bound = 2 * (1 - stats.norm.cdf(ts))
    excess_ratio = empirical_tail / np.clip(gaussian_bound, 1e-300, None)

    # Gap 1 fix: bias-corrected reference. Standardizing K draws by their OWN sample std
    # (df = K-1) mechanically inflates apparent tail mass vs. N(0,1) even for exactly
    # Gaussian data -- Student's-t(df) is the standard reference for this small-sample
    # z-score-vs-t-statistic effect. Report the SAME exceedance ratio against t(df).
    t_bound = 2 * (1 - stats.t.cdf(ts, df=df))
    excess_ratio_vs_t = empirical_tail / np.clip(t_bound, 1e-300, None)

    # KS test of standardized scalars against N(0,1).
    ks_stat, ks_pvalue = stats.kstest(std_scalars, "norm")

    # Excess kurtosis: 0 for exact Gaussian, positive => heavier-than-Gaussian tails.
    kurtosis = stats.kurtosis(std_scalars, fisher=True)

    return {
        "label": label,
        "n_theta_samples": int(len(theta_norms)),
        "n_std_scalars": int(len(std_scalars)),
        "d": int(d),
        "draws_per_client": int(draws_per_client),
        "standardization_df": int(df),
        "sigma_norm": float(sigma_norm),
        "hill_tail_index_alpha": float(hill_alpha),
        "tail_check_t": ts.tolist(),
        "empirical_tail_prob": empirical_tail.tolist(),
        "gaussian_bound_2exp": gaussian_bound.tolist(),
        "excess_ratio_empirical_over_gaussian_bound": excess_ratio.tolist(),
        "t_dist_bound_df": t_bound.tolist(),
        "excess_ratio_empirical_over_t_dist_bound": excess_ratio_vs_t.tolist(),
        "ks_statistic_vs_normal": float(ks_stat),
        "ks_pvalue_vs_normal": float(ks_pvalue),
        "excess_kurtosis": float(kurtosis),
    }


def qq_plot(std_scalars, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    stats.probplot(std_scalars, dist="norm", plot=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
