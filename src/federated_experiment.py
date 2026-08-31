"""Shared federated-training harness wiring data/models/attacks into ByzClip21SGD2M.

Design note on how attacks map onto Algorithm 1's client roles:
  - IPM is a vector-injection attack: Byzantine clients bypass the honest v_i/g_i/clip
    pipeline entirely and transmit an arbitrary crafted c_i (per Algorithm 1's "for
    Byzantine i: c_i = arbitrary_vector()" line). This is implemented via the
    `byzantine_fn` hook in ByzClip21SGD2M.step, operating on the round's honest
    transmitted c_i vectors (omniscient-coalition attack model).
  - Label-flipping is a DATA-poisoning attack, not a protocol deviation: the attacker
    follows the honest v_i/g_i/clip/noise pipeline exactly, but computes its local
    stochastic gradient on systematically mislabeled data. Mechanically this means
    label-flip attackers are simulated as additional "honest-pipeline" clients (they go
    through ByzClip21SGD2M's honest v_i update) whose data loader applies
    attacks.apply_label_flip to every label before computing the loss. There is no
    n_byzantine > 0 in the ByzClip21SGD2M sense for this attack type.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

from byz_clip21_sgd2m import ByzClip21SGD2M
from byz_clip21_sgd2m_adaptive import AdaptiveClip21SGD2M
from byz_clip21_sgd2m_warmstart import WarmStartClip21SGD2M
from byz_clip21_sgd2m_bandclip import BandClip21SGD2M
from byz_clip21_sgd2m_tailadaptive import TailAdaptiveClip21SGD2M
from byz_clip21_sgd2m_evtfloor import EVTFloorClip21SGD2M
from byz_clip_sgd import ByzClipSGD
from safe_dshb import SafeDSHB
from robust_aggregators import apply_ragg
from attacks import ipm_byzantine_batch, apply_label_flip
from data import make_client_loaders, InfiniteLoaderIter, partition_iid


def compute_sigma_omega(tau, epsilon, T, delta=1e-5):
    """sigma_omega = (tau/epsilon) * sqrt(T * log(1/delta)), per the task spec.

    Privacy amplification by sub-sampling is deliberately NOT applied, matching the
    paper's stated design choice ("we disable privacy amplification by sub-sampling").
    """
    if epsilon is None or epsilon == float("inf"):
        return 0.0
    return (tau / epsilon) * math.sqrt(T * math.log(1.0 / delta))


class FlatModel:
    """Wraps an nn.Module to expose flat-vector get/set of parameters and flat gradients."""

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.d = sum(p.numel() for p in model.parameters())

    def get_flat(self):
        return nn_utils.parameters_to_vector(self.model.parameters()).detach().clone()

    def set_flat(self, x):
        nn_utils.vector_to_parameters(x, self.model.parameters())

    def flat_grad(self, x, xb, yb, loss_fn):
        self.set_flat(x)
        self.model.zero_grad()
        out = self.model(xb.to(self.device))
        loss = loss_fn(out, yb.to(self.device))
        loss.backward()
        grad = nn_utils.parameters_to_vector([p.grad for p in self.model.parameters()]).detach().clone()
        return grad, loss.item()

    @torch.no_grad()
    def evaluate(self, x, loader):
        self.set_flat(x)
        self.model.eval()
        correct, total = 0, 0
        for xb, yb in loader:
            out = self.model(xb.to(self.device))
            pred = out.argmax(dim=1).cpu()
            correct += (pred == yb).sum().item()
            total += yb.shape[0]
        self.model.train()
        return correct / total


def run_experiment(
    model_ctor,
    train_dataset,
    test_dataset,
    n_regular,
    n_byzantine,
    attack_type,          # None | "ipm" | "label_flip"
    beta,
    beta_hat,
    gamma,
    tau,
    epsilon,              # None disables DP noise (sigma_omega=0)
    ragg_name,
    T,
    batch_size,
    seed,
    ablation=None,
    partition_fn=None,
    partition_kwargs=None,
    num_classes=10,
    device="cpu",
    eval_every=None,
    delta=1e-5,
):
    """Run one full Byz-Clip21-SGD2M training run and return a results dict.

    `n_regular` regular (never-poisoned) clients always exist. For attack_type="ipm",
    `n_byzantine` additional vector-injection attackers are added (algorithm's B set).
    For attack_type="label_flip", `n_byzantine` additional clients are added to the
    HONEST pipeline but trained on flipped labels (see module docstring). For
    attack_type=None, n_byzantine is ignored (treated as 0).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if attack_type == "label_flip":
        n_pipeline_honest = n_regular + n_byzantine
        n_algo_byzantine = 0
    else:
        n_pipeline_honest = n_regular
        n_algo_byzantine = n_byzantine if attack_type == "ipm" else 0

    n_total_clients = n_pipeline_honest + n_algo_byzantine
    partition_fn = partition_fn or partition_iid
    partition_kwargs = partition_kwargs or {}
    shards = partition_fn(train_dataset, n_total_clients, seed=seed, **partition_kwargs)
    loaders = make_client_loaders(train_dataset, shards, batch_size, seed=seed)
    iters = [InfiniteLoaderIter(loader) for loader in loaders]

    is_flip_client = [False] * n_pipeline_honest
    if attack_type == "label_flip":
        for i in range(n_regular, n_pipeline_honest):
            is_flip_client[i] = True

    model = model_ctor(num_classes=num_classes)
    flat_model = FlatModel(model, device)
    d = flat_model.d
    loss_fn = nn.CrossEntropyLoss()

    sigma_omega = compute_sigma_omega(tau if tau != float("inf") else 1.0, epsilon, T, delta) if epsilon else 0.0
    if ablation == "no_clip_no_dp":
        sigma_omega = 0.0  # tau forced to inf inside ByzClip21SGD2M too

    def grad_fn(x):
        grads = torch.zeros(n_pipeline_honest, d)
        for i in range(n_pipeline_honest):
            xb, yb = iters[i].next_batch()
            if is_flip_client[i]:
                yb = apply_label_flip(yb, num_classes)
            g, _ = flat_model.flat_grad(x, xb, yb, loss_fn)
            grads[i] = g
        return grads

    byzantine_fn = None
    if n_algo_byzantine > 0:
        byzantine_fn = lambda honest_c: ipm_byzantine_batch(honest_c, n_algo_byzantine, scale=-10.0)

    ragg_fn = lambda vectors, num_byz: apply_ragg(ragg_name, vectors, num_byz)

    algo = ByzClip21SGD2M(
        d=d, n_honest=n_pipeline_honest, n_byzantine=n_algo_byzantine,
        beta=beta, beta_hat=beta_hat, gamma=gamma, tau=tau, sigma_omega=sigma_omega,
        ragg_fn=ragg_fn, device=device, ablation=ablation,
    )
    x0 = flat_model.get_flat()
    algo.set_x(x0)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    eval_every = eval_every or max(1, T // 10)
    accuracy_trace = []
    loss_diverged = False

    for t in range(T):
        x = algo.step(grad_fn, byzantine_fn)
        if not torch.isfinite(x).all():
            loss_diverged = True
            break
        if (t + 1) % eval_every == 0 or t == T - 1:
            acc = flat_model.evaluate(x, test_loader)
            accuracy_trace.append({"round": t + 1, "test_acc": acc})

    final_acc = accuracy_trace[-1]["test_acc"] if accuracy_trace and not loss_diverged else 0.0

    return {
        "final_test_acc": final_acc,
        "accuracy_trace": accuracy_trace,
        "diverged": loss_diverged,
        "sigma_omega": sigma_omega,
        "config": {
            "n_regular": n_regular, "n_byzantine": n_byzantine, "attack_type": attack_type,
            "beta": beta, "beta_hat": beta_hat, "gamma": gamma, "tau": tau, "epsilon": epsilon,
            "ragg_name": ragg_name, "T": T, "batch_size": batch_size, "seed": seed,
            "ablation": ablation, "d": d,
        },
    }


def run_adaptive_experiment(
    model_ctor,
    train_dataset,
    test_dataset,
    n_regular,
    n_byzantine,
    attack_type,          # None | "ipm" | "label_flip"
    beta,
    beta_hat,
    gamma,
    T,
    batch_size,
    seed,
    tau_init=1.0,
    tau_min=0.01,
    tau_max=50.0,
    target_quantile=0.5,
    quantile_lr=0.2,
    tau_warmup_rounds=5,
    ragg_name="trimmed_mean",
    partition_fn=None,
    partition_kwargs=None,
    num_classes=10,
    device="cpu",
    eval_every=None,
):
    """Same harness as run_experiment, but driving AdaptiveClip21SGD2M instead of the
    fixed-tau ByzClip21SGD2M. No `epsilon`/DP argument: this variant is not privacy-
    accounted for a time-varying tau (see byz_clip21_sgd2m_adaptive module docstring),
    so it is only ever run with sigma_omega=0.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if attack_type == "label_flip":
        n_pipeline_honest = n_regular + n_byzantine
        n_algo_byzantine = 0
    else:
        n_pipeline_honest = n_regular
        n_algo_byzantine = n_byzantine if attack_type == "ipm" else 0

    n_total_clients = n_pipeline_honest + n_algo_byzantine
    partition_fn = partition_fn or partition_iid
    partition_kwargs = partition_kwargs or {}
    shards = partition_fn(train_dataset, n_total_clients, seed=seed, **partition_kwargs)
    loaders = make_client_loaders(train_dataset, shards, batch_size, seed=seed)
    iters = [InfiniteLoaderIter(loader) for loader in loaders]

    is_flip_client = [False] * n_pipeline_honest
    if attack_type == "label_flip":
        for i in range(n_regular, n_pipeline_honest):
            is_flip_client[i] = True

    model = model_ctor(num_classes=num_classes)
    flat_model = FlatModel(model, device)
    d = flat_model.d
    loss_fn = nn.CrossEntropyLoss()

    def grad_fn(x):
        grads = torch.zeros(n_pipeline_honest, d)
        for i in range(n_pipeline_honest):
            xb, yb = iters[i].next_batch()
            if is_flip_client[i]:
                yb = apply_label_flip(yb, num_classes)
            g, _ = flat_model.flat_grad(x, xb, yb, loss_fn)
            grads[i] = g
        return grads

    byzantine_fn = None
    if n_algo_byzantine > 0:
        byzantine_fn = lambda honest_c: ipm_byzantine_batch(honest_c, n_algo_byzantine, scale=-10.0)

    ragg_fn = lambda vectors, num_byz: apply_ragg(ragg_name, vectors, num_byz)

    algo = AdaptiveClip21SGD2M(
        d=d, n_honest=n_pipeline_honest, n_byzantine=n_algo_byzantine,
        beta=beta, beta_hat=beta_hat, gamma=gamma, sigma_omega=0.0,
        ragg_fn=ragg_fn, device=device,
        tau_init=tau_init, tau_min=tau_min, tau_max=tau_max,
        target_quantile=target_quantile, quantile_lr=quantile_lr,
        tau_warmup_rounds=tau_warmup_rounds,
    )
    x0 = flat_model.get_flat()
    algo.set_x(x0)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    eval_every = eval_every or max(1, T // 10)
    accuracy_trace = []
    tau_trace = []
    loss_diverged = False

    for t in range(T):
        x = algo.step(grad_fn, byzantine_fn)
        tau_trace.append(algo.tau)
        if not torch.isfinite(x).all():
            loss_diverged = True
            break
        if (t + 1) % eval_every == 0 or t == T - 1:
            acc = flat_model.evaluate(x, test_loader)
            accuracy_trace.append({"round": t + 1, "test_acc": acc})

    final_acc = accuracy_trace[-1]["test_acc"] if accuracy_trace and not loss_diverged else 0.0

    return {
        "final_test_acc": final_acc,
        "accuracy_trace": accuracy_trace,
        "diverged": loss_diverged,
        "final_tau": tau_trace[-1] if tau_trace else tau_init,
        "tau_trace_summary": {
            "min": min(tau_trace) if tau_trace else None,
            "max": max(tau_trace) if tau_trace else None,
        },
        "config": {
            "n_regular": n_regular, "n_byzantine": n_byzantine, "attack_type": attack_type,
            "beta": beta, "beta_hat": beta_hat, "gamma": gamma,
            "tau_init": tau_init, "tau_min": tau_min, "tau_max": tau_max,
            "target_quantile": target_quantile, "quantile_lr": quantile_lr,
            "tau_warmup_rounds": tau_warmup_rounds,
            "ragg_name": ragg_name, "T": T, "batch_size": batch_size, "seed": seed, "d": d,
        },
    }


def run_warmstart_experiment(
    model_ctor,
    train_dataset,
    test_dataset,
    n_regular,
    n_byzantine,
    attack_type,          # None | "ipm" | "label_flip"
    beta,
    beta_hat,
    gamma,
    tau,
    epsilon,              # None disables DP noise (sigma_omega=0)
    ragg_name,
    T,
    batch_size,
    seed,
    partition_fn=None,
    partition_kwargs=None,
    num_classes=10,
    device="cpu",
    eval_every=None,
    delta=1e-5,
):
    """Same harness as run_experiment, but driving WarmStartClip21SGD2M (fast-start
    beta/beta_hat schedule, target rates = beta/beta_hat) instead of the fixed-rate
    ByzClip21SGD2M. tau stays fixed throughout, so DP calibration is unaffected and
    epsilon works exactly as in run_experiment.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if attack_type == "label_flip":
        n_pipeline_honest = n_regular + n_byzantine
        n_algo_byzantine = 0
    else:
        n_pipeline_honest = n_regular
        n_algo_byzantine = n_byzantine if attack_type == "ipm" else 0

    n_total_clients = n_pipeline_honest + n_algo_byzantine
    partition_fn = partition_fn or partition_iid
    partition_kwargs = partition_kwargs or {}
    shards = partition_fn(train_dataset, n_total_clients, seed=seed, **partition_kwargs)
    loaders = make_client_loaders(train_dataset, shards, batch_size, seed=seed)
    iters = [InfiniteLoaderIter(loader) for loader in loaders]

    is_flip_client = [False] * n_pipeline_honest
    if attack_type == "label_flip":
        for i in range(n_regular, n_pipeline_honest):
            is_flip_client[i] = True

    model = model_ctor(num_classes=num_classes)
    flat_model = FlatModel(model, device)
    d = flat_model.d
    loss_fn = nn.CrossEntropyLoss()

    sigma_omega = compute_sigma_omega(tau if tau != float("inf") else 1.0, epsilon, T, delta) if epsilon else 0.0

    def grad_fn(x):
        grads = torch.zeros(n_pipeline_honest, d)
        for i in range(n_pipeline_honest):
            xb, yb = iters[i].next_batch()
            if is_flip_client[i]:
                yb = apply_label_flip(yb, num_classes)
            g, _ = flat_model.flat_grad(x, xb, yb, loss_fn)
            grads[i] = g
        return grads

    byzantine_fn = None
    if n_algo_byzantine > 0:
        byzantine_fn = lambda honest_c: ipm_byzantine_batch(honest_c, n_algo_byzantine, scale=-10.0)

    ragg_fn = lambda vectors, num_byz: apply_ragg(ragg_name, vectors, num_byz)

    algo = WarmStartClip21SGD2M(
        d=d, n_honest=n_pipeline_honest, n_byzantine=n_algo_byzantine,
        beta=beta, beta_hat=beta_hat, gamma=gamma, tau=tau, sigma_omega=sigma_omega,
        ragg_fn=ragg_fn, device=device,
    )
    x0 = flat_model.get_flat()
    algo.set_x(x0)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    eval_every = eval_every or max(1, T // 10)
    accuracy_trace = []
    loss_diverged = False

    for t in range(T):
        x = algo.step(grad_fn, byzantine_fn)
        if not torch.isfinite(x).all():
            loss_diverged = True
            break
        if (t + 1) % eval_every == 0 or t == T - 1:
            acc = flat_model.evaluate(x, test_loader)
            accuracy_trace.append({"round": t + 1, "test_acc": acc})

    final_acc = accuracy_trace[-1]["test_acc"] if accuracy_trace and not loss_diverged else 0.0

    return {
        "final_test_acc": final_acc,
        "accuracy_trace": accuracy_trace,
        "diverged": loss_diverged,
        "sigma_omega": sigma_omega,
        "config": {
            "n_regular": n_regular, "n_byzantine": n_byzantine, "attack_type": attack_type,
            "beta": beta, "beta_hat": beta_hat, "gamma": gamma, "tau": tau, "epsilon": epsilon,
            "ragg_name": ragg_name, "T": T, "batch_size": batch_size, "seed": seed, "d": d,
        },
    }


def run_bandclip_experiment(
    model_ctor,
    train_dataset,
    test_dataset,
    n_regular,
    n_byzantine,
    attack_type,          # None | "ipm" | "label_flip"
    beta,
    beta_hat,
    gamma,
    tau,
    tau_floor,
    epsilon,              # None disables DP noise (sigma_omega=0)
    ragg_name,
    T,
    batch_size,
    seed,
    partition_fn=None,
    partition_kwargs=None,
    num_classes=10,
    device="cpu",
    eval_every=None,
    delta=1e-5,
):
    """Same harness as run_experiment, but driving BandClip21SGD2M (floor-and-ceiling
    clip_band instead of ceiling-only clip_tau). tau stays fixed throughout, so DP
    calibration is unaffected and epsilon works exactly as in run_experiment.
    tau_floor=0.0 reduces this to the base algorithm exactly.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if attack_type == "label_flip":
        n_pipeline_honest = n_regular + n_byzantine
        n_algo_byzantine = 0
    else:
        n_pipeline_honest = n_regular
        n_algo_byzantine = n_byzantine if attack_type == "ipm" else 0

    n_total_clients = n_pipeline_honest + n_algo_byzantine
    partition_fn = partition_fn or partition_iid
    partition_kwargs = partition_kwargs or {}
    shards = partition_fn(train_dataset, n_total_clients, seed=seed, **partition_kwargs)
    loaders = make_client_loaders(train_dataset, shards, batch_size, seed=seed)
    iters = [InfiniteLoaderIter(loader) for loader in loaders]

    is_flip_client = [False] * n_pipeline_honest
    if attack_type == "label_flip":
        for i in range(n_regular, n_pipeline_honest):
            is_flip_client[i] = True

    model = model_ctor(num_classes=num_classes)
    flat_model = FlatModel(model, device)
    d = flat_model.d
    loss_fn = nn.CrossEntropyLoss()

    sigma_omega = compute_sigma_omega(tau if tau != float("inf") else 1.0, epsilon, T, delta) if epsilon else 0.0

    def grad_fn(x):
        grads = torch.zeros(n_pipeline_honest, d)
        for i in range(n_pipeline_honest):
            xb, yb = iters[i].next_batch()
            if is_flip_client[i]:
                yb = apply_label_flip(yb, num_classes)
            g, _ = flat_model.flat_grad(x, xb, yb, loss_fn)
            grads[i] = g
        return grads

    byzantine_fn = None
    if n_algo_byzantine > 0:
        byzantine_fn = lambda honest_c: ipm_byzantine_batch(honest_c, n_algo_byzantine, scale=-10.0)

    ragg_fn = lambda vectors, num_byz: apply_ragg(ragg_name, vectors, num_byz)

    algo = BandClip21SGD2M(
        d=d, n_honest=n_pipeline_honest, n_byzantine=n_algo_byzantine,
        beta=beta, beta_hat=beta_hat, gamma=gamma, tau=tau, sigma_omega=sigma_omega,
        ragg_fn=ragg_fn, device=device, tau_floor=tau_floor,
    )
    x0 = flat_model.get_flat()
    algo.set_x(x0)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    eval_every = eval_every or max(1, T // 10)
    accuracy_trace = []
    loss_diverged = False

    for t in range(T):
        x = algo.step(grad_fn, byzantine_fn)
        if not torch.isfinite(x).all():
            loss_diverged = True
            break
        if (t + 1) % eval_every == 0 or t == T - 1:
            acc = flat_model.evaluate(x, test_loader)
            accuracy_trace.append({"round": t + 1, "test_acc": acc})

    final_acc = accuracy_trace[-1]["test_acc"] if accuracy_trace and not loss_diverged else 0.0

    return {
        "final_test_acc": final_acc,
        "accuracy_trace": accuracy_trace,
        "diverged": loss_diverged,
        "sigma_omega": sigma_omega,
        "config": {
            "n_regular": n_regular, "n_byzantine": n_byzantine, "attack_type": attack_type,
            "beta": beta, "beta_hat": beta_hat, "gamma": gamma, "tau": tau,
            "tau_floor": tau_floor, "epsilon": epsilon,
            "ragg_name": ragg_name, "T": T, "batch_size": batch_size, "seed": seed, "d": d,
        },
    }


def run_localsteps_experiment(
    model_ctor,
    train_dataset,
    test_dataset,
    n_regular,
    n_byzantine,
    attack_type,          # None | "ipm" | "label_flip"
    beta,
    beta_hat,
    gamma,
    tau,
    tau_floor,
    epsilon,              # None disables DP noise (sigma_omega=0)
    ragg_name,
    T,
    batch_size,
    seed,
    local_steps=1,
    local_lr=0.01,
    partition_fn=None,
    partition_kwargs=None,
    num_classes=10,
    device="cpu",
    eval_every=None,
    delta=1e-5,
):
    """BandClip21SGD2M (floor-and-ceiling clip; tau_floor=0 recovers the base algorithm's
    ceiling-only clip_tau) driven by MULTI-STEP local pseudo-gradients instead of a
    single fresh minibatch gradient per client per round.

    Every other experiment in this codebase (run_experiment, vanilla_control_sweep.py,
    etc.) uses exactly local_steps=1 -- one minibatch, one gradient, per client, per
    communication round. That is the standard FedSGD limit, not FedAvg's actual
    mechanism (McMahan et al. 2017): FedAvg's whole point is to trade communication
    rounds for local computation by taking multiple local SGD steps before each
    communication round. This function does exactly that: each client runs `local_steps`
    consecutive SGD steps (fresh minibatch each step, learning rate `local_lr`) on a
    LOCAL copy of the current global model, and the resulting model displacement is
    converted to a pseudo-gradient the same way Reddi et al. 2020 ("Adaptive Federated
    Optimization") formalize a FedAvg server update as a generalized gradient step:
        pseudo_grad_i = (x - x_i_after_E_local_steps) / (local_steps * local_lr)
    This is a strict generalization of the existing single-step protocol: at
    local_steps=1, pseudo_grad_i reduces algebraically to exactly the fresh minibatch
    gradient run_experiment already computes (x_i_after_1_step = x - local_lr*grad, so
    (x - x_i_after_1_step)/(1*local_lr) = grad). The pseudo-gradient feeds into the same
    v_i/g_i EF21 double-momentum pipeline, clipped by clip_band, exactly as in
    BandClip21SGD2M -- no other mechanism changes.

    local_lr defaults to 0.01, the one learning rate this session's own
    centralized_sanity_check.py independently found stable for this exact architecture
    (SmallCNN on CIFAR-10): lr=0.1 diverged to chance over 10 epochs of plain centralized
    SGD on this un-batch-normalized net; lr=0.01 reached 65.2%. Reusing that
    already-validated value here rather than guessing a new one.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if attack_type == "label_flip":
        n_pipeline_honest = n_regular + n_byzantine
        n_algo_byzantine = 0
    else:
        n_pipeline_honest = n_regular
        n_algo_byzantine = n_byzantine if attack_type == "ipm" else 0

    n_total_clients = n_pipeline_honest + n_algo_byzantine
    partition_fn = partition_fn or partition_iid
    partition_kwargs = partition_kwargs or {}
    shards = partition_fn(train_dataset, n_total_clients, seed=seed, **partition_kwargs)
    loaders = make_client_loaders(train_dataset, shards, batch_size, seed=seed)
    iters = [InfiniteLoaderIter(loader) for loader in loaders]

    is_flip_client = [False] * n_pipeline_honest
    if attack_type == "label_flip":
        for i in range(n_regular, n_pipeline_honest):
            is_flip_client[i] = True

    model = model_ctor(num_classes=num_classes)
    flat_model = FlatModel(model, device)
    d = flat_model.d
    loss_fn = nn.CrossEntropyLoss()

    sigma_omega = compute_sigma_omega(tau if tau != float("inf") else 1.0, epsilon, T, delta) if epsilon else 0.0

    def grad_fn(x):
        pseudo_grads = torch.zeros(n_pipeline_honest, d)
        for i in range(n_pipeline_honest):
            x_local = x.clone()
            for _e in range(local_steps):
                xb, yb = iters[i].next_batch()
                yb_use = apply_label_flip(yb, num_classes) if is_flip_client[i] else yb
                g, _ = flat_model.flat_grad(x_local, xb, yb_use, loss_fn)
                x_local = x_local - local_lr * g
            pseudo_grads[i] = (x - x_local) / (local_steps * local_lr)
        return pseudo_grads

    byzantine_fn = None
    if n_algo_byzantine > 0:
        byzantine_fn = lambda honest_c: ipm_byzantine_batch(honest_c, n_algo_byzantine, scale=-10.0)

    ragg_fn = lambda vectors, num_byz: apply_ragg(ragg_name, vectors, num_byz)

    algo = BandClip21SGD2M(
        d=d, n_honest=n_pipeline_honest, n_byzantine=n_algo_byzantine,
        beta=beta, beta_hat=beta_hat, gamma=gamma, tau=tau, sigma_omega=sigma_omega,
        ragg_fn=ragg_fn, device=device, tau_floor=tau_floor,
    )
    x0 = flat_model.get_flat()
    algo.set_x(x0)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    eval_every = eval_every or max(1, T // 10)
    accuracy_trace = []
    loss_diverged = False

    for t in range(T):
        x = algo.step(grad_fn, byzantine_fn)
        if not torch.isfinite(x).all():
            loss_diverged = True
            break
        if (t + 1) % eval_every == 0 or t == T - 1:
            acc = flat_model.evaluate(x, test_loader)
            accuracy_trace.append({"round": t + 1, "test_acc": acc})

    final_acc = accuracy_trace[-1]["test_acc"] if accuracy_trace and not loss_diverged else 0.0

    return {
        "final_test_acc": final_acc,
        "accuracy_trace": accuracy_trace,
        "diverged": loss_diverged,
        "sigma_omega": sigma_omega,
        "config": {
            "n_regular": n_regular, "n_byzantine": n_byzantine, "attack_type": attack_type,
            "beta": beta, "beta_hat": beta_hat, "gamma": gamma, "tau": tau,
            "tau_floor": tau_floor, "epsilon": epsilon, "local_steps": local_steps,
            "local_lr": local_lr,
            "ragg_name": ragg_name, "T": T, "batch_size": batch_size, "seed": seed, "d": d,
        },
    }


def run_tailadaptive_experiment(
    model_ctor,
    train_dataset,
    test_dataset,
    n_regular,
    n_byzantine,
    attack_type,          # None | "ipm" | "label_flip"
    beta,
    beta_hat,
    gamma,
    T,
    batch_size,
    seed,
    mode="evt_quantile",
    tau_init=1.0,
    tau_min=1e-3,
    tau_max=100.0,
    schedule_c=1.0,
    q_target=0.05,
    window_rounds=10,
    n_byz_assumed=0,
    tail_warmup_rounds=10,
    ragg_name="trimmed_mean",
    partition_fn=None,
    partition_kwargs=None,
    num_classes=10,
    device="cpu",
    eval_every=None,
):
    """Drives TailAdaptiveClip21SGD2M (src/byz_clip21_sgd2m_tailadaptive.py). No epsilon/
    DP argument: not privacy-accounted for a time-varying tau, so sigma_omega=0 always.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if attack_type == "label_flip":
        n_pipeline_honest = n_regular + n_byzantine
        n_algo_byzantine = 0
    else:
        n_pipeline_honest = n_regular
        n_algo_byzantine = n_byzantine if attack_type == "ipm" else 0

    n_total_clients = n_pipeline_honest + n_algo_byzantine
    partition_fn = partition_fn or partition_iid
    partition_kwargs = partition_kwargs or {}
    shards = partition_fn(train_dataset, n_total_clients, seed=seed, **partition_kwargs)
    loaders = make_client_loaders(train_dataset, shards, batch_size, seed=seed)
    iters = [InfiniteLoaderIter(loader) for loader in loaders]

    is_flip_client = [False] * n_pipeline_honest
    if attack_type == "label_flip":
        for i in range(n_regular, n_pipeline_honest):
            is_flip_client[i] = True

    model = model_ctor(num_classes=num_classes)
    flat_model = FlatModel(model, device)
    d = flat_model.d
    loss_fn = nn.CrossEntropyLoss()

    def grad_fn(x):
        grads = torch.zeros(n_pipeline_honest, d)
        for i in range(n_pipeline_honest):
            xb, yb = iters[i].next_batch()
            if is_flip_client[i]:
                yb = apply_label_flip(yb, num_classes)
            g, _ = flat_model.flat_grad(x, xb, yb, loss_fn)
            grads[i] = g
        return grads

    byzantine_fn = None
    if n_algo_byzantine > 0:
        byzantine_fn = lambda honest_c: ipm_byzantine_batch(honest_c, n_algo_byzantine, scale=-10.0)

    ragg_fn = lambda vectors, num_byz: apply_ragg(ragg_name, vectors, num_byz)

    algo = TailAdaptiveClip21SGD2M(
        d=d, n_honest=n_pipeline_honest, n_byzantine=n_algo_byzantine,
        beta=beta, beta_hat=beta_hat, gamma=gamma, sigma_omega=0.0,
        ragg_fn=ragg_fn, device=device, mode=mode,
        tau_init=tau_init, tau_min=tau_min, tau_max=tau_max, schedule_c=schedule_c,
        q_target=q_target, window_rounds=window_rounds, n_byz_assumed=n_byz_assumed,
        tail_warmup_rounds=tail_warmup_rounds,
    )
    x0 = flat_model.get_flat()
    algo.set_x(x0)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    eval_every = eval_every or max(1, T // 10)
    accuracy_trace = []
    loss_diverged = False

    for t in range(T):
        x = algo.step(grad_fn, byzantine_fn)
        if not torch.isfinite(x).all():
            loss_diverged = True
            break
        if (t + 1) % eval_every == 0 or t == T - 1:
            acc = flat_model.evaluate(x, test_loader)
            accuracy_trace.append({"round": t + 1, "test_acc": acc})

    final_acc = accuracy_trace[-1]["test_acc"] if accuracy_trace and not loss_diverged else 0.0

    return {
        "final_test_acc": final_acc,
        "accuracy_trace": accuracy_trace,
        "diverged": loss_diverged,
        "final_tau": algo.tau_trace[-1] if algo.tau_trace else tau_init,
        "final_alpha_hat": algo.alpha_trace[-1] if algo.alpha_trace else None,
        "tau_trace_summary": {
            "min": min(algo.tau_trace) if algo.tau_trace else None,
            "max": max(algo.tau_trace) if algo.tau_trace else None,
        },
        "config": {
            "n_regular": n_regular, "n_byzantine": n_byzantine, "attack_type": attack_type,
            "beta": beta, "beta_hat": beta_hat, "gamma": gamma, "mode": mode,
            "tau_init": tau_init, "tau_min": tau_min, "tau_max": tau_max,
            "schedule_c": schedule_c, "q_target": q_target, "window_rounds": window_rounds,
            "n_byz_assumed": n_byz_assumed, "tail_warmup_rounds": tail_warmup_rounds,
            "ragg_name": ragg_name, "T": T, "batch_size": batch_size, "seed": seed, "d": d,
        },
    }


def run_evtfloor_experiment(
    model_ctor,
    train_dataset,
    test_dataset,
    n_regular,
    n_byzantine,
    attack_type,          # None | "ipm" | "label_flip"
    beta,
    beta_hat,
    gamma,
    tau,
    epsilon,              # None disables DP noise (sigma_omega=0)
    ragg_name,
    T,
    batch_size,
    seed,
    q_target=0.05,
    floor_min=0.0,
    floor_max=None,
    window_rounds=10,
    n_byz_assumed=0,
    floor_warmup_rounds=10,
    partition_fn=None,
    partition_kwargs=None,
    num_classes=10,
    device="cpu",
    eval_every=None,
    delta=1e-5,
):
    """Drives EVTFloorClip21SGD2M (src/byz_clip21_sgd2m_evtfloor.py): fixed ceiling tau
    (as in run_experiment), EVT-quantile-calibrated floor. tau is fixed throughout, so DP
    calibration is unaffected and epsilon works exactly as in run_experiment.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if attack_type == "label_flip":
        n_pipeline_honest = n_regular + n_byzantine
        n_algo_byzantine = 0
    else:
        n_pipeline_honest = n_regular
        n_algo_byzantine = n_byzantine if attack_type == "ipm" else 0

    n_total_clients = n_pipeline_honest + n_algo_byzantine
    partition_fn = partition_fn or partition_iid
    partition_kwargs = partition_kwargs or {}
    shards = partition_fn(train_dataset, n_total_clients, seed=seed, **partition_kwargs)
    loaders = make_client_loaders(train_dataset, shards, batch_size, seed=seed)
    iters = [InfiniteLoaderIter(loader) for loader in loaders]

    is_flip_client = [False] * n_pipeline_honest
    if attack_type == "label_flip":
        for i in range(n_regular, n_pipeline_honest):
            is_flip_client[i] = True

    model = model_ctor(num_classes=num_classes)
    flat_model = FlatModel(model, device)
    d = flat_model.d
    loss_fn = nn.CrossEntropyLoss()

    sigma_omega = compute_sigma_omega(tau if tau != float("inf") else 1.0, epsilon, T, delta) if epsilon else 0.0

    def grad_fn(x):
        grads = torch.zeros(n_pipeline_honest, d)
        for i in range(n_pipeline_honest):
            xb, yb = iters[i].next_batch()
            if is_flip_client[i]:
                yb = apply_label_flip(yb, num_classes)
            g, _ = flat_model.flat_grad(x, xb, yb, loss_fn)
            grads[i] = g
        return grads

    byzantine_fn = None
    if n_algo_byzantine > 0:
        byzantine_fn = lambda honest_c: ipm_byzantine_batch(honest_c, n_algo_byzantine, scale=-10.0)

    ragg_fn = lambda vectors, num_byz: apply_ragg(ragg_name, vectors, num_byz)

    algo = EVTFloorClip21SGD2M(
        d=d, n_honest=n_pipeline_honest, n_byzantine=n_algo_byzantine,
        beta=beta, beta_hat=beta_hat, gamma=gamma, tau=tau, sigma_omega=sigma_omega,
        ragg_fn=ragg_fn, device=device,
        q_target=q_target, floor_min=floor_min, floor_max=floor_max,
        window_rounds=window_rounds, n_byz_assumed=n_byz_assumed,
        floor_warmup_rounds=floor_warmup_rounds,
    )
    x0 = flat_model.get_flat()
    algo.set_x(x0)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    eval_every = eval_every or max(1, T // 10)
    accuracy_trace = []
    loss_diverged = False

    for t in range(T):
        x = algo.step(grad_fn, byzantine_fn)
        if not torch.isfinite(x).all():
            loss_diverged = True
            break
        if (t + 1) % eval_every == 0 or t == T - 1:
            acc = flat_model.evaluate(x, test_loader)
            accuracy_trace.append({"round": t + 1, "test_acc": acc})

    final_acc = accuracy_trace[-1]["test_acc"] if accuracy_trace and not loss_diverged else 0.0

    return {
        "final_test_acc": final_acc,
        "accuracy_trace": accuracy_trace,
        "diverged": loss_diverged,
        "sigma_omega": sigma_omega,
        "final_floor": algo.floor_trace[-1] if algo.floor_trace else 0.0,
        "final_alpha_hat": algo.alpha_trace[-1] if algo.alpha_trace else None,
        "config": {
            "n_regular": n_regular, "n_byzantine": n_byzantine, "attack_type": attack_type,
            "beta": beta, "beta_hat": beta_hat, "gamma": gamma, "tau": tau, "epsilon": epsilon,
            "q_target": q_target, "floor_min": floor_min, "floor_max": floor_max,
            "window_rounds": window_rounds, "n_byz_assumed": n_byz_assumed,
            "floor_warmup_rounds": floor_warmup_rounds,
            "ragg_name": ragg_name, "T": T, "batch_size": batch_size, "seed": seed, "d": d,
        },
    }


def run_baseline_experiment(
    algo_name,            # "byz_clip_sgd" | "safe_dshb"
    model_ctor,
    train_dataset,
    test_dataset,
    n_regular,
    n_byzantine,
    attack_type,          # None | "ipm" | "label_flip"
    gamma,
    tau,
    epsilon,              # None disables DP noise (sigma_omega=0)
    ragg_name,
    T,
    batch_size,
    seed,
    beta=0.1,             # Safe-DSHB's client-momentum, per the source paper Sec. 6:
                           # "For Byz-Clip21-SGD2M and Safe-DSHB, we fix ... beta=0.1".
                           # Ignored for Byz-Clip-SGD (it has no momentum parameter).
    partition_fn=None,
    partition_kwargs=None,
    num_classes=10,
    device="cpu",
    eval_every=None,
    delta=1e-5,
):
    """Run one full external-baseline (Byz-Clip-SGD or Safe-DSHB, arXiv:2603.23472
    Appendix Algorithm 3 / Algorithm 4) training run, under the identical harness
    (data partitioning, attack simulation, DP-noise calibration, evaluation) already
    used for Byz-Clip21-SGD2M in `run_experiment`, so results are directly comparable.
    """
    if algo_name not in ("byz_clip_sgd", "safe_dshb"):
        raise ValueError(f"Unknown algo_name '{algo_name}'")

    torch.manual_seed(seed)
    np.random.seed(seed)

    if attack_type == "label_flip":
        n_pipeline_honest = n_regular + n_byzantine
        n_algo_byzantine = 0
    else:
        n_pipeline_honest = n_regular
        n_algo_byzantine = n_byzantine if attack_type == "ipm" else 0

    n_total_clients = n_pipeline_honest + n_algo_byzantine
    partition_fn = partition_fn or partition_iid
    partition_kwargs = partition_kwargs or {}
    shards = partition_fn(train_dataset, n_total_clients, seed=seed, **partition_kwargs)
    loaders = make_client_loaders(train_dataset, shards, batch_size, seed=seed)
    iters = [InfiniteLoaderIter(loader) for loader in loaders]

    is_flip_client = [False] * n_pipeline_honest
    if attack_type == "label_flip":
        for i in range(n_regular, n_pipeline_honest):
            is_flip_client[i] = True

    model = model_ctor(num_classes=num_classes)
    flat_model = FlatModel(model, device)
    d = flat_model.d
    loss_fn = nn.CrossEntropyLoss()

    sigma_omega = compute_sigma_omega(tau if tau != float("inf") else 1.0, epsilon, T, delta) if epsilon else 0.0

    def grad_fn(x):
        grads = torch.zeros(n_pipeline_honest, d)
        for i in range(n_pipeline_honest):
            xb, yb = iters[i].next_batch()
            if is_flip_client[i]:
                yb = apply_label_flip(yb, num_classes)
            g, _ = flat_model.flat_grad(x, xb, yb, loss_fn)
            grads[i] = g
        return grads

    byzantine_fn = None
    if n_algo_byzantine > 0:
        byzantine_fn = lambda honest_c: ipm_byzantine_batch(honest_c, n_algo_byzantine, scale=-10.0)

    ragg_fn = lambda vectors, num_byz: apply_ragg(ragg_name, vectors, num_byz)

    if algo_name == "byz_clip_sgd":
        algo = ByzClipSGD(
            d=d, n_honest=n_pipeline_honest, n_byzantine=n_algo_byzantine,
            gamma=gamma, tau=tau, sigma_omega=sigma_omega, ragg_fn=ragg_fn, device=device,
        )
    else:
        algo = SafeDSHB(
            d=d, n_honest=n_pipeline_honest, n_byzantine=n_algo_byzantine,
            beta=beta, gamma=gamma, tau=tau, sigma_omega=sigma_omega, ragg_fn=ragg_fn, device=device,
        )

    x0 = flat_model.get_flat()
    algo.set_x(x0)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    eval_every = eval_every or max(1, T // 10)
    accuracy_trace = []
    loss_diverged = False

    for t in range(T):
        x = algo.step(grad_fn, byzantine_fn)
        if not torch.isfinite(x).all():
            loss_diverged = True
            break
        if (t + 1) % eval_every == 0 or t == T - 1:
            acc = flat_model.evaluate(x, test_loader)
            accuracy_trace.append({"round": t + 1, "test_acc": acc})

    final_acc = accuracy_trace[-1]["test_acc"] if accuracy_trace and not loss_diverged else 0.0

    return {
        "final_test_acc": final_acc,
        "accuracy_trace": accuracy_trace,
        "diverged": loss_diverged,
        "sigma_omega": sigma_omega,
        "config": {
            "algo_name": algo_name,
            "n_regular": n_regular, "n_byzantine": n_byzantine, "attack_type": attack_type,
            "beta": beta if algo_name == "safe_dshb" else None,
            "gamma": gamma, "tau": tau, "epsilon": epsilon,
            "ragg_name": ragg_name, "T": T, "batch_size": batch_size, "seed": seed, "d": d,
        },
    }
