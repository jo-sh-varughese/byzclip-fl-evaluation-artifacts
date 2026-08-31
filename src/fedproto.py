"""FedProto (Tan et al. 2022, AAAI, arXiv:2105.00243) -- reference implementation from
the paper's own algorithm description, not from the original authors' code (none was
consulted; this is an independent implementation from the published equations, in the
same spirit as this project's Safe-DSHB/Byz-Clip-SGD implementations from pseudocode).

Algorithm, following the paper's own notation:
  - Each client i keeps its OWN full local model (embedding f_i + classifier), never a
    single shared model -- FedProto aggregates PROTOTYPES, not gradients or weights.
  - Local prototype for class j on client i: C_i^(j) = mean over client i's class-j
    samples of f_i(x) (the embedding, from model.embed(x)).
  - Client loss: L = CE(classifier(embed(x)), y) + lambda * sum_j d(C_i^(j), Cbar^(j))
    (L2 distance, summed over classes present in the current batch), where Cbar^(j) is
    the GLOBAL prototype from the previous round (skipped for a class with no global
    prototype yet, i.e. round 0).
  - Server aggregation: Cbar^(j) = weighted mean over clients holding class j of their
    C_i^(j), weighted by each client's class-j sample count.

No Byzantine-robustness or DP mechanism in the original paper. This module adds an
OPTIONAL, clearly-separate Byzantine stress test (not part of FedProto itself): a
coalition of Byzantine clients can report an IPM-crafted prototype
(src/attacks.py:ipm_byzantine_batch) for every class instead of a genuine one, applying
the same omniscient-coalition attack model used throughout the rest of this project,
since FedProto's plain weighted-mean aggregation has no defense against this by design.
"""

import numpy as np
import torch
import torch.nn as nn

from attacks import ipm_byzantine_batch


def compute_local_prototypes(model, xb, yb, num_classes, device="cpu"):
    """One batch's per-class prototype contributions: returns (sums, counts), both
    length-num_classes lists of (embed_dim,) tensors / ints, to be accumulated across
    batches by the caller (this function does not average -- averaging must happen
    after seeing all of a client's data for the round, not per batch).
    """
    with torch.no_grad():
        z = model.embed(xb.to(device))
    embed_dim = z.shape[1]
    sums = [torch.zeros(embed_dim) for _ in range(num_classes)]
    counts = [0 for _ in range(num_classes)]
    for c in range(num_classes):
        mask = (yb == c)
        if mask.any():
            sums[c] += z[mask].sum(dim=0)
            counts[c] += int(mask.sum().item())
    return sums, counts


def fedproto_local_update(model, loader, global_protos, num_classes, lam, lr,
                           local_steps, device="cpu"):
    """Runs `local_steps` SGD steps (fresh batch each step, matching this project's
    step-budgeted convention rather than a full local epoch -- see the calling script
    for why full local epochs are not used at this compute scale) with the combined
    classification + prototype-regularization loss, then returns this client's own
    end-of-round local prototypes (recomputed from the updated model over the same
    batches actually seen this round, accumulated on the fly).

    global_protos: dict {class_idx: (embed_dim,) tensor} of the CURRENT global
        prototypes (empty dict on round 0).
    Returns: (proto_sums, proto_counts) as in compute_local_prototypes, accumulated
        across all local_steps batches.
    """
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    ce = nn.CrossEntropyLoss()
    embed_dim = None
    proto_sums, proto_counts = None, None

    for _ in range(local_steps):
        xb, yb = loader()
        xb, yb = xb.to(device), yb.to(device)

        z = model.embed(xb)
        if embed_dim is None:
            embed_dim = z.shape[1]
            proto_sums = [torch.zeros(embed_dim) for _ in range(num_classes)]
            proto_counts = [0 for _ in range(num_classes)]
        logits = model.classify(z)
        loss = ce(logits, yb)

        if global_protos:
            reg = torch.zeros((), device=device)
            present_classes = torch.unique(yb).tolist()
            for c in present_classes:
                if c in global_protos:
                    mask = (yb == c)
                    local_c_proto = z[mask].mean(dim=0)
                    reg = reg + torch.norm(local_c_proto - global_protos[c].to(device), p=2)
            loss = loss + lam * reg

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            z_detached = z.detach()
        for c in range(num_classes):
            mask = (yb == c)
            if mask.any():
                proto_sums[c] += z_detached[mask].sum(dim=0)
                proto_counts[c] += int(mask.sum().item())

    return proto_sums, proto_counts


def aggregate_prototypes(all_proto_sums, all_proto_counts, num_classes):
    """all_proto_sums/counts: lists (one per client) of length-num_classes
    sums/counts, as returned by fedproto_local_update. Weighted mean per class over
    clients that hold at least one sample of that class.
    """
    global_protos = {}
    for c in range(num_classes):
        total_count = sum(counts[c] for counts in all_proto_counts)
        if total_count > 0:
            total_sum = sum(sums[c] for sums in all_proto_sums)
            global_protos[c] = total_sum / total_count
    return global_protos


def byzantine_prototype_attack(honest_proto_sums, honest_proto_counts, num_classes,
                                n_byzantine, scale=-10.0, nominal_count=50):
    """For each class, craft an IPM-style adversarial prototype from the honest
    clients' own per-class MEAN prototypes (sum/count) for that class, and return
    n_byzantine identical copies (the omniscient-coalition assumption already used for
    IPM elsewhere in this project) with a nominal per-class sample count (Byzantine
    clients can freely lie about their own data statistics, so an arbitrary but fixed
    weight is as legitimate a modeling choice as any specific one).
    """
    byz_sums = [torch.zeros_like(honest_proto_sums[0][c]) for c in range(num_classes)]
    byz_counts = [0 for _ in range(num_classes)]
    for c in range(num_classes):
        total_count = sum(counts[c] for counts in honest_proto_counts)
        if total_count == 0:
            continue
        honest_mean = sum(sums[c] for sums in honest_proto_sums) / total_count
        crafted = ipm_byzantine_batch(honest_mean.unsqueeze(0), n_byzantine, scale=scale)
        byz_sums[c] = crafted.sum(dim=0) * nominal_count / max(n_byzantine, 1)
        byz_counts[c] = nominal_count
    return [byz_sums] * n_byzantine, [byz_counts] * n_byzantine
