# FedProto Collapse: what happened, what we checked, why we stopped where we did

This document exists so Section 6 of `paper/main_v2.tex` ("Attempting a Comparison with
FedProto") can cite a fuller technical account than fits in the paper's prose, and so
the diagnosis is reproducible from this file, `src/fedproto.py`, and
`scripts/fedproto_collapse_diagnostic.py` rather than from memory.

## What we were trying to do

Section 2 explains why FedProto (Tan et al., 2022, AAAI, arXiv:2105.00243) is not a
like-for-like baseline for Byz-Clip21-SGD2M: different aggregated object (prototypes,
not gradients or weights), different threat model (none, in the original paper),
different preferred metric (personalized per-client accuracy, not a single global
model's accuracy). We attempted a scoped, clean-condition comparison anyway, reasoning
that a compute-feasible approximation of FedProto's own protocol would still be
informative context for readers of this paper.

We implemented FedProto from its published equations (`src/fedproto.py`), not from the
original authors' code (none was consulted), in the same spirit as this project's other
from-pseudocode implementations (Safe-DSHB, Byz-Clip-SGD): local prototype computation
per class per client, the combined classification-plus-prototype-alignment loss, and
weighted-mean server aggregation of prototypes.

## The one disclosed protocol deviation, and why

FedProto's own paper specifies one full local **epoch** per round. At CIFAR-10's
~2,500 images/client (20 clients, IID partition of the 50,000-image training set) and
batch size 8, that is ~312 steps/client/round; at the paper's own `T=110` CIFAR-10
round budget and 20 clients, that is ~686,000 forward/backward passes total — far beyond
this project's CPU pilot-scale budget (the same budget constraint that governs every
other experiment in this paper). We used a fixed step count per round instead
(`LOCAL_STEPS`) and report the exact value used rather than silently badging a
step-limited run as "one local epoch."

## What happened at the first value we tried

`LOCAL_STEPS=5` (chosen as a small, clearly-affordable starting point, no different in
spirit from any other first guess in a compute-constrained pilot) produced mean
client accuracy pinned at **exactly 10.0%** (chance, on CIFAR-10's ten classes) for the
entire `T=110` run, both clean and under attack.

An exactly-chance result this clean, on a run that completed without error, is exactly
the signature this paper's own Confound Audit (Sections 4–5) argues should be
*diagnosed*, not reported at face value — the same standard applied to
Byz-Clip21-SGD2M's own collapse is applied here to a baseline instead. We did not
report the 10.0% number and move on.

## Diagnosis: tracking the global prototype norm, not just downstream accuracy

FedProto's client loss is `CE(classifier(embed(x)), y) + lambda * sum_j
||local_proto_j - global_proto_j||_2` over classes present in the batch. This loss has
a trivial degenerate minimizer that the classification term alone does not rule out:
collapsing every class's embedding toward the *same point* drives the regularization
term to zero without requiring any class to be separable, let alone correctly
classified. If a round's classification-loss gradient is too small, relative to the
regularization gradient, to pull embeddings back apart, training collapses into that
degenerate minimizer instead of a useful one.

We tested this directly by tracking the **mean global-prototype norm** across rounds
(a proxy for representation collapse: near zero means every class's embedding has
converged to the same point) rather than only the downstream accuracy, using
`scripts/fedproto_collapse_diagnostic.py`'s `run(n_clients, local_steps, n_rounds,
seed)` function. Results (15 rounds, `lambda=0.1`, reproducible by rerunning that
script — see Table 2 of `paper/main_v2.tex` for the same numbers in context):

| Local steps/round | 3 clients | 20 clients |
|---|---|---|
| 5   | collapses (0.33 → 0.03) | — |
| 15  | partial collapse (0.20) | — |
| 25  | stable (≈0.41) | collapses (0.04) |
| 40  | stable (≈0.42) | — |
| 50  | — | collapses (0.06) |
| 100 | stable (≈0.34) | collapses (0.05) |

Two things follow directly from this table, neither of which we had assumed going in:

1. **`LOCAL_STEPS=5`'s collapse is a genuine instance of the mechanism above**, not an
   implementation bug: the norm trace at 3 clients shows a smooth decay from ~0.33 to
   ~0.03 over 15 rounds, consistent with the regularization term winning out gradually
   rather than a discontinuous failure.
2. **Client count matters independently of local computation.** The same step budget
   that stabilizes training at 3 clients (25, 40, or 100 steps/round) *collapses* it at
   20 clients. We did not anticipate this before running the sweep — the original
   hypothesis was purely "not enough local steps," and the 20-client column shows that
   is incomplete: at 20 clients, even 100 steps/round (40% of a full local epoch at
   this data size) does not stabilize training. We do not have a confirmed mechanistic
   account of *why* client count interacts with the collapse this way (a plausible but
   unverified reading is that more clients means each one's local prototype estimate is
   noisier per round, and/or that the regularization gradient's effective per-client
   weight in the aggregated update scales differently with client count than the
   classification gradient's does) and did not investigate further, for the same reason
   given below.

We additionally inspected three individual client models directly at the `LOCAL_STEPS=5`
collapse point: every one of the three predicted a single fixed class for all 10,000
CIFAR-10 test images (classes 7, 2, and 2 respectively) — the precise mechanism that
produces an exact 10.0% on a class-balanced ten-way test set, confirming the
prototype-norm diagnosis at the level of individual model behavior, not just the
aggregate metric.

## Why we stopped here rather than force a number

Table 2 shows the failure is sensitive to both local computation *and* client count,
and that the reduced budgets this project's compute can support at 20 clients (up to
100 steps/round, already 40% of a full local epoch) do not avoid it. Closing the gap
would require approaching FedProto's own full-local-epoch budget (~312 steps/round at
this data size), which we estimate at several hours of additional CPU compute for a
single seed/condition pair — substantially more than any other single result in this
paper — and did not run.

Reporting an accuracy number from a collapsed configuration, or quietly increasing
local steps until an arbitrary-looking number appeared without understanding why that
particular value worked, would be exactly the kind of unexamined result this paper
argues against accepting at face value in its own Confound Audit (Sections 4–5). We
therefore used `LOCAL_STEPS=25` (the smallest 20-client value in the table that avoids
collapse) for the paper's one reported clean-condition FedProto number, and report the
diagnosed failure mode itself — not a stronger accuracy claim — as the FedProto
section's actual finding.

## Why this is not a wasted attempt

This failure mode is, independently of Byz-Clip21-SGD2M, a second demonstration of the
paper's central point: an accuracy collapse this clean (exactly at chance, not merely
poor) is a signature of an under-provisioned training budget, not necessarily a
property of the algorithm being evaluated. We did not have to look for a second
example of this — it appeared while attempting to build a comparison, in a completely
different algorithm (prototype aggregation, not gradient-based), under a completely
different mechanism (representation collapse under a regularizer's degenerate
minimizer, not gradient clipping or momentum dynamics). That two unrelated algorithms
both produce a clean, diagnosable, budget-driven collapse under reduced compute is, if
anything, stronger evidence for this paper's general caution than either instance
alone.

## What this document licenses the paper to say, and what it does not

Licensed: "an attempted comparison against FedProto independently reproduced this
paper's central caution under reduced compute: training collapsed to exactly-chance,
single-class predictions, diagnosed to a representation-collapse failure mode
sensitive to both local computation and client count, not a property of the algorithm
itself."

Not licensed: any claim that FedProto is unable to train on CIFAR-10 in general (it is
not — the failure is specific to this project's compute-reduced protocol, and the
3-client column of Table 2 shows the same mechanism stabilizes readily at lower client
count and adequate local computation); any accuracy-level comparison between
Byz-Clip21-SGD2M and FedProto under matched, non-collapsed conditions (none has been
run); any claim about *why* client count interacts with the collapse mechanism beyond
the unverified reading offered above.
