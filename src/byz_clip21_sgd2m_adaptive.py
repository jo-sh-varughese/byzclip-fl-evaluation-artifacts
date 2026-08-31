"""Adaptive-threshold variant of Byz-Clip21-SGD2M (see byz_clip21_sgd2m.py for the base
algorithm and citation).

Motivation, grounded in this repository's own diagnostics, not asserted a priori:
Byz-Clip21-SGD2M uses ONE FIXED clipping threshold tau for the entire run, hand-tuned per
dataset. This session's confound_check sweep (results/confound_check/) ruled out
learning-rate mistransfer and missing normalization as explanations for CIFAR-10's
collapse relative to MNIST under this exact algorithm, holding tau=1.0 fixed throughout.
The paper's own measured Hill/kurtosis statistics (paper/main_preprint.tex, H1) show
CIFAR-10's gradient-noise scale is heavier-tailed than MNIST's. A heavier tail also means
the scale of ||v_i - g_i|| is less stationary round to round -- a single fixed tau, tuned
on MNIST's lighter-tailed, more stationary distribution, has no mechanism to track a scale
that moves during training: it is either too tight early (discarding most of the true
signal and biasing the EF21 reference state g_i), or too loose later (admitting the very
tail events the clipping exists to bound).

AC21 replaces the fixed tau with the quantile-tracking mechanism of Andrew et al. 2021
("Differentially Private Learning with Adaptive Clipping"), adapted to this algorithm's
EF21/double-momentum structure: each round, every HONEST client's diff v_i - g_i either
does or does not exceed the current threshold tau_t (a single bit b_i in {0,1}); tau moves
geometrically toward whatever value keeps the fraction of honest clients exceeding it near
a target quantile q:
    tau_{t+1} = clip( tau_t * exp(-eta * (frac_exceed_t - q)), tau_min, tau_max )
This tracks the CURRENT scale of the diff distribution directly. An earlier version of
this file tried to proxy that scale from the norm of the algorithm's accumulated
g_global direction; that direction lives at a much smaller, differently-scaled magnitude
(it is damped by beta_hat and gamma), so the threshold collapsed to its floor and clipped
away nearly all signal (measured: instant collapse to chance, tau pinned at tau_min within
5 rounds). The quantile-tracking form is scale-free and is what the design comment above
should have used from the start -- documented here, not silently fixed, per this
project's own norm of writing measured negative results into the code that found them.

Causal ordering: tau_t is fixed for the ENTIRE duration of round t (used to clip round
t's diff); only AFTER round t's clip decision is finalized does that round's exceed
fraction fold into tau_{t+1}. tau_t never depends on information computed during round t.

Byzantine-robustness scope: Byzantine clients never contribute an exceed bit, because
(per federated_experiment.py's own module docstring) IPM-style attackers bypass the
honest v_i/g_i pipeline entirely and inject an arbitrary c_i directly -- they have no
diff to compare against tau in the first place. So the exceed-fraction statistic is
computed only over honest clients, and needs no separate robustness argument beyond the
base algorithm's own honest/Byzantine separation.

Scoped limitation, stated plainly: joint compatibility with DP is NOT addressed here. A
time-varying tau_t requires re-deriving federated_experiment.compute_sigma_omega's fixed-
tau privacy accounting (Andrew et al. 2021 spend a small additional privacy budget on the
exceed-bit query itself for exactly this reason). This pilot runs AC21 with sigma_omega=0
only, and the constructor raises if given sigma_omega > 0 so this cannot be silently
violated.
"""

import math

import torch

from byz_clip21_sgd2m import clip_tau


class AdaptiveClip21SGD2M:
    def __init__(self, d, n_honest, n_byzantine, beta, beta_hat, gamma, sigma_omega,
                 ragg_fn, device="cpu",
                 tau_init=1.0, tau_min=0.01, tau_max=50.0,
                 target_quantile=0.5, quantile_lr=0.2, tau_warmup_rounds=5):
        """
        Args:
            d, n_honest, n_byzantine, beta, beta_hat, gamma, ragg_fn, device: as in
                ByzClip21SGD2M.
            sigma_omega: must be 0 (raises otherwise -- see module docstring).
            tau_init: fixed threshold for the first `tau_warmup_rounds` rounds.
            tau_min, tau_max: hard clamp on the adaptive threshold.
            target_quantile: the exceed-fraction tau is steered toward (0.5 = track the
                median honest diff norm).
            quantile_lr: step size of the geometric update (Andrew et al. 2021 use a
                similar-order learning rate for the analogous DP-SGD mechanism).
            tau_warmup_rounds: rounds run at tau_init before adaptation begins, so the
                EF21 state has some history before the threshold starts moving.
        """
        if sigma_omega and sigma_omega > 0:
            raise ValueError(
                "AdaptiveClip21SGD2M's tau_t is not privacy-accounted for a fixed-T DP "
                "budget; run with sigma_omega=0. See module docstring."
            )
        self.d = d
        self.n_honest = n_honest
        self.n_byzantine = n_byzantine
        self.n = n_honest + n_byzantine
        self.beta = beta
        self.beta_hat = beta_hat
        self.gamma = gamma
        self.sigma_omega = sigma_omega
        self.ragg_fn = ragg_fn
        self.device = device

        self.tau = tau_init
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.target_quantile = target_quantile
        self.quantile_lr = quantile_lr
        self.tau_warmup_rounds = tau_warmup_rounds

        self.x = torch.zeros(d, device=device)
        self.v = torch.zeros(n_honest, d, device=device)
        self.g_local = torch.zeros(n_honest, d, device=device)
        self.m = torch.zeros(self.n, d, device=device)
        self.g_global = torch.zeros(d, device=device)
        self._round = 0
        self.tau_trace = []

    def set_x(self, x0):
        self.x = x0.clone().to(self.device)

    def step(self, grad_fn, byzantine_fn=None):
        self.x = self.x - self.gamma * self.g_global

        grads = grad_fn(self.x)
        assert grads.shape == (self.n_honest, self.d)

        self.v = (1 - self.beta) * self.v + self.beta * grads
        diff = self.v - self.g_local
        diff_norms = diff.norm(dim=1)

        clipped_diff = clip_tau(diff, self.tau)
        c_honest = clipped_diff  # sigma_omega == 0 enforced in __init__, so no DP noise term
        self.g_local = self.g_local + self.beta_hat * clipped_diff

        if self.n_byzantine > 0:
            assert byzantine_fn is not None, "byzantine_fn required when n_byzantine > 0"
            c_byz = byzantine_fn(c_honest)
            assert c_byz.shape == (self.n_byzantine, self.d)
            c_all = torch.cat([c_honest, c_byz], dim=0)
        else:
            c_all = c_honest

        self.m = self.m + self.beta_hat * c_all
        self.g_global = self.ragg_fn(self.m, self.n_byzantine)

        if self._round >= self.tau_warmup_rounds:
            frac_exceed = (diff_norms > self.tau).float().mean().item()
            self.tau = float(min(max(
                self.tau * math.exp(-self.quantile_lr * (frac_exceed - self.target_quantile)),
                self.tau_min), self.tau_max))
        self.tau_trace.append(self.tau)
        self._round += 1

        return self.x
