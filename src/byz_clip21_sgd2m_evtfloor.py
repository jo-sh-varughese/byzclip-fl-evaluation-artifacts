"""EVT-Floor Clip21-SGD2M: applies the EVT-quantile formula from
byz_clip21_sgd2m_tailadaptive.py to the FLOOR of clip_band, not the ceiling.

Why the floor and not the ceiling: this session tested FOUR independent ceiling
mechanisms on raw-pixel CIFAR-10 -- the base algorithm's fixed tau=1.0, adaptive
quantile-tracking (byz_clip21_sgd2m_adaptive.py), a growing moment-bound schedule, and
an EVT-quantile-calibrated ceiling (both in byz_clip21_sgd2m_tailadaptive.py) -- and all
four produced accuracy statistically indistinguishable from each other (results/
adaptive_clip_check, results/tailadaptive_check, results/evt_quantile_pilot). The ONE
mechanism that showed a real, if modest and not-yet-significant, positive signal all
session was the FLOOR half of band-clipping (results/bandclip_scaleup: clean
17.1%->19.3% p=0.105, ipm 16.3%->18.0% p=0.064, hand-tuned floor=0.8) -- boosting
undersized updates, not capping oversized ones. This module keeps that mechanism (it is
the one with actual empirical support) and replaces its hand-tuned constant with the
same EVT-quantile derivation used for the (unhelpful) ceiling attempt:
    tau_floor_t = sigma_hat_t * q_target^(-1/alpha_hat_t)
computed online, per round, from the same Byzantine-robust tail-index estimator
(src/tail_index_estimator.py). The ceiling stays FIXED at the base algorithm's own tau
(no adaptive-ceiling mechanism tested this session showed any effect, so there is no
motivation to make it adaptive too -- see docs/TAIL_ADAPTIVE_CLIPPING_THEORY.md).

This is the version of "tail-index-calibrated clipping" this session's own evidence
actually supports testing at scale -- not a first guess kept for its own sake.
"""

import torch

from byz_clip21_sgd2m_bandclip import clip_band
from byz_clip21_sgd2m_tailadaptive import evt_quantile_tau
from tail_index_estimator import OnlineTailEstimator


class EVTFloorClip21SGD2M:
    def __init__(self, d, n_honest, n_byzantine, beta, beta_hat, gamma, tau, sigma_omega,
                 ragg_fn, device="cpu",
                 q_target=0.05, floor_min=0.0, floor_max=None,
                 window_rounds=10, n_byz_assumed=0, floor_warmup_rounds=10):
        """
        Args: as ByzClip21SGD2M, plus:
            q_target: evt_quantile_tau's target exceedance probability, now applied to
                the floor -- smaller q_target -> a larger, more generous floor.
            floor_min: hard lower clamp on the floor (0.0 = can shrink to "no floor",
                recovering the base algorithm's plain clip_tau exactly).
            floor_max: hard upper clamp on the floor; defaults to `tau` itself (the
                floor can never exceed the fixed ceiling -- clip_band already handles
                floor > ceiling as an edge case, but capping here keeps the two bounds
                meaningfully distinct by construction).
            window_rounds, n_byz_assumed: OnlineTailEstimator's parameters.
            floor_warmup_rounds: rounds run with floor=0 before the estimator takes over.
        """
        self.d = d
        self.n_honest = n_honest
        self.n_byzantine = n_byzantine
        self.n = n_honest + n_byzantine
        self.beta = beta
        self.beta_hat = beta_hat
        self.gamma = gamma
        self.tau = tau  # fixed ceiling
        self.sigma_omega = sigma_omega
        self.ragg_fn = ragg_fn
        self.device = device

        self.tau_floor = 0.0
        self.floor_min = floor_min
        self.floor_max = floor_max if floor_max is not None else tau
        self.q_target = q_target
        self.floor_warmup_rounds = floor_warmup_rounds
        self.estimator = OnlineTailEstimator(window_rounds=window_rounds, n_byz_assumed=n_byz_assumed)

        self.x = torch.zeros(d, device=device)
        self.v = torch.zeros(n_honest, d, device=device)
        self.g_local = torch.zeros(n_honest, d, device=device)
        self.m = torch.zeros(self.n, d, device=device)
        self.g_global = torch.zeros(d, device=device)
        self._round = 0
        self.floor_trace = []
        self.alpha_trace = []

    def set_x(self, x0):
        self.x = x0.clone().to(self.device)

    def step(self, grad_fn, byzantine_fn=None):
        self.x = self.x - self.gamma * self.g_global

        grads = grad_fn(self.x)
        assert grads.shape == (self.n_honest, self.d)

        self.v = (1 - self.beta) * self.v + self.beta * grads
        diff = self.v - self.g_local
        diff_norms = diff.norm(dim=1)

        clipped_diff = clip_band(diff, self.tau_floor, self.tau)

        if self.sigma_omega > 0:
            omega = torch.randn(self.n_honest, self.d, device=self.device) * self.sigma_omega
        else:
            omega = torch.zeros(self.n_honest, self.d, device=self.device)

        c_honest = clipped_diff + omega
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

        self.estimator.update(diff_norms.detach().cpu().numpy())
        alpha_hat, sigma_hat = self.estimator.get_estimate()
        self.alpha_trace.append(alpha_hat)
        if self._round >= self.floor_warmup_rounds and alpha_hat is not None:
            self.tau_floor = evt_quantile_tau(alpha_hat, sigma_hat, q_target=self.q_target,
                                               tau_min=self.floor_min, tau_max=self.floor_max)
        self.floor_trace.append(self.tau_floor)
        self._round += 1

        return self.x
