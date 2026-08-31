"""Tail-Adaptive Clip21-SGD2M: a clipping threshold derived from an online,
Byzantine-robust estimate of the gradient-noise tail index, replacing Byz-Clip21-SGD2M's
single hand-tuned constant tau. See docs/TAIL_ADAPTIVE_CLIPPING_THEORY.md for the full
account, including a documented design failure this file's history records rather than
hides (this project's own convention -- see byz_clip21_sgd2m_adaptive.py's docstring for
the same pattern with an earlier mechanism).

Two threshold modes, both implemented, `mode="evt_quantile"` the one actually recommended:

`mode="moment_schedule"` -- the FIRST design tried, kept for the historical record: a
literal instantiation of the growing-clip-threshold prescription from bounded-p-th-moment
heavy-tailed-SGD theory (arXiv:2410.16561 Thm. 3, h ~ T^(2p/(3p-2)) sigma^(3p/(3p-2)),
p in (1,2]). MEASURED THIS SESSION to be behaviorally a no-op: every Hill-alpha estimate
observed in this codebase (MNIST ~4.5-11, CIFAR-10 ~3.2-8.3, both clean and under IPM)
sits above 2, so mapping alpha into the theorem's required p in (1,2] range saturates the
clamp at the SAME boundary (p=2) on every round regardless of the actual measured value
-- there is no real adaptivity left once that happens. Confirmed empirically:
results/tailadaptive_check and results/mnist_tailadaptive_check both show accuracy
IDENTICAL to the fixed-tau baseline to 3-4 decimal places, Wilcoxon p=1.000 on every
condition -- not a small effect, a provable no-op, because tau grows past the honest
diff-norm ceiling (already measured at ~0.04-0.16 for CIFAR-10) within the first few
post-warmup rounds and never returns.

`mode="evt_quantile"` -- the corrected design, grounded in the SAME extreme-value theory
that the Hill estimator itself comes from (not imported from a differently-scoped, non-
matching theorem): for a regularly-varying right tail with index alpha,
P(X > x) ~ (x_ref / x)^alpha, so the value x achieving a target exceedance probability
q, relative to a reference scale x_ref, is x = x_ref * q^(-1/alpha). Using the online
robust scale estimate sigma_hat as x_ref:
    tau_t = sigma_hat_t * q_target ^ (-1 / alpha_hat_t)
This uses the FULL measured range of alpha (no saturating clamp -- alpha enters as a
smooth exponent, not a value compared against an unrelated theorem's assumption), and,
critically, ties tau to the CURRENT robust scale each round rather than letting it grow
unboundedly with the round count. That distinction matters for a reason specific to the
Byzantine setting and not addressed by the single-machine literature this design started
from: a threshold that grows without bound eventually admits an adversarial contribution
of ANY magnitude, at which point Byz-Clip21-SGD2M's own Byzantine-robustness argument
(which relies on tau bounding any single client's contribution) no longer holds. Tying
tau_t to sigma_hat_t -- a robust statistic of the CURRENT honest distribution, not an
ever-increasing function of elapsed rounds -- keeps the threshold's Byzantine-robustness
role intact by construction, at the cost of not literally reproducing the moment-bound
theorem's asymptotic guarantee. See docs/TAIL_ADAPTIVE_CLIPPING_THEORY.md for exactly
what this does and does not prove.
"""

import torch

from byz_clip21_sgd2m import clip_tau
from tail_index_estimator import OnlineTailEstimator


def schedule_tau_moment(round_t, alpha_hat, sigma_hat, c=1.0, tau_min=1e-3, tau_max=100.0):
    """h ~ c * (t+1)^(2p/(3p-2)) * sigma^(3p/(3p-2)), p = clamp(alpha_hat, 1+eps, 2).
    Kept for the historical record -- see module docstring: measured to be a no-op given
    this codebase's actual alpha_hat range (always > 2, saturating the clamp).
    """
    p = min(max(alpha_hat, 1.0 + 1e-3), 2.0)
    t_exp = 2 * p / (3 * p - 2)
    sigma_exp = 3 * p / (3 * p - 2)
    tau = c * ((round_t + 1) ** t_exp) * (max(sigma_hat, 1e-8) ** sigma_exp)
    return float(min(max(tau, tau_min), tau_max))


def evt_quantile_tau(alpha_hat, sigma_hat, q_target=0.05, tau_min=1e-3, tau_max=100.0):
    """tau = sigma_hat * q_target^(-1/alpha_hat) -- the Pareto-tail quantile implied by
    the measured tail index, relative to the current robust scale. q_target is the one
    interpretable design parameter: "treat anything past the top q_target fraction, under
    a regularly-varying tail with the measured index, as the kind of event clipping
    should guard against." Smaller alpha_hat (heavier measured tail) -> larger multiplier
    on sigma_hat, matching the standard EVT intuition that heavier tails need a more
    generous threshold relative to their own typical scale.
    """
    alpha = max(alpha_hat, 0.5)  # guard against a pathological near-zero estimate
    k = q_target ** (-1.0 / alpha)
    tau = sigma_hat * k
    return float(min(max(tau, tau_min), tau_max))


class TailAdaptiveClip21SGD2M:
    def __init__(self, d, n_honest, n_byzantine, beta, beta_hat, gamma, sigma_omega,
                 ragg_fn, device="cpu",
                 mode="evt_quantile",
                 tau_init=1.0, tau_min=1e-3, tau_max=100.0,
                 schedule_c=1.0, q_target=0.05,
                 window_rounds=10, n_byz_assumed=0, tail_warmup_rounds=10):
        """
        Args: as ByzClip21SGD2M, plus:
            mode: "evt_quantile" (recommended, see module docstring) or "moment_schedule"
                (kept for comparison; measured to be a no-op in this codebase).
            tau_init: fixed threshold used during tail_warmup_rounds.
            tau_min, tau_max: hard clamp on tau in either mode.
            schedule_c: schedule_tau_moment's own constant (ignored in evt_quantile mode).
            q_target: evt_quantile_tau's target exceedance probability (ignored in
                moment_schedule mode).
            window_rounds, n_byz_assumed: OnlineTailEstimator's parameters.
            tail_warmup_rounds: rounds run at tau_init before the estimator takes over.
        sigma_omega must be 0: a time-varying tau is not privacy-accounted for a fixed-T
        DP budget, in either mode.
        """
        if sigma_omega and sigma_omega > 0:
            raise ValueError(
                "TailAdaptiveClip21SGD2M's tau_t is not privacy-accounted for a fixed-T "
                "DP budget; run with sigma_omega=0."
            )
        if mode not in ("evt_quantile", "moment_schedule"):
            raise ValueError(f"Unknown mode '{mode}'")
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
        self.mode = mode

        self.tau = tau_init
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.schedule_c = schedule_c
        self.q_target = q_target
        self.tail_warmup_rounds = tail_warmup_rounds
        self.estimator = OnlineTailEstimator(window_rounds=window_rounds, n_byz_assumed=n_byz_assumed)

        self.x = torch.zeros(d, device=device)
        self.v = torch.zeros(n_honest, d, device=device)
        self.g_local = torch.zeros(n_honest, d, device=device)
        self.m = torch.zeros(self.n, d, device=device)
        self.g_global = torch.zeros(d, device=device)
        self._round = 0
        self.tau_trace = []
        self.alpha_trace = []
        self.sigma_trace = []

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
        c_honest = clipped_diff  # sigma_omega == 0 enforced in __init__
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
        self.sigma_trace.append(sigma_hat)
        if self._round >= self.tail_warmup_rounds and alpha_hat is not None:
            if self.mode == "evt_quantile":
                self.tau = evt_quantile_tau(alpha_hat, sigma_hat, q_target=self.q_target,
                                             tau_min=self.tau_min, tau_max=self.tau_max)
            else:
                self.tau = schedule_tau_moment(self._round, alpha_hat, sigma_hat,
                                                c=self.schedule_c, tau_min=self.tau_min, tau_max=self.tau_max)
        self.tau_trace.append(self.tau)
        self._round += 1

        return self.x
