"""Warm-started (bias-corrected) variant of Byz-Clip21-SGD2M (see byz_clip21_sgd2m.py).

Motivation, following directly from this session's own negative result: the adaptive-
clipping pilot (src/byz_clip21_sgd2m_adaptive.py, results/adaptive_clip_check/) found
that tau=1.0 was NEVER the active constraint on CIFAR-10 for this architecture -- the
quantile tracker raised tau to its ceiling within a handful of rounds because almost no
honest client's diff ever exceeded 1.0 in the first place. That rules out the clipping
threshold as the bottleneck, which redirects attention to the algorithm's other free
mechanism: the double exponential-moving-average structure itself.

v_i^{t+1} = (1-beta) v_i^t + beta * grad_i^{t+1}          (client momentum, beta=0.1)
g_i^{t+1} = g_i^t + beta_hat * clip(v_i^{t+1} - g_i^t)    (server EF21 reference, beta_hat=0.1)

Both v_i and g_i are zero-initialized. With beta=0.1, v_i needs ~1/beta=10 rounds before
it reflects the true gradient at more than ~65% weight (1-(1-beta)^10 =~ 0.65); g_i then
separately chases v_i at its OWN beta_hat-scaled rate, so the two cold-starts compound --
by the time g_i (which is what m, and hence g_global, and hence the x-update, are actually
built from) has caught up, a meaningful fraction of a short T=80 budget can be spent in a
throttled warm-up regime, before either momentum term is honest about the gradient's true
scale. This is the textbook cold-start-bias problem EMA-based optimizers usually correct
for (e.g. Adam's m_t/(1-beta1^t)); Byz-Clip21-SGD2M has no such correction anywhere. The
tax is the same fixed ~10-20 rounds for both datasets, but it is a much larger fraction of
a short, already-hard CIFAR-10 budget than of MNIST's easier one -- a plausible mechanism
for exactly the dataset-asymmetry this whole project has been chasing, distinct from
clipping, DP noise, learning rate, or normalization (all now separately tested).

Fix: replace the fixed beta / beta_hat with a schedule that starts at a fast, unbiased
cumulative-average rate and decays to the target value:
    rate(t) = max(target_rate, 1 / (t + 2))
At t=0 this is a plain (unbiased) running average of the first sample; it matches a
cumulative moving average until t+2 exceeds 1/target_rate (~10 rounds for target=0.1),
then clamps to the fixed EMA rate the base algorithm uses for the rest of training. This
is algebraically the same debiasing effect as Adam's correction, applied here as a
schedule rather than a post-hoc division, because g_i's update is not a plain EMA of an
i.i.d. sequence (it is an EF21 error-feedback recursion on the CLIPPED diff) and dividing
its raw value by a bias factor after the fact is not the right correction for that
recursion; scheduling the rate itself is. Applied identically to beta (v_i's update) and
beta_hat (both the g_i update and the m accumulation step, since m_i^{t+1} = m_i^t +
beta_hat * c_i^{t+1} is likewise a running accumulation that should start at an unbiased
rate). No other mechanism (clipping, DP noise, aggregation) is touched.
"""

import torch

from byz_clip21_sgd2m import clip_tau


def warmstart_rate(t, target_rate):
    """rate(t) = max(target_rate, 1/(t+2)); t is 0-indexed round number."""
    return max(target_rate, 1.0 / (t + 2))


class WarmStartClip21SGD2M:
    def __init__(self, d, n_honest, n_byzantine, beta, beta_hat, gamma, tau, sigma_omega,
                 ragg_fn, device="cpu"):
        """Same parameters as ByzClip21SGD2M; `beta`/`beta_hat` here are the TARGET
        (asymptotic) rates the schedule decays to, matching the base algorithm's meaning
        exactly once warm-up has elapsed.
        """
        self.d = d
        self.n_honest = n_honest
        self.n_byzantine = n_byzantine
        self.n = n_honest + n_byzantine
        self.beta = beta
        self.beta_hat = beta_hat
        self.gamma = gamma
        self.tau = tau
        self.sigma_omega = sigma_omega
        self.ragg_fn = ragg_fn
        self.device = device

        self.x = torch.zeros(d, device=device)
        self.v = torch.zeros(n_honest, d, device=device)
        self.g_local = torch.zeros(n_honest, d, device=device)
        self.m = torch.zeros(self.n, d, device=device)
        self.g_global = torch.zeros(d, device=device)
        self._round = 0

    def set_x(self, x0):
        self.x = x0.clone().to(self.device)

    def step(self, grad_fn, byzantine_fn=None):
        self.x = self.x - self.gamma * self.g_global

        grads = grad_fn(self.x)
        assert grads.shape == (self.n_honest, self.d)

        beta_t = warmstart_rate(self._round, self.beta)
        beta_hat_t = warmstart_rate(self._round, self.beta_hat)

        self.v = (1 - beta_t) * self.v + beta_t * grads

        diff = self.v - self.g_local
        clipped_diff = clip_tau(diff, self.tau)

        if self.sigma_omega > 0:
            omega = torch.randn(self.n_honest, self.d, device=self.device) * self.sigma_omega
        else:
            omega = torch.zeros(self.n_honest, self.d, device=self.device)

        c_honest = clipped_diff + omega
        self.g_local = self.g_local + beta_hat_t * clipped_diff

        if self.n_byzantine > 0:
            assert byzantine_fn is not None, "byzantine_fn required when n_byzantine > 0"
            c_byz = byzantine_fn(c_honest)
            assert c_byz.shape == (self.n_byzantine, self.d)
            c_all = torch.cat([c_honest, c_byz], dim=0)
        else:
            c_all = c_honest

        self.m = self.m + beta_hat_t * c_all
        self.g_global = self.ragg_fn(self.m, self.n_byzantine)

        self._round += 1
        return self.x
