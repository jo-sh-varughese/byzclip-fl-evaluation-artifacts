"""Safe-DSHB (Algorithm 4 of Islamov, Malinovsky, Gaponov, Lucchi, Richtarik,
Gorbunov, arXiv:2603.23472, Appendix F.1, p.67), attributed there to Allouah, Farhadkhani,
Guerraoui, Gupta, Pinot, Stephan, "On the Privacy-Robustness-Utility Trilemma in
Distributed Learning", ICML 2023 (arXiv:2302.04787) -- see main.bib entry
`allouah2023trilemma`.

Verbatim pseudocode extracted from the paper PDF (Algorithm 4, page 67):

    Input: x^0 in X, momentum beta in (0,1], step-size gamma > 0, m_i^0 in R^d,
    clipping tau > 0, DP-noise variance sigma_omega^2 >= 0
    for t = 0, ..., T-1 do
        x^{t+1} = x^t - gamma * g^t
        for i in G do
            g_i^{t+1} = clip_tau(grad f_i(x^{t+1}, xi_i^{t+1}))
            omega_i^{t+1} ~ N(0, sigma_omega^2 I)
            m_i^{t+1} = (1-beta) m_i^t + beta * (g_i^{t+1} + omega_i^{t+1})
        for i in B do
            m_i^{t+1} = (*) sends arbitrary vector
        g^{t+1} = RAgg(m_1^{t+1}, ..., m_n^{t+1})

Per the paper's own Appendix F.1 (p.68): "The method of Allouah et al. [2023b]
originally performs per-example clipping (line 5); to align with our analysis under
sigma-sub-Gaussian noise, i.e., without assuming a finite-sum structure for f_i, we
instead clip the stochastic gradient" -- i.e. the paper itself already deviates from
the original Allouah et al. per-example clipping in favor of clipping the (single)
stochastic gradient, and that is the variant given in Algorithm 4 above. We follow
Algorithm 4 exactly, not the original per-example-clipping Allouah et al. version, per
the source paper's own stated adaptation for its comparison.

Single-momentum (beta) client-side smoothing of the noised, clipped gradient, then
robust aggregation -- no server-side EF21-style double momentum (beta_hat), unlike
Byz-Clip21-SGD2M.
"""

import torch

from byz_clip21_sgd2m import clip_tau


class SafeDSHB:
    def __init__(self, d, n_honest, n_byzantine, beta, gamma, tau, sigma_omega,
                 ragg_fn, device="cpu"):
        """
        Args:
            d: parameter dimension (flattened).
            n_honest, n_byzantine: client counts.
            beta: client-momentum coefficient, beta in (0, 1].
            gamma: step size.
            tau: clipping threshold (float('inf') disables clipping).
            sigma_omega: DP noise std parameter (0 disables DP noise).
            ragg_fn: callable(vectors: (n,d) tensor, num_byzantine:int) -> (d,) tensor.
        """
        self.d = d
        self.n_honest = n_honest
        self.n_byzantine = n_byzantine
        self.n = n_honest + n_byzantine
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.sigma_omega = sigma_omega
        self.ragg_fn = ragg_fn
        self.device = device

        self.x = torch.zeros(d, device=device)
        self.m = torch.zeros(n_honest, d, device=device)  # m_i^t, honest-only (m_i^0 = 0)
        self.g_global = torch.zeros(d, device=device)     # g^t used for the x update

    def set_x(self, x0):
        self.x = x0.clone().to(self.device)

    def step(self, grad_fn, byzantine_fn=None):
        """
        Args:
            grad_fn: callable(x: (d,) tensor) -> (n_honest, d) tensor of fresh stochastic
                     gradients grad_f_i(x, xi_i) for each honest client, evaluated at the
                     NEW x (x^{t+1}), per the algorithm's ordering.
            byzantine_fn: callable(honest_m: (n_honest, d) tensor) -> (n_byzantine, d)
                          tensor of attacker-controlled m_i vectors, given this round's
                          honest transmitted m_i vectors (omniscient-coalition attack
                          model). Required when n_byzantine > 0.

        Returns:
            new x (x^{t+1}).
        """
        self.x = self.x - self.gamma * self.g_global

        grads = grad_fn(self.x)
        assert grads.shape == (self.n_honest, self.d)

        g_honest = clip_tau(grads, self.tau)

        if self.sigma_omega > 0:
            omega = torch.randn(self.n_honest, self.d, device=self.device) * self.sigma_omega
        else:
            omega = torch.zeros(self.n_honest, self.d, device=self.device)

        self.m = (1 - self.beta) * self.m + self.beta * (g_honest + omega)
        m_honest = self.m

        if self.n_byzantine > 0:
            assert byzantine_fn is not None, "byzantine_fn required when n_byzantine > 0"
            m_byz = byzantine_fn(m_honest)
            assert m_byz.shape == (self.n_byzantine, self.d)
            m_all = torch.cat([m_honest, m_byz], dim=0)
        else:
            m_all = m_honest

        self.g_global = self.ragg_fn(m_all, self.n_byzantine)

        return self.x
