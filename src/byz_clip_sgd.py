"""Byz-Clip-SGD (Algorithm 3 of Islamov, Malinovsky, Gaponov, Lucchi, Richtarik,
Gorbunov, arXiv:2603.23472, Appendix F.1, p.67).

Verbatim pseudocode extracted from the paper PDF (Algorithm 3, page 67):

    Input: x^0 in X, step-size gamma > 0, clipping parameter tau > 0, DP-noise
    variance sigma_omega^2 >= 0
    for t = 0, ..., T-1 do
        x^{t+1} = x^t - gamma * g^t
        for i in G do
            g_i^{t+1} = clip_tau(grad f_i(x^{t+1}, xi_i^{t+1}))
            omega_i^{t+1} ~ N(0, sigma_omega^2 I)
            m_i^{t+1} = g_i^{t+1} + omega_i^{t+1}
        for i in B do
            m_i^{t+1} = (*) sends arbitrary vector
        g^{t+1} = RAgg(m_1^{t+1}, ..., m_n^{t+1})

Per the paper's own Appendix F.1 (p.68): "Byz-Clip-SGD is a variant of DP-SGD in which
a robust aggregation rule replaces server-side averaging." It has no client-side
momentum and no server-side EF21-style double momentum -- each round's message is the
freshly clipped-and-noised stochastic gradient itself.
"""

import torch

from byz_clip21_sgd2m import clip_tau


class ByzClipSGD:
    def __init__(self, d, n_honest, n_byzantine, gamma, tau, sigma_omega,
                 ragg_fn, device="cpu"):
        """
        Args:
            d: parameter dimension (flattened).
            n_honest, n_byzantine: client counts.
            gamma: step size.
            tau: clipping threshold (float('inf') disables clipping).
            sigma_omega: DP noise std parameter (0 disables DP noise).
            ragg_fn: callable(vectors: (n,d) tensor, num_byzantine:int) -> (d,) tensor.
        """
        self.d = d
        self.n_honest = n_honest
        self.n_byzantine = n_byzantine
        self.n = n_honest + n_byzantine
        self.gamma = gamma
        self.tau = tau
        self.sigma_omega = sigma_omega
        self.ragg_fn = ragg_fn
        self.device = device

        self.x = torch.zeros(d, device=device)
        self.g_global = torch.zeros(d, device=device)  # g^t used for the x update

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

        m_honest = g_honest + omega

        if self.n_byzantine > 0:
            assert byzantine_fn is not None, "byzantine_fn required when n_byzantine > 0"
            m_byz = byzantine_fn(m_honest)
            assert m_byz.shape == (self.n_byzantine, self.d)
            m_all = torch.cat([m_honest, m_byz], dim=0)
        else:
            m_all = m_honest

        self.g_global = self.ragg_fn(m_all, self.n_byzantine)

        return self.x
