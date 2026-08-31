"""Byz-Clip21-SGD2M (Algorithm 1) plus two labeled ablations of it.

Implements, line by line, the pseudocode in the task prompt (representing Algorithm 1
of Islamov, Malinovsky, Gaponov, Lucchi, Richtarik, Gorbunov, arXiv:2603.23472):

for t = 0..T-1:
    x^{t+1} = x^t - gamma * g^t
    for honest i:
        v_i^{t+1} = (1-beta) v_i^t + beta * grad_f_i(x^{t+1}, xi_i^{t+1})
        omega_i^{t+1} ~ N(0, sigma_omega^2 I)
        c_i^{t+1} = clip_tau(v_i^{t+1} - g_i^t) + omega_i^{t+1}
        g_i^{t+1} = g_i^t + beta_hat * clip_tau(v_i^{t+1} - g_i^t)
    for byzantine i: c_i^{t+1} = arbitrary_vector()
    for all i: m_i^{t+1} = m_i^t + beta_hat * c_i^{t+1}
    g^{t+1} = RAgg(m_1^{t+1}, ..., m_n^{t+1})

NOTE the two distinct "clip-then-something" operations, per the task spec: `c_i` =
clip(v_i - g_i) + DP noise (noise added AFTER clipping); `g_i` = g_i + beta_hat *
clip(v_i - g_i), which uses the SAME clipped difference but WITHOUT the DP noise --
g_i is a noise-free EF21-style reference state, not the noisy client message.

Ablations (of THIS algorithm, NOT reproductions of the paper's external Safe-DSHB /
Byz-Clip-SGD baselines, for which we do not have verified appendix pseudocode):
  - "no_momentum": beta_hat = 1 (removes server-side double-momentum smoothing).
  - "no_clip_no_dp": tau = inf, sigma_omega = 0 (removes DP noise and clipping,
    isolating the pure-Byzantine-robustness regime -- matches the paper's own
    Theorem 5.2 special case per the task spec).
"""

import torch


def clip_tau(vec, tau):
    """L2-norm clipping: clip_tau(x) = (tau/||x||) x if ||x||>tau else x.

    Row-wise when `vec` is 2D (one row per client).
    """
    if tau == float("inf"):
        return vec
    if vec.dim() == 1:
        norm = vec.norm()
        if norm > tau:
            return vec * (tau / norm)
        return vec
    norms = vec.norm(dim=1, keepdim=True)
    scale = torch.clamp(tau / norms.clamp_min(1e-12), max=1.0)
    return vec * scale


class ByzClip21SGD2M:
    def __init__(self, d, n_honest, n_byzantine, beta, beta_hat, gamma, tau, sigma_omega,
                 ragg_fn, device="cpu", ablation=None):
        """
        Args:
            d: parameter dimension (flattened).
            n_honest, n_byzantine: client counts.
            beta: client-momentum coefficient.
            beta_hat: server EF21/momentum coefficient.
            gamma: step size.
            tau: clipping threshold (float('inf') disables clipping).
            sigma_omega: DP noise std parameter (0 disables DP noise).
            ragg_fn: callable(vectors: (n,d) tensor, num_byzantine:int) -> (d,) tensor.
            ablation: None | "no_momentum" | "no_clip_no_dp".
        """
        if ablation == "no_momentum":
            beta_hat = 1.0
        elif ablation == "no_clip_no_dp":
            tau = float("inf")
            sigma_omega = 0.0
        elif ablation is not None:
            raise ValueError(f"Unknown ablation '{ablation}'")

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
        self.ablation = ablation

        self.x = torch.zeros(d, device=device)
        self.v = torch.zeros(n_honest, d, device=device)
        self.g_local = torch.zeros(n_honest, d, device=device)  # g_i^t, honest-only EF21 state
        self.m = torch.zeros(self.n, d, device=device)          # m_i^t, all clients
        self.g_global = torch.zeros(d, device=device)           # g^t used for the x update

    def set_x(self, x0):
        self.x = x0.clone().to(self.device)

    def step(self, grad_fn, byzantine_fn=None):
        """
        Args:
            grad_fn: callable(x: (d,) tensor) -> (n_honest, d) tensor of fresh stochastic
                     gradients grad_f_i(x, xi_i) for each honest client, evaluated at the
                     NEW x (x^{t+1}), per the algorithm's ordering.
            byzantine_fn: callable(honest_c: (n_honest, d) tensor) -> (n_byzantine, d)
                          tensor of attacker-controlled c_i vectors, given this round's
                          honest *transmitted* c_i vectors (omniscient-coalition attack
                          model). Required when n_byzantine > 0.

        Returns:
            new x (x^{t+1}).
        """
        self.x = self.x - self.gamma * self.g_global

        grads = grad_fn(self.x)
        assert grads.shape == (self.n_honest, self.d)

        self.v = (1 - self.beta) * self.v + self.beta * grads

        diff = self.v - self.g_local
        clipped_diff = clip_tau(diff, self.tau)

        if self.sigma_omega > 0:
            omega = torch.randn(self.n_honest, self.d, device=self.device) * self.sigma_omega
        else:
            omega = torch.zeros(self.n_honest, self.d, device=self.device)

        c_honest = clipped_diff + omega                             # c_i: clip THEN add noise
        self.g_local = self.g_local + self.beta_hat * clipped_diff  # g_i: same clipped diff, NO noise

        if self.n_byzantine > 0:
            assert byzantine_fn is not None, "byzantine_fn required when n_byzantine > 0"
            c_byz = byzantine_fn(c_honest)
            assert c_byz.shape == (self.n_byzantine, self.d)
            c_all = torch.cat([c_honest, c_byz], dim=0)
        else:
            c_all = c_honest

        self.m = self.m + self.beta_hat * c_all
        self.g_global = self.ragg_fn(self.m, self.n_byzantine)

        return self.x
