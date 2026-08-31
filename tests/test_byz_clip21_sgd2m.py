"""Unit tests for Algorithm 1, verified against an analytically-derived reduction.

Toy problem: n_honest quadratic clients f_i(x) = 0.5 ||x - a_i||^2, full (deterministic)
gradients grad_f_i(x) = x - a_i (no stochastic minibatch noise -- orthogonal to what this
test checks, which is the algorithm's bookkeeping, not variance behavior).

Analytic fact used for the test (derived from the pseudocode, not assumed):
When tau=inf and sigma_omega=0 (ablation "no_clip_no_dp"), c_i^{t+1} = v_i^{t+1} - g_i^t
exactly (no clipping), so
    m_i^{t+1} = m_i^t + beta_hat*(v_i^{t+1} - g_i^t) = m_i^t + (g_i^{t+1} - g_i^t)
Since m_i^0 = g_i^0 = 0, by induction m_i^t = g_i^t for ALL t -- an exact invariant of
the EF21-style bookkeeping whenever clipping and DP noise are both disabled, independent
of beta_hat. This test checks that invariant holds to floating-point precision at every
step.

Further, with beta_hat = 1 (ablation "no_momentum") ALSO applied on top, clip(v_i-g_i)
degenerates further: g_i^{t+1} = g_i^t + 1*(v_i^{t+1}-g_i^t) = v_i^{t+1} exactly, so
m_i^{t+1} = v_i^{t+1} exactly, and (with delta_byz=0, i.e. n_byzantine=0, using trimmed
mean which reduces to the plain mean at f=0) g^{t+1} = mean_i(v_i^{t+1}). This is
distributed SGD with per-client momentum on gradients (v_i), averaged across clients --
the closest analytically-clean reduction to "standard FedAvg" available from this
pseudocode (Byz-Clip21-SGD2M's own beta-momentum layer does not vanish under tau=inf,
sigma_omega=0 alone -- only combined with beta_hat=1 does the algorithm collapse to
plain momentum-SGD averaging). We verify x's trajectory matches a directly-computed
reference implementation of that reduced recursion.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from byz_clip21_sgd2m import ByzClip21SGD2M
from robust_aggregators import apply_ragg


def make_ragg(name):
    return lambda vectors, num_byzantine: apply_ragg(name, vectors, num_byzantine)


def test_m_equals_g_invariant():
    torch.manual_seed(0)
    n_honest, d = 5, 8
    a = torch.randn(n_honest, d)  # client optima

    algo = ByzClip21SGD2M(
        d=d, n_honest=n_honest, n_byzantine=0,
        beta=0.3, beta_hat=0.2, gamma=0.05, tau=float("inf"), sigma_omega=0.0,
        ragg_fn=make_ragg("trimmed_mean"), ablation=None,
    )
    algo.set_x(torch.randn(d))

    for t in range(30):
        grad_fn = lambda x, a=a: x.unsqueeze(0) - a  # full gradient x - a_i for every client
        algo.step(grad_fn)
        assert torch.allclose(algo.m, algo.g_local, atol=1e-6), f"m != g at t={t}"

    print("m_i^t == g_i^t invariant holds for all t under tau=inf, sigma_omega=0.")


def test_reduces_to_momentum_sgd_averaging():
    torch.manual_seed(1)
    n_honest, d = 4, 6
    a = torch.randn(n_honest, d)
    beta = 0.4
    gamma = 0.1
    x0 = torch.randn(d)

    algo = ByzClip21SGD2M(
        d=d, n_honest=n_honest, n_byzantine=0,
        beta=beta, beta_hat=1.0, gamma=gamma, tau=float("inf"), sigma_omega=0.0,
        ragg_fn=make_ragg("trimmed_mean"), ablation=None,
    )
    algo.set_x(x0)

    # Reference: directly compute the reduced recursion.
    x_ref = x0.clone()
    v_ref = torch.zeros(n_honest, d)
    g_ref = torch.zeros(d)  # this plays the role of g^t in the reduced recursion

    for t in range(25):
        x_algo = algo.step(lambda x, a=a: x.unsqueeze(0) - a)

        x_ref = x_ref - gamma * g_ref
        grads_ref = x_ref.unsqueeze(0) - a
        v_ref = (1 - beta) * v_ref + beta * grads_ref
        g_ref = v_ref.mean(dim=0)  # RAgg(v_1,...,v_n) with f=0 == plain mean

        assert torch.allclose(x_algo, x_ref, atol=1e-5), f"x mismatch at t={t}: {x_algo} vs {x_ref}"

    print("Reduces to plain distributed momentum-SGD-with-mean-aggregation when "
          "beta_hat=1, tau=inf, sigma_omega=0, delta_byz=0 (closest analytic reduction "
          "to 'standard FedAvg' obtainable from this pseudocode).")


def test_zero_byzantine_zero_grad_stays_at_x0():
    d = 4
    algo = ByzClip21SGD2M(
        d=d, n_honest=3, n_byzantine=0,
        beta=0.1, beta_hat=0.1, gamma=0.01, tau=1.0, sigma_omega=0.0,
        ragg_fn=make_ragg("trimmed_mean"),
    )
    x0 = torch.ones(d) * 2.0
    algo.set_x(x0)
    for _ in range(10):
        algo.step(lambda x: torch.zeros(3, d))
    assert torch.allclose(algo.x, x0), algo.x
    print("Zero-gradient fixed point holds (no drift when all client gradients are zero).")


if __name__ == "__main__":
    test_m_equals_g_invariant()
    test_reduces_to_momentum_sgd_averaging()
    test_zero_byzantine_zero_grad_stays_at_x0()
    print("All Algorithm 1 unit tests passed.")
