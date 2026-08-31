"""Unit tests for the Byz-Clip-SGD (Algorithm 3) and Safe-DSHB (Algorithm 4) baselines,
verified against the special-case relationship stated by their own pseudocode
(arXiv:2603.23472, Appendix F.1, p.67): Safe-DSHB with client-momentum beta=1 collapses
its recursion m_i^{t+1} = (1-beta)*m_i^t + beta*(g_i^{t+1}+omega_i^{t+1}) to exactly
m_i^{t+1} = g_i^{t+1}+omega_i^{t+1}, which is Byz-Clip-SGD's own m_i^{t+1} update
verbatim. This is not an approximation -- it is an algebraic identity of the two
pseudocode blocks, so the two trajectories must match exactly (up to floating point and
matched RNG draws for the DP noise).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from byz_clip_sgd import ByzClipSGD
from safe_dshb import SafeDSHB
from robust_aggregators import apply_ragg


def make_ragg(name):
    return lambda vectors, num_byzantine: apply_ragg(name, vectors, num_byzantine)


def test_safe_dshb_beta1_reduces_to_byz_clip_sgd_no_dp():
    torch.manual_seed(0)
    n_honest, d = 5, 8
    a = torch.randn(n_honest, d)
    gamma, tau = 0.05, 1.0
    x0 = torch.randn(d)

    byz_clip = ByzClipSGD(
        d=d, n_honest=n_honest, n_byzantine=0, gamma=gamma, tau=tau, sigma_omega=0.0,
        ragg_fn=make_ragg("trimmed_mean"),
    )
    byz_clip.set_x(x0)

    safe_dshb = SafeDSHB(
        d=d, n_honest=n_honest, n_byzantine=0, beta=1.0, gamma=gamma, tau=tau, sigma_omega=0.0,
        ragg_fn=make_ragg("trimmed_mean"),
    )
    safe_dshb.set_x(x0)

    grad_fn = lambda x, a=a: x.unsqueeze(0) - a

    for t in range(30):
        x1 = byz_clip.step(grad_fn)
        x2 = safe_dshb.step(grad_fn)
        assert torch.allclose(x1, x2, atol=1e-6), f"mismatch at t={t}: {x1} vs {x2}"

    print("Safe-DSHB(beta=1) == Byz-Clip-SGD exactly, for all t (no DP noise, no Byzantines).")


def test_safe_dshb_beta1_reduces_to_byz_clip_sgd_with_dp_and_byzantines():
    """Same identity, now with DP noise and an IPM-style Byzantine attacker present.
    DP noise draws must line up round-by-round, so we seed identically before each step.
    """
    n_honest, n_byz, d = 4, 1, 6
    a = torch.randn(n_honest, d)
    gamma, tau, sigma_omega = 0.02, 0.5, 0.1
    x0 = torch.randn(d)

    byz_clip = ByzClipSGD(
        d=d, n_honest=n_honest, n_byzantine=n_byz, gamma=gamma, tau=tau, sigma_omega=sigma_omega,
        ragg_fn=make_ragg("trimmed_mean"),
    )
    byz_clip.set_x(x0)

    safe_dshb = SafeDSHB(
        d=d, n_honest=n_honest, n_byzantine=n_byz, beta=1.0, gamma=gamma, tau=tau, sigma_omega=sigma_omega,
        ragg_fn=make_ragg("trimmed_mean"),
    )
    safe_dshb.set_x(x0)

    grad_fn = lambda x, a=a: x.unsqueeze(0) - a
    byzantine_fn = lambda honest_m: (-10.0 * honest_m.mean(dim=0)).unsqueeze(0).repeat(n_byz, 1)

    for t in range(20):
        torch.manual_seed(100 + t)
        x1 = byz_clip.step(grad_fn, byzantine_fn)
        torch.manual_seed(100 + t)
        x2 = safe_dshb.step(grad_fn, byzantine_fn)
        assert torch.allclose(x1, x2, atol=1e-6), f"mismatch at t={t}: {x1} vs {x2}"

    print("Safe-DSHB(beta=1) == Byz-Clip-SGD exactly under DP noise + IPM Byzantines too.")


def test_byz_clip_sgd_zero_grad_stays_at_x0():
    d = 4
    algo = ByzClipSGD(
        d=d, n_honest=3, n_byzantine=0, gamma=0.01, tau=1.0, sigma_omega=0.0,
        ragg_fn=make_ragg("trimmed_mean"),
    )
    x0 = torch.ones(d) * 2.0
    algo.set_x(x0)
    for _ in range(10):
        algo.step(lambda x: torch.zeros(3, d))
    assert torch.allclose(algo.x, x0), algo.x
    print("Byz-Clip-SGD: zero-gradient fixed point holds.")


def test_safe_dshb_zero_grad_stays_at_x0():
    d = 4
    algo = SafeDSHB(
        d=d, n_honest=3, n_byzantine=0, beta=0.3, gamma=0.01, tau=1.0, sigma_omega=0.0,
        ragg_fn=make_ragg("trimmed_mean"),
    )
    x0 = torch.ones(d) * 2.0
    algo.set_x(x0)
    for _ in range(10):
        algo.step(lambda x: torch.zeros(3, d))
    assert torch.allclose(algo.x, x0), algo.x
    print("Safe-DSHB: zero-gradient fixed point holds (momentum stays at zero too).")


if __name__ == "__main__":
    test_safe_dshb_beta1_reduces_to_byz_clip_sgd_no_dp()
    test_safe_dshb_beta1_reduces_to_byz_clip_sgd_with_dp_and_byzantines()
    test_byz_clip_sgd_zero_grad_stays_at_x0()
    test_safe_dshb_zero_grad_stays_at_x0()
    print("All baseline unit tests passed.")
