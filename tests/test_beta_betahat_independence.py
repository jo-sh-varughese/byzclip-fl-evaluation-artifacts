"""Checks that beta (client momentum) and beta_hat (server EF21 momentum) in
ByzClip21SGD2M are genuinely independent variables/config fields, not aliased.

Code locations under test (src/byz_clip21_sgd2m.py):
  - __init__ (lines ~77-78): self.beta = beta ; self.beta_hat = beta_hat
  - step() line 114: self.v = (1 - self.beta) * self.v + self.beta * grads
  - step() line 125: self.g_local = self.g_local + self.beta_hat * clipped_diff
  - step() line 135: self.m = self.m + self.beta_hat * c_all

Method:
  1. Storage independence: construct with distinct beta/beta_hat, mutate one
     attribute post-construction, confirm the other is unaffected (rules out
     shared-reference/property aliasing).
  2. Functional isolation at line 114: fix beta=1.0 (so v depends only on the
     fresh grads, never on beta_hat by construction of the update rule) and
     vary beta_hat across two runs -> v trajectory must be identical in both,
     since line 114 must not read beta_hat.
  3. Functional isolation at lines 125/135: fix beta_hat=0.0 (so g_local and m
     must stay exactly zero for all t if beta_hat -- not beta -- gates those
     updates) and vary beta across two runs -> g_local and m must stay
     identically zero in both, and identical to each other's runs.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from byz_clip21_sgd2m import ByzClip21SGD2M
from robust_aggregators import apply_ragg


def make_ragg(name):
    return lambda vectors, num_byzantine: apply_ragg(name, vectors, num_byzantine)


def test_storage_independence():
    algo = ByzClip21SGD2M(
        d=4, n_honest=3, n_byzantine=0,
        beta=0.3, beta_hat=0.7, gamma=0.05, tau=float("inf"), sigma_omega=0.0,
        ragg_fn=make_ragg("trimmed_mean"),
    )
    assert algo.beta == 0.3
    assert algo.beta_hat == 0.7
    assert algo.beta != algo.beta_hat

    algo.beta_hat = 0.99
    assert algo.beta == 0.3, "mutating beta_hat changed beta -- aliased!"

    algo.beta = 0.11
    assert algo.beta_hat == 0.99, "mutating beta changed beta_hat -- aliased!"

    print("PASS storage_independence: self.beta and self.beta_hat are distinct attributes.")


def test_v_update_ignores_beta_hat():
    # gamma=0 freezes x across steps so the grad_fn input sequence is identical
    # in both runs regardless of beta_hat; this isolates line 114 (the v-update)
    # from the *indirect*, algorithmically-expected coupling where beta_hat
    # affects g_global -> x -> grads in later rounds. Without freezing x, that
    # legitimate indirect coupling would be misread as aliasing.
    torch.manual_seed(0)
    n_honest, d = 5, 6
    a = torch.randn(n_honest, d)

    def run(beta_hat_value):
        algo = ByzClip21SGD2M(
            d=d, n_honest=n_honest, n_byzantine=0,
            beta=1.0, beta_hat=beta_hat_value, gamma=0.0,
            tau=float("inf"), sigma_omega=0.0,
            ragg_fn=make_ragg("trimmed_mean"),
        )
        algo.set_x(torch.zeros(d))
        v_trace = []
        for _ in range(10):
            algo.step(lambda x, a=a: x.unsqueeze(0) - a)
            v_trace.append(algo.v.clone())
        return v_trace

    trace_low = run(beta_hat_value=0.05)
    trace_high = run(beta_hat_value=0.95)

    for t, (v_lo, v_hi) in enumerate(zip(trace_low, trace_high)):
        assert torch.allclose(v_lo, v_hi, atol=0.0), (
            f"v trajectory at t={t} differs when only beta_hat changes "
            f"(beta fixed at 1.0) -- line 114 is reading beta_hat, i.e. "
            f"beta/beta_hat are aliased at the v-update."
        )

    print("PASS v_update_ignores_beta_hat: line 114's v-update depends only on beta.")


def test_g_local_and_m_ignore_beta():
    torch.manual_seed(1)
    n_honest, d = 4, 5
    a = torch.randn(n_honest, d)

    def run(beta_value):
        algo = ByzClip21SGD2M(
            d=d, n_honest=n_honest, n_byzantine=0,
            beta=beta_value, beta_hat=0.0, gamma=0.05,
            tau=float("inf"), sigma_omega=0.0,
            ragg_fn=make_ragg("trimmed_mean"),
        )
        algo.set_x(torch.randn(d))
        for _ in range(10):
            algo.step(lambda x, a=a: x.unsqueeze(0) - a)
        return algo.g_local.clone(), algo.m.clone()

    g_local_lo, m_lo = run(beta_value=0.1)
    g_local_hi, m_hi = run(beta_value=0.9)

    zeros_local = torch.zeros_like(g_local_lo)
    zeros_m = torch.zeros_like(m_lo)

    assert torch.allclose(g_local_lo, zeros_local, atol=1e-12), (
        "g_local drifted from zero with beta_hat=0.0 -- line 125 is reading "
        "beta instead of/in addition to beta_hat."
    )
    assert torch.allclose(g_local_hi, zeros_local, atol=1e-12), (
        "g_local drifted from zero with beta_hat=0.0 -- line 125 is reading "
        "beta instead of/in addition to beta_hat."
    )
    assert torch.allclose(m_lo, zeros_m, atol=1e-12), (
        "m drifted from zero with beta_hat=0.0 -- line 135 is reading beta "
        "instead of/in addition to beta_hat."
    )
    assert torch.allclose(m_hi, zeros_m, atol=1e-12), (
        "m drifted from zero with beta_hat=0.0 -- line 135 is reading beta "
        "instead of/in addition to beta_hat."
    )

    print("PASS g_local_and_m_ignore_beta: lines 125/135 depend only on beta_hat.")


if __name__ == "__main__":
    test_storage_independence()
    test_v_update_ignores_beta_hat()
    test_g_local_and_m_ignore_beta()
    print("All beta/beta_hat independence checks passed.")
