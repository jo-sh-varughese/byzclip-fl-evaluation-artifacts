"""Floor-and-ceiling ("band") clipping variant of Byz-Clip21-SGD2M (see
byz_clip21_sgd2m.py). Grounded in two independent lines of literature (checked this
session, not assumed): (1) Byzantine-robust FL via gradient normalization -- Fed-NGA
(Wu et al., arXiv:2408.09539) aggregates g_m/||g_m|| rather than raw gradients,
specifically so a client's influence cannot be manipulated by magnitude, only by
direction, which the robust aggregator already defends against; (2) heavy-tailed SGD
theory (arXiv:2410.16561, arXiv:2410.13849) showing that normalization -- not clipping
alone -- is what is needed when a heavy tail also means small, non-stationary typical
magnitudes, because clipping is a one-sided ceiling that leaves already-small updates
untouched.

Why a ceiling-only fix (clip_tau) cannot be the whole story here, following directly
from this session's own measurement: the adaptive-clipping pilot
(byz_clip21_sgd2m_adaptive.py, results/adaptive_clip_check/) found tau=1.0 was NEVER
the active constraint on CIFAR-10 for this architecture -- honest diff norms almost
never exceeded it (the quantile tracker raised tau to its ceiling within a handful of
rounds chasing a quantile it could never satisfy at that scale). A ceiling clip is a
no-op on a distribution that never reaches the ceiling. What the normalization
literature above suggests is missing is the opposite bound: a FLOOR, so a naturally
tiny, heavy-tailed diff is boosted to a controlled minimum "useful" step magnitude
instead of injecting whatever arbitrarily small amount of signal happened to be
achieved that round.

Pure normalization (rescaling every diff to a fixed norm, full stop, as Fed-NGA does
for raw gradients) is not adapted directly here because of a risk specific to this
algorithm's error-feedback structure: g_i chases v_i via g_i += beta_hat * diff, and
forcing diff to a fixed nonzero norm even once g_i has nearly converged to v_i would
inject a perpetual, undamped correction of that size every round -- the state would
oscillate around v_i indefinitely rather than settling. A ceiling-and-floor BAND avoids
this: once real progress pushes a diff's norm above the floor (i.e. the EF21 state is
still meaningfully behind, the normal/intended operating regime), clip_band leaves it
untouched exactly as the base algorithm's clip_tau already does; only diffs BELOW the
floor -- the regime our own diagnostics show CIFAR-10 spends most of its time in -- get
rescaled up. tau_floor=0 recovers the base algorithm's clip_tau exactly (a strict
generalization, not a replacement).
"""

import torch

from byz_clip21_sgd2m import ByzClip21SGD2M


def clip_band(vec, tau_min, tau_max):
    """Row-wise (or 1D) norm rescaling, direction always preserved:
      - norm > tau_max: shrink to tau_max (standard ceiling clip -- Byzantine/outlier
        protection, identical to byz_clip21_sgd2m.clip_tau).
      - norm < tau_min: grow to tau_min (new floor -- prevents throttling by a
        naturally tiny, heavy-tailed-noise-driven update).
      - otherwise: unchanged.
    """
    if vec.dim() == 1:
        norm = vec.norm()
        if tau_max != float("inf") and norm > tau_max:
            return vec * (tau_max / norm)
        if tau_min > 0 and norm < tau_min and norm > 1e-12:
            return vec * (tau_min / norm)
        return vec
    norms = vec.norm(dim=1, keepdim=True).clamp_min(1e-12)
    scale = torch.ones_like(norms)
    if tau_max != float("inf"):
        scale = torch.where(norms > tau_max, tau_max / norms, scale)
    if tau_min > 0:
        scale = torch.where(norms < tau_min, tau_min / norms, scale)
    return vec * scale


class BandClip21SGD2M(ByzClip21SGD2M):
    """ByzClip21SGD2M with clip_tau replaced by clip_band(tau_min=tau_floor, tau_max=tau).
    tau_floor=0.0 (the default) makes this bit-for-bit identical to the base algorithm.
    """

    def __init__(self, *args, tau_floor=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau_floor = tau_floor

    def step(self, grad_fn, byzantine_fn=None):
        self.x = self.x - self.gamma * self.g_global

        grads = grad_fn(self.x)
        assert grads.shape == (self.n_honest, self.d)

        self.v = (1 - self.beta) * self.v + self.beta * grads

        diff = self.v - self.g_local
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

        return self.x
