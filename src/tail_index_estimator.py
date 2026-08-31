"""Online, Byzantine-robust tail-index and scale estimator, for use INSIDE a training
loop (as opposed to subgaussian_analysis.py's collect_gradient_noise/fit_and_report,
which is an offline, post-hoc diagnostic run separately from training with extra
fresh-gradient draws at fixed snapshots). This module answers a different question:
at round t, using only information the algorithm already computes, what is the CURRENT
tail index and scale of the honest client update-norm distribution -- so the clipping
mechanism can be calibrated to it, rather than to a value hand-tuned once offline.

Byzantine robustness: label-flip attackers in this codebase go through the honest
v_i/g_i/diff pipeline (federated_experiment.py's module docstring), so a malicious
client's diff CAN appear in the per-round norm sample and skew a naive pooled estimate.
Robustness here is deliberately minimal and reuses an existing convention rather than
inventing a new one: trim the top `n_byz_assumed` largest per-round norms before adding
them to the rolling window, mirroring this project's own trimmed-mean robust aggregator
(src/robust_aggregators.py) rather than a bespoke mechanism. IPM-style attackers, who
bypass the diff pipeline entirely and inject c_i directly (never computing a v_i/g_i
diff at all -- see federated_experiment.py), never enter this sample in the first
place, so no separate handling is needed for that attack type.
"""

import numpy as np

from subgaussian_analysis import hill_estimator


class OnlineTailEstimator:
    def __init__(self, window_rounds=10, n_byz_assumed=0, tail_fraction=0.25, min_samples=30):
        """
        Args:
            window_rounds: how many past rounds' (trimmed) per-client norms to pool for
                the estimate -- a rolling window, not the whole history, so the estimate
                tracks a noise distribution that may itself drift over training.
            n_byz_assumed: number of largest per-round norms to discard before pooling,
                a public, conservative upper bound on the Byzantine count (the same kind
                of assumption robust aggregators already require).
            tail_fraction: fraction of the pooled window used as the Hill estimator's
                tail (see subgaussian_analysis.hill_estimator); higher than that
                function's offline default (0.1) because the online window is much
                smaller, so a larger fraction is needed to get enough tail points.
            min_samples: minimum pooled sample size before an estimate is trusted; below
                this, get_estimate() returns None and the caller should fall back to a
                fixed default (this project's own diagnostic pipeline requires n>=1
                snapshot but effectively pools hundreds of samples -- min_samples=30 is
                a conservative floor for a rolling window this much smaller).
        """
        self.window_rounds = window_rounds
        self.n_byz_assumed = n_byz_assumed
        self.tail_fraction = tail_fraction
        self.min_samples = min_samples
        self._window = []  # list of 1D numpy arrays, one per round, oldest first

    def update(self, per_client_norms):
        """per_client_norms: 1D array-like of honest-pipeline diff norms for this round."""
        norms = np.asarray(per_client_norms, dtype=float)
        if self.n_byz_assumed > 0 and len(norms) > self.n_byz_assumed:
            norms = np.sort(norms)[: len(norms) - self.n_byz_assumed]  # drop top-k largest
        self._window.append(norms)
        if len(self._window) > self.window_rounds:
            self._window.pop(0)

    def get_estimate(self):
        """Returns (alpha_hat, sigma_hat) or (None, None) if not enough pooled data yet.
        alpha_hat: Hill tail-index estimate (larger = lighter tail).
        sigma_hat: robust scale estimate (trimmed mean of the pooled window), analogous
            to the sigma in a bounded-p-th-moment noise assumption.
        """
        pooled = np.concatenate(self._window) if self._window else np.array([])
        if len(pooled) < self.min_samples:
            return None, None
        alpha_hat = hill_estimator(pooled, tail_fraction=self.tail_fraction)
        if not np.isfinite(alpha_hat) or alpha_hat <= 0:
            return None, None
        sorted_pooled = np.sort(pooled)
        trim = max(int(0.1 * len(sorted_pooled)), 0)
        core = sorted_pooled[trim: len(sorted_pooled) - trim] if trim > 0 else sorted_pooled
        sigma_hat = float(core.mean()) if len(core) > 0 else float(pooled.mean())
        return float(alpha_hat), sigma_hat
