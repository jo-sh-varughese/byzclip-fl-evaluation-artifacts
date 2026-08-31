# Tail-Adaptive Clipping: what is proven, what is conjectured

This document exists so the claim made by `src/byz_clip21_sgd2m_tailadaptive.py` is never
stated more strongly in the paper than it actually is. Read it before writing any theorem
statement about this algorithm.

## The gap this is trying to close

Islamov et al. (arXiv:2603.23472, Byz-Clip21-SGD2M) prove high-probability convergence
under a σ-sub-Gaussian gradient-noise assumption, with ONE clipping threshold `tau`
fixed for the entire run. Their own Conclusion lists extending to heavy-tailed noise as
future work. This repository's own H1 measurement (`paper/main_preprint.tex`,
`src/subgaussian_analysis.py`) found CIFAR-10's gradient noise reads consistently
heavier-tailed than MNIST's (Hill α ≈ 5.4–5.9 vs. ≈ 8.0, bias-corrected, stable across a
5× draw-count increase and a synthetic-control check). Nobody — not the source paper, not
this project until now — has connected that measurement to the algorithm's own `tau`.

## What the cited literature actually proves

"Revisiting Gradient Normalization and Clipping for Nonconvex SGD under Heavy-Tailed
Noise" (arXiv:2410.16561) assumes a bounded p-th moment on gradient noise,
`E||∇f(w,ξ) − ∇f(w)||^p ≤ σ^p`, for `p ∈ (1, 2]` — strictly weaker than bounded variance
(which is `p = 2`). Under that assumption, and under an *individual* Lipschitz-continuity
condition, their Theorem 3 (NSGDC) prescribes a clipping parameter

```
h = 2 * ( γL·T^(2p/(3p−2)) · max{1, σ^(3p/(3p−2)) · 1[σ²T ≥ LΔ]} + ||∇f(w⁰)|| )
```

i.e. a threshold that **grows** with the round count `T`, with both the `T`-exponent and
the `σ`-exponent depending on `p` — larger for smaller `p` (heavier tails). They also
prove gradient normalization alone suffices for convergence without any clipping under
that same Lipschitz condition, and that combining normalization with clipping gives the
best rates specifically when `σ` is large. A fixed, small, dataset-independent constant
(Byz-Clip21-SGD2M's own `tau = 1.0`) is exactly the naive baseline this line of work is
arguing against.

## Iteration 1 (`mode="moment_schedule"`) — tried, measured, and kept as a documented failure

`tau_t = c · (t+1)^(2p/(3p−2)) · σ̂_t^(3p/(3p−2))`, with `p = clip(α̂_t, 1+ε, 2)`.

**This is a no-op in this codebase, measured, not assumed.** Every Hill-α estimate this
session ever produced — MNIST clean ≈4.5–11, MNIST under IPM ≈2.6–4.2, CIFAR-10 clean
≈3.2–8.3 — sits above 2. Clamping `p` into `(1,2]` therefore saturates at the SAME
boundary value (`p=2`) on every single round, for every dataset, every condition: there
is no real α-dependence left once that happens, only a fixed functional shape. Worse,
because `tau_t` grows monotonically with `t` and the honest diff-norm ceiling it is being
compared against is small and roughly stationary (CIFAR-10 ≈0.04–0.16, directly measured
in `results/confound_check`), `tau_t` exceeds that ceiling within the first few
post-warmup rounds and never returns below it — clipping becomes permanently inactive.
`results/tailadaptive_check/` and `results/mnist_tailadaptive_check/` confirm this
directly: accuracy identical to the fixed-τ baseline to 3–4 decimal places on every
condition, Wilcoxon p=1.000 across the board (not "not significant" — literally no
detected difference). This is a real result worth keeping visible (this project's own
convention — see `byz_clip21_sgd2m_adaptive.py`'s docstring for the same pattern with an
earlier mechanism), not something to quietly delete and pretend didn't happen.

## Iteration 2 (`mode="evt_quantile"`, recommended) — the corrected design

`tau_t = σ̂_t · q_target^(−1/α̂_t)`.

This drops the imported moment-bound theorem (arXiv:2410.16561's exact regime doesn't
match this codebase's measured α anyway — see above) in favor of the SAME extreme-value
theory the Hill estimator itself already comes from: for a regularly-varying tail with
index α, `P(X > x) ~ (x_ref/x)^α`, so the value achieving a target exceedance probability
`q` relative to a reference scale `x_ref` is `x_ref · q^(−1/α)`. Using the robust scale
estimate `σ̂_t` as `x_ref` gives a threshold that (a) uses the FULL measured range of α as
a smooth exponent — no saturating clamp, so MNIST and CIFAR-10, or clean and IPM, can
genuinely produce different thresholds — and (b) stays proportional to the CURRENT
robust scale each round rather than growing without bound in `t`.

Point (b) matters for a reason the single-machine literature this design started from
does not need to consider: an unboundedly growing `tau_t` eventually admits an
adversarial contribution of any magnitude, at which point Byz-Clip21-SGD2M's own
Byzantine-robustness argument (which relies on `tau` bounding any single client's
contribution) stops holding. **A threshold that is heavy-tail-optimal in the
single-machine sense and a threshold that preserves Byzantine robustness are in direct
tension once you require the threshold to grow with `T`** — this project has not seen
that tension stated elsewhere and believes it is a genuine, citable observation in its
own right, independent of whatever `evt_quantile_tau`'s empirical performance turns out
to be. Tying `tau_t` to `σ̂_t` sidesteps the tension (bounded scale ⇒ bounded threshold)
at the cost of not literally reproducing the moment-bound theorem's asymptotic guarantee.

## Where the honest gap is — read this before claiming a theorem

1. **The setting is different from what any cited theorem proves.** Neither
   arXiv:2410.16561 (single-machine NSGDC) nor standard Hill-estimator EVT theory (i.i.d.
   tail behavior, not a moving target inside a federated, Byzantine, EF21 recursion)
   was derived for this combination. A full convergence proof for
   `TailAdaptiveClip21SGD2M` (either mode) would need to combine:
   - Islamov et al.'s EF21 + Byzantine-robust-aggregation proof structure (their
     Algorithm 1 / Theorem 5.2-style argument) for the `g_i`/`m_i`/`RAgg` recursion, and
   - a heavy-tailed high-probability argument (Gorbunov et al. 2020-style, or the
     arXiv:2410.16561 argument, adapted to a quantile- rather than moment-based
     threshold) for the noise term inside that recursion, and
   - a concentration argument for `α̂_t`/`σ̂_t` themselves being *estimated online from a
     rolling window* rather than known constants — the threshold depends on noisy,
     evolving estimates of its own inputs, which none of the above proofs handle as
     stated.
   That combination has not been done, here or, to this project's knowledge, anywhere
   else. It is real, substantial, separate theoretical work — a conjecture with a stated
   reduction, not a checked theorem.
2. **`n_byz_assumed` trimming is a convention, not a proven robust estimator.** The Hill
   estimator's own robustness under adversarial corruption of its input sample (as
   opposed to the aggregator's robustness, which Islamov et al. do prove) has not been
   separately analyzed here.
3. **`q_target` is a design knob, not a derived optimum.** EVT gives the FORM of the
   quantile relationship; it does not, by itself, say which exceedance probability is
   "correct" for this joint Byzantine/DP/federated setting. Treat `q_target` as a
   hyperparameter to sweep, not a theoretically fixed value.

## What this document licenses the paper to say

- "We propose a clipping schedule whose functional form is motivated by, and directly
  analogous to, the heavy-tailed clipped-SGD literature's own prescription, instantiated
  with an online, Byzantine-robust estimate of the tail index and scale."
- "We conjecture, but do not prove, that this schedule extends Byz-Clip21-SGD2M's
  convergence guarantee to a genuinely heavy-tailed noise regime; a full proof would
  need to combine [the three items above], which we leave as future theoretical work."

## What it does NOT license

- Any sentence of the form "we prove Theorem N: TailAdaptiveClip21SGD2M converges under
  heavy-tailed noise" without the three-part reduction above spelled out as *unproven*.
- Treating the measured Hill α values (5.4–8.0) as evidence the algorithm operates in the
  cited theorem's assumed `p ∈ (1,2]` regime. They don't; say so.
