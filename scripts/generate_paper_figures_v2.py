"""Figures for paper/main_v2.tex, generated directly from the JSON result files (never
hand-typed numbers) so they cannot drift from the text they illustrate.

Three figures, deliberately kept separate rather than merged into one chart, because
the audit mechanisms and the resolution numbers are not comparable on the same axis:

fig_audit_delta.pdf: for each of the 8 audited mechanisms, the accuracy DELTA from its
OWN correctly seed-matched baseline (not one shared baseline bar across mechanisms
tested at different n -- the n=3 audit mechanisms share one 3-seed baseline, the n=10
ones use a different 10-seed baseline; mixing them in one absolute-accuracy chart would
visually suggest, e.g., that the n=3 adaptive-quantile-ceiling mechanism "beats" the
n=10 baseline bar, when it is in fact numerically IDENTICAL to its own matched baseline).

fig_resolution_absolute.pdf: absolute accuracy through the resolution's own progression
(raw-pixel baseline -> Stage 1 backbone -> Stage 2 backbone -> +floor+local-steps ->
+extended rounds), where absolute numbers are the right comparison because each step
is a genuinely different training regime, not a matched-seed ablation of the same one.

fig_round_trace.pdf: mean accuracy vs. communication round for the T=300 pretrained-head
combo, both conditions, averaged across all n=10 seeds.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_acc(dirname, fname):
    with open(os.path.join(RESULTS, dirname, fname)) as f:
        return json.load(f)["final_test_acc"]


def load_accs(dirname, pattern, seeds):
    return np.array([load_acc(dirname, pattern.format(seed=s)) for s in seeds])


# ---------------------------------------------------------------------------
# Figure 1: audit mechanisms, delta from EACH mechanism's own matched baseline
# ---------------------------------------------------------------------------
base3_clean = load_accs("adaptive_clip_check", "baseline__clean__seed{seed}.json", range(3))
base3_ipm = load_accs("adaptive_clip_check", "baseline__ipm__seed{seed}.json", range(3))
base10_clean = load_accs("tailadaptive_check", "baseline__clean__seed{seed}.json", range(10))
base10_ipm = np.concatenate([
    load_accs("tailadaptive_check", "baseline__ipm__seed{seed}.json", range(3)),
    load_accs("evtfloor_scaleup", "baseline__ipm__seed{seed}.json", range(3, 10)),
])

rows = []


def add(name, base_clean, clean_accs, base_ipm, ipm_accs):
    rows.append((
        name,
        (clean_accs.mean() - base_clean.mean()) * 100,
        (ipm_accs.mean() - base_ipm.mean()) * 100,
    ))


add("Adaptive quantile ceiling ($q{=}0.5$)", base3_clean,
    load_accs("adaptive_clip_check", "ac21__clean__q0.5__seed{seed}.json", range(3)),
    base3_ipm, load_accs("adaptive_clip_check", "ac21__ipm__q0.5__seed{seed}.json", range(3)))
add("Growing schedule ceiling", base10_clean,
    load_accs("tailadaptive_check", "tailadaptive__clean__seed{seed}.json", range(10)),
    load_accs("tailadaptive_check", "baseline__ipm__seed{seed}.json", range(3)),
    load_accs("tailadaptive_check", "tailadaptive__ipm__seed{seed}.json", range(3)))
add("EVT-quantile ceiling ($q{=}0.05$)", base3_clean,
    load_accs("evt_quantile_pilot", "evt__clean__q0.05__seed{seed}.json", range(3)),
    base3_ipm, load_accs("evt_quantile_pilot", "evt__ipm__q0.05__seed{seed}.json", range(3)))
add("Momentum warm-start",
    load_accs("warmstart_scaleup", "baseline__clean__seed{seed}.json", range(10)),
    load_accs("warmstart_scaleup", "warmstart__clean__seed{seed}.json", range(10)),
    load_accs("warmstart_scaleup", "baseline__ipm__seed{seed}.json", range(10)),
    load_accs("warmstart_scaleup", "warmstart__ipm__seed{seed}.json", range(10)))
# (add's signature is (name, base_clean, clean_accs, base_ipm, ipm_accs) -- matches above)
add("Hand-tuned floor (0.8)", base10_clean,
    load_accs("bandclip_scaleup", "bandclip__clean__floor0.8__seed{seed}.json", range(10)),
    base10_ipm, load_accs("bandclip_scaleup", "bandclip__ipm__floor0.8__seed{seed}.json", range(10)))
add("EVT-quantile floor ($q{=}0.001$)", base10_clean,
    load_accs("evtfloor_scaleup", "evtfloor__clean__q0.001__seed{seed}.json", range(10)),
    base10_ipm, load_accs("evtfloor_scaleup", "evtfloor__ipm__q0.001__seed{seed}.json", range(10)))

names = [r[0] for r in rows]
clean_deltas = [r[1] for r in rows]
ipm_deltas = [r[2] for r in rows]

y = np.arange(len(names))
h = 0.35
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.barh(y + h / 2, clean_deltas, height=h, label="Clean", color="#4C72B0")
ax.barh(y - h / 2, ipm_deltas, height=h, label="IPM attack", color="#C44E52")
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=8)
ax.invert_yaxis()
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Accuracy change from each mechanism's own matched baseline (percentage points)")
ax.legend(fontsize=8, loc="lower right")
ax.set_title("Six clipping/momentum mechanisms vs. their own matched baselines\n(none reaches significance; see Table 1)", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_audit_delta.pdf"))
plt.close(fig)
print("Wrote fig_audit_delta.pdf")
for n, cd, idd in rows:
    print(f"  {n}: clean_delta={cd:+.2f}pp  ipm_delta={idd:+.2f}pp")

# ---------------------------------------------------------------------------
# Figure 2: resolution progression, absolute accuracy
# ---------------------------------------------------------------------------
stages = [
    ("Raw-pixel baseline\n($T{=}80$)",
     base10_clean, base10_ipm),
    ("Pretrained backbone\n(Stage 1, no aug)",
     load_accs("pretrained_head_check", "baseline__clean__seed{seed}.json", range(10)),
     load_accs("pretrained_head_check", "baseline__ipm__seed{seed}.json", range(10))),
    ("Pretrained backbone\n(Stage 2, deep+BN+aug)",
     load_accs("pretrained_head_check_v2", "baseline__clean__seed{seed}.json", range(10)),
     load_accs("pretrained_head_check_v2", "baseline__ipm__seed{seed}.json", range(10))),
    ("+ floor-clip + local steps\n($T{=}80$)",
     load_accs("pretrained_head_check_v2", "combo__clean__seed{seed}.json", range(10)),
     load_accs("pretrained_head_check_v2", "combo__ipm__seed{seed}.json", range(10))),
    ("+ extended rounds\n($T{=}300$)",
     load_accs("t300_scaleup", "combo__clean__T300__seed{seed}.json", range(10)),
     load_accs("t300_scaleup", "combo__ipm__T300__seed{seed}.json", range(10))),
]

names2 = [s[0] for s in stages]
clean_means2 = [s[1].mean() * 100 for s in stages]
clean_stds2 = [s[1].std() * 100 for s in stages]
ipm_means2 = [s[2].mean() * 100 for s in stages]
ipm_stds2 = [s[2].std() * 100 for s in stages]

x = np.arange(len(names2))
w = 0.35
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(x - w / 2, clean_means2, width=w, yerr=clean_stds2, label="Clean", color="#4C72B0", capsize=3)
ax.bar(x + w / 2, ipm_means2, width=w, yerr=ipm_stds2, label="IPM attack", color="#C44E52", capsize=3)
ax.set_xticks(x)
ax.set_xticklabels(names2, fontsize=7.5)
ax.axhline(10, color="gray", linestyle=":", linewidth=1, label="chance (10%)")
ax.set_ylabel("CIFAR-10 test accuracy (%)")
ax.set_title("The resolution: each step's absolute accuracy ($n{=}10$)", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_resolution_absolute.pdf"))
plt.close(fig)
print("Wrote fig_resolution_absolute.pdf")
for n, cm, cs, im, ist in zip(names2, clean_means2, clean_stds2, ipm_means2, ipm_stds2):
    print(f"  {n}: clean={cm:.2f}+-{cs:.2f}  ipm={im:.2f}+-{ist:.2f}")

# ---------------------------------------------------------------------------
# Figure 3: accuracy-vs-round trace at T=300, averaged across seeds
# ---------------------------------------------------------------------------
def load_traces(dirname, pattern, n):
    traces = []
    for s in range(n):
        with open(os.path.join(RESULTS, dirname, pattern.format(seed=s))) as f:
            d = json.load(f)
        traces.append(d["accuracy_trace"])
    return traces


def mean_trace(traces):
    rounds = [pt["round"] for pt in traces[0]]
    accs = np.array([[pt["test_acc"] for pt in tr] for tr in traces])
    return rounds, accs.mean(axis=0) * 100, accs.std(axis=0) * 100


clean_traces = load_traces("t300_scaleup", "combo__clean__T300__seed{seed}.json", 10)
ipm_traces = load_traces("t300_scaleup", "combo__ipm__T300__seed{seed}.json", 10)
r_clean, m_clean, s_clean = mean_trace(clean_traces)
r_ipm, m_ipm, s_ipm = mean_trace(ipm_traces)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(r_clean, m_clean, label="Clean", color="#4C72B0")
ax.fill_between(r_clean, m_clean - s_clean, m_clean + s_clean, color="#4C72B0", alpha=0.2)
ax.plot(r_ipm, m_ipm, label="IPM attack", color="#C44E52")
ax.fill_between(r_ipm, m_ipm - s_ipm, m_ipm + s_ipm, color="#C44E52", alpha=0.2)
ax.axvline(80, color="gray", linestyle="--", linewidth=1, label="$T{=}80$ (audit budget)")
ax.set_xlabel("Communication round")
ax.set_ylabel("CIFAR-10 test accuracy (%)")
ax.set_title("Pretrained-backbone resolution, $T{=}300$ ($n{=}10$, mean$\\pm$std)", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_round_trace.pdf"))
plt.close(fig)
print("Wrote fig_round_trace.pdf")
print(f"  clean checkpoints: {list(zip(r_clean, np.round(m_clean, 2)))}")
print(f"  ipm checkpoints: {list(zip(r_ipm, np.round(m_ipm, 2)))}")
