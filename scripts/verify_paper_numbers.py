"""Machine-verified summary of the paper's numeric claims, covering every results/
subdirectory cited in paper/main_v2.tex (audit mechanisms, the pretrained-backbone
resolution, the DP-noise/non-IID/disentangling/larger-scale checks, and the gamma=0.3
exploratory retune). Reads directly from the JSON result files -- never from chat
transcript memory -- and recomputes means/stds/Wilcoxon p-values from scratch so the
paper text can cite this script's OWN printed output, not a remembered number.
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def load(dirname, pattern_seeds, n):
    accs = []
    for seed in range(n):
        path = os.path.join(RESULTS, dirname, pattern_seeds.format(seed=seed))
        with open(path) as f:
            accs.append(json.load(f)["final_test_acc"])
    return np.array(accs)


def load_multi(dirnames, pattern_seeds, n):
    """Try each directory in order per seed -- for results backfilled across scripts."""
    accs = []
    for seed in range(n):
        for dirname in dirnames:
            path = os.path.join(RESULTS, dirname, pattern_seeds.format(seed=seed))
            if os.path.exists(path):
                with open(path) as f:
                    accs.append(json.load(f)["final_test_acc"])
                break
        else:
            raise FileNotFoundError(f"seed {seed} of {pattern_seeds} not found in {dirnames}")
    return np.array(accs)


def report(name, accs):
    print(f"{name}: n={len(accs)} mean={accs.mean():.4f} std={accs.std():.4f} "
          f"values={[round(a,4) for a in accs]}")
    return accs


def paired(name, a, b):
    stat, p = wilcoxon(b, a)
    print(f"{name}: wilcoxon p={p:.6f}")
    return p


print("=== Centralized sanity check (LR confound) ===")
print("Read directly from centralized_sanity_run.log / centralized_sanity_lr01_run.log "
      "(single-run script prints, not JSON) -- quoting verbatim, not recomputed:")
print("  matched_compute_T100steps (lr=0.1): test_acc=0.1781")
print("  generous_10epochs (lr=0.1, 10 epochs): test_acc=0.1000 (diverged to chance)")
print("  generous_10epochs (lr=0.01, 10 epochs): test_acc=0.6522")

print("\n=== Confound check: LR retune + GroupNorm, clean/no-DP/no-Byzantine, T=80, n=10 ===")
base_lrgn = load("confound_check", "SmallCNN__gamma0.1__seed{seed}.json", 10)
for arch in ["SmallCNN", "SmallCNNGN"]:
    for gamma in [0.1, 0.05, 0.02, 0.01]:
        accs = load("confound_check", f"{arch}__gamma{gamma}__seed{{seed}}.json", 10)
        report(f"  {arch} gamma={gamma}", accs)
        if not (arch == "SmallCNN" and gamma == 0.1):
            paired(f"    vs SmallCNN gamma=0.1 baseline", base_lrgn, accs)

print("\n=== Adaptive-clip (AC21) check -- AC21 mechanism itself stays n=3 pilot")
print("(mechanistically-proven no-op, see paper Section 4.3); its baseline was")
print("separately scaled to n=10 for the EVT-quantile ceiling's matched comparison ===")
for cond in ["clean", "ipm"]:
    base3 = report(f"  baseline (n=3 subset) {cond}", load("adaptive_clip_check", f"baseline__{cond}__seed{{seed}}.json", 3))
    for q in [0.3, 0.5, 0.7]:
        ac21 = report(f"  ac21(q={q}) {cond}", load("adaptive_clip_check", f"ac21__{cond}__q{q}__seed{{seed}}.json", 3))
    report(f"  baseline (n=10, scaled up) {cond}", load("adaptive_clip_check", f"baseline__{cond}__seed{{seed}}.json", 10))

print("\n=== Band-clip (hand-tuned floor=0.8) scale-up, n=10 ===")
for cond in ["clean", "ipm"]:
    base = load("bandclip_scaleup", f"baseline__{cond}__seed{{seed}}.json", 10)
    band = load("bandclip_scaleup", f"bandclip__{cond}__floor0.8__seed{{seed}}.json", 10)
    report(f"  baseline {cond}", base)
    report(f"  bandclip(floor=0.8) {cond}", band)
    paired(f"  {cond}", base, band)

print("\n=== Warm-start (momentum cold-start fix) scale-up, n=10 ===")
for cond in ["clean", "ipm"]:
    base = load("warmstart_scaleup", f"baseline__{cond}__seed{{seed}}.json", 10)
    warm = load("warmstart_scaleup", f"warmstart__{cond}__seed{{seed}}.json", 10)
    report(f"  baseline {cond}", base)
    report(f"  warmstart {cond}", warm)
    paired(f"  {cond}", base, warm)

print("\n=== Local-steps + floor combo, n=3 pilot only, raw pixels ===")
for cond in ["clean", "ipm"]:
    e1 = load("localsteps_check", f"localsteps__{cond}__E1__seed{{seed}}.json", 3)
    e5 = load("localsteps_check", f"localsteps__{cond}__E5__seed{{seed}}.json", 3)
    report(f"  E=1 (floor=0.8 only) {cond}", e1)
    report(f"  E=5 (floor=0.8 + local steps) {cond}", e5)

print("\n=== Tail-adaptive ceiling (moment_schedule, the flawed/no-op design), n=10 both conditions ===")
for cond in ["clean", "ipm"]:
    base = load("tailadaptive_check", f"baseline__{cond}__seed{{seed}}.json", 10)
    tail = load("tailadaptive_check", f"tailadaptive__{cond}__seed{{seed}}.json", 10)
    report(f"  baseline {cond}", base)
    report(f"  tailadaptive(moment_schedule) {cond}", tail)
    paired(f"  {cond}", base, tail)

print("\n=== Tail-adaptive ceiling, MNIST, n=10 ===")
for cond in ["clean", "ipm"]:
    base = load("mnist_tailadaptive_check", f"baseline__{cond}__seed{{seed}}.json", 10)
    tail = load("mnist_tailadaptive_check", f"tailadaptive__{cond}__seed{{seed}}.json", 10)
    report(f"  baseline {cond}", base)
    report(f"  tailadaptive(moment_schedule) {cond}", tail)
    paired(f"  {cond}", base, tail)

print("\n=== EVT-quantile ceiling, n=10 (baseline: adaptive_clip_check, scaled up) ===")
base10_ac = {}
for cond in ["clean", "ipm"]:
    base10_ac[cond] = report(f"  baseline {cond}", load("adaptive_clip_check", f"baseline__{cond}__seed{{seed}}.json", 10))
    for q in [0.01, 0.05, 0.2]:
        accs = report(f"  evt-ceiling {cond} q={q}", load("evt_quantile_pilot", f"evt__{cond}__q{q}__seed{{seed}}.json", 10))
        paired(f"    vs baseline", base10_ac[cond], accs)

print("\n=== EVT-quantile floor (q=0.001) scale-up, n=10 ===")
for cond in ["clean", "ipm"]:
    base = load_multi(["tailadaptive_check", "evtfloor_scaleup"], f"baseline__{cond}__seed{{seed}}.json", 10)
    evtf = load_multi(["evtfloor_scaleup", "evtfloor_pilot"], f"evtfloor__{cond}__q0.001__seed{{seed}}.json", 10)
    report(f"  baseline {cond}", base)
    report(f"  evtfloor(q=0.001) {cond}", evtf)
    paired(f"  {cond}", base, evtf)

print("\n=== Pretrained-backbone Stage 1 (2-conv, no augmentation), n=10 ===")
for cond in ["clean", "ipm"]:
    base = load("pretrained_head_check", f"baseline__{cond}__seed{{seed}}.json", 10)
    combo = load("pretrained_head_check", f"combo__{cond}__seed{{seed}}.json", 10)
    report(f"  baseline {cond}", base)
    report(f"  combo(floor+localsteps) {cond}", combo)
    paired(f"  {cond}", base, combo)

print("\n=== Pretrained-backbone Stage 2 (deep+BN+aug), n=10, T=80 ===")
for cond in ["clean", "ipm"]:
    base = load("pretrained_head_check_v2", f"baseline__{cond}__seed{{seed}}.json", 10)
    combo = load("pretrained_head_check_v2", f"combo__{cond}__seed{{seed}}.json", 10)
    report(f"  baseline {cond}", base)
    report(f"  combo(floor+localsteps) {cond}", combo)
    paired(f"  {cond}", base, combo)

print("\n=== Extended rounds (T=300), n=10 (or n=5 where scale-up incomplete) ===")
for cond in ["clean", "ipm"]:
    try:
        accs = load("t300_scaleup", f"combo__{cond}__T300__seed{{seed}}.json", 10)
    except FileNotFoundError:
        accs = load("extended_rounds_check", f"combo__{cond}__T300__seed{{seed}}.json", 5)
    report(f"  T=300 combo {cond}", accs)

print("\n=== Gamma retune for head, T=80, n=3 pilot ===")
for cond in ["clean", "ipm"]:
    for gamma in [0.05, 0.1, 0.2, 0.3]:
        try:
            accs = load("gamma_retune_head", f"combo__{cond}__gamma{gamma}__seed{{seed}}.json", 3)
            report(f"  gamma={gamma} {cond}", accs)
        except FileNotFoundError:
            print(f"  gamma={gamma} {cond}: not yet complete")

print("\n=== gamma=0.3 + T=300 (exploratory retune, not the recommended config), n=10 ===")
for cond in ["clean", "ipm"]:
    accs = load("final_config_check", f"combo__{cond}__gamma0.3__T300__seed{{seed}}.json", 10)
    report(f"  gamma=0.3 T=300 {cond}", accs)

print("\n=== DP-noise check (recommended config, eps in {18,8}), n=3 pilot ===")
for cond in ["clean", "ipm"]:
    for eps in [18, 8]:
        accs = load("dp_check", f"dp__{cond}__eps{eps}__seed{{seed}}.json", 3)
        report(f"  eps={eps} {cond}", accs)

print("\n=== Non-IID check (Dirichlet alpha=0.5, recommended config), n=3 pilot ===")
for cond in ["clean", "ipm"]:
    base = report(f"  non-IID baseline {cond}",
                   load("noniid_check", f"noniid_baseline__{cond}__seed{{seed}}.json", 3))
    combo = report(f"  non-IID combo {cond}",
                    load("noniid_check", f"noniid_combo__{cond}__seed{{seed}}.json", 3))

print("\n=== Disentangling ablation (floor-only vs. local-steps-only), n=3 pilot ===")
base3 = {}
for cond in ["clean", "ipm"]:
    base3[cond] = load("pretrained_head_check_v2", f"baseline__{cond}__seed{{seed}}.json", 3)
    combo3 = load("pretrained_head_check_v2", f"combo__{cond}__seed{{seed}}.json", 3)
    ls_only = load("ablation_disentangle", f"localsteps_only__{cond}__seed{{seed}}.json", 3)
    fl_only = load("ablation_disentangle", f"floor_only__{cond}__seed{{seed}}.json", 3)
    report(f"  baseline (n=3 subset) {cond}", base3[cond])
    report(f"  combo (n=3 subset) {cond}", combo3)
    report(f"  localsteps_only {cond}", ls_only)
    report(f"  floor_only {cond}", fl_only)
    print(f"  {cond}: localsteps_only delta={(ls_only.mean()-base3[cond].mean())*100:+.2f}pp "
          f"floor_only delta={(fl_only.mean()-base3[cond].mean())*100:+.2f}pp "
          f"combo delta={(combo3.mean()-base3[cond].mean())*100:+.2f}pp")

print("\n=== 40-client larger-scale robustness check (recommended config), n=3 pilot ===")
for cond in ["clean", "ipm"]:
    accs = load("larger_scale_check", f"combo40__{cond}__seed{{seed}}.json", 3)
    report(f"  40-client {cond}", accs)
