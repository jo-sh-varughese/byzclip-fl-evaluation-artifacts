"""Generates the paper's result figures as vector PDFs (paper/figures/).

Palette: fixed categorical order per the project's dataviz convention --
slot1 blue #2a78d6 (clean), slot2 orange #eb6834 (ipm), slot3 aqua #1baf7a
(label_flip) -- never reassigned across figures, never cycled. Static
print output (light surface only; no dark-mode/hover layer, which don't
apply to a PDF figure embedded in a paper).

Run again after the CIFAR-10 seed scale-up finishes to regenerate with
final n=10/n=5 numbers.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = os.path.join(os.path.dirname(__file__), "..")
MNIST_DIR = os.path.join(ROOT, "results", "mnist", "partB_scaleup")
CIFAR_DIR = os.path.join(ROOT, "results", "cifar10")
FIG_DIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BLUE = "#2a78d6"     # slot 1 -- clean
ORANGE = "#eb6834"   # slot 2 -- ipm
AQUA = "#1baf7a"      # slot 3 -- label_flip
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9,
})

COND_COLOR = {"clean": BLUE, "ipm": ORANGE, "label_flip": AQUA}
COND_LABEL = {"clean": "clean", "ipm": "IPM attack", "label_flip": "label-flip attack"}


def load_acc(results_dir, stage, name):
    path = os.path.join(results_dir, f"{stage}__{name}.json")
    with open(path) as f:
        return json.load(f)["final_test_acc"]


def load_cell(results_dir, stage, template, seeds):
    vals = []
    for s in seeds:
        try:
            vals.append(load_acc(results_dir, stage, template.format(seed=s)))
        except FileNotFoundError:
            break
    return vals


def mean_std(vals):
    n = len(vals)
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / max(n - 1, 1)
    return m, var ** 0.5, n


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def grouped_bar_figure(cells, conditions, group_labels, title, ylabel, out_path, chance_line=0.10):
    """cells: dict[(group_idx, condition)] -> (mean, std, n)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.1), dpi=200)
    n_groups = len(group_labels)
    n_cond = len(conditions)
    width = 0.8 / n_cond
    x = range(n_groups)

    for i, cond in enumerate(conditions):
        means = [cells[(g, cond)][0] for g in range(n_groups)]
        stds = [cells[(g, cond)][1] for g in range(n_groups)]
        ns = [cells[(g, cond)][2] for g in range(n_groups)]
        offsets = [xi + (i - (n_cond - 1) / 2) * width for xi in x]
        ax.bar(offsets, means, width=width * 0.92, color=COND_COLOR[cond],
               label=f"{COND_LABEL[cond]} (n={ns[0]})", zorder=3)
        ax.errorbar(offsets, means, yerr=stds, fmt="none", ecolor=INK, elinewidth=1,
                     capsize=2.5, capthick=1, zorder=4)

    ax.axhline(chance_line, color=INK_MUTED, linewidth=1, linestyle=(0, (3, 2)), zorder=2)
    ax.text(n_groups - 0.52, chance_line + 0.012, "chance (10%)", color=INK_MUTED,
            fontsize=7.5, ha="right", va="bottom")

    ax.set_xticks(list(x))
    ax.set_xticklabels(group_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK, fontsize=10, loc="left", pad=10)
    ax.set_ylim(0, max(0.75, max(c[0] + c[1] for c in cells.values()) * 1.15))
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=7.5, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_c1_vs_b2():
    """Side-by-side: MNIST B2 vs CIFAR-10 C1, same y-scale, same conditions/epsilons."""
    conditions = ["clean", "ipm", "label_flip"]
    epsilons = [8, 18]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), dpi=200, sharey=True)
    for ax, (dataset_name, results_dir, stage) in zip(
        axes, [("MNIST (B2)", MNIST_DIR, "B2_main"), ("CIFAR-10 (C1)", CIFAR_DIR, "C1_main_iid")]
    ):
        n_cond = len(conditions)
        width = 0.8 / n_cond
        x = range(len(epsilons))
        max_n_seen = 0
        for i, cond in enumerate(conditions):
            cells = [load_cell(results_dir, stage, f"{cond}_eps{eps}_seed{{seed}}", range(10)) for eps in epsilons]
            stats = [mean_std(c) if c else (0, 0, 0) for c in cells]
            means = [s[0] for s in stats]
            stds = [s[1] for s in stats]
            ns = [s[2] for s in stats]
            max_n_seen = max(max_n_seen, max(ns) if ns else 0)
            offsets = [xi + (i - (n_cond - 1) / 2) * width for xi in x]
            ax.bar(offsets, means, width=width * 0.92, color=COND_COLOR[cond],
                   label=COND_LABEL[cond] if ax is axes[0] else None, zorder=3)
            ax.errorbar(offsets, means, yerr=stds, fmt="none", ecolor=INK, elinewidth=1,
                         capsize=2.5, capthick=1, zorder=4)
        ax.axhline(0.10, color=INK_MUTED, linewidth=1, linestyle=(0, (3, 2)), zorder=2)
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"$\\varepsilon$={e}" for e in epsilons])
        ax.set_title(f"{dataset_name}, n={max_n_seen}/cell", color=INK, fontsize=10, loc="left", pad=8)
        style_axes(ax)

    axes[0].set_ylabel("final test accuracy")
    axes[0].set_ylim(0, 0.75)
    axes[1].text(1.45, 0.115, "chance (10%)", color=INK_MUTED, fontsize=7.5, ha="right", va="bottom")
    axes[0].legend(frameon=False, loc="upper left", fontsize=7.5)
    fig.suptitle("DP+Byzantine main sweep: MNIST reaches above chance, CIFAR-10 does not",
                 fontsize=10, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIG_DIR, "fig_mnist_vs_cifar_main_sweep.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_ablation_recovery():
    """MNIST B5 vs CIFAR-10 C3: does no_clip_no_dp recover accuracy under IPM attack?"""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), dpi=200)

    # MNIST B5: no_momentum vs no_clip_no_dp (both IPM-active ablations, no direct "clean" arm on disk)
    mnist_cells = {
        "no_momentum": load_cell(MNIST_DIR, "B5_ablation", "no_momentum_seed{seed}", range(10)),
        "no_clip_no_dp": load_cell(MNIST_DIR, "B5_ablation", "no_clip_no_dp_seed{seed}", range(10)),
    }
    # CIFAR-10 C3: clean control vs no_clip_no_dp+ipm ablation (paired arms, by design)
    cifar_cells = {
        "clean\ncontrol": load_cell(CIFAR_DIR, "C3_ablation", "clean_no_dp_T80_control_seed{seed}", range(10)),
        "no_clip_no_dp\n+ ipm": load_cell(CIFAR_DIR, "C3_ablation", "no_clip_no_dp_ipm_seed{seed}", range(10)),
    }

    for ax, (title, cells, colors) in zip(
        axes,
        [("MNIST (B5), IPM active", mnist_cells, [ORANGE, BLUE]),
         ("CIFAR-10 (C3)", cifar_cells, [BLUE, ORANGE])],
    ):
        labels = list(cells.keys())
        stats = [mean_std(cells[l]) if cells[l] else (0, 0, 0) for l in labels]
        means = [s[0] for s in stats]
        stds = [s[1] for s in stats]
        ns = [s[2] for s in stats]
        xpos = range(len(labels))
        ax.bar(xpos, means, width=0.55, color=colors[: len(labels)], zorder=3)
        ax.errorbar(xpos, means, yerr=stds, fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=4)
        for xi, m, s, n in zip(xpos, means, stds, ns):
            ax.text(xi, m + s + 0.02, f"n={n}", ha="center", fontsize=7.5, color=INK_SECONDARY)
        ax.axhline(0.10, color=INK_MUTED, linewidth=1, linestyle=(0, (3, 2)), zorder=2)
        ax.set_xticks(list(xpos))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
        ax.set_ylim(0, 0.85)
        style_axes(ax)

    axes[0].set_ylabel("final test accuracy")
    fig.suptitle("Isolating ablation: does removing clip+DP noise recover accuracy under attack?",
                 fontsize=10, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIG_DIR, "fig_ablation_recovery.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_convergence_traces():
    """Round-by-round accuracy trace: MNIST vs CIFAR-10, clean condition, seed 0."""
    fig, ax = plt.subplots(figsize=(5.2, 3.1), dpi=200)
    for label, results_dir, stage, name, color in [
        ("MNIST, clean (eps=8)", MNIST_DIR, "B2_main", "clean_eps8_seed0", BLUE),
        ("CIFAR-10, clean (eps=8)", CIFAR_DIR, "C1_main_iid", "clean_eps8_seed0", ORANGE),
    ]:
        path = os.path.join(results_dir, f"{stage}__{name}.json")
        with open(path) as f:
            data = json.load(f)
        rounds = [p["round"] for p in data["accuracy_trace"]]
        accs = [p["test_acc"] for p in data["accuracy_trace"]]
        ax.plot(rounds, accs, color=color, linewidth=2, marker="o", markersize=4, label=label, zorder=3)

    ax.axhline(0.10, color=INK_MUTED, linewidth=1, linestyle=(0, (3, 2)), zorder=2)
    ax.text(78, 0.115, "chance (10%)", color=INK_MUTED, fontsize=7.5, ha="right", va="bottom")
    ax.set_xlabel("communication round")
    ax.set_ylabel("test accuracy")
    ax.set_title("Convergence trace, seed 0, T=80", color=INK, fontsize=10, loc="left", pad=10)
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_convergence_trace.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_c1_vs_b2()
    fig_ablation_recovery()
    fig_convergence_traces()
    print("\nAll figures generated. Re-run after the CIFAR-10 scale-up completes for final n.")
