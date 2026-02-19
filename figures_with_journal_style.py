"""
figures_with_journal_style.py
==============================
Demonstrates how journal_style.py replaces the manual mpl.rcParams blocks
in figures.ipynb, plus additional use cases.

Structure
---------
  Part 1  â€” Direct mapping: reproduce every figure from figures.ipynb
  Part 2  â€” Additional examples beyond the original notebook

Run as a script:
    python figures_with_journal_style.py

Or copy individual cells into a Jupyter notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# The only style import needed â€” replaces all mpl.rcParams.update({...}) blocks
from journal_style import journal_style, set_color_style, set_grayscale_style, reset_style

np.random.seed(0)

# ============================================================
# Shared synthetic data (same as figures.ipynb)
# ============================================================

x         = np.linspace(0, 10, 60)
y         = 2 * x + np.random.normal(0, 5, size=len(x))
categories = ["A", "B", "C"]
means      = [5, 7, 4]
errors     = [0.8, 1.2, 0.6]
sample1    = np.random.normal(0, 1, 200)
sample2    = np.random.normal(1, 1, 200)
heat       = np.random.rand(10, 10)
perf       = np.abs(np.random.randn(30, 3)) + 1
ratio      = perf / perf.min(axis=1, keepdims=True)


# ============================================================
# PART 1 â€” Direct mapping from figures.ipynb
# ============================================================
# BEFORE (figures.ipynb):
#   mpl.rcParams.update({... 20 lines ...})
#   LINESTYLES = ['-', '--', ':', '-.']
#   MARKERS    = ['o', 's', '^', 'D']
#   GRAY_COLORS = ['black', 'dimgray', 'gray', 'darkgray']
#
# AFTER (journal_style):
#   with journal_style("nature", mode="grayscale") as p:
#       ...use p.linestyles, p.markers, p.colors, p.figsize_double...
# ============================================================

# --------------------------------------------------
# Figure 1 â€” Grayscale advanced panel (figures.ipynb cell 2)
#   Scatter + CI / Error bars / ECDF / Heatmap / Violin / Performance profile
# --------------------------------------------------
with journal_style("nature", mode="grayscale") as p:

    fig = plt.figure(figsize=p.figsize_double)   # replaces hard-coded (7.2, 8)
    gs  = fig.add_gridspec(4, 2, hspace=0.7, wspace=0.35)

    # --- Scatter + model fit + CI ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(x, y, s=10, color=p.colors[0])
    slope, intercept, *_ = stats.linregress(x, y)
    line = slope * x + intercept
    ci   = 1.96 * np.std(y - line)
    ax1.plot(x, line, color=p.colors[0], linestyle="-")
    ax1.fill_between(x, line - ci, line + ci, color=p.colors[1], alpha=p.fill_alpha)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("Model fit + CI")

    # --- Point + error bar ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.errorbar(categories, means, yerr=errors,
                 fmt=p.markers[0], color=p.colors[0], capsize=3)
    ax2.set_ylabel("Value")
    ax2.set_title("Mean Â± CI")

    # --- ECDF ---
    ax3 = fig.add_subplot(gs[1, 0])
    for data, ls, mk in zip([sample1, sample2], p.linestyles, p.markers):
        sd   = np.sort(data)
        ecdf = np.arange(1, len(sd) + 1) / len(sd)
        ax3.plot(sd, ecdf, linestyle=ls, marker=mk, markevery=25, color=p.colors[0])
    ax3.set_xlabel("Value")
    ax3.set_ylabel("ECDF")
    ax3.set_title("ECDF comparison")

    # --- Heatmap ---
    ax4 = fig.add_subplot(gs[1, 1])
    sns.heatmap(heat, cmap="cividis", cbar=False, ax=ax4)
    ax4.set_title("Heatmap")

    # --- Violin ---
    ax5 = fig.add_subplot(gs[2, :])
    sns.violinplot(data=[sample1, sample2], inner="box", color=p.colors[1], ax=ax5)
    ax5.set_xticks([0, 1])
    ax5.set_xticklabels(["Method A", "Method B"])
    ax5.set_ylabel("Error")
    ax5.set_title("Distribution comparison")

    # --- Performance profile ---
    ax6 = fig.add_subplot(gs[3, :])
    for i, (col, ls, mk) in enumerate(p.cycle(ratio.shape[1])):
        r = np.sort(ratio[:, i])
        prob = np.arange(1, len(r) + 1) / len(r)
        ax6.step(r, prob, where="post", linestyle=ls, color=col, label=f"Alg {i+1}")
    ax6.set_xlabel("Performance ratio")
    ax6.set_ylabel("Proportion of instances")
    ax6.set_title("Performance profile")
    ax6.legend()

    fig.savefig("fig1_grayscale_panel.pdf")
    fig.savefig("fig1_grayscale_panel.tiff")
    plt.close(fig)
    print(f"Fig 1 saved  | journal=nature | figsize={p.figsize_double}")


# --------------------------------------------------
# Figure 2 â€” Color panel (figures.ipynb cell 1 equivalent)
#   Box / Bar / Multi-line / Residuals â€” switched to COLOR mode
# --------------------------------------------------
with journal_style("nature", mode="color") as p:

    box_data = {cat: np.random.normal(m, e * 3, 50)
                for cat, m, e in zip(categories, means, errors)}

    fig = plt.figure(figsize=p.figsize_double)
    gs  = fig.add_gridspec(3, 2, hspace=0.65, wspace=0.35)

    # --- Box plot ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.boxplot(list(box_data.values()), tick_labels=list(box_data.keys()))
    ax1.set_ylabel("Value")
    ax1.set_title("Box plot")

    # --- Bar + error ---
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (cat, m, e, col) in enumerate(zip(categories, means, errors, p.colors)):
        ax2.bar(i, m, yerr=e, color=col, capsize=3, width=0.6)
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("Value")
    ax2.set_title("Bar + error")

    # --- Multi-line series using palette.cycle() ---
    ax3 = fig.add_subplot(gs[1, :])
    t = np.linspace(0, 2 * np.pi, 120)
    for i, (col, ls, mk) in enumerate(p.cycle(4)):
        ax3.plot(t, np.sin(t + i * 0.5) * (1 - 0.15 * i),
                 color=col, linestyle=ls, marker=mk, markevery=20,
                 label=f"Series {i+1}")
    ax3.set_xlabel("t (rad)")
    ax3.set_ylabel("Amplitude")
    ax3.set_title("Multi-series (palette.cycle)")
    ax3.legend(ncol=2)

    # --- Scatter + residuals ---
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(x, y, s=8, color=p.colors[0], label="Data")
    slope, intercept, *_ = stats.linregress(x, y)
    ax4.plot(x, slope * x + intercept, color=p.colors[1], label="Fit")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_title("Scatter + fit")
    ax4.legend()

    ax5 = fig.add_subplot(gs[2, 1])
    residuals = y - (slope * x + intercept)
    markerline, stemlines, baseline = ax5.stem(x, residuals)
    plt.setp(stemlines, color=p.colors[2], linewidth=0.8)
    plt.setp(markerline, color=p.colors[2], markersize=3)
    plt.setp(baseline, color="black", linewidth=0.5)
    ax5.axhline(0, lw=0.5, color="black")
    ax5.set_xlabel("X")
    ax5.set_ylabel("Residual")
    ax5.set_title("Residuals")

    fig.savefig("fig2_color_panel.pdf")
    plt.close(fig)
    print(f"Fig 2 saved  | journal=nature | figsize={p.figsize_double}")


# ============================================================
# PART 2 â€” Additional examples
# ============================================================


# --------------------------------------------------
# Example A â€” Single-column figure (Elsevier submission)
#   Demonstrates switching journal preset + single column width
# --------------------------------------------------
with journal_style("elsevier", mode="color") as p:

    fig, ax = plt.subplots(figsize=p.figsize_single)   # 3.54 Ã— 2.66 in

    for i, (col, ls, mk) in enumerate(p.cycle(3)):
        ax.plot(t, np.cos(t + i * 0.7),
                color=col, linestyle=ls, marker=mk, markevery=15,
                label=f"Condition {i+1}")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Response")
    ax.set_title("Elsevier â€” single column")
    ax.legend()

    fig.savefig("figA_elsevier_single.pdf")
    plt.close(fig)
    print(f"Fig A saved  | journal=elsevier | figsize={p.figsize_single}")


# --------------------------------------------------
# Example B â€” IEEE double-column with custom color palette
#   Shows how to inject domain-specific colors (e.g. brand palette)
# --------------------------------------------------
CUSTOM_COLORS = ["#003366", "#E31937", "#00843D", "#FF8200"]   # e.g. institution brand

with journal_style("ieee", mode="color", colors=CUSTOM_COLORS) as p:

    fig, axes = plt.subplots(1, 2, figsize=p.figsize_double)

    # Left: CDF of runtime
    runtimes = [np.random.gamma(shape=k, scale=0.5, size=300) for k in [2, 4, 6, 8]]
    for idx, (rt, (col, ls, mk)) in enumerate(zip(runtimes, p.cycle(4))):
        sd = np.sort(rt)
        axes[0].plot(sd, np.linspace(0, 1, len(sd)),
                     color=col, linestyle=ls, label=f"k={2*(len(runtimes)-idx)}")
    axes[0].set_xlabel("Runtime (s)")
    axes[0].set_ylabel("CDF")
    axes[0].set_title("Runtime distribution")
    axes[0].legend()

    # Right: stacked bar (throughput breakdown)
    stages = ["Preprocess", "Solve", "Post"]
    times  = np.array([[1.2, 3.5, 0.8],
                        [1.5, 2.8, 0.9],
                        [0.9, 4.1, 0.7],
                        [1.1, 3.0, 1.0]])
    bottom = np.zeros(len(runtimes))
    for j, (stage, col) in enumerate(zip(stages, p.colors)):
        axes[1].bar(range(len(runtimes)), times[:, j],
                    bottom=bottom, color=col, label=stage, width=0.6)
        bottom += times[:, j]
    axes[1].set_xticks(range(len(runtimes)))
    axes[1].set_xticklabels([f"Cfg {k+1}" for k in range(len(runtimes))])
    axes[1].set_ylabel("Time (s)")
    axes[1].set_title("Stacked runtime breakdown")
    axes[1].legend()

    fig.savefig("figB_ieee_custom_colors.pdf")
    plt.close(fig)
    print(f"Fig B saved  | journal=ieee  | figsize={p.figsize_double}")


# --------------------------------------------------
# Example C â€” PLOS ONE: grayscale + fill_between confidence bands
#   Demonstrates fill_alpha and hatches for accessibility
# --------------------------------------------------
with journal_style("plos", mode="grayscale") as p:

    fig, ax = plt.subplots(figsize=p.figsize_single)

    ts = np.linspace(0, 4 * np.pi, 200)
    signals = [np.sin(ts) * np.exp(-0.1 * ts),
               np.cos(ts) * np.exp(-0.08 * ts)]
    labels  = ["Damped sine", "Damped cosine"]

    for sig, label, (col, ls, mk) in zip(signals, labels, p.cycle(2)):
        noise = np.random.normal(0, 0.05, len(ts))
        ax.plot(ts, sig, color=col, linestyle=ls, label=label)
        ax.fill_between(ts, sig - 0.15, sig + 0.15,
                        color=col, alpha=p.fill_alpha, linewidth=0)

    ax.set_xlabel("t (rad)")
    ax.set_ylabel("Amplitude")
    ax.set_title("PLOS â€” grayscale + CI bands")
    ax.legend()

    fig.savefig("figC_plos_grayscale_bands.pdf")
    plt.close(fig)
    print(f"Fig C saved  | journal=plos  | figsize={p.figsize_single}")


# --------------------------------------------------
# Example D â€” Mixed styles in the same script WITHOUT global pollution
#   key: each `with` block restores rcParams on exit
# --------------------------------------------------
print("\nFig D: verifying no style leakage between journals")

with journal_style("acs", mode="color") as p_acs:
    fig, ax = plt.subplots(figsize=p_acs.figsize_single)
    ax.plot(t[:60], np.sin(t[:60]), color=p_acs.colors[0])
    ax.set_title(f"ACS color  | width={p_acs.figsize_single[0]:.2f} in")
    fig.savefig("figD_acs_single.pdf")
    plt.close(fig)

with journal_style("nature", mode="grayscale") as p_nat:
    fig, ax = plt.subplots(figsize=p_nat.figsize_double)
    for col, ls, mk in p_nat.cycle(3):
        ax.plot(t[:60], np.sin(t[:60] + 0.5), color=col, linestyle=ls, marker=mk, markevery=10)
    ax.set_title(f"Nature grayscale | width={p_nat.figsize_double[0]:.2f} in")
    fig.savefig("figD_nature_double.pdf")
    plt.close(fig)

print("  ACS  single col:", round(p_acs.figsize_single[0], 2), "in  â†’  nature double:", round(p_nat.figsize_double[0], 2), "in")
print("  rcParams restored to default after each block.")
print("\nAll figures saved to current directory.")
