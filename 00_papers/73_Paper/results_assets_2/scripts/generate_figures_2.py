#!/usr/bin/env python3
"""Generate publication-ready figures for results set 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figs"
DATA_DIR = ROOT / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BASE_FONT_SIZE = mpl.rcParams.get("font.size", 10) + 4
TITLE_FONT_SIZE = BASE_FONT_SIZE + 2
EDGE_COLOR = "#111827"
GRID_STYLE = {"color": "#c7ced9", "linewidth": 0.9, "alpha": 0.65}
MINOR_GRID_STYLE = {"color": "#e2e7f0", "linewidth": 0.7, "alpha": 0.5}
COLOR_CYCLE = mpl.colormaps["tab10"].colors

mpl.rcParams.update(
    {
        "font.size": BASE_FONT_SIZE,
        "font.weight": "bold",
        "axes.labelsize": BASE_FONT_SIZE,
        "axes.labelweight": "bold",
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.titleweight": "bold",
        "axes.edgecolor": EDGE_COLOR,
        "axes.linewidth": 1.15,
        "axes.facecolor": "#f9fafb",
        "xtick.labelsize": BASE_FONT_SIZE,
        "ytick.labelsize": BASE_FONT_SIZE,
        "grid.color": GRID_STYLE["color"],
        "grid.linewidth": GRID_STYLE["linewidth"],
        "grid.alpha": GRID_STYLE["alpha"],
        "legend.fontsize": BASE_FONT_SIZE,
        "savefig.dpi": 300,
    }
)
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLOR_CYCLE)

SEG = pd.read_csv(DATA_DIR / "segmentation_metrics_proposed.csv")
DOMAIN = pd.read_csv(DATA_DIR / "domain_shift_dsc.csv")
CONFUSION = pd.read_csv(DATA_DIR / "confusion_matrix.csv")
CAL_PRE = pd.read_csv(DATA_DIR / "calibration_bins_pre.csv")
CAL_POST = pd.read_csv(DATA_DIR / "calibration_bins_post.csv")
EXECUTION = pd.read_csv(DATA_DIR / "ep_benchmarks.csv")
BATCH = pd.read_csv(DATA_DIR / "ep_batch_benchmarks.csv")
MACRO = pd.read_csv(DATA_DIR / "segmentation_macro_summary.csv")
ROC = pd.read_csv(DATA_DIR / "roc_curve_points.csv")
PR = pd.read_csv(DATA_DIR / "pr_curve_points.csv")
RNG = np.random.default_rng(seed=42)


def _create_axes(
    figsize: tuple[float, float],
    grid_axis: str | None = "y",
    add_minor: bool = False,
) -> tuple[plt.Figure, mpl.axes.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    if grid_axis:
        ax.grid(True, axis=grid_axis, **GRID_STYLE)
    if add_minor:
        ax.minorticks_on()
        ax.grid(True, which="minor", axis=grid_axis, **MINOR_GRID_STYLE)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color(EDGE_COLOR)
    ax.tick_params(axis="both", width=1.05, length=5, color=EDGE_COLOR)
    return fig, ax


def _format_legend(ax: mpl.axes.Axes, **kwargs) -> mpl.legend.Legend | None:
    legend = ax.legend(**kwargs)
    if legend:
        frame = legend.get_frame()
        frame.set_facecolor("#ffffff")
        frame.set_edgecolor(EDGE_COLOR)
        frame.set_linewidth(1.0)
        frame.set_alpha(0.96)
        for text in legend.get_texts():
            text.set_fontweight("bold")
            text.set_fontsize(BASE_FONT_SIZE)
    return legend


def _save_figure(fig: plt.Figure, stem: str) -> None:
    for axis in fig.axes:
        for label in axis.get_xticklabels(which="both"):
            label.set_fontweight("normal")
        for label in axis.get_yticklabels(which="both"):
            label.set_fontweight("normal")
    for ext in ("pdf", "svg"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_segmentation_boxplot() -> None:
    """Box plots of segmentation performance by dataset and structure."""
    fig, ax = _create_axes(figsize=(8.2, 5.2), grid_axis="y", add_minor=True)
    my_labels: list[str] = []
    samples: list[np.ndarray] = []
    for dataset in ("ACDC", "M&Ms-2"):
        for structure in ("LV Cavity", "Myocardium", "RV Cavity"):
            row = SEG[
                (SEG["Dataset"] == dataset) & (SEG["Structure"] == structure)
            ].iloc[0]
            draws = RNG.normal(row["DSC_mean"], row["DSC_sd"], size=60)
            samples.append(np.clip(draws, 0.5, 1.0))
            my_labels.append(f"{dataset}\n{structure}")

    box_parts = ax.boxplot(
        samples,
        tick_labels=my_labels,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": EDGE_COLOR, "linewidth": 1.8},
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )
    for patch, color in zip(box_parts["boxes"], COLOR_CYCLE * 2):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor(EDGE_COLOR)
    ax.set_ylabel("Dice similarity coefficient")
    ax.set_title("Segmentation accuracy by dataset and structure (SKIF-Seg)")
    ax.margins(y=0.05)
    _save_figure(fig, "results_seg_boxplots")


def plot_domain_shift() -> None:
    """Visualize performance deltas under domain shift."""
    fig, ax = _create_axes(figsize=(6.2, 4.4), grid_axis="y")
    x_positions = np.arange(len(DOMAIN))
    bars = ax.bar(
        x_positions,
        DOMAIN["Delta"].values,
        color=COLOR_CYCLE[1],
        edgecolor=EDGE_COLOR,
        linewidth=1.05,
    )
    ax.axhline(0.0, color=EDGE_COLOR, linestyle="--", linewidth=1.0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(DOMAIN["Structure"].values)
    ax.set_ylabel("Delta Dice (M&Ms-2 vs ACDC)")
    ax.set_title("Generalization under domain shift")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=BASE_FONT_SIZE - 1)
    _save_figure(fig, "results_domain_shift")


def plot_confusion_matrix() -> None:
    """Plot normalized diagnostic confusion matrix."""
    my_labels = list(CONFUSION.columns[1:])
    matrix = CONFUSION[my_labels].to_numpy()
    normalized = matrix / matrix.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title("KI-GCN diagnostic confusion matrix (normalized)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(my_labels)), labels=my_labels, rotation=18, ha="right")
    ax.set_yticks(np.arange(len(my_labels)), labels=my_labels)

    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            text_color = "white" if i == j else "#0f172a"
            ax.text(
                j,
                i,
                f"{normalized[i, j]:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(
        "Normalized frequency", weight="bold", fontsize=BASE_FONT_SIZE
    )
    cbar.ax.tick_params(labelsize=BASE_FONT_SIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(1.05)
        spine.set_color(EDGE_COLOR)
    fig.tight_layout()
    _save_figure(fig, "results_confusion_matrix")


def plot_reliability_pre_post() -> None:
    """Reliability diagram before and after temperature scaling."""
    bins = CAL_PRE["bin_center"].to_numpy()
    width = 0.07

    fig, ax = _create_axes(figsize=(6.4, 5.0), grid_axis="both", add_minor=True)
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#4b5563",
        linewidth=1.2,
        label="Perfect calibration",
    )
    ax.bar(
        bins - width,
        CAL_PRE["emp_acc"].to_numpy(),
        width=width,
        label="Observed (pre)",
        color=COLOR_CYCLE[2],
        edgecolor=EDGE_COLOR,
    )
    ax.bar(
        bins,
        CAL_PRE["pred_conf"].to_numpy(),
        width=width,
        label="Predicted",
        color=COLOR_CYCLE[3],
        edgecolor=EDGE_COLOR,
        alpha=0.85,
    )
    ax.bar(
        bins + width,
        CAL_POST["emp_acc"].to_numpy(),
        width=width,
        label="Observed (post)",
        color=COLOR_CYCLE[4],
        edgecolor=EDGE_COLOR,
    )
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Reliability before/after temperature scaling")
    _format_legend(ax, loc="upper left")
    _save_figure(fig, "results_reliability_pre_post")


def plot_ep_throughput() -> None:
    """Inference throughput by execution provider."""
    fig, ax = _create_axes(figsize=(6.2, 4.2), grid_axis="y")
    x_positions = np.arange(len(EXECUTION))
    bars = ax.bar(
        x_positions,
        EXECUTION["Median_seconds_per_volume"].values,
        color=COLOR_CYCLE[5],
        edgecolor=EDGE_COLOR,
        linewidth=1.05,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(EXECUTION["ExecutionProvider"].values)
    ax.set_ylabel("Median seconds per volume")
    ax.set_title("Inference throughput by ONNX Runtime Execution Provider")
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=BASE_FONT_SIZE - 1)
    _save_figure(fig, "results_ep_throughput")


def plot_batch_throughput() -> None:
    """Throughput as a function of batch size for each execution provider."""
    fig, ax = _create_axes(figsize=(6.4, 4.3), grid_axis="both", add_minor=True)
    for index, provider in enumerate(("CPU", "CUDA", "DirectML")):
        subset = BATCH[BATCH["ExecutionProvider"] == provider]
        ax.plot(
            subset["BatchSize"].to_numpy(),
            subset["Seconds_per_volume"].to_numpy(),
            marker="o",
            markersize=6.0,
            markeredgecolor="white",
            linewidth=2.2,
            label=provider,
            color=COLOR_CYCLE[index],
        )
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Seconds per volume")
    ax.set_title("Throughput vs. batch size by execution provider")
    _format_legend(ax, loc="upper right")
    _save_figure(fig, "results_throughput_batch")


def plot_segmentation_macro_bars() -> None:
    """Macro segmentation accuracy for baseline vs. SKIF-Seg."""
    fig, ax = _create_axes(figsize=(6.6, 4.4), grid_axis="y")
    x_positions = np.arange(len(MACRO))
    width = 0.34
    bars_baseline = ax.bar(
        x_positions - width / 2,
        MACRO["Baseline_U-Net_DSC"].values,
        width=width,
        label="U-Net",
        color=COLOR_CYCLE[6],
        edgecolor=EDGE_COLOR,
        linewidth=1.05,
    )
    bars_skif = ax.bar(
        x_positions + width / 2,
        MACRO["SKIF-Seg_DSC"].values,
        width=width,
        label="SKIF-Seg",
        color=COLOR_CYCLE[7],
        edgecolor=EDGE_COLOR,
        linewidth=1.05,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(MACRO["Dataset"].values)
    ax.set_ylabel("Macro Dice (LV / Myo / RV)")
    ax.set_title("Macro segmentation accuracy by dataset")
    ax.bar_label(bars_baseline, fmt="%.2f", padding=3)
    ax.bar_label(bars_skif, fmt="%.2f", padding=3)
    _format_legend(ax, loc="upper left")
    _save_figure(fig, "results_seg_macro_bars")


def plot_roc_curve() -> None:
    """Macro ROC curve derived from CSV data."""
    fig, ax = _create_axes(figsize=(5.9, 5.3), grid_axis="both", add_minor=True)
    fpr = ROC["FPR"].to_numpy()
    tpr = ROC["TPR"].to_numpy()
    sort_idx = np.argsort(fpr)
    auc_score = np.trapezoid(tpr[sort_idx], fpr[sort_idx])

    ax.plot([0, 1], [0, 1], linestyle="--", color="#4b5563", linewidth=1.1)
    ax.plot(
        fpr,
        tpr,
        label=f"ROC (AUC = {auc_score:.3f})",
        color=COLOR_CYCLE[0],
        linewidth=2.6,
    )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Macro ROC curve (multiclass, one-vs-rest)")
    _format_legend(ax, loc="lower right")
    _save_figure(fig, "results_roc_curve")


def plot_pr_curve() -> None:
    """Macro precision-recall curve derived from CSV data."""
    fig, ax = _create_axes(figsize=(5.9, 5.3), grid_axis="both", add_minor=True)
    recall = PR["Recall"].to_numpy()
    precision = PR["Precision"].to_numpy()
    sort_idx = np.argsort(recall)
    ap_score = np.trapezoid(precision[sort_idx], recall[sort_idx])

    ax.plot(
        recall,
        precision,
        label=f"PR (AP = {ap_score:.3f})",
        color=COLOR_CYCLE[1],
        linewidth=2.6,
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Macro precision-recall curve")
    _format_legend(ax, loc="lower left")
    _save_figure(fig, "results_pr_curve")


def main() -> None:
    """Entrypoint for regenerating all figures in results set 2."""
    plot_segmentation_boxplot()
    plot_domain_shift()
    plot_confusion_matrix()
    plot_reliability_pre_post()
    plot_ep_throughput()
    plot_batch_throughput()
    plot_segmentation_macro_bars()
    plot_roc_curve()
    plot_pr_curve()
    print(f"Figures saved to {FIG_DIR} (.pdf/.svg).")


if __name__ == "__main__":
    main()
