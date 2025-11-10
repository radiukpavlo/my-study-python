#!/usr/bin/env python3
"""Render publication-ready diagnostic figures directly from the CSV datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "img"
DATA_DIR = ROOT / "data"

IMG_DIR.mkdir(parents=True, exist_ok=True)

BASE_FONT_SIZE = mpl.rcParams.get("font.size", 10) + 2
mpl.rcParams.update(
    {
        "font.size": BASE_FONT_SIZE,
        "font.weight": "bold",
        "axes.labelsize": BASE_FONT_SIZE,
        "axes.labelweight": "bold",
        "axes.titlesize": BASE_FONT_SIZE + 2,
        "axes.titleweight": "bold",
        "xtick.labelsize": BASE_FONT_SIZE - 1,
        "ytick.labelsize": BASE_FONT_SIZE - 1,
        "legend.fontsize": BASE_FONT_SIZE - 1,
        "legend.edgecolor": "0.65",
        "legend.facecolor": "1.0",
        "legend.framealpha": 0.9,
        "axes.edgecolor": "0.35",
        "axes.linewidth": 1.0,
        "grid.color": "0.78",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.55,
        "savefig.dpi": 220,
        "svg.fonttype": "none",
    }
)

COLOR_CYCLE = mpl.colormaps["tab10"].colors
MAJOR_GRID_STYLE = {"color": "0.78", "linewidth": 0.85, "alpha": 0.6}
MINOR_GRID_STYLE = {"color": "0.9", "linewidth": 0.6, "alpha": 0.4}
CONFUSION_CLASSES = ["NOR", "HCM", "DCM", "MINF", "ARV"]


def _style_axis(
    ax: mpl.axes.Axes,
    grid_axis: str = "both",
    add_minor: bool = False,
) -> None:
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(True, axis=grid_axis, **MAJOR_GRID_STYLE)
    if add_minor:
        ax.minorticks_on()
        ax.grid(True, which="minor", axis=grid_axis, **MINOR_GRID_STYLE)
    for spine in ax.spines.values():
        spine.set_color("0.35")
        spine.set_linewidth(1.0)


def _create_axes(
    figsize: tuple[float, float],
    grid_axis: str = "both",
    add_minor: bool = False,
) -> tuple[plt.Figure, mpl.axes.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    _style_axis(ax, grid_axis=grid_axis, add_minor=add_minor)
    return fig, ax


def _format_legend(ax: mpl.axes.Axes, **kwargs) -> mpl.legend.Legend | None:
    legend = ax.legend(**kwargs)
    if legend:
        frame = legend.get_frame()
        frame.set_linewidth(0.9)
        frame.set_edgecolor("0.65")
    return legend


def _save_svg(fig: plt.Figure, stem: str) -> None:
    out_path = IMG_DIR / f"{stem}.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _load_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, **kwargs)


def plot_segmentation_macro() -> None:
    df = _load_csv("segmentation_macro.csv")
    indices = np.arange(len(df))
    width = 0.32
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), sharex=False)
    for ax in axes:
        _style_axis(ax, grid_axis="y")

    axes[0].bar(
        indices - width / 2,
        df["U-Net_DSC"],
        width=width,
        label="U-Net",
        color=COLOR_CYCLE[0],
        edgecolor="0.25",
        linewidth=0.8,
    )
    axes[0].bar(
        indices + width / 2,
        df["SKIFSeg_DSC"],
        width=width,
        label="SKIF-Seg",
        color=COLOR_CYCLE[1],
        edgecolor="0.25",
        linewidth=0.8,
    )
    axes[0].set_xticks(indices)
    axes[0].set_xticklabels(df["Dataset"])
    axes[0].set_ylabel("Dice coefficient")
    axes[0].set_ylim(0.88, 0.96)
    axes[0].set_title("Macro Dice by dataset")
    _format_legend(axes[0], loc="lower left")

    axes[1].bar(
        indices - width / 2,
        df["U-Net_HD95_mm"],
        width=width,
        label="U-Net",
        color=COLOR_CYCLE[0],
        edgecolor="0.25",
        linewidth=0.8,
    )
    axes[1].bar(
        indices + width / 2,
        df["SKIFSeg_HD95_mm"],
        width=width,
        label="SKIF-Seg",
        color=COLOR_CYCLE[1],
        edgecolor="0.25",
        linewidth=0.8,
    )
    axes[1].set_xticks(indices)
    axes[1].set_xticklabels(df["Dataset"])
    axes[1].set_ylabel("HD95 (mm) ↓")
    axes[1].set_ylim(5.0, 10.5)
    axes[1].set_title("Macro HD95 by dataset")
    _format_legend(axes[1], loc="upper right")

    fig.suptitle("Segmentation macro-level comparison", y=1.02)
    fig.tight_layout()
    _save_svg(fig, "segmentation_macro")


def plot_structure_performance() -> None:
    dataset_files = [
        ("ACDC", "segmentation_acdc.csv"),
        ("M&Ms-2", "segmentation_mnms2.csv"),
    ]
    long_df = []
    for dataset, filename in dataset_files:
        df = _load_csv(filename)
        df = df.assign(Dataset=dataset)
        long_df.append(df)
    combined = pd.concat(long_df, ignore_index=True)

    structures = combined["Structure"].unique()
    width = 0.32
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), sharey=True)

    for ax in axes:
        _style_axis(ax, grid_axis="y")
        ax.set_ylim(0.84, 0.97)
        ax.set_ylabel("Dice coefficient")

    for ax, dataset in zip(axes, ["ACDC", "M&Ms-2"], strict=True):
        subset = combined[combined["Dataset"] == dataset]
        idx = np.arange(len(structures))
        axes_values = subset.set_index("Structure").reindex(structures)
        ax.bar(
            idx - width / 2,
            axes_values["DSC_Unet"],
            width=width,
            label="U-Net",
            color=COLOR_CYCLE[2],
            edgecolor="0.25",
            linewidth=0.8,
        )
        ax.bar(
            idx + width / 2,
            axes_values["DSC_SKIFSeg"],
            width=width,
            label="SKIF-Seg",
            color=COLOR_CYCLE[3],
            edgecolor="0.25",
            linewidth=0.8,
        )
        ax.set_xticks(idx)
        ax.set_xticklabels(structures, rotation=15, ha="right")
        ax.set_title(f"{dataset} structures")

    _format_legend(axes[0], loc="lower left")
    fig.tight_layout()
    _save_svg(fig, "segmentation_structures")


def plot_domain_shift() -> None:
    df = _load_csv("domain_shift.csv")
    fig, ax = _create_axes(figsize=(5.2, 3.4), grid_axis="x")
    ax.axvline(0.0, color="0.3", linewidth=1.0, linestyle="--")
    ax.barh(
        df["Structure"],
        df["DeltaDice"],
        color=COLOR_CYCLE[4],
        edgecolor="0.25",
        linewidth=0.8,
    )
    ax.set_xlabel("Δ Dice (M&Ms-2 − ACDC)")
    ax.set_ylabel("Structure")
    ax.set_title("Domain shift impact on Dice")
    _save_svg(fig, "domain_shift")


def plot_roc_curve() -> None:
    roc = _load_csv("roc_curve.csv")
    auc_value = float(np.trapezoid(roc["tpr"], roc["fpr"]))
    fig, ax = _create_axes(figsize=(5.6, 4.2), add_minor=True)
    ax.plot(
        roc["fpr"],
        roc["tpr"],
        color=COLOR_CYCLE[5],
        linewidth=2.8,
        label=f"Macro ROC (AUC = {auc_value:.3f})",
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="0.4", linewidth=1.2, label="Chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("KI-GCN macro ROC curve")
    _format_legend(ax, loc="lower right")
    _save_svg(fig, "diagnostic_roc")


def plot_pr_curve() -> None:
    pr = _load_csv("pr_curve.csv")
    ap_value = float(np.trapezoid(pr["precision"], pr["recall"]))
    fig, ax = _create_axes(figsize=(5.6, 4.2), add_minor=True)
    ax.plot(
        pr["recall"],
        pr["precision"],
        color=COLOR_CYCLE[6],
        linewidth=2.8,
        label=f"Macro PR (AP = {ap_value:.3f})",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.85, 1.0)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("KI-GCN macro precision–recall")
    _format_legend(ax, loc="lower left")
    _save_svg(fig, "diagnostic_pr")


def plot_confusion_matrix() -> None:
    cm = _load_csv("confusion_matrix.csv", header=None).to_numpy()
    fig, ax = _create_axes(figsize=(5.0, 4.6), grid_axis=None)
    heatmap = ax.imshow(cm, cmap="Purples", interpolation="nearest")
    vmax = cm.max()
    for (i, j), value in np.ndenumerate(cm):
        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            color="white" if value > vmax * 0.6 else "black",
        )
    ax.set_xticks(range(len(CONFUSION_CLASSES)))
    ax.set_xticklabels(CONFUSION_CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CONFUSION_CLASSES)))
    ax.set_yticklabels(CONFUSION_CLASSES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Normalized confusion matrix")
    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.set_ylabel("Probability", rotation=270, labelpad=18, weight="bold")
    _save_svg(fig, "diagnostic_confusion")


def plot_reliability() -> None:
    pre = _load_csv("calibration_pre.csv")
    post = _load_csv("calibration_post.csv")
    fig, ax = _create_axes(figsize=(6.0, 4.0), add_minor=True)
    ax.plot(
        pre["confidence"],
        pre["accuracy"],
        marker="o",
        linewidth=2.4,
        markersize=6,
        color=COLOR_CYCLE[7],
        label="Pre-scaling",
    )
    ax.plot(
        post["confidence"],
        post["accuracy"],
        marker="s",
        linewidth=2.4,
        markersize=6,
        color=COLOR_CYCLE[8 % len(COLOR_CYCLE)],
        label="Post-scaling",
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="0.4", linewidth=1.1, label="Ideal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.75, 1.0)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    _format_legend(ax, loc="upper left")
    _save_svg(fig, "reliability")


def plot_execution_providers() -> None:
    df = _load_csv("ep_benchmark.csv")
    fig, ax1 = _create_axes(figsize=(6.0, 3.6), grid_axis="y")
    idx = np.arange(len(df))
    width = 0.4
    ax1.bar(
        idx,
        df["Median_s"],
        width=width,
        color=COLOR_CYCLE[0],
        edgecolor="0.25",
        linewidth=0.8,
        label="Median latency (s)",
    )
    ax1.set_xticks(idx)
    ax1.set_xticklabels(df["EP"])
    ax1.set_ylabel("Seconds per volume")

    ax2 = ax1.twinx()
    ax2.plot(
        idx,
        df["PassRate_percent"],
        marker="D",
        linewidth=1.8,
        color=COLOR_CYCLE[3],
        label="Pass rate (%)",
    )
    ax2.set_ylabel("Certification pass rate (%)")
    ax2.set_ylim(94, 101)

    ax1.set_title("Execution provider throughput")
    _format_legend(ax1, loc="upper left")
    _format_legend(ax2, loc="upper right")
    fig.tight_layout()
    _save_svg(fig, "ep_throughput")


def plot_sota_acdc() -> None:
    df = _load_csv("sota_acdc.csv")
    metrics = ["LV", "Myo", "RV"]
    idx = np.arange(len(metrics))
    width = 0.18
    fig, ax = _create_axes(figsize=(7.0, 4.2), grid_axis="y")

    for offset, (method, row) in enumerate(df.iterrows()):
        ax.bar(
            idx + (offset - len(df) / 2) * width,
            row[metrics],
            width=width,
            label=method,
            edgecolor="0.25",
            linewidth=0.6,
        )
    ax.set_xticks(idx)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Dice coefficient")
    ax.set_ylim(0.88, 0.98)
    ax.set_title("ACDC structure-wise SOTA comparison")
    _format_legend(ax, loc="lower center", ncol=2)
    _save_svg(fig, "acdc_sota_structures")


def main() -> None:
    plot_segmentation_macro()
    plot_structure_performance()
    plot_domain_shift()
    plot_roc_curve()
    plot_pr_curve()
    plot_confusion_matrix()
    plot_reliability()
    plot_execution_providers()
    plot_sota_acdc()
    print(f"Figures written to {IMG_DIR}")


if __name__ == "__main__":
    main()
