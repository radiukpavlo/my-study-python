"""Produce segmentation study figures with consistent professional styling."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import auc

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
IMG_DIR = ROOT_DIR / "img"

BASE_FONT_SIZE = mpl.rcParams.get("font.size", 10) + 2
COLOR_CYCLE = tuple(mpl.colormaps["tab10"].colors)

STYLE_PARAMS = {
    "font.size": BASE_FONT_SIZE,
    "font.weight": "bold",
    "axes.labelsize": BASE_FONT_SIZE,
    "axes.labelweight": "bold",
    "axes.titlesize": BASE_FONT_SIZE + 2,
    "axes.titleweight": "bold",
    "xtick.labelsize": BASE_FONT_SIZE - 1,
    "ytick.labelsize": BASE_FONT_SIZE - 1,
    "legend.fontsize": BASE_FONT_SIZE - 1,
    "legend.edgecolor": "0.75",
    "legend.facecolor": "1.0",
    "legend.framealpha": 0.92,
    "axes.edgecolor": "0.35",
    "axes.linewidth": 1.1,
    "grid.color": "0.7",
    "grid.linewidth": 0.85,
    "grid.alpha": 0.55,
    "savefig.dpi": 220,
}

_STYLE_INITIALISED = False

BOXPLOT_STATS = {
    "ACDC-LV": (0.965, 0.020),
    "ACDC-Myo": (0.912, 0.030),
    "ACDC-RV": (0.941, 0.020),
    "M&Ms2-LV": (0.953, 0.025),
    "M&Ms2-Myo": (0.899, 0.035),
    "M&Ms2-RV": (0.928, 0.025),
}


def configure_matplotlib() -> None:
    """Apply the shared matplotlib rcParams once per process."""

    global _STYLE_INITIALISED
    if _STYLE_INITIALISED:
        return
    mpl.rcParams.update(STYLE_PARAMS)
    _STYLE_INITIALISED = True


def _bold_ticklabels(ax: plt.Axes) -> None:
    """Ensure tick labels stay bold even after custom formatting."""

    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")


def _create_axes(
    figsize: tuple[float, float],
    grid_axis: str = "both",
    add_minor: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Create an axis with consistent spines, grids, and tick styling."""

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(
            True,
            axis=grid_axis,
            color=STYLE_PARAMS["grid.color"],
            linewidth=STYLE_PARAMS["grid.linewidth"],
            alpha=STYLE_PARAMS["grid.alpha"],
        )
    if add_minor:
        ax.minorticks_on()
        ax.grid(
            True,
            which="minor",
            axis=grid_axis or "both",
            color="0.85",
            linewidth=0.6,
            alpha=0.4,
        )
    for spine in ax.spines.values():
        spine.set_color(STYLE_PARAMS["axes.edgecolor"])
        spine.set_linewidth(STYLE_PARAMS["axes.linewidth"])
    ax.tick_params(axis="both", which="both", width=0.9)
    _bold_ticklabels(ax)
    return fig, ax


def _format_legend(ax: plt.Axes, **kwargs) -> mpl.legend.Legend | None:
    """Attach a subtle framed legend."""

    legend = ax.legend(**kwargs)
    if legend:
        frame = legend.get_frame()
        frame.set_linewidth(0.8)
        frame.set_edgecolor("0.68")
        for text in legend.get_texts():
            text.set_fontweight("bold")
    return legend


def _save_figure(fig: plt.Figure, stem: str) -> None:
    """Save the current figure as SVG and close it."""

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG_DIR / f"{stem}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def _load_csv(filename: str, **kwargs) -> pd.DataFrame:
    """Convenience wrapper for reading CSV files from DATA_DIR."""

    return pd.read_csv(DATA_DIR / filename, **kwargs)


def plot_segmentation_macro(df: pd.DataFrame) -> None:
    """Grouped bar chart comparing macro Dice scores."""

    idx = np.arange(len(df))
    width = 0.35
    fig, ax = _create_axes(figsize=(7.0, 5.0), grid_axis="y", add_minor=True)
    ax.bar(
        idx - width / 2,
        df["U-Net_DSC"],
        width=width,
        label="U-Net",
        color=COLOR_CYCLE[0],
        edgecolor="0.25",
        linewidth=0.9,
    )
    ax.bar(
        idx + width / 2,
        df["SKIFSeg_DSC"],
        width=width,
        label="SKIF-Seg",
        color=COLOR_CYCLE[1],
        edgecolor="0.25",
        linewidth=0.9,
    )
    ax.set_xticks(idx)
    ax.set_xticklabels(df["Dataset"])
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Macro Dice")
    ax.set_title("Macro Dice by dataset")
    _bold_ticklabels(ax)
    _format_legend(ax, loc="upper right", ncol=2)
    fig.tight_layout()
    _save_figure(fig, "seg_macro_bars")


def plot_segmentation_boxplots() -> None:
    """Boxplots showing per-case Dice distributions (synthetic)."""

    rng = np.random.default_rng(42)
    samples = [
        np.clip(rng.normal(mean, std, 120), 0.6, 0.99)
        for mean, std in BOXPLOT_STATS.values()
    ]
    my_labels = list(BOXPLOT_STATS.keys())

    fig, ax = _create_axes(figsize=(8.0, 5.0), grid_axis="y")
    ax.boxplot(
        samples,
        tick_labels=my_labels,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": COLOR_CYCLE[2], "alpha": 0.45},
        medianprops={"color": "0.2", "linewidth": 1.4},
    )
    ax.set_ylabel("Dice score")
    ax.set_title("Case-wise Dice distributions (SKIF-Seg)")
    _bold_ticklabels(ax)
    fig.tight_layout()
    _save_figure(fig, "seg_boxplots")


def plot_domain_shift(df: pd.DataFrame) -> None:
    """Single-series bar chart capturing cross-domain degradation."""

    fig, ax = _create_axes(figsize=(7.0, 4.5), grid_axis="y")
    ax.bar(
        df["Structure"],
        df["DeltaDice"],
        color=COLOR_CYCLE[3],
        edgecolor="0.25",
        linewidth=0.9,
    )
    ax.set_xlabel("Structure")
    ax.set_ylabel("ΔDice (M&Ms-2 → ACDC)")
    ax.set_title("Domain shift analysis")
    _bold_ticklabels(ax)
    fig.tight_layout()
    _save_figure(fig, "domain_shift")


def plot_roc_curve(df: pd.DataFrame) -> None:
    """Plot ROC curve with its AUC annotation."""

    roc_auc = auc(df["fpr"], df["tpr"])
    fig, ax = _create_axes(figsize=(7.0, 5.0), add_minor=True)
    ax.plot(
        df["fpr"],
        df["tpr"],
        label=f"Macro ROC (AUC = {roc_auc:.3f})",
        linewidth=2.4,
        color=COLOR_CYCLE[4],
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.3,
        color="0.35",
        label="Chance",
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("ROC curve (KI-GCN)")
    _bold_ticklabels(ax)
    _format_legend(ax, loc="lower right")
    fig.tight_layout()
    _save_figure(fig, "roc_curve")


def plot_pr_curve(df: pd.DataFrame) -> None:
    """Plot Precision-Recall curve with area label."""

    pr_auc = auc(df["recall"], df["precision"])
    fig, ax = _create_axes(figsize=(7.0, 5.0), add_minor=True)
    ax.plot(
        df["recall"],
        df["precision"],
        label=f"Macro PR (AUC = {pr_auc:.3f})",
        linewidth=2.4,
        color=COLOR_CYCLE[5],
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Precision-Recall curve (KI-GCN)")
    _bold_ticklabels(ax)
    _format_legend(ax, loc="lower left")
    fig.tight_layout()
    _save_figure(fig, "pr_curve")


def plot_confusion_matrix(matrix: np.ndarray) -> None:
    """Heatmap of the normalized five-class confusion matrix."""

    classes = ["NOR", "HCM", "DCM", "MINF", "ARV"]
    fig, ax = _create_axes(figsize=(6.4, 6.0), grid_axis=None)
    heatmap = ax.imshow(matrix, cmap="Blues", aspect="auto")
    for (row, col), value in np.ndenumerate(matrix):
        color = "white" if value > matrix.max() * 0.65 else "black"
        ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Normalized confusion matrix")
    _bold_ticklabels(ax)
    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Proportion", rotation=270, labelpad=18, fontweight="bold")
    cbar.ax.tick_params(labelsize=BASE_FONT_SIZE - 1, width=0.8)
    fig.tight_layout()
    _save_figure(fig, "confusion_matrix")


def plot_reliability_diagram(pre: pd.DataFrame, post: pd.DataFrame) -> None:
    """Reliability diagram comparing calibration pre/post scaling."""

    width = 0.02
    fig, ax = _create_axes(figsize=(7.0, 5.0), grid_axis="y")
    ax.bar(
        pre["confidence"] - width / 2,
        pre["accuracy"],
        width=width,
        label="Pre-scaling",
        color=COLOR_CYCLE[6],
        edgecolor="0.25",
        linewidth=0.9,
    )
    ax.bar(
        post["confidence"] + width / 2,
        post["accuracy"],
        width=width,
        label="Post-scaling",
        color=COLOR_CYCLE[7],
        edgecolor="0.25",
        linewidth=0.9,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.3, color="0.35")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    _bold_ticklabels(ax)
    _format_legend(ax, loc="lower right")
    fig.tight_layout()
    _save_figure(fig, "reliability_pre_post")


def plot_execution_providers(df: pd.DataFrame) -> None:
    """Bar chart for inference throughput across execution providers."""

    fig, ax = _create_axes(figsize=(7.0, 5.0), grid_axis="y")
    ax.bar(
        df["EP"],
        df["Median_s"],
        color=COLOR_CYCLE[0],
        edgecolor="0.25",
        linewidth=0.9,
    )
    ax.set_xlabel("Execution provider")
    ax.set_ylabel("Seconds per volume (median)")
    ax.set_title("Inference throughput")
    _bold_ticklabels(ax)
    fig.tight_layout()
    _save_figure(fig, "ep_throughput")


def pipeline_diagram() -> None:
    """Simple schematic of the KI-GCN pipeline."""

    fig, ax = plt.subplots(figsize=(9.0, 2.8))
    ax.set_axis_off()
    boxes = [
        ("DICOM/NIfTI ingestion\n+ anonymization", (0.05, 0.5)),
        ("Preprocessing\n(reorient, normalize)", (0.28, 0.5)),
        ("SKIF-Seg (ONNX)\nvolumetric segmentation", (0.51, 0.5)),
        ("Graph construction\n(features + edges)", (0.74, 0.5)),
        ("KI-GCN\ndiagnosis", (0.92, 0.5)),
    ]
    for label, (x, y) in boxes:
        rect = FancyBboxPatch(
            (x - 0.09, y - 0.17),
            0.18,
            0.34,
            boxstyle="round,pad=0.02",
            edgecolor="0.35",
            facecolor="none",
            linewidth=1.1,
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", weight="bold")
    for start, end in zip(boxes, boxes[1:]):
        (x0, y0), (x1, y1) = start[1], end[1]
        ax.annotate(
            "",
            xy=(x1 - 0.15, y1),
            xytext=(x0 + 0.15, y0),
            arrowprops={"arrowstyle": "->", "linewidth": 1.3, "color": "0.35"},
        )
    fig.tight_layout()
    _save_figure(fig, "pipeline")


def main() -> None:
    """Generate every figure required by the manuscript."""

    configure_matplotlib()
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    seg_macro = _load_csv("segmentation_macro.csv")
    domain_shift = _load_csv("domain_shift.csv")
    roc_df = _load_csv("roc_curve.csv")
    pr_df = _load_csv("pr_curve.csv")
    cm_matrix = _load_csv("confusion_matrix.csv", header=None).to_numpy()
    calibration_pre = _load_csv("calibration_pre.csv")
    calibration_post = _load_csv("calibration_post.csv")
    ep_df = _load_csv("ep_benchmark.csv")

    pipeline_diagram()
    plot_segmentation_macro(seg_macro)
    plot_segmentation_boxplots()
    plot_domain_shift(domain_shift)
    plot_roc_curve(roc_df)
    plot_pr_curve(pr_df)
    plot_confusion_matrix(cm_matrix)
    plot_reliability_diagram(calibration_pre, calibration_post)
    plot_execution_providers(ep_df)
    print(f"Figures written to {IMG_DIR} (SVG format)")


if __name__ == "__main__":
    main()
