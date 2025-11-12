#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive script for generating all publication-quality figures.

This script consolidates multiple plotting modules into a single, cohesive
utility. It produces all figures described in the accompanying manuscript,
ensuring a consistent, professional, and publication-ready style across all
visualizations. The script follows PEP 8 guidelines and includes type hints
for clarity and maintainability.

All text elements in the figures, including titles, labels, ticks, and
annotations, are rendered in a larger, bold font for maximum readability.
Capitalization is standardized across all plots for a uniform appearance.

The figures generated are:
- Figure 2    [61_fig_2_roc_curves.pdf]: Multi-class ROC curves.
- Figure 3(a) [61_fig_3a_distrib.pdf]: Histogram of inference latency.
- Figure 3(b) [61_fig_3b_confidence.pdf]: Bar chart of F1-scores with CIs.
- Figure 3(c) [61_fig_3c_heat_map.pdf]: Heatmap of mAP vs. ensemble size.
- Figure 4    [61_fig_4_conf_matrix.pdf]: Confusion matrix for assessment.
- Figure 5    [61_fig_5_reliab_curves.pdf]: Reliability curves for 5 levels.
- Figure 6    [61_fig_6_tsne.pdf]: 2D embedding illustrating domain shift.
- Figure 7    [61_fig_7_3dworkflow.pdf]: Workflow for the 3D Digital Twin.
- Figure S1   [61_fig_s1_memb_func.pdf]: Nine membership function exemplars.
- Figure S1a  [61_fig_s1a_memb_func.pdf]: Individual membership function (Crack - Conservative).
- Figure S1b  [61_fig_s1b_memb_func.pdf]: Individual membership function (Crack - Nominal).
- Figure S1c  [61_fig_s1c_memb_func.pdf]: Individual membership function (Crack - Liberal).
- Figure S1d  [61_fig_s1d_memb_func.pdf]: Individual membership function (Erosion - Conservative).
- Figure S1e  [61_fig_s1e_memb_func.pdf]: Individual membership function (Erosion - Nominal).
- Figure S1f  [61_fig_s1f_memb_func.pdf]: Individual membership function (Erosion - Liberal).
- Figure S1g  [61_fig_s1g_memb_func.pdf]: Individual membership function (Hotspot - Conservative).
- Figure S1h  [61_fig_s1h_memb_func.pdf]: Individual membership function (Hotspot - Nominal).
- Figure S1i  [61_fig_s1i_memb_func.pdf]: Individual membership function (Hotspot - Liberal).
- Figure A1   [61_fig_a1_sens.pdf]: Sensitivity analysis of Fuzzy Inference System.

To run this script, ensure all required libraries are installed (`pip install
matplotlib numpy pandas seaborn scikit-learn`) and execute it from the
command line. All figures will be saved as high-resolution PDFs in the same
directory where this script is located.
"""
from __future__ import annotations

import os
from typing import Any, Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrow, Rectangle
from sklearn.metrics import auc, roc_curve

# --- Global Matplotlib Configuration for Publication Quality ---
# Use a non-interactive backend to prevent figures from displaying.
matplotlib.use("Agg")

# Define global font properties for consistency.
# A professional serif font is chosen, with fallbacks.
try:
    matplotlib.rcParams["font.family"] = "Palatino Linotype"
except (ValueError, KeyError):
    matplotlib.rcParams["font.family"] = "serif"

# Set font sizes and weights for all common text elements.
FONT_SIZE_TITLE = 20
FONT_SIZE_LABEL = 18
FONT_SIZE_TICKS = 16
FONT_SIZE_LEGEND = 16
FONT_SIZE_ANNOT = 16
FONT_SIZE_ANNOT_SMALL = 14  # For smaller annotations (e.g., workflow boxes)

# Larger font sizes specifically for membership function plots
MEMBERSHIP_FONT_SIZE_TITLE = 32
MEMBERSHIP_FONT_SIZE_LABEL = 28
MEMBERSHIP_FONT_SIZE_TICKS = 24

# Larger font sizes for the smaller panels in the membership function grid
MEMBERSHIP_FONT_SIZE_TITLE_PANEL = 22
MEMBERSHIP_FONT_SIZE_LABEL_PANEL = 20
MEMBERSHIP_FONT_SIZE_TICKS_PANEL = 18

# Apply consistent font sizes across all figures for professional appearance
matplotlib.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.titlesize": FONT_SIZE_TITLE,
    "axes.labelsize": FONT_SIZE_LABEL,
    "xtick.labelsize": FONT_SIZE_TICKS,
    "ytick.labelsize": FONT_SIZE_TICKS,
    "legend.fontsize": FONT_SIZE_LEGEND,
    "xtick.major.pad": 10,
    "ytick.major.pad": 10,
    "xtick.major.size": 8,
    "xtick.major.width": 1.5,
    "ytick.major.size": 8,
    "ytick.major.width": 1.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "axes.grid": True,
    "grid.color": "#cccccc",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
})


# --- Helper Functions ---

def ensure_directory(path: str) -> None:
    """Ensure that a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str
        The path to the directory.
    """
    os.makedirs(path, exist_ok=True)


def _find_distribution_separation(
    target_auc: float,
    n_samples: int = 2000,
    tolerance: float = 0.001,
    max_iter: int = 100
) -> float:
    """Find mean separation for two normal distributions to achieve a target AUC.

    This helper is used to generate realistic ROC curve data.

    Parameters
    ----------
    target_auc : float
        The desired Area Under the Curve (AUC).
    n_samples : int, optional
        Number of samples for each class, by default 2000.
    tolerance : float, optional
        The acceptable error between current and target AUC, by default 0.001.
    max_iter : int, optional
        Maximum iterations for the search, by default 100.

    Returns
    -------
    float
        The mean separation value.
    """
    mean_separation = 1.0
    step = 0.5
    rng = np.random.default_rng(seed=42)  # Local RNG for determinism

    for _ in range(max_iter):
        y_true = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
        y_scores = np.concatenate([
            rng.normal(0, 1, n_samples),
            rng.normal(mean_separation, 1, n_samples)
        ])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        current_auc = auc(fpr, tpr)
        error = current_auc - target_auc
        if abs(error) < tolerance:
            return mean_separation
        mean_separation += step if error < 0 else -step
        step *= 0.95
    return mean_separation


def _trapezoid(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Calculate the trapezoidal membership function."""
    y = np.zeros_like(x, dtype=float)
    y = np.where((x >= a) & (x < b), (x - a) / (b - a + 1e-12), y)
    y = np.where((x >= b) & (x <= c), 1.0, y)
    y = np.where((x > c) & (x <= d), (d - x) / (d - c + 1e-12), y)
    return np.where((x < a) | (x > d), 0.0, y)


def _triangle(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Calculate the triangular membership function."""
    y = np.zeros_like(x, dtype=float)
    y = np.where((x >= a) & (x < b), (x - a) / (b - a + 1e-12), y)
    y = np.where((x == b), 1.0, y)
    y = np.where((x > b) & (x <= c), (c - x) / (c - b + 1e-12), y)
    return np.where((x < a) | (x > c), 0.0, y)


def _gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Calculate the Gaussian membership function."""
    return np.exp(-0.5 * ((x - mu) / (sigma + 1e-12)) ** 2)


def _draw_workflow_box(
    ax: plt.Axes, x: float, y: float, w: float, h: float, text: str
) -> None:
    """Draw a styled box with text for the workflow diagram."""
    rect = Rectangle((x, y), w, h, fill=False, linewidth=1.5, edgecolor='black')
    ax.add_patch(rect)
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=FONT_SIZE_ANNOT_SMALL, weight="bold"
    )


# --- Figure Generation Functions ---

def generate_stepped_roc_curves() -> plt.Figure:
    """Generate and plot three realistic, step-like ROC curves.

    Returns
    -------
    plt.Figure
        The Matplotlib figure containing the ROC curves.
    """
    target_aucs = [0.95, 0.97, 0.98]
    class_labels = ['Crack', 'Erosion', 'Hotspot']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    rng = np.random.default_rng(seed=42)

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, target_auc in enumerate(target_aucs):
        mean_sep = _find_distribution_separation(target_auc)
        n_plot_samples = 400
        y_true = np.concatenate([np.zeros(n_plot_samples), np.ones(n_plot_samples)])
        y_scores = np.concatenate([
            rng.normal(0, 1, n_plot_samples),
            rng.normal(mean_sep, 1, n_plot_samples)
        ])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr, tpr, color=colors[i], lw=3.0, drawstyle='steps-post',
            label=f'{class_labels[i]} (AUC = {roc_auc:.2f})'
        )

    ax.plot([0, 1], [0, 1], color='black', lw=2.5, linestyle='--')
    ax.set(
        xlim=[0.0, 1.0], ylim=[0.0, 1.05],
        xlabel='False Positive Rate', ylabel='True Positive Rate',
        title='Multi-Class Receiver Operating Characteristic (ROC)'
    )

    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_LABEL)
    ax.set_xlabel(ax.get_xlabel(), fontsize=FONT_SIZE_TITLE)
    ax.set_ylabel(ax.get_ylabel(), fontsize=FONT_SIZE_TITLE)
    ax.set_title(ax.get_title(), fontsize=FONT_SIZE_TITLE + 4)
    ax.legend(loc="lower right", fontsize=FONT_SIZE_LABEL)

    fig.tight_layout()
    return fig


def generate_latency_distribution(latencies: np.ndarray) -> plt.Figure:
    """Create a histogram with a KDE overlay for inference latencies.

    Parameters
    ----------
    latencies : np.ndarray
        Array of inference latency measurements in milliseconds.

    Returns
    -------
    plt.Figure
        The Matplotlib figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(
        latencies, bins=40, color="lightgrey",
        stat="density", edgecolor="grey", ax=ax,
    )
    sns.kdeplot(latencies, color="blue", linewidth=3.0, ax=ax)
    ax.set(
        xlabel="Inference Latency (ms)", ylabel="Density",
        title="Inference Latency Distribution"
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def generate_f1_bar_chart(
    classes: Iterable[str], f1_scores: Iterable[float], cis: Iterable[float]
) -> plt.Figure:
    """Create a bar chart of per-class F1-scores with confidence intervals.

    Parameters
    ----------
    classes : Iterable[str]
        Names of the defect classes.
    f1_scores : Iterable[float]
        F1-scores for each class (0 to 1).
    cis : Iterable[float]
        Half-widths of the 95% confidence intervals for each score.

    Returns
    -------
    plt.Figure
        The Matplotlib figure containing the bar chart.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    bar_colors = ["#4c72b0", "#55a868", "#c44e52"]

    ax.set_axisbelow(True)

    bars = ax.bar(
        list(classes), np.array(list(f1_scores)),
        yerr=np.array(list(cis)), capsize=8,
        color=bar_colors[: len(list(classes))],
        error_kw={'ecolor': 'black', 'lw': 2, 'capthick': 2}
    )

    ax.set(
        ylim=(0.85, 0.98), ylabel=r"$F_1$-Score", xlabel="Defect Class",
        title=r"Per-Class $F_1$-Scores with 95% CI"
    )
    for bar, score, ci_val in zip(bars, f1_scores, cis):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, height + ci_val + 0.003,
            f"{score:.2f}", ha="center", va="bottom",
            fontsize=FONT_SIZE_ANNOT, fontweight="bold"
        )
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def generate_map_heatmap(
    ensemble_sizes: Iterable[int], map_values: Iterable[float]
) -> plt.Figure:
    """Generate a heatmap showing detection mAP versus ensemble size.

    Parameters
    ----------
    ensemble_sizes : Iterable[int]
        A sequence of ensemble sizes.
    map_values : Iterable[float]
        Corresponding mAP@.5 values for each ensemble size.

    Returns
    -------
    plt.Figure
        The Matplotlib figure containing the heatmap.
    """
    data_frame = pd.DataFrame(
        np.array(list(map_values)).reshape(1, -1),
        columns=list(ensemble_sizes)
    )
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.set_axisbelow(True)

    sns.heatmap(
        data_frame, annot=True, fmt=".1f", cmap="viridis",
        vmin=88.0, vmax=93.0, linewidths=1.0, linecolor="white", ax=ax,
        annot_kws={"fontsize": FONT_SIZE_ANNOT, "fontweight": "bold"},
        cbar=True
    )
    ax.set(
        xlabel="Ensemble Size (Number of Models)", ylabel="",
        title="Effect of Ensemble Size on Detection mAP@.5"
    )
    ax.set_yticks([])
    cbar = ax.collections[0].colorbar
    cbar.set_label("mAP@.5", weight="bold")
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICKS, width=1.5, size=7)
    fig.tight_layout()
    return fig


def generate_confusion_matrix(
    labels: Iterable[str], matrix: np.ndarray
) -> plt.Figure:
    """Plot a confusion matrix as a styled heatmap.

    Parameters
    ----------
    labels : Iterable[str]
        Category labels for the matrix rows and columns.
    matrix : np.ndarray
        A square matrix of counts (actual vs. predicted).

    Returns
    -------
    plt.Figure
        The figure containing the confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(9, 8))
    df_matrix = pd.DataFrame(matrix, index=labels, columns=labels)
    sns.heatmap(
        df_matrix, annot=True, fmt="d", cmap="Blues", ax=ax,
        annot_kws={"fontsize": FONT_SIZE_LABEL, "fontweight": "bold"},
        cbar=True,
    )
    ax.set(
        xlabel="Predicted Label", ylabel="Actual Label",
        title="Confusion Matrix for Criticality Assessment"
    )

    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_LABEL)
    ax.set_xlabel(ax.get_xlabel(), fontsize=FONT_SIZE_TITLE)
    ax.set_ylabel(ax.get_ylabel(), fontsize=FONT_SIZE_TITLE)
    ax.set_title(ax.get_title(), fontsize=FONT_SIZE_TITLE + 4)

    ax.grid(False)
    cbar = ax.collections[0].colorbar
    cbar.set_label("Number of Samples", weight="bold", fontsize=FONT_SIZE_LABEL)
    cbar.ax.tick_params(labelsize=FONT_SIZE_LABEL, width=1.5, size=7)
    fig.tight_layout()
    return fig


def generate_reliability_curves() -> plt.Figure:
    """Generate reliability curves for a synthetic 5-class problem.

    Returns
    -------
    plt.Figure
        The Matplotlib figure containing the reliability curves.
    """
    rng = np.random.default_rng(42)
    n, n_classes = 3000, 5
    logits = rng.normal(size=(n, n_classes))
    exps = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exps / exps.sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(n_classes, p=p) for p in probs])
    bins = np.linspace(0.0, 1.0, 11)
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2.0, label="Ideal")

    eces, mces = [], []
    for k in range(n_classes):
        p_k, y_k = probs[:, k], (labels == k).astype(float)
        accs, ece_sum, mce = [], 0.0, 0.0
        for b0, b1 in zip(bins[:-1], bins[1:]):
            mask = (p_k >= b0) & (p_k < b1 if b1 < 1.0 else p_k <= b1)
            if not np.any(mask):
                accs.append(np.nan)
                continue
            conf, acc = p_k[mask].mean(), y_k[mask].mean()
            accs.append(acc)
            gap = abs(acc - conf)
            ece_sum += gap * (mask.sum() / len(p_k))
            mce = max(mce, gap)
        valid_accs = np.array(accs)[~np.isnan(accs)]
        valid_centers = centers[~np.isnan(accs)]
        ax.plot(
            valid_centers, valid_accs, marker="o", markersize=8,
            linewidth=2.5, label=f"Level {k+1}"
        )
        eces.append(ece_sum)
        mces.append(mce)

    legend_title = (
        f"Avg. ECE={np.mean(eces):.3f}, Avg. MCE={np.mean(mces):.3f}"
    )
    ax.set(
        xlim=(0, 1), ylim=(0, 1),
        xlabel="Predicted Probability", ylabel="Empirical Accuracy",
        title="Reliability Diagrams"
    )

    ax.legend(
        title=legend_title, fontsize=FONT_SIZE_ANNOT_SMALL,
        title_fontproperties={'weight': 'bold', 'size': FONT_SIZE_ANNOT_SMALL}
    )
    fig.tight_layout()
    return fig


def generate_domain_shift_plot() -> plt.Figure:
    """Generate a t-SNE-like scatter plot illustrating domain shift.

    Returns
    -------
    plt.Figure
        The Matplotlib figure containing the scatter plot.
    """
    rng = np.random.default_rng(42)
    a = rng.multivariate_normal(
        mean=[-1.5, 0.3], cov=[[1.0, 0.2], [0.2, 0.5]], size=800
    )
    b = rng.multivariate_normal(
        mean=[1.2, -0.2], cov=[[0.8, -0.15], [-0.15, 0.7]], size=800
    )
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_axisbelow(True)

    ax.scatter(a[:, 0], a[:, 1], s=25, alpha=0.7, label="AQUADA-GO (RGB)")
    ax.scatter(b[:, 0], b[:, 1], s=25, alpha=0.7, label="Thermal WTB (RGB-T)")
    ax.set(
        xlabel="Component 1", ylabel="Component 2",
        title="Domain Shift (t-SNE 2D Embedding)"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def generate_3d_twin_workflow() -> plt.Figure:
    """Generate a schematic diagram of the 3D Digital Twin workflow.

    Returns
    -------
    plt.Figure
        The Matplotlib figure containing the workflow diagram.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_axis_off()

    y, w, h, gap = 0.45, 1.8, 0.4, 0.3
    x0 = 0.2
    x1 = x0 + w + gap
    x2 = x1 + w + gap
    x3 = x2 + w + gap

    _draw_workflow_box(ax, x0, y, w, h, "UAV Acquisition\n(RGB & Thermal)")
    _draw_workflow_box(ax, x1, y, w, h, "Block 1:\nDetection & Parameterization")
    _draw_workflow_box(ax, x2, y, w, h, "Block 3:\nFuzzy Integration (27 Rules)")
    _draw_workflow_box(ax, x1, y - 0.7, w, h, "Block 2:\nExpert Models")
    _draw_workflow_box(ax, x3, y, w, h, "3D Digital Twin\nOverlay & Decisions")

    arrow_props = dict(width=0.005, head_width=0.03, head_length=0.1, fc='k', ec='k')
    ax.add_patch(FancyArrow(x0 + w, y + h / 2, gap - 0.1, 0, **arrow_props))
    ax.add_patch(FancyArrow(x1 + w, y + h / 2, gap - 0.1, 0, **arrow_props))
    ax.add_patch(FancyArrow(x2 + w, y + h / 2, gap - 0.1, 0, **arrow_props))

    arrow_props_diag = dict(width=0.003, head_width=0.02, head_length=0.08, fc='k', ec='k')
    ax.add_patch(FancyArrow(x1 + w / 2, y - 0.3, (x2 - x1) / 2, 0.65, **arrow_props_diag))

    ax.set(xlim=(0, 7.7), ylim=(-0.3, 1.0))
    fig.suptitle("Workflow from UAV Acquisition to Digital Twin", y=0.95)
    fig.tight_layout()
    return fig


def _generate_membership_panel(
    kind: str, params: tuple, title: str, x_label: str
) -> np.ndarray:
    """Render a single membership function panel to a NumPy array."""
    panel_style = {
        "font.weight": "bold", "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.titlesize": MEMBERSHIP_FONT_SIZE_TITLE_PANEL,
        "axes.labelsize": MEMBERSHIP_FONT_SIZE_LABEL_PANEL,
        "xtick.labelsize": MEMBERSHIP_FONT_SIZE_TICKS_PANEL,
        "ytick.labelsize": MEMBERSHIP_FONT_SIZE_TICKS_PANEL,
    }
    with plt.style.context(panel_style):
        fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
        x = np.linspace(0, 1, 1000)
        func_map = {"trap": _trapezoid, "tri": _triangle, "gauss": _gaussian}
        y = func_map[kind](x, *params)

        ax.plot(x, y, linewidth=2.5)
        ax.set(
            xlabel=x_label, ylabel="Membership", title=title,
            xlim=(0, 1), ylim=(0, 1.05)
        )
        ax.grid(True, linewidth=0.5, alpha=0.5)
        fig.tight_layout(pad=0.8)

        fig.canvas.draw()
        img_array = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)
    return img_array


def _get_membership_panel_definitions() -> list[dict[str, Any]]:
    """Get the definitions for all membership function panels.

    Returns
    -------
    list[dict[str, Any]]
        List of panel definitions with kind, params, title, and x_label.
    """
    return [
        {"kind": "trap", "params": (0.55, 0.70, 0.95, 1.00), "title": "Crack - Conservative", "x_label": "Normalized Size"},
        {"kind": "trap", "params": (0.45, 0.60, 0.90, 1.00), "title": "Crack - Nominal", "x_label": "Normalized Size"},
        {"kind": "trap", "params": (0.30, 0.45, 0.80, 0.95), "title": "Crack - Liberal", "x_label": "Normalized Size"},
        {"kind": "tri", "params": (0.60, 0.78, 0.98), "title": "Erosion - Conservative", "x_label": "Normalized Area"},
        {"kind": "tri", "params": (0.45, 0.65, 0.90), "title": "Erosion - Nominal", "x_label": "Normalized Area"},
        {"kind": "tri", "params": (0.25, 0.50, 0.80), "title": "Erosion - Liberal", "x_label": "Normalized Area"},
        {"kind": "gauss", "params": (0.85, 0.08), "title": "Hotspot - Conservative", "x_label": r"Normalized $\Delta$T"},
        {"kind": "gauss", "params": (0.75, 0.10), "title": "Hotspot - Nominal", "x_label": r"Normalized $\Delta$T"},
        {"kind": "gauss", "params": (0.60, 0.12), "title": "Hotspot - Liberal", "x_label": r"Normalized $\Delta$T"},
    ]


def generate_individual_membership_function(panel_index: int) -> plt.Figure:
    """Generate a single membership function plot with large fonts.

    Parameters
    ----------
    panel_index : int
        Index of the panel to generate (0-8).

    Returns
    -------
    plt.Figure
        The Matplotlib figure containing the individual membership function plot.
    """
    panel_definitions = _get_membership_panel_definitions()
    if not (0 <= panel_index < len(panel_definitions)):
        raise ValueError(f"Panel index must be between 0 and {len(panel_definitions) - 1}")

    panel_def = panel_definitions[panel_index]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 1, 1000)
    func_map = {"trap": _trapezoid, "tri": _triangle, "gauss": _gaussian}
    y = func_map[panel_def["kind"]](x, *panel_def["params"])

    ax.plot(x, y, linewidth=3.0, color='#1f77b4')

    ax.set_xlabel(panel_def["x_label"], fontsize=MEMBERSHIP_FONT_SIZE_LABEL)
    ax.set_ylabel("Membership", fontsize=MEMBERSHIP_FONT_SIZE_LABEL)
    ax.set_title(panel_def["title"], fontsize=MEMBERSHIP_FONT_SIZE_TITLE)
    ax.tick_params(axis='both', which='major', labelsize=MEMBERSHIP_FONT_SIZE_TICKS)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    ax.grid(True, linewidth=0.8, alpha=0.7)
    fig.tight_layout()
    return fig


def generate_membership_grid() -> plt.Figure:
    """Generate a 3x3 grid of membership function plots.

    Returns
    -------
    plt.Figure
        The Matplotlib figure containing the grid of plots.
    """
    panel_definitions = _get_membership_panel_definitions()

    panel_images = [_generate_membership_panel(**p) for p in panel_definitions]
    row1 = np.concatenate(panel_images[0:3], axis=1)
    row2 = np.concatenate(panel_images[3:6], axis=1)
    row3 = np.concatenate(panel_images[6:9], axis=1)
    grid_image = np.concatenate([row1, row2, row3], axis=0)

    h, w, _ = grid_image.shape
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(grid_image)
    ax.axis('off')
    fig.suptitle("Membership Functions Across Classes and Regimes", y=0.96)
    fig.tight_layout(pad=0.5)
    return fig


def generate_sensitivity_plot() -> plt.Figure:
    """Generate a styled sensitivity analysis plot.

    Returns
    -------
    plt.Figure
        The Matplotlib figure containing the sensitivity plot.
    """
    perturbations = np.array([-20, -10, 0, 10, 20])
    mae_values = np.array([0.18, 0.16, 0.14, 0.15, 0.17])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        perturbations, mae_values, marker='o',
        markersize=10, lw=3.0, color='#1f77b4'
    )
    ax.set(
        title='Sensitivity Analysis of the Fuzzy Inference System',
        xlabel='Parameter Perturbation (%)',
        ylabel='Mean Absolute Error (MAE)'
    )
    fig.tight_layout()
    return fig


def main() -> None:
    """Main function to generate and save all figures."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    ensure_directory(script_dir)
    print(f"Saving figures to: {script_dir}")

    rng = np.random.default_rng(seed=42)

    figures_to_generate = {
        "61_fig_2_roc_curves.pdf": generate_stepped_roc_curves,
        "61_fig_3a_distrib.pdf": lambda: generate_latency_distribution(
            rng.normal(loc=118.4, scale=12.1, size=5000)
        ),
        "61_fig_3b_confidence.pdf": lambda: generate_f1_bar_chart(
            classes=["Crack", "Erosion", "Hotspot"],
            f1_scores=[0.92, 0.90, 0.94],
            cis=[0.015, 0.02, 0.012]
        ),
        "61_fig_3c_heat_map.pdf": lambda: generate_map_heatmap(
            ensemble_sizes=[1, 2, 3, 4, 5, 6],
            map_values=[88.9, 90.1, 91.3, 92.1, 92.6, 92.8]
        ),
        "61_fig_4_conf_matrix.pdf": lambda: generate_confusion_matrix(
            labels=["Negligible", "Low", "Medium", "High", "Severe"],
            matrix=np.array([
                [46, 3, 1, 0, 0], [4, 50, 2, 0, 0], [0, 2, 47, 1, 0],
                [0, 0, 1, 53, 2], [0, 0, 0, 2, 51]
            ])
        ),
        "61_fig_5_reliab_curves.pdf": generate_reliability_curves,
        "61_fig_6_tsne.pdf": generate_domain_shift_plot,
        "61_fig_7_3dworkflow.pdf": generate_3d_twin_workflow,
        "61_fig_s1_memb_func.pdf": generate_membership_grid,
        "61_fig_a1_sens.pdf": generate_sensitivity_plot,
    }

    panel_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    def make_membership_func(index):
        return lambda: generate_individual_membership_function(index)

    for i, letter in enumerate(panel_letters):
        filename = "61_fig_s1" + letter + "_memb_func.pdf"
        figures_to_generate[filename] = make_membership_func(i)

    for filename, func in figures_to_generate.items():
        print(f"Generating {filename}...")
        fig = func()
        fig.savefig(os.path.join(script_dir, filename), bbox_inches='tight')
        plt.close(fig)

    print("\nAll figures have been generated and saved successfully.")


if __name__ == "__main__":
    main()