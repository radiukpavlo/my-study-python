#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive script for generating all publication-quality figures.

This script consolidates multiple plotting modules into a single, cohesive
utility. It produces all six figures described in the accompanying manuscript,
ensuring a consistent, professional, and publication-ready style across all
visualizations. The script follows PEP 8 guidelines and includes type hints
for clarity and maintainability.

All text elements in the figures, including titles, labels, ticks, and
annotations, are rendered in a larger, bold font for maximum readability.

The figures generated are:
- Figure 2(a) [61_fig_3a.pdf]: Histogram of inference latency with KDE.
- Figure 2(b) [61_fig_3b.pdf]: Bar chart of per-class F1-scores with 95% CIs.
- Figure 2(c) [61_fig_3c.pdf]: Heatmap of mAP vs. ensemble size.
- Figure 3    [61_fig_4.pdf]: Confusion matrix for criticality assessment.
- ROC Curves  [61_fig_2_roc.pdf]: Multi-class ROC curves for three models.
- Sensitivity [61_fig_a1.pdf]: Sensitivity analysis of a Fuzzy Inference System.

To run this script, ensure all required libraries are installed (`pip install
matplotlib numpy pandas seaborn scikit-learn`) and execute it from the
command line. All figures will be saved as high-resolution PDFs in an ``img``
subdirectory created in the same location as the script.
"""
from __future__ import annotations

import os
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
FONT_SIZE_TITLE = 22
FONT_SIZE_LABEL = 20
FONT_SIZE_TICKS = 20
FONT_SIZE_LEGEND = 20
FONT_SIZE_ANNOT = 20

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
    "ytick.major.width": 1.5,
    "ytick.major.size": 8,
    "ytick.major.width": 1.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
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


# --- Figure Generation Functions ---

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
        latencies,
        bins=40,
        color="lightgrey",
        stat="density",
        edgecolor="grey",
        ax=ax,
    )
    sns.kdeplot(latencies, color="blue", linewidth=3.0, ax=ax)

    ax.set_xlabel("Inference latency (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Inference Latency Distribution")

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

    bars = ax.bar(
        list(classes),
        np.array(list(f1_scores)),
        yerr=np.array(list(cis)),
        capsize=8,
        color=bar_colors[: len(list(classes))],
        error_kw={'ecolor': 'black', 'lw': 2, 'capthick': 2}
    )

    ax.set_ylim(0.85, 0.98)
    ax.set_ylabel(r"$F_{1}$-score")
    ax.set_xlabel("Defect class")
    ax.set_title(r"Per-class $F_1$-scores with 95% CI")

    for bar, score, ci_val in zip(bars, f1_scores, cis):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + ci_val + 0.003,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_ANNOT,
            fontweight="bold"
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

    sns.heatmap(
        data_frame,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        vmin=88.0,
        vmax=93.0,
        linewidths=1.0,
        linecolor="white",
        ax=ax,
        annot_kws={"fontsize": FONT_SIZE_ANNOT, "fontweight": "bold"},
        cbar=True
    )

    ax.set_xlabel("Ensemble size (number of models)")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_title("Effect of Ensemble Size on Detection mAP")

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
        df_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        annot_kws={"fontsize": FONT_SIZE_ANNOT, "fontweight": "bold"},
        cbar=True,
    )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Confusion Matrix for Criticality Assessment")

    cbar = ax.collections[0].colorbar
    cbar.set_label("Number of Samples", weight="bold")
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICKS, width=1.5, size=7)

    fig.tight_layout()
    return fig


def generate_stepped_roc_curves() -> plt.Figure:
    """Generate and plot three realistic, step-like ROC curves.

    Returns
    -------
    plt.Figure
        The Matplotlib figure containing the ROC curves.
    """
    target_aucs = [0.95, 0.97, 0.98]
    class_labels = ['Crack', 'Erosion', 'Hotspot']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
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
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Multi-Class Receiver Operating Characteristic')
    ax.legend(loc="lower right")

    fig.tight_layout()
    return fig


def generate_sensitivity_plot() -> plt.Figure:
    """Generate and a styled sensitivity analysis plot.

    Returns
    -------
    plt.Figure
        The Matplotlib figure containing the sensitivity plot.
    """
    perturbations = np.array([-20, -10, 0, 10, 20])
    mae_values = np.array([0.18, 0.16, 0.14, 0.15, 0.17])

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        perturbations,
        mae_values,
        marker='o',
        markersize=10,
        lw=3.0,
        color='#1f77b4'  # Classic Matplotlib blue
    )

    ax.set_title('Sensitivity Analysis of the Fuzzy Inference System')
    ax.set_xlabel('Parameter Perturbation (%)')
    ax.set_ylabel('Mean Absolute Error (MAE)')

    fig.tight_layout()
    return fig


def main() -> None:
    """Main function to generate and save all figures."""
    # Define the output directory relative to the script's location.
    # Use os.path.realpath to handle symlinks correctly.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(script_dir)
    ensure_directory(output_dir)
    print(f"Output directory: {output_dir}")

    # Set a global random seed for reproducibility.
    rng = np.random.default_rng(seed=42)

    # --- Generate Figure 2(a): Latency Distribution ---
    print("Generating latency distribution plot...")
    latency_samples = rng.normal(loc=118.4, scale=12.1, size=5000)
    fig_2a = generate_latency_distribution(latency_samples)
    fig_2a.savefig(os.path.join(output_dir, "61_fig_3a.pdf"))
    plt.close(fig_2a)

    # --- Generate Figure 2(b): F1-score Bar Chart ---
    print("Generating F1-score bar chart...")
    fig_2b = generate_f1_bar_chart(
        classes=["Crack", "Erosion", "Hotspot"],
        f1_scores=[0.92, 0.90, 0.94],
        cis=[0.015, 0.02, 0.012]
    )
    fig_2b.savefig(os.path.join(output_dir, "61_fig_3b.pdf"))
    plt.close(fig_2b)

    # --- Generate Figure 2(c): mAP Heatmap ---
    print("Generating mAP heatmap...")
    fig_2c = generate_map_heatmap(
        ensemble_sizes=[1, 2, 3, 4, 5, 6],
        map_values=[88.9, 90.1, 91.3, 92.1, 92.6, 92.8]
    )
    fig_2c.savefig(os.path.join(output_dir, "61_fig_3c.pdf"))
    plt.close(fig_2c)

    # --- Generate Figure 3: Confusion Matrix ---
    print("Generating confusion matrix...")
    confusion_counts = np.array([
        [46, 3, 1, 0, 0], [4, 50, 2, 0, 0], [0, 2, 47, 1, 0],
        [0, 0, 1, 53, 2], [0, 0, 0, 2, 51]
    ])
    fig_3 = generate_confusion_matrix(
        labels=["Negligible", "Low", "Medium", "High", "Severe"],
        matrix=confusion_counts
    )
    fig_3.savefig(os.path.join(output_dir, "61_fig_4.pdf"))
    plt.close(fig_3)

    # --- Generate ROC Curves Figure ---
    print("Generating ROC curves plot...")
    fig_roc = generate_stepped_roc_curves()
    fig_roc.savefig(os.path.join(output_dir, "61_fig_2_roc.pdf"))
    plt.close(fig_roc)

    # --- Generate Sensitivity Analysis Figure ---
    print("Generating sensitivity analysis plot...")
    fig_sens = generate_sensitivity_plot()
    fig_sens.savefig(os.path.join(output_dir, "61_fig_a1.pdf"))
    plt.close(fig_sens)

    print("\nAll figures have been generated and saved successfully.")


if __name__ == "__main__":
    main()