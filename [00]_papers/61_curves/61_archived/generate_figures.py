"""Module for generating research figures.

This script produces the four publication-quality figures described in the
accompanying manuscript. It includes functions to create each figure
separately and a main entry point that saves them to a designated
directory. The script follows PEP 8 style guidelines for readability
and maintainability.

Figures generated:
* Figure 2(a): Histogram of inference latency with KDE overlay.
* Figure 2(b): Bar chart of per-class F1-scores with confidence intervals.
* Figure 2(c): Heatmap showing the effect of ensemble size on mAP.
* Figure 3: Confusion matrix for five-level criticality assessment.

To run this module as a script, execute it directly. It will save all
figures into a ``img`` subdirectory relative to the script
location. Alternatively, import the individual functions into a
Jupyter Notebook and call them with a path to save the images.
"""
from __future__ import annotations
import os
from typing import Iterable
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Configure matplotlib aesthetics ---
# Set font family, size, and weight globally for publication-quality style
matplotlib.rcParams["font.family"] = "Palatino Linotype"
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["font.weight"] = "bold"
# Use a non-interactive backend to facilitate saving without display.
matplotlib.use("Agg")


def ensure_directory(path: str) -> None:
    """Ensure that the given directory exists.

    Parameters
    ----------
    path : str
        The path to the directory that should be created if it does not
        already exist.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


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
    fig, ax = plt.subplots(figsize=(7.0, 5.0)) # Increased figure size
    # Plot histogram normalized to density to align with KDE.
    sns.histplot(
        latencies,
        bins=40,
        color="lightgrey",
        stat="density",
        edgecolor="grey",
        ax=ax,
    )
    # Overlay KDE for the distribution.
    sns.kdeplot(latencies, color="blue", linewidth=2.0, ax=ax)

    # Enhanced axis labels and title with bold font
    ax.set_xlabel("Inference latency (ms)", fontweight="bold", labelpad=5)
    ax.set_ylabel("Density", fontweight="bold", labelpad=5)
    ax.set_title("Inference Latency Distribution", fontsize=18, fontweight="bold")

    # Improve aesthetics
    sns.despine(ax=ax)
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.5)

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
        The F1-scores associated with each class. Values should be between
        0 and 1.
    cis : Iterable[float]
        Half-widths of the 95% confidence intervals for each score.

    Returns
    -------
    plt.Figure
        The Matplotlib figure containing the bar chart.
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.0)) # Increased figure size
    classes = list(classes)
    f1_scores = np.array(list(f1_scores))
    ci = np.array(list(cis))
    bar_colors = ["#4c72b0", "#55a868", "#c44e52"]

    bars = ax.bar(
        classes,
        f1_scores,
        yerr=ci,
        capsize=5,
        color=bar_colors[: len(classes)],
        error_kw=dict(ecolor='black', lw=1.5, capsize=5, capthick=1.5) # Style error bars
    )

    ax.set_ylim(0.85, 0.97)

    # Enhanced axis labels and title with bold font
    ax.set_ylabel(r"$F_{1}$-score", fontweight="bold", labelpad=5)
    ax.set_xlabel("Defect class", fontweight="bold", labelpad=5)
    ax.set_title("Per-class $F_1$-scores with 95% CI", fontsize=18, fontweight="bold")

    # Annotate bars with their respective values (increased font size).
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(0.004, ci[bars.index(bar)] + 0.002), # Adjust text position above error bar
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=14, # Increased font size for annotations
            fontweight="bold"
        )

    # Improve aesthetics
    sns.despine(ax=ax)
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    # Increase tick label size and make them bold
    ax.tick_params(axis='both', which='major', labelsize=14, labelcolor='black')

    fig.tight_layout()
    return fig


def generate_mAP_heatmap(
    ensemble_sizes: Iterable[int], mAP_values: Iterable[float]
) -> plt.Figure:
    """Generate a heatmap showing detection mAP versus ensemble size.

    Parameters
    ----------
    ensemble_sizes : Iterable[int]
        A sequence of ensemble sizes (e.g., [1, 2, 3, 4, 5, 6]).
    mAP_values : Iterable[float]
        Corresponding mAP@.5 values for each ensemble size.

    Returns
    -------
    plt.Figure
        The Matplotlib figure containing the heatmap.
    """
    sizes = list(ensemble_sizes)
    values = np.array(list(mAP_values))
    # Reshape into a single-row DataFrame for heatmap plotting.
    data_frame = pd.DataFrame(values.reshape(1, -1), columns=sizes)

    # Increased figure size
    fig, ax = plt.subplots(figsize=(8.0, 4.0))

    # Create heatmap with annotations
    sns.heatmap(
        data_frame,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        cbar_kws={"label": "mAP@.5"},
        vmin=88.0,
        vmax=93.0,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        # Style the annotations inside the heatmap cells
        annot_kws={"fontsize":14, "fontweight":"bold"}
    )

    # Enhanced axis labels and title with bold font
    ax.set_xlabel("Ensemble size (number of models)", fontweight="bold", labelpad=10)
    # Remove the y-axis ticks and label because there is only one row.
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_title("Effect of Ensemble Size on Detection mAP", fontsize=18, fontweight="bold")

    # Increase tick label size and make them bold
    ax.tick_params(axis='x', which='major', labelsize=14, labelcolor='black')
    # Access the colorbar object and set its label properties
    cbar = ax.collections[0].colorbar
    cbar.set_label("mAP@.5", fontweight="bold", labelpad=10)
    cbar.ax.tick_params(labelsize=14)

    fig.tight_layout()
    return fig


def generate_confusion_matrix(
    labels: Iterable[str], matrix: np.ndarray
) -> plt.Figure:
    """Plot a confusion matrix as a heatmap.

    Parameters
    ----------
    labels : Iterable[str]
        The category labels for the matrix rows and columns.
    matrix : np.ndarray
        A square matrix of counts where rows correspond to the actual
        labels and columns correspond to the predicted labels.

    Returns
    -------
    plt.Figure
        The figure containing the confusion matrix heatmap.
    """
    # Increased figure size
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    df_matrix = pd.DataFrame(matrix, index=labels, columns=labels)

    # Create heatmap with annotations
    sns.heatmap(
        df_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Count"},
        ax=ax,
        # Style the annotations inside the heatmap cells
        annot_kws={"fontsize": 14, "fontweight": "bold"}
    )

    # Enhanced axis labels and title with bold font
    ax.set_xlabel("Predicted label", fontweight="bold", labelpad=10)
    ax.set_ylabel("Actual label", fontweight="bold", labelpad=10)
    ax.set_title("Confusion Matrix for Criticality Assessment", fontsize=18, fontweight="bold")

    # Increase tick label size and make them bold
    ax.tick_params(axis='both', which='major', labelsize=14, labelcolor='black')
    # Rotate tick labels if needed (they usually fit for square matrices like this)
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Access the colorbar object and set its label properties
    cbar = ax.collections[0].colorbar
    cbar.set_label("Count", fontweight="bold", labelpad=10)
    cbar.ax.tick_params(labelsize=14)

    fig.tight_layout()
    return fig


def main() -> None:
    """Main function to generate and save all figures.

    This function defines the synthetic data used in the manuscript and
    calls the plotting functions to produce each figure. The output
    images are saved into a ``img`` directory located
    relative to where this script is executed.
    """
    # Set the directory for saving figures.
    output_dir = os.path.join(os.path.dirname(__file__), "img")
    ensure_directory(output_dir)

    # Set a random seed for reproducibility of synthetic data.
    rng = np.random.default_rng(seed=42)

    # ----- Figure 2(a): latency distribution -----
    mean_latency = 118.4
    std_latency = 12.1
    sample_size = 5_000
    latency_samples = rng.normal(loc=mean_latency, scale=std_latency, size=sample_size)
    fig2a = generate_latency_distribution(latency_samples)
    fig2a_path = os.path.join(output_dir, "61_fig_3a.pdf")
    fig2a.savefig(fig2a_path, dpi=300, bbox_inches='tight') # Added bbox_inches='tight'
    plt.close(fig2a)

    # ----- Figure 2(b): F1-score bar chart -----
    defect_classes = ["Crack", "Erosion", "Hotspot"]
    f1_scores = [0.92, 0.90, 0.94]
    ci_widths = [0.015, 0.02, 0.012]
    fig2b = generate_f1_bar_chart(defect_classes, f1_scores, ci_widths)
    fig2b_path = os.path.join(output_dir, "61_fig_3b.pdf")
    fig2b.savefig(fig2b_path, dpi=300, bbox_inches='tight')
    plt.close(fig2b)

    # ----- Figure 2(c): mAP heatmap -----
    ensemble_sizes = [1, 2, 3, 4, 5, 6]
    mAP_vals = [88.9, 90.1, 91.3, 92.1, 92.6, 92.8]
    fig2c = generate_mAP_heatmap(ensemble_sizes, mAP_vals)
    fig2c_path = os.path.join(output_dir, "61_fig_3c.pdf")
    fig2c.savefig(fig2c_path, dpi=300, bbox_inches='tight')
    plt.close(fig2c)

    # ----- Figure 3: confusion matrix -----
    labels = ["Negligible", "Low", "Medium", "High", "Severe"]
    confusion_counts = np.array(
        [
            [46, 3, 1, 0, 0],
            [4, 50, 2, 0, 0],
            [0, 2, 47, 1, 0],
            [0, 0, 1, 53, 2],
            [0, 0, 0, 2, 51],
        ],
        dtype=int,
    )
    fig3 = generate_confusion_matrix(labels, confusion_counts)
    fig3_path = os.path.join(output_dir, "61_fig_4.pdf")
    fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close(fig3)

    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()