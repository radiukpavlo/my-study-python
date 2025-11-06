import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Union


def plot_roc_curves(output_path: Union[str, Path, None] = None) -> None:
    """
    Recreates the ROC curves from 67_fig_10.png and saves them as a PDF.
    """
    # Match the typography choices used in 67_conf_matrices.py
    plt.style.use("default")
    font_properties = {
        "family": "sans-serif",
        "size": 16,
        "weight": "bold",
    }
    plt.rc("font", **font_properties)
    plt.rc("axes", titlesize=18)
    plt.rc("xtick", labelsize=14)
    plt.rc("ytick", labelsize=14)

    roc_configs: List[Dict[str, Any]] = [
        {
            "label": "1 layer - Faster R-CNN",
            "color": "#d62728",
            "points": [
                (0.00, 0.00),
                (0.01, 0.30),
                (0.02, 0.50),
                (0.03, 0.60),
                (0.05, 0.71),
                (0.06, 0.76),
                (0.08, 0.82),
                (0.10, 0.85),
                (0.12, 0.88),
                (0.14, 0.90),
                (0.16, 0.93),
                (0.18, 0.947),
                (0.20, 0.957),
                (0.23, 0.968),
                (0.26, 0.975),
                (0.30, 0.984),
                (0.35, 0.991),
                (0.40, 0.995),
                (0.50, 0.998),
                (0.65, 1.00),
                (1.00, 1.00),
            ],
            "target_auc": 0.95,
        },
        {
            "label": "2 layer - YOLOv11m",
            "color": "#000000",
            "points": [
                (0.00, 0.00),
                (0.01, 0.18),
                (0.02, 0.36),
                (0.03, 0.49),
                (0.05, 0.62),
                (0.07, 0.69),
                (0.09, 0.76),
                (0.11, 0.81),
                (0.13, 0.85),
                (0.16, 0.89),
                (0.19, 0.92),
                (0.23, 0.94),
                (0.27, 0.955),
                (0.31, 0.968),
                (0.35, 0.978),
                (0.40, 0.986),
                (0.45, 0.991),
                (0.55, 0.995),
                (0.70, 0.999),
                (1.00, 1.00),
            ],
            "target_auc": 0.93,
        },
        {
            "label": "3 layer - YOLOv11s",
            "color": "#2ca02c",
            "points": [
                (0.00, 0.00),
                (0.01, 0.26),
                (0.02, 0.50),
                (0.03, 0.62),
                (0.04, 0.71),
                (0.05, 0.78),
                (0.06, 0.82),
                (0.08, 0.86),
                (0.10, 0.89),
                (0.12, 0.92),
                (0.14, 0.948),
                (0.16, 0.964),
                (0.18, 0.973),
                (0.22, 0.985),
                (0.26, 0.991),
                (0.30, 0.995),
                (0.35, 0.998),
                (0.40, 1.00),
                (0.50, 1.00),
                (1.00, 1.00),
            ],
            "target_auc": 0.96,
        },
    ]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Reference diagonal for random classifier performance
    ax.plot([0, 1], [0, 1], linestyle="--", color="#b22222", linewidth=1.5, label="_nolegend_")

    for config in roc_configs:
        points = np.array(config["points"], dtype=float)
        fpr = points[:, 0]
        tpr = points[:, 1]
        auc_estimate = np.trapezoid(tpr, fpr)
        legend_label = f"{config['label']} (AUC = {config['target_auc']:.2f})"
        ax.plot(
            fpr,
            tpr,
            color=config["color"],
            linewidth=2.5,
            label=legend_label,
        )
        print(f"{config['label']} approximated AUC: {auc_estimate:.4f}")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax.set_title("ROC Curve Comparison", weight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=16, weight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=16, weight="bold")

    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.6)

    # Apply bold formatting to tick labels to match the updated style
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    legend = ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=12)
    for text in legend.get_texts():
        text.set_fontweight("bold")

    plt.tight_layout()

    destination = Path(output_path) if output_path is not None else Path(__file__).with_name("67_fig_10.pdf")
    fig.savefig(destination, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"ROC curves saved to '{destination}'")


if __name__ == "__main__":
    plot_roc_curves()
