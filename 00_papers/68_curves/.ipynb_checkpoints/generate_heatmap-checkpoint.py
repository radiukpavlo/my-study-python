# File: generate_heatmap.py

import ternary_diagram
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
from typing import Dict, Tuple, List

# --- Type Hint Aliases ---
MetricsDict = Dict[str, float]
WeightsTuple = Tuple[float, float, float]


def calculate_quality_scores(
    alpha: float,
    beta: float,
    gamma: float,
    hdbscan_metrics: MetricsDict,
    kmeans_metrics: MetricsDict
) -> Tuple[float, float]:
    """Calculate composite quality scores based on manuscript Equations 20 & 21."""
    quality_hdbscan = (alpha * hdbscan_metrics['Silhouette'] +
                       beta * hdbscan_metrics['Stability'] +
                       gamma * hdbscan_metrics['Interpretability'])

    quality_kmeans = (alpha * kmeans_metrics['Silhouette'] +
                      beta * kmeans_metrics['Compactness'] +
                      gamma * kmeans_metrics['Separation'])

    return quality_hdbscan, quality_kmeans


def choose_strategy(
    quality_hdbscan: float,
    quality_kmeans: float,
    tolerance: float
) -> int:
    """Implement the tolerant decision rule from manuscript Equation 19."""
    if quality_hdbscan > quality_kmeans + tolerance:
        return 0  # HDBSCAN is significantly better
    elif quality_kmeans > quality_hdbscan + tolerance:
        return 1  # k-means is significantly better
    else:
        return 2  # Scores are within tolerance, choose Hybrid


def generate_heatmap_data(
    steps: int,
    hdbscan_metrics: MetricsDict,
    kmeans_metrics: MetricsDict,
    tolerance: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate simplex points and corresponding strategy codes for the heatmap.

    This function manually generates points on a triangular grid (simplex)
    without relying on external library helpers.
    """
    weights: List[WeightsTuple] = []
    for i in range(steps + 1):
        for j in range(steps - i + 1):
            k = steps - i - j
            # Normalize to ensure weights sum to 1
            alpha = i / float(steps)
            beta = j / float(steps)
            gamma = k / float(steps)
            weights.append((alpha, beta, gamma))

    values = []
    for alpha, beta, gamma in weights:
        q_h, q_k = calculate_quality_scores(
            alpha, beta, gamma, hdbscan_metrics, kmeans_metrics
        )
        strategy = choose_strategy(q_h, q_k, tolerance)
        values.append(strategy)

    return np.array(weights), np.array(values)


def generate_and_save_heatmap() -> None:
    """Configure, generate, and save the publication-quality ternary heatmap."""
    # --- Configuration based on the Manuscript ---
    hdbscan_metrics: MetricsDict = {
        'Silhouette': 0.52, 'Stability': 0.90, 'Interpretability': 0.85
    }
    kmeans_metrics: MetricsDict = {
        'Silhouette': 0.57, 'Compactness': 0.88, 'Separation': 0.80
    }
    delta_tolerance = 0.05
    heatmap_steps = 100

    # --- Data Generation ---
    weights, values = generate_heatmap_data(
        heatmap_steps, hdbscan_metrics, kmeans_metrics, delta_tolerance
    )

    # Project the 3D simplex weights into 2D Cartesian coordinates
    x_coords, y_coords = ternary_diagram.project_points(weights)

    # --- Plotting with ternary-diagram ---
    fig, ax = ternary_diagram.create_axes(
        labels=[
            r"$\alpha$ (Silhouette Score)",
            r"$\beta$ (Stability / Compactness)",
            r"$\gamma$ (Interpretability / Separation)"
        ],
        label_fontsize=14,
        edge_colors='black',
        linewidth=2.0,
    )
    fig.set_size_inches(11, 10)
    fig.suptitle(
        "Appendix B: Strategy Selection Sensitivity to Weighting Factors",
        fontsize=18, y=0.98
    )

    cmap = plt.get_cmap('viridis', 3)
    levels = [-0.5, 0.5, 1.5, 2.5]  # Boundaries for the three categories

    contour = ax.tricontourf(x_coords, y_coords, values, levels=levels, cmap=cmap)
    ternary_diagram.grid(ax, color="white", linewidth=0.7, linestyle='--')

    cbar = fig.colorbar(contour, ax=ax, ticks=[0, 1, 2])
    cbar.set_ticklabels(['HDBSCAN-first', 'k-means-first', 'Hybrid'])
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Selected Strategy", size=14)

    equal_weight_point = (1/3., 1/3., 1/3.)
    x_eq, y_eq = ternary_diagram.project_point(equal_weight_point)
    ax.scatter(
        x_eq, y_eq, marker='*', color='red', s=300, edgecolor='black',
        zorder=10, label='Equal Weights (Manuscript Default)'
    )
    ax.legend(fontsize=12, loc='upper right', bbox_to_anchor=(0.95, 1.0))

    q_h, q_k = calculate_quality_scores(1/3, 1/3, 1/3, hdbscan_metrics,
                                        kmeans_metrics)
    strategy_code = choose_strategy(q_h, q_k, delta_tolerance)
    strategy_label = ['HDBSCAN-first', 'k-means-first', 'Hybrid'][strategy_code]
    print(f"Heatmap Script: At equal weights (1/3, 1/3, 1/3), the chosen "
          f"strategy is: {strategy_label}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = "appendix_b_weight_sensitivity.png"
    fig.savefig(output_filename, dpi=300)
    print(f"Successfully saved heatmap to {output_filename}")


if __name__ == '__main__':
    generate_and_save_heatmap()