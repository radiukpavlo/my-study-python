# File: generate_heatmap.py

"""Generate manuscript-aligned weight sensitivity heatmap for the voting mechanism."""

from pathlib import Path
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
import numpy as np

# Reset and configure rcParams for smaller text but larger plotted objects
# Make all text bold (including math text)
plt.rcParams.update({
    'font.size': 18,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titlesize': 20,
    'axes.titleweight': 'bold',
    'legend.fontsize': 18,
    'legend.title_fontsize': 18,
    'mathtext.default': 'bf',
})

# --- Type Hint Aliases ---
MetricsDict = Dict[str, float]
WeightsTuple = Tuple[float, float, float]

# --- Manuscript Configuration ---
SIMPLEX_STEP = 0.1  # Increments across the probability simplex
DELTA_TOLERANCE = 0.03  # delta_tolerance from the manuscript subsection

# Vertices of an equilateral triangle used for barycentric projection
TRIANGLE_VERTICES = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, np.sqrt(3.0) / 2.0],
])

# Offsets to keep corner labels readable
LABEL_OFFSET_X = 0.08
LABEL_OFFSET_Y = 0.07
# Inset bottom-edge labels (alpha/beta) toward the center of the base
EDGE_LABEL_INSET = 0.15

# Visual scaling factors for larger objects
TRIANGLE_LINEWIDTH = 3.0
GRID_LINEWIDTH = 1.2
GRID_ALPHA = 0.6
MARKER_SIZE = 450


def project_point(weights: WeightsTuple) -> Tuple[float, float]:
    """Project a single barycentric coordinate triple onto the triangle."""
    weights_array = np.asarray(weights, dtype=float)
    coords = weights_array @ TRIANGLE_VERTICES
    return float(coords[0]), float(coords[1])


def project_points(weight_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised projection of multiple barycentric coordinates."""
    weights_2d = np.asarray(weight_array, dtype=float)
    coords = weights_2d @ TRIANGLE_VERTICES
    return coords[:, 0], coords[:, 1]


def create_axes(
    labels: List[str],
    label_fontsize: int = 16,
    edge_colors: str = 'black',
    linewidth: float = TRIANGLE_LINEWIDTH,
):
    """Create a Matplotlib axis resembling a ternary diagram."""
    fig, ax = plt.subplots()

    triangle = Polygon(
        TRIANGLE_VERTICES,
        closed=True,
        edgecolor=edge_colors,
        facecolor='none',
        linewidth=linewidth,
    )
    ax.add_patch(triangle)
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, TRIANGLE_VERTICES[2, 1] + 0.12)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if labels:
        # Place alpha and beta labels inset from the corners toward the center of the base
        ax.text(
            EDGE_LABEL_INSET,
            -LABEL_OFFSET_Y,
            labels[0],
            fontsize=label_fontsize,
            fontweight='bold',
            ha='center',
            va='top',
        )
        ax.text(
            1.0 - EDGE_LABEL_INSET,
            -LABEL_OFFSET_Y,
            labels[1],
            fontsize=label_fontsize,
            fontweight='bold',
            ha='center',
            va='top',
        )
        # Gamma label remains above the apex
        ax.text(
            0.5,
            TRIANGLE_VERTICES[2, 1] + LABEL_OFFSET_Y,
            labels[2],
            fontsize=label_fontsize,
            fontweight='bold',
            ha='center',
            va='bottom',
        )

    return fig, ax


def grid(
    ax,
    subdivisions: int = 10,
    **line_kwargs,
) -> None:
    """Draw internal grid lines for the ternary diagram."""
    if subdivisions < 2:
        return

    processed_kwargs = dict(line_kwargs)
    if 'color' in processed_kwargs:
        processed_kwargs['colors'] = processed_kwargs.pop('color')
    if 'linewidth' in processed_kwargs:
        processed_kwargs['linewidths'] = processed_kwargs.pop('linewidth')
    if 'linestyle' in processed_kwargs:
        processed_kwargs['linestyles'] = processed_kwargs.pop('linestyle')

    defaults = {
        'colors': (1.0, 1.0, 1.0, GRID_ALPHA),
        'linewidths': GRID_LINEWIDTH,
        'linestyles': '--'
    }
    for key, value in defaults.items():
        processed_kwargs.setdefault(key, value)

    lines = []
    for i in range(1, subdivisions):
        t = i / subdivisions
        lines.append([
            project_point((t, 1.0 - t, 0.0)),
            project_point((t, 0.0, 1.0 - t)),
        ])
        lines.append([
            project_point((1.0 - t, t, 0.0)),
            project_point((0.0, t, 1.0 - t)),
        ])
        lines.append([
            project_point((1.0 - t, 0.0, t)),
            project_point((0.0, 1.0 - t, t)),
        ])

    ax.add_collection(LineCollection(lines, **processed_kwargs))


def calculate_quality_scores(
    alpha: float,
    beta: float,
    gamma: float,
    hdbscan_metrics: MetricsDict,
    kmeans_metrics: MetricsDict
) -> Tuple[float, float]:
    """Calculate composite quality scores using manuscript-aligned metrics."""
    quality_hdbscan = (
        alpha * hdbscan_metrics['Silhouette'] +
        beta * hdbscan_metrics['TemporalCoherence'] +
        gamma * hdbscan_metrics['CalinskiHarabasz']
    )

    quality_kmeans = (
        alpha * kmeans_metrics['Silhouette'] +
        beta * kmeans_metrics['TemporalCoherence'] +
        gamma * kmeans_metrics['CalinskiHarabasz']
    )

    return quality_hdbscan, quality_kmeans


def choose_strategy(
    quality_hdbscan: float,
    quality_kmeans: float,
    tolerance: float
) -> int:
    """Implement the tolerant decision rule from manuscript Equation 19."""
    if quality_hdbscan > quality_kmeans + tolerance:
        return 0  # HDBSCAN is significantly better
    if quality_kmeans > quality_hdbscan + tolerance:
        return 1  # k-means is significantly better
    return 2  # Scores are within tolerance, choose Hybrid


def generate_simplex_weights(step: float) -> np.ndarray:
    """Generate weight combinations across the simplex in the requested increments."""
    subdivisions = int(round(1.0 / step))
    weights: List[WeightsTuple] = []
    for i in range(subdivisions + 1):
        for j in range(subdivisions - i + 1):
            k = subdivisions - i - j
            weight = (i / subdivisions, j / subdivisions, k / subdivisions)
            weights.append(weight)
    return np.array(weights, dtype=float)


def evaluate_strategies(
    weights: np.ndarray,
    hdbscan_metrics: MetricsDict,
    kmeans_metrics: MetricsDict,
    tolerance: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute strategy codes and score deltas for every simplex point."""
    codes = np.empty(len(weights), dtype=int)
    deltas = np.empty(len(weights), dtype=float)
    for idx, (alpha, beta, gamma) in enumerate(weights):
        quality_hdbscan, quality_kmeans = calculate_quality_scores(
            alpha, beta, gamma, hdbscan_metrics, kmeans_metrics
        )
        deltas[idx] = abs(quality_hdbscan - quality_kmeans)
        codes[idx] = choose_strategy(quality_hdbscan, quality_kmeans, tolerance)
    return codes, deltas


def generate_and_save_heatmap() -> None:
    """Configure, generate, and save the manuscript weight sensitivity heatmap."""
    hdbscan_metrics: MetricsDict = {
        'Silhouette': 0.52,
        'TemporalCoherence': 0.93,
        'CalinskiHarabasz': 0.88,
    }
    kmeans_metrics: MetricsDict = {
        'Silhouette': 0.57,
        'TemporalCoherence': 0.81,
        'CalinskiHarabasz': 0.83,
    }

    weights = generate_simplex_weights(SIMPLEX_STEP)
    strategy_codes, score_deltas = evaluate_strategies(
        weights, hdbscan_metrics, kmeans_metrics, DELTA_TOLERANCE
    )

    x_coords, y_coords = project_points(weights)

    fig, ax = create_axes(
        labels=[
            r"$\alpha$ (Compactness / Silhouette)",
            r"$\beta$ (Stability / Temporal Coherence)",
            r"$\gamma$ (Separability / Calinski-Harabasz)"
        ],
        label_fontsize=20,
        edge_colors='black',
        linewidth=TRIANGLE_LINEWIDTH,
    )
    fig.set_size_inches(13, 11)
    # fig.suptitle(
    #     "Weight Sensitivity Analysis for the Voting Mechanism",
    #     fontsize=18,
    #     fontweight='bold',
    #     y=0.98,
    # )

    cmap = plt.get_cmap('viridis', 3)
    levels = [-0.5, 0.5, 1.5, 2.5]
    contour = ax.tricontourf(
        x_coords,
        y_coords,
        strategy_codes.astype(float),
        levels=levels,
        cmap=cmap,
    )
    grid(
        ax,
        subdivisions=int(round(1.0 / SIMPLEX_STEP)),
        color="white",
        linewidth=GRID_LINEWIDTH,
        linestyle='--',
    )

    cbar = fig.colorbar(
        contour,
        ax=ax,
        ticks=[0, 1, 2],
        fraction=0.04,
        pad=0.02,
        shrink=0.82,
    )
    cbar.set_ticklabels(['HDBSCAN-first', 'k-means-first', 'Hybrid'])
    cbar.ax.tick_params(labelsize=16)
    for t in cbar.ax.get_yticklabels():
        t.set_fontweight('bold')
    cbar.set_label(
        r"Selected Strategy ($\delta_{\text{tolerance}} = 0.03$)",
        size=16,
        fontweight='bold',
    )

    equal_weight_point = (1 / 3.0, 1 / 3.0, 1 / 3.0)
    x_eq, y_eq = project_point(equal_weight_point)
    ax.scatter(
        x_eq,
        y_eq,
        marker='*',
        color='red',
        s=MARKER_SIZE,
        edgecolor='black',
        zorder=10,
        label='Equal Weights',
    )
    legend = ax.legend(
        loc='upper right',
        bbox_to_anchor=(0.95, 0.8),
        frameon=True,
        edgecolor='black',
        facecolor='white',
        prop={'weight': 'bold', 'size': 18},
    )
    legend.get_frame().set_alpha(0.9)

    # Tight layout without extra margins
    fig.tight_layout(pad=0.0)

    output_dir = Path(__file__).resolve().parent / 'img'
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / 'appendix_b_weight_sensitivity.png'
    pdf_path = Path(__file__).resolve().parent / '68_fig_a1.pdf'

    fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.0)
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.0)

    strategy_labels = ['HDBSCAN-first', 'k-means-first', 'Hybrid']
    counts = {label: int(np.count_nonzero(strategy_codes == idx))
              for idx, label in enumerate(strategy_labels)}
    non_hybrid_mask = strategy_codes != 2
    min_gap = float(score_deltas[non_hybrid_mask].min()) if np.any(non_hybrid_mask) else 0.0

    print(f"Heatmap: evaluated {len(weights)} simplex points with step {SIMPLEX_STEP}.")
    for label in strategy_labels:
        print(f"  {label}: {counts[label]} combinations")
    print(f"  Hybrid threshold triggered for {counts['Hybrid']} combinations (|delta| <= {DELTA_TOLERANCE}).")
    print(f"  Smallest decisive gap outside tolerance: {min_gap:.4f}.")

    q_h, q_k = calculate_quality_scores(1/3, 1/3, 1/3, hdbscan_metrics, kmeans_metrics)
    strategy_code = choose_strategy(q_h, q_k, DELTA_TOLERANCE)
    strategy_label = strategy_labels[strategy_code]
    print(
        "Reference weights (1/3, 1/3, 1/3) produce "
        f"{strategy_label} with |delta|={abs(q_h - q_k):.4f}."
    )
    print(f"Saved heatmap PNG to {png_path}")
    print(f"Saved vector PDF to {pdf_path}")


if __name__ == '__main__':
    generate_and_save_heatmap()
