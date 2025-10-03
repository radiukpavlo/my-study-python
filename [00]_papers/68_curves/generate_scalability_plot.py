# File: generate_scalability_plot.py

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def model_and_plot_scalability() -> None:
    """Model scalability and generate a publication-quality log-linear plot."""
    # --- Data Points and Complexity Modeling ---
    k_known = 132
    runtime_known = 6.84  # seconds
    k_values = np.array([64, k_known, 256])

    # Model 1: O(K log K) complexity for accelerated HDBSCAN
    # Runtime = c * K * log2(K). Find scaling constant 'c'.
    c_nlogn = runtime_known / (k_known * np.log2(k_known))
    runtimes_nlogn = c_nlogn * k_values * np.log2(k_values)

    # Model 2: O(K^2) complexity (naive case) for visual comparison
    # Runtime = c * K^2. Set 'c' to pass through the known point.
    c_quad = runtime_known / (k_known**2)

    # --- Theoretical Curve Generation for Visualization ---
    k_smooth = np.linspace(50, 300, 500)
    runtime_curve_nlogn = c_nlogn * k_smooth * np.log2(k_smooth)
    runtime_curve_quad = c_quad * k_smooth**2

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(k_smooth, runtime_curve_quad, 'r--', linewidth=2,
            label=r'Theoretical $O(K^2)$ Scaling (Worst-Case)')
    ax.plot(k_smooth, runtime_curve_nlogn, 'b-', linewidth=2.5,
            label=r'Fitted $O(K \log K)$ Scaling (Accelerated HDBSCAN)')
    ax.plot(k_values, runtimes_nlogn, 'ko', markersize=9, mfc='white', mew=2,
            label='Runtime Data Points (Actual & Extrapolated)')

    ax.set_yscale('log')
    ax.set_title(
        'Computational Scalability of the Adaptive Cascade Approach (Section 4.4)',
        fontsize=18, pad=15
    )
    ax.set_xlabel('K (Number of Time Windows)', fontsize=14)
    ax.set_ylabel('Estimated Runtime (seconds, log scale)', fontsize=14)

    ax.annotate(
        f'Actual Measurement\nFrom Experiment\n({k_known} windows, {runtime_known}s)',
        xy=(k_known, runtime_known),
        xytext=(k_known + 40, runtime_known - 3),
        arrowprops=dict(facecolor='black', shrink=0.05,
                        width=1.5, headwidth=10),
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", fc="azure",
                  ec="black", lw=1, alpha=0.9)
    )

    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, which="both", ls="--", alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim(50, 300)

    plt.tight_layout()
    output_path = Path(__file__).resolve().with_name("figure_scalability_curve.png")
    fig.savefig(output_path, dpi=300)
    print(f"Successfully saved scalability plot to {output_path}")


if __name__ == '__main__':
    model_and_plot_scalability()



