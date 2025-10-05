from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASELINE_K = 132
BASELINE_RUNTIME_SECONDS = 6.84
K_MIN = 50
K_MAX = 2000
NUM_POINTS = 600

ANNOTATION_ARROW_PROPS = {
    "arrowstyle": "->",
    "color": "black",
    "lw": 2,
}
ANNOTATION_BOX_KWARGS = {
    "boxstyle": "round,pad=0.25",
    "fc": "white",
    "ec": "black",
    "lw": 1.5,
    "alpha": 0.95,
}


@dataclass(frozen=True)
class ProjectionLabel:
    """Collect placement options for a projected-runtime annotation."""

    k_value: int
    xytext: tuple[float, float] = (36.0, 28.0)
    textcoords: str = "offset points"
    ha: str | None = None
    va: str | None = None


PROJECTION_LABELS: tuple[ProjectionLabel, ...] = (
    ProjectionLabel(k_value=500, xytext=(42, -50)),
    ProjectionLabel(k_value=1000, xytext=(42, -50)),
    ProjectionLabel(k_value=2000, xytext=(-20, -50)),
)


def configure_plot_style() -> None:
    """Ensure all text is bold and slightly larger for publication clarity."""
    base_size = 16
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": base_size + 2,
            "font.weight": "bold",
            "axes.labelsize": base_size + 4,
            "axes.labelweight": "bold",
            "axes.titlesize": base_size + 8,
            "axes.titleweight": "bold",
            "legend.fontsize": base_size + 2,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.5,
        }
    )


def _annotation_text_kwargs() -> dict[str, object]:
    return {
        "fontsize": plt.rcParams["font.size"] + 2,
        "fontweight": "bold",
        "bbox": dict(ANNOTATION_BOX_KWARGS),
    }


def project_runtime(
    k_values: np.ndarray,
    *,
    baseline_k: float = BASELINE_K,
    baseline_runtime: float = BASELINE_RUNTIME_SECONDS,
) -> np.ndarray:
    """Project runtime under O(K log K) complexity anchored to the measured point."""
    scaling_constant = baseline_runtime / (baseline_k * np.log(baseline_k))
    return scaling_constant * k_values * np.log(k_values)


def prepare_runtime_curve(
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    num_points: int = NUM_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    k_curve = np.linspace(k_min, k_max, num_points)
    runtime_curve = project_runtime(k_curve)
    return k_curve, runtime_curve


def _resolve_alignment(label: ProjectionLabel) -> tuple[str, str]:
    if label.textcoords != "offset points":
        return label.ha or "left", label.va or "bottom"

    x_offset, y_offset = label.xytext
    ha = label.ha or ("left" if x_offset >= 0 else "right")
    va = label.va or ("bottom" if y_offset >= 0 else "top")
    return ha, va


def annotate_projection(ax: plt.Axes, label: ProjectionLabel) -> None:
    projected_runtime = project_runtime(np.array([label.k_value]))[0]
    ha, va = _resolve_alignment(label)

    ax.annotate(
        f"Projected ~= {projected_runtime:0.1f}s\nfor K = {label.k_value}",
        xy=(label.k_value, projected_runtime),
        xytext=label.xytext,
        textcoords=label.textcoords,
        ha=ha,
        va=va,
        arrowprops=ANNOTATION_ARROW_PROPS,
        **_annotation_text_kwargs(),
    )


def create_scalability_plot() -> plt.Figure:
    configure_plot_style()
    k_curve, runtime_curve = prepare_runtime_curve()

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(
        k_curve,
        runtime_curve,
        color="#1f77b4",
        linewidth=3,
        label=r"Projected $O(K \log K)$ runtime",
    )

    ax.scatter(
        [BASELINE_K],
        [BASELINE_RUNTIME_SECONDS],
        s=120,
        color="black",
        edgecolor="white",
        linewidth=2.2,
        zorder=5,
        label="Measured baseline",
    )

    baseline_annotation_kwargs = _annotation_text_kwargs()
    ax.annotate(
        f"Measured {BASELINE_RUNTIME_SECONDS:0.2f}s\nat K = {BASELINE_K}",
        xy=(BASELINE_K, BASELINE_RUNTIME_SECONDS),
        xytext=(BASELINE_K * 2.5, BASELINE_RUNTIME_SECONDS * 0.4),
        arrowprops=ANNOTATION_ARROW_PROPS,
        ha="left",
        va="bottom",
        **baseline_annotation_kwargs,
    )

    for projection_label in PROJECTION_LABELS:
        annotate_projection(ax, projection_label)

    ax.set_yscale("log")
    ax.set_xlim(K_MIN, K_MAX)
    ax.set_xlabel(r"Number of time windows ($K$)")
    ax.set_ylabel("Runtime (seconds, log scale)")

    ax.grid(True, which="both", linestyle="--", linewidth=1.1, alpha=0.7)
    ax.legend(
        loc="upper left",
        frameon=False,
        prop={"weight": "bold", "size": plt.rcParams["legend.fontsize"]},
    )

    ax.tick_params(axis="both", which="major", length=7, width=1.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    fig.tight_layout()
    return fig


def model_and_plot_scalability(*, show: bool = False) -> None:
    fig = create_scalability_plot()
    output_dir = Path(__file__).resolve().parent / "img"
    output_dir.mkdir(parents=True, exist_ok=True)

    # png_path = "68_fig_a6.png"
    pdf_path = "68_fig_a6.pdf"

    # fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0)

    # print(f"Saved PNG plot to {png_path}")
    print(f"Saved PDF plot to {pdf_path}")

    plt.show()


if __name__ == "__main__":
    model_and_plot_scalability()
