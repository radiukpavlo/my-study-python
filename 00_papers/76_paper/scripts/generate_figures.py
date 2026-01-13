#!/usr/bin/env python3
"""Generate publication-ready figures for the CEUR manuscript."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ArrowStyle, FancyArrowPatch, Rectangle

BASE_DIR = Path(__file__).resolve().parent.parent
FIGS_DIR = BASE_DIR / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DATA_M2 = {
    "Crack": {"mAP50": 0.93, "Precision": 0.91, "Recall": 0.89},
    "Soiling": {"mAP50": 0.90, "Precision": 0.88, "Recall": 0.86},
    "Delamination": {"mAP50": 0.89, "Precision": 0.87, "Recall": 0.85},
    "Average": {"mAP50": 0.91, "Precision": 0.89, "Recall": 0.87},
}
DATA_M3 = {
    "Crack": {"mAP50": 0.87, "Precision": 0.86, "Recall": 0.84},
    "Soiling": {"mAP50": 0.90, "Precision": 0.88, "Recall": 0.86},
    "Delamination": {"mAP50": 0.93, "Precision": 0.90, "Recall": 0.88},
    "Average": {"mAP50": 0.90, "Precision": 0.88, "Recall": 0.86},
}
ENSEMBLE_BEFORE_AFTER = {
    "Crack": (0.93, 0.96),
    "Soiling": (0.90, 0.90),
    "Delamination": (0.93, 0.95),
}
HARDWARE = [
    ("NVIDIA Jetson Orin Nano", 100, 0.95, 0.93),
    ("Ambarella H2", 60, 0.70, 0.75),
    ("Qualcomm QCS605", 80, 0.85, 0.82),
]
FLIGHT_HEIGHT = [(5, 98, 96), (10, 93, 90), (15, 84, 79)]
FLIGHT_SPEED = [(3, 94, 92), (5, 93, 90), (7, 91, 88), (10, 85, 82)]
TIME_OF_DAY = [("08:00-10:00", 93, 90), ("12:00-14:00", 96, 94), ("17:00-19:00", 92, 89)]
WEATHER = [("Clear", 92, 89), ("Cloudy", 96, 94)]
SOTA_MAP = [
    ("Di Tommaso et al. (IR hotspots) AP@0.5", 0.669),
    ("Dotenco et al. (defect cls) F1", 0.939),
    ("Xie et al. ST-YOLO mAP@0.5", 0.966),
    ("This work (ensemble) mAP@0.5", 0.960),
]

BASE_FONT_SIZE = mpl.rcParams.get("font.size", 10) + 2
TITLE_FONT_SIZE = BASE_FONT_SIZE + 2
LEGEND_FONT_SIZE = max(BASE_FONT_SIZE - 2, 10)
FONT_SMALL = max(BASE_FONT_SIZE - 2, 10)
FONT_MEDIUM = BASE_FONT_SIZE - 1

COLOR_CYCLE = mpl.colormaps["tab10"].colors
MAJOR_GRID_STYLE = {"color": "#c7ccd6", "linewidth": 0.9, "alpha": 0.7}
DARK_EDGE_COLOR = "#1f2937"
DARK_TEXT_COLOR = "#0f172a"

mpl.rcParams.update(
    {
        "font.size": BASE_FONT_SIZE,
        "font.weight": "bold",
        "axes.labelsize": BASE_FONT_SIZE,
        "axes.labelweight": "bold",
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.titleweight": "bold",
        "xtick.labelsize": BASE_FONT_SIZE,
        "ytick.labelsize": BASE_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "legend.framealpha": 0.92,
        "axes.edgecolor": DARK_EDGE_COLOR,
        "axes.labelcolor": DARK_TEXT_COLOR,
        "axes.titlecolor": DARK_TEXT_COLOR,
        "axes.linewidth": 1.1,
        "grid.color": MAJOR_GRID_STYLE["color"],
        "grid.linewidth": MAJOR_GRID_STYLE["linewidth"],
        "grid.alpha": MAJOR_GRID_STYLE["alpha"],
        "savefig.dpi": 240,
    }
)
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLOR_CYCLE)


def _enforce_bold_text(ax: mpl.axes.Axes) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for text in ax.texts:
        text.set_fontweight("bold")
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontweight("bold")


def _create_axes(
    figsize: tuple[float, float],
    grid_axis: str = "y",
) -> tuple[plt.Figure, mpl.axes.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#fbfbf8")
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(True, axis=grid_axis, **MAJOR_GRID_STYLE)
    ax.tick_params(axis="both", width=1.05, length=5, color=DARK_EDGE_COLOR)
    for spine in ax.spines.values():
        spine.set_color(DARK_EDGE_COLOR)
        spine.set_linewidth(1.1)
    return fig, ax


def _save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    for ax in fig.axes:
        _enforce_bold_text(ax)
    for ext in ("pdf", "svg"):
        fig.savefig(FIGS_DIR / f"{stem}.{ext}", format=ext, bbox_inches="tight")
    plt.close(fig)


def _add_box(
    ax: mpl.axes.Axes,
    text: str,
    x: float,
    y: float,
    w: float = 0.26,
    h: float = 0.08,
) -> None:
    ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=1.2, edgecolor=DARK_EDGE_COLOR))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FONT_MEDIUM)


def _add_arrow(ax: mpl.axes.Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=ArrowStyle("->", head_length=6, head_width=3),
            linewidth=1.2,
            mutation_scale=10,
            color=DARK_EDGE_COLOR,
        )
    )


def fig_cps_architecture() -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    tiers = [
        ("UAV tier\n(sensing + onboard inference)", 0.02, 0.62, 0.30, 0.33),
        ("Edge tier\n(local server at PV plant)", 0.35, 0.62, 0.30, 0.33),
        ("Cloud tier\n(long-term analytics)", 0.68, 0.62, 0.30, 0.33),
        ("Operator & SCADA\n(HMI / alarms / work orders)", 0.35, 0.08, 0.30, 0.33),
    ]
    for label, x, y, w, h in tiers:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=1.5, edgecolor=DARK_EDGE_COLOR))
        ax.text(x + w / 2, y + h - 0.03, label, ha="center", va="top", fontsize=FONT_MEDIUM)

    _add_box(ax, "RGB camera", 0.04, 0.78)
    _add_box(ax, "Thermal camera", 0.04, 0.69)
    _add_box(ax, "Jetson Orin\n(YOLOv11-seg)", 0.04, 0.60)

    _add_box(ax, "Mission cache\n+ RTK geotagging", 0.37, 0.78)
    _add_box(ax, "Aggregation\n+ de-duplication", 0.37, 0.69)
    _add_box(ax, "SCADA gateway\n(MQTT/OPC-UA)", 0.37, 0.60)

    _add_box(ax, "Object store\n(images + masks)", 0.70, 0.78)
    _add_box(ax, "Analytics\n(dashboards, trends)", 0.70, 0.69)
    _add_box(ax, "Model registry\n+ updates", 0.70, 0.60)

    _add_box(ax, "HMI dashboard", 0.37, 0.24)
    _add_box(ax, "Alarm rules\n+ fire-risk logic", 0.37, 0.15)

    _add_arrow(ax, 0.32, 0.75, 0.35, 0.75)
    _add_arrow(ax, 0.32, 0.66, 0.35, 0.66)
    _add_arrow(ax, 0.65, 0.75, 0.68, 0.75)
    _add_arrow(ax, 0.65, 0.66, 0.68, 0.66)
    _add_arrow(ax, 0.50, 0.62, 0.50, 0.41)

    _save_figure(fig, "cps_architecture")


def fig_glare_geometry() -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot([0.1, 0.9], [0.2, 0.2], linewidth=2, color=DARK_EDGE_COLOR)
    ax.text(0.5, 0.14, "PV module plane", ha="center", fontsize=FONT_SMALL)
    ax.arrow(0.5, 0.2, 0, 0.5, head_width=0.02, length_includes_head=True, color=DARK_EDGE_COLOR)
    ax.text(0.52, 0.55, "n", fontsize=FONT_MEDIUM)
    ax.arrow(
        0.2,
        0.9,
        0.25,
        -0.65,
        head_width=0.02,
        length_includes_head=True,
        color=DARK_EDGE_COLOR,
    )
    ax.text(0.18, 0.92, "s (sun ray)", fontsize=FONT_SMALL)
    ax.arrow(
        0.45,
        0.25,
        0.35,
        0.55,
        head_width=0.02,
        length_includes_head=True,
        color=DARK_EDGE_COLOR,
    )
    ax.text(0.82, 0.82, "r (specular)", fontsize=FONT_SMALL, ha="right")
    ax.arrow(
        0.45,
        0.25,
        0.45,
        0.10,
        head_width=0.02,
        length_includes_head=True,
        color=DARK_EDGE_COLOR,
    )
    ax.text(0.92, 0.35, "v* (camera)\nrotated away", fontsize=FONT_SMALL, ha="right")
    _save_figure(fig, "glare_geometry")


def _plot_map(dataset: dict[str, dict[str, float]], title: str, stem: str) -> None:
    labels = ["Crack", "Soiling", "Delamination", "Average"]
    vals = [dataset[key]["mAP50"] for key in labels]
    fig, ax = _create_axes(figsize=(6, 3.5), grid_axis="y")
    bars = ax.bar(labels, vals, edgecolor=DARK_EDGE_COLOR, linewidth=1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("mAP@0.5")
    ax.set_title(title)
    ax.bar_label(bars, padding=3, fontsize=FONT_SMALL)
    _save_figure(fig, stem)


def fig_map_charts() -> None:
    _plot_map(DATA_M2, "Thermogram M2 (two-color palette): detection quality", "map_m2")
    _plot_map(DATA_M3, "Thermogram M3 (three-color palette): detection quality", "map_m3")


def fig_ensemble_gain() -> None:
    labels = list(ENSEMBLE_BEFORE_AFTER.keys())
    before = [ENSEMBLE_BEFORE_AFTER[key][0] for key in labels]
    after = [ENSEMBLE_BEFORE_AFTER[key][1] for key in labels]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = _create_axes(figsize=(6, 3.5), grid_axis="y")
    ax.bar(
        x - width / 2,
        before,
        width,
        label="Before ensemble",
        edgecolor=DARK_EDGE_COLOR,
        linewidth=1.0,
    )
    ax.bar(
        x + width / 2,
        after,
        width,
        label="After ensemble",
        edgecolor=DARK_EDGE_COLOR,
        linewidth=1.0,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Ensemble fusion improves mAP@0.5")
    ax.legend()
    _save_figure(fig, "ensemble_gain")


def fig_hardware_tradeoff() -> None:
    fig, ax = _create_axes(figsize=(6, 3.5), grid_axis="y")
    fps = [entry[1] for entry in HARDWARE]
    map_values = [entry[2] for entry in HARDWARE]
    ax.scatter(fps, map_values, s=70, edgecolor=DARK_EDGE_COLOR, linewidth=0.8)
    for name, fps_value, map_value, _ in HARDWARE:
        ax.annotate(
            name,
            (fps_value, map_value),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=FONT_SMALL,
        )
    ax.set_xlabel("Throughput (FPS)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Edge hardware tradeoff (thermogram inference)")
    ax.set_xlim(40, 110)
    ax.set_ylim(0.6, 1.0)
    _save_figure(fig, "hardware_tradeoff")


def _plot_param(
    data: Sequence[Sequence[object]],
    xlabel: str,
    title: str,
    stem: str,
) -> None:
    if isinstance(data[0][0], str):
        labels = [str(entry[0]) for entry in data]
        accuracy = [float(entry[1]) for entry in data]
        recall = [float(entry[2]) for entry in data]
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = _create_axes(figsize=(6, 3.5), grid_axis="y")
        ax.bar(
            x - width / 2,
            accuracy,
            width,
            label="Accuracy (%)",
            edgecolor=DARK_EDGE_COLOR,
            linewidth=1.0,
        )
        ax.bar(
            x + width / 2,
            recall,
            width,
            label="Recall (%)",
            edgecolor=DARK_EDGE_COLOR,
            linewidth=1.0,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
    else:
        xvals = [float(entry[0]) for entry in data]
        accuracy = [float(entry[1]) for entry in data]
        recall = [float(entry[2]) for entry in data]
        fig, ax = _create_axes(figsize=(6, 3.5), grid_axis="y")
        ax.plot(xvals, accuracy, marker="o", linewidth=2.4, label="Accuracy (%)")
        ax.plot(xvals, recall, marker="o", linewidth=2.4, label="Recall (%)")
        ax.set_xticks(xvals)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_ylim(70, 100)
    ax.legend()
    _save_figure(fig, stem)


def fig_flight_params() -> None:
    _plot_param(
        FLIGHT_HEIGHT,
        "Flight altitude (m)",
        "Effect of altitude on detection quality",
        "flight_height",
    )
    _plot_param(
        FLIGHT_SPEED,
        "Flight speed (m/s)",
        "Effect of speed on detection quality",
        "flight_speed",
    )
    _plot_param(TIME_OF_DAY, "Time window", "Time-of-day sensitivity", "time_of_day")
    _plot_param(WEATHER, "Weather", "Weather sensitivity", "weather")


def fig_sota_comparison() -> None:
    fig, ax = _create_axes(figsize=(7, 3.5), grid_axis="x")
    labels = [entry[0] for entry in SOTA_MAP]
    values = [entry[1] for entry in SOTA_MAP]
    bars = ax.barh(labels, values, edgecolor=DARK_EDGE_COLOR, linewidth=1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Reported score (mAP@0.5 or F1)")
    ax.set_title("Representative quantitative comparison (reported by authors)")
    ax.bar_label(bars, padding=4, fontsize=FONT_SMALL)
    _save_figure(fig, "sota_comparison")


def main() -> None:
    fig_cps_architecture()
    fig_glare_geometry()
    fig_map_charts()
    fig_ensemble_gain()
    fig_hardware_tradeoff()
    fig_flight_params()
    fig_sota_comparison()
    print(f"Done. Figures written to: {FIGS_DIR} (.pdf/.svg)")


if __name__ == "__main__":
    main()
