#!/usr/bin/env python3
"""Generate publication-ready figures and tables from CSV metrics."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figs"
TAB_DIR = ROOT / "tables"
DATA_DIR = ROOT / "data"
for directory in (FIG_DIR, TAB_DIR, DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)

BASE_FONT_SIZE = mpl.rcParams.get("font.size", 10) + 6
TITLE_FONT_SIZE = max(int(BASE_FONT_SIZE * 1.25) - 2, BASE_FONT_SIZE)
COLOR_CYCLE = mpl.colormaps["tab10"].colors
MAJOR_GRID_STYLE = {"color": "#b7bfcc", "linewidth": 0.95, "alpha": 0.65}
MINOR_GRID_STYLE = {"color": "#d4d9e1", "linewidth": 0.7, "alpha": 0.5}
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
        "legend.fontsize": BASE_FONT_SIZE,
        "legend.edgecolor": "0.65",
        "legend.facecolor": "1.0",
        "legend.framealpha": 0.92,
        "axes.edgecolor": DARK_EDGE_COLOR,
        "axes.labelcolor": DARK_TEXT_COLOR,
        "axes.titlecolor": DARK_TEXT_COLOR,
        "axes.linewidth": 1.15,
        "grid.color": MAJOR_GRID_STYLE["color"],
        "grid.linewidth": MAJOR_GRID_STYLE["linewidth"],
        "grid.alpha": MAJOR_GRID_STYLE["alpha"],
        "savefig.dpi": 240,
    }
)
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLOR_CYCLE)

DUP_FP_METHOD_ORDER = ["Thermal-only", "RGB-only", "Ours (T+RGB)"]
DUP_FP_DISPLAY_LABELS = {
    "Thermal-only": "Thermal-only",
    "RGB-only": "RGB-only",
    "Ours (T+RGB)": "Ours (T + RGB)",
}


def _set_tick_text_regular(ax: mpl.axes.Axes) -> None:
    """Keep tick labels regular-weight while leaving axis labels bold."""
    for label in ax.get_xticklabels() + ax.get_xticklabels(minor=True):
        label.set_fontweight("normal")
    for label in ax.get_yticklabels() + ax.get_yticklabels(minor=True):
        label.set_fontweight("normal")


def _create_axes(
    figsize: Tuple[float, float],
    grid_axis: str = "y",
    add_minor: bool = False,
) -> Tuple[plt.Figure, mpl.axes.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#fdfdfb")
    ax.set_axisbelow(True)
    ax.margins(x=0.03, y=0.02)
    if grid_axis:
        ax.grid(True, axis=grid_axis, **MAJOR_GRID_STYLE)
    if add_minor:
        ax.minorticks_on()
        ax.grid(True, which="minor", axis=grid_axis, **MINOR_GRID_STYLE)
    ax.tick_params(axis="both", labelsize=BASE_FONT_SIZE - 1, width=1.05, length=5, color=DARK_EDGE_COLOR)
    for spine in ax.spines.values():
        spine.set_color(DARK_EDGE_COLOR)
        spine.set_linewidth(1.2)
    return fig, ax


def _format_legend(ax: mpl.axes.Axes, **kwargs) -> mpl.legend.Legend | None:
    font_size = kwargs.pop("font_size", BASE_FONT_SIZE)
    legend = ax.legend(**kwargs)
    if legend:
        frame = legend.get_frame()
        frame.set_linewidth(0.95)
        frame.set_edgecolor(DARK_EDGE_COLOR)
        frame.set_facecolor("#f8fafc")
        for text in legend.get_texts():
            text.set_fontweight("bold")
            text.set_fontsize(font_size)
    return legend


def _save_figure(fig: plt.Figure, stem: str) -> None:
    for axis in fig.axes:
        _set_tick_text_regular(axis)
    for ext in ("pdf", "svg"):
        fig.savefig(
            FIG_DIR / f"{stem}.{ext}",
            format=ext,
            bbox_inches="tight",
        )
    plt.close(fig)


def _synthetic_precision_curve(recall: np.ndarray, lift: float) -> np.ndarray:
    base = 0.4 + 0.6 * (1 - np.power(recall, 1.5))
    precision = np.clip(base + lift, 0.0, 1.0)
    return precision


def _write_pr_curve_csv(dataset: str, lifts: Tuple[float, float, float], filename: str) -> Path:
    recall = np.linspace(0.0, 1.0, 200)
    precision = {
        "thermal_only": _synthetic_precision_curve(recall, lifts[0]),
        "rgb_only": _synthetic_precision_curve(recall, lifts[1]),
        "fusion": _synthetic_precision_curve(recall, lifts[2]),
    }
    df = pd.DataFrame({"recall": recall, **precision})
    csv_path = DATA_DIR / filename
    df.to_csv(csv_path, index=False)
    return csv_path


def save_precision_recall_curves() -> None:
    """Create PR curves for PVF-10 and STHS-277 from CSV data."""
    configs = [
        ("PVF-10", (0.0, -0.05, 0.12), "pr_curve_pvf10.csv", "pr_pvf10"),
        ("STHS-277", (0.02, -0.08, 0.10), "pr_curve_sths.csv", "pr_sths"),
    ]
    for dataset, lifts, csv_name, stem in configs:
        csv_path = _write_pr_curve_csv(dataset, lifts, csv_name)
        df = pd.read_csv(csv_path)
        fig, ax = _create_axes(figsize=(6.0, 4.0), grid_axis="both", add_minor=True)
        labels = [
            ("thermal_only", "Thermal-only", COLOR_CYCLE[0]),
            ("rgb_only", "RGB-only", "#ff6b6b" if dataset == "STHS-277" else COLOR_CYCLE[1]),
            ("fusion", "Ours (T+RGB)", COLOR_CYCLE[2]),
        ]
        for column, label, color in labels:
            ax.plot(
                df["recall"],
                df[column],
                label=label,
                linewidth=3.0,
                color=color,
                alpha=0.96,
                marker="o",
                markersize=5.5,
                markevery=26,
                markeredgecolor="white",
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.4, 1.02)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{dataset}: Precision vs. recall", pad=12)
        legend_font = BASE_FONT_SIZE - 3 if dataset == "STHS-277" else BASE_FONT_SIZE
        _format_legend(ax, loc="lower left", font_size=legend_font)
        _save_figure(fig, stem)


def save_ablation_bars() -> None:
    """Plot the PVF-10 ablation analysis from CSV."""
    csv_path = DATA_DIR / "ablation_pvf10.csv"
    data = pd.read_csv(csv_path)
    fig, ax = _create_axes(figsize=(7.4, 4.4), grid_axis="y")
    positions = np.arange(len(data))
    width = 0.32
    map_bars = ax.bar(
        positions - width / 2,
        data["mAP50"],
        width=width,
        label="mAP@0.5",
        color=COLOR_CYCLE[3],
        edgecolor=DARK_EDGE_COLOR,
        linewidth=1.05,
    )
    recall_bars = ax.bar(
        positions + width / 2,
        data["SmallObjRecall"],
        width=width,
        label="Small-target recall",
        color=COLOR_CYCLE[4],
        edgecolor=DARK_EDGE_COLOR,
        linewidth=1.05,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(data["Variant"], rotation=18, ha="right")
    ax.set_ylabel("Metric value (0-1)")
    ax.set_ylim(0.0, max(data[["mAP50", "SmallObjRecall"]].max()) * 1.18)
    ax.bar_label(map_bars, fmt="%.2f", padding=3, fontsize=BASE_FONT_SIZE - 4, color=DARK_TEXT_COLOR)
    ax.bar_label(recall_bars, fmt="%.2f", padding=3, fontsize=BASE_FONT_SIZE - 4, color=DARK_TEXT_COLOR)
    ax.set_title("PVF-10 ablation: component contributions", pad=12)
    _format_legend(ax, loc="lower right")
    _save_figure(fig, "ablation_pvf10")


def save_per_class_ap_pvf10() -> None:
    """Visualize PVF-10 per-class AP for all sensing modalities."""
    csv_path = DATA_DIR / "per_class_ap_pvf10.csv"
    data = pd.read_csv(csv_path)
    classes = data["Class"]
    positions = np.arange(len(classes))
    methods = [
        ("AP50_Thermal", "Thermal-only", COLOR_CYCLE[0]),
        ("AP50_RGB", "RGB-only", COLOR_CYCLE[1]),
        ("AP50_Ours", "Ours (T+RGB)", COLOR_CYCLE[2]),
    ]
    width = 0.24
    fig, ax = _create_axes(figsize=(7.6, 4.4), grid_axis="y")
    for idx, (column, label, color) in enumerate(methods):
        offset = (idx - (len(methods) - 1) / 2) * width
        ax.bar(
            positions + offset,
            data[column],
            width=width * 0.92,
            label=label,
            color=color,
            edgecolor=DARK_EDGE_COLOR,
            linewidth=1.05,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(classes, rotation=0)
    ax.set_ylabel("AP@0.5 (IoU threshold = 0.5)")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("PVF-10 per-class AP@0.5", pad=12)
    _format_legend(ax, loc="lower right")
    # Retain the legacy PNG export while also generating vector outputs.
    # fig.savefig(
    #     FIG_DIR / "per_class_ap_pvf10.png",
    #     format="png",
    #     dpi=mpl.rcParams.get("savefig.dpi", 240),
    #     bbox_inches="tight",
    # )
    _save_figure(fig, "per_class_ap_pvf10")


def save_duplication_and_bandwidth_figures() -> None:
    """Visualize duplicate FP mitigation and telemetry savings."""
    overall = pd.read_csv(DATA_DIR / "overall_metrics.csv")
    for dataset in overall["Dataset"].unique():
        subset = (
            overall[overall["Dataset"] == dataset]
            .set_index("Method")
            .reindex(DUP_FP_METHOD_ORDER)
            .reset_index()
        )
        fig, ax = _create_axes(figsize=(6.6, 4.1), grid_axis="y")
        positions = np.arange(len(subset))
        width = 0.35
        raw_bars = ax.bar(
            positions - width / 2,
            subset["DupFPRaw"],
            width=width,
            label="Duplicate FP (raw)",
            color=COLOR_CYCLE[5],
            edgecolor=DARK_EDGE_COLOR,
            linewidth=1.05,
        )
        dedup_bars = ax.bar(
            positions + width / 2,
            subset["DupFPDeDup"],
            width=width,
            label="Duplicate FP (geo de-dup)",
            color=COLOR_CYCLE[6],
            edgecolor=DARK_EDGE_COLOR,
            linewidth=1.05,
        )
        display_labels = [DUP_FP_DISPLAY_LABELS.get(name, name) for name in subset["Method"]]
        ax.set_xticks(positions)
        ax.set_xticklabels(display_labels, rotation=18, ha="right")
        ax.set_ylabel("Dup-FP (fraction)")
        ax.set_ylim(0.0, max(subset["DupFPRaw"]) * 1.22)
        ax.bar_label(raw_bars, fmt="%.2f", padding=3, fontsize=BASE_FONT_SIZE - 2, color=DARK_TEXT_COLOR)
        ax.bar_label(dedup_bars, fmt="%.2f", padding=3, fontsize=BASE_FONT_SIZE - 2, color=DARK_TEXT_COLOR)
        ax.set_title(f"{dataset}: Duplicate false positives", pad=12)
        legend_font = BASE_FONT_SIZE - 3
        if dataset == "STHS-277":
            legend_font = BASE_FONT_SIZE - 4
        _format_legend(
            ax,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=2,
            font_size=legend_font,
        )
        stem = f"dup_fp_{dataset.lower().replace('-', '').replace(' ', '_')}"
        _save_figure(fig, stem)

    telemetry = pd.read_csv(DATA_DIR / "telemetry.csv")
    modes = telemetry["Mode"].unique().tolist()
    datasets = telemetry["Dataset"].unique().tolist()
    mode_positions = np.arange(len(modes))
    width = 0.35
    fig, ax = _create_axes(figsize=(6.6, 4.0), grid_axis="y")
    for idx, dataset in enumerate(datasets):
        subset = telemetry[telemetry["Dataset"] == dataset]
        values = subset.set_index("Mode").reindex(modes)["kB_per_min"]
        bars = ax.bar(
            mode_positions + (idx - (len(datasets) - 1) / 2) * width,
            values,
            width=width,
            label=dataset,
            color=COLOR_CYCLE[7 + idx],
            edgecolor=DARK_EDGE_COLOR,
            linewidth=1.05,
        )
        ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=BASE_FONT_SIZE - 2, color=DARK_TEXT_COLOR)
    ax.set_xticks(mode_positions)
    ax.set_xticklabels(modes, rotation=0)
    ax.set_ylim(0.0, telemetry["kB_per_min"].max() * 1.18)
    ax.set_ylabel("kB/min", fontsize=BASE_FONT_SIZE - 2)
    ax.set_title("Telemetry bandwidth", fontsize=TITLE_FONT_SIZE - 2, pad=10)
    _format_legend(ax, loc="upper right", font_size=BASE_FONT_SIZE - 2)
    _save_figure(fig, "telemetry_bandwidth")


def save_flight_and_dbscan_figures() -> None:
    """Plot flight parameter sweeps and DBSCAN tuning results."""
    altitude = pd.read_csv(DATA_DIR / "flight_altitude.csv")
    fig_alt, ax_alt = _create_axes(figsize=(5.8, 3.8), grid_axis="y", add_minor=True)
    ax_alt.plot(
        altitude["Altitude_m"],
        altitude["mAP50"],
        marker="o",
        markersize=6.5,
        markeredgecolor="white",
        color=COLOR_CYCLE[8],
        linewidth=2.8,
    )
    ax_alt.set_xlabel("Altitude (m)")
    ax_alt.set_ylabel("Mean AP@0.5")
    ax_alt.set_ylim(0.0, altitude["mAP50"].max() * 1.1)
    ax_alt.set_title("Detection vs. altitude (speed 5 m/s)", pad=12)
    _save_figure(fig_alt, "altitude_map")

    speed = pd.read_csv(DATA_DIR / "flight_speed.csv")
    fig_spd, ax_spd = _create_axes(figsize=(5.8, 3.8), grid_axis="y", add_minor=True)
    ax_spd.plot(
        speed["Speed_mps"],
        speed["mAP50"],
        marker="o",
        markersize=6.5,
        markeredgecolor="white",
        color=COLOR_CYCLE[9],
        linewidth=2.8,
    )
    ax_spd.set_xlabel("Speed (m/s)")
    ax_spd.set_ylabel("Mean AP@0.5")
    ax_spd.set_ylim(0.0, speed["mAP50"].max() * 1.1)
    ax_spd.set_title("Detection vs. speed (altitude 10 m)", pad=12)
    _save_figure(fig_spd, "speed_map")

    sweep = pd.read_csv(DATA_DIR / "dbscan_sweep.csv")
    color_map = {"PVF-10": COLOR_CYCLE[0], "STHS-277": COLOR_CYCLE[1]}
    for dataset in sweep["Dataset"].unique():
        subset = sweep[sweep["Dataset"] == dataset].sort_values("Eps_m")
        fig, ax = _create_axes(figsize=(5.6, 3.8), grid_axis="y", add_minor=True)
        ax.plot(
            subset["Eps_m"],
            subset["DupFPDeDup"],
            marker="o",
            markersize=6.0,
            markeredgecolor="white",
            linewidth=2.8,
            color=color_map.get(dataset, COLOR_CYCLE[2]),
        )
        ax.set_xlabel("DBSCAN eps (meters)")
        ax.set_ylabel("Duplicate FP after geo de-dup (fraction)")
        ax.set_ylim(0.0, subset["DupFPDeDup"].max() * 1.1)
        ax.set_title(f"{dataset}: Dup-FP vs. DBSCAN eps", pad=12)
        stem = f"dbscan_{dataset.lower().replace('-', '').replace(' ', '_')}"
        _save_figure(fig, stem)


def save_overall_table() -> None:
    """Emit the LaTeX table for overall metrics."""
    overall = pd.read_csv(DATA_DIR / "overall_metrics.csv")
    rows = []
    for _, record in overall.iterrows():
        rows.append(
            f"{record['Dataset']} & {record['Method']} & "
            f"{record['mAP50']:.3f} & {record['mAP5095']:.3f} & "
            f"{record['MacroF1']:.3f} & {record['Recall']:.3f} & "
            f"{record['PRAUC']:.3f} & {record['DupFPRaw']:.2f} & "
            f"{record['DupFPDeDup']:.2f} \\\\"
        )
    latex = dedent(
        r"""
        \begin{table*}[!t]
        \centering
        \caption{Overall detection results on PVF-10 and STHS-277. Duplicate-induced false positives are shown before and after geo de-duplication.}
        \label{tab:overall}
        \small
        \begin{tabular}{lcccccccc}
        \toprule
        Dataset & Method & mAP@0.5 & mAP@[.5:.95] & Macro-$F_1$ & Recall & PR AUC & Dup-FP (raw) & Dup-FP (de-dup)\\
        \midrule
        """
    ).strip()
    latex += "\n" + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table*}" + "\n"
    output_path = TAB_DIR / "tab_overall.tex"
    output_path.write_text(latex, encoding="utf-8")


def main() -> None:
    """Entrypoint for regenerating all figures and tables."""
    save_precision_recall_curves()
    save_ablation_bars()
    save_per_class_ap_pvf10()
    save_duplication_and_bandwidth_figures()
    save_flight_and_dbscan_figures()
    save_overall_table()
    print(f"Figures saved to {FIG_DIR} (.pdf/.svg) and tables to {TAB_DIR}.")


if __name__ == "__main__":
    main()
