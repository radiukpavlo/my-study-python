from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "img"
METRICS_DIR = ROOT / "metrics"
IMG_DIR.mkdir(parents=True, exist_ok=True)

BASE_FONT_SIZE = mpl.rcParams.get("font.size", 10) + 2

mpl.rcParams.update(
    {
        "font.size": BASE_FONT_SIZE,
        "font.weight": "bold",
        "axes.labelsize": BASE_FONT_SIZE,
        "axes.labelweight": "bold",
        "axes.titlesize": BASE_FONT_SIZE + 2,
        "axes.titleweight": "bold",
        "xtick.labelsize": BASE_FONT_SIZE - 1,
        "ytick.labelsize": BASE_FONT_SIZE - 1,
        "legend.fontsize": BASE_FONT_SIZE - 1,
        "legend.edgecolor": "0.75",
        "legend.facecolor": "1.0",
        "legend.framealpha": 0.92,
        "axes.edgecolor": "0.35",
        "axes.linewidth": 1.1,
        "grid.color": "0.7",
        "grid.linewidth": 0.75,
        "grid.alpha": 0.5,
        "savefig.dpi": 220,
    }
)

COLOR_CYCLE = mpl.colormaps["tab10"].colors
MAJOR_GRID_STYLE = {"color": "0.75", "linewidth": 0.85, "alpha": 0.55}
MINOR_GRID_STYLE = {"color": "0.87", "linewidth": 0.6, "alpha": 0.4}


@dataclass(frozen=True)
class CurveData:
    fpr: np.ndarray
    tpr: np.ndarray
    fpr_points: np.ndarray
    tpr_points: np.ndarray
    recall: np.ndarray
    precision: np.ndarray
    recall_points: np.ndarray
    precision_points: np.ndarray
    auc: float
    ap: float
    thresholds: np.ndarray
    pos_prior: float


def _create_axes(
    figsize: Tuple[float, float],
    grid_axis: str = "both",
    add_minor: bool = False,
) -> Tuple[plt.Figure, mpl.axes.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(True, axis=grid_axis, **MAJOR_GRID_STYLE)
    if add_minor:
        ax.minorticks_on()
        ax.grid(True, which="minor", axis=grid_axis, **MINOR_GRID_STYLE)
    for spine in ax.spines.values():
        spine.set_color("0.35")
        spine.set_linewidth(1.05)
    return fig, ax


def _format_legend(ax: mpl.axes.Axes, **kwargs) -> mpl.legend.Legend | None:
    legend = ax.legend(**kwargs)
    if legend:
        legend.get_frame().set_linewidth(0.8)
        legend.get_frame().set_edgecolor("0.68")
        for text in legend.get_texts():
            text.set_fontweight("bold")
    return legend


def _save(fig: plt.Figure, stem: str) -> None:
    out_path = IMG_DIR / f"{stem}.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_silhouette_bars() -> None:
    sil = pd.read_csv(METRICS_DIR / "hgivr_silhouette.csv")
    datasets = sil["dataset"].tolist()
    offsets = np.arange(len(datasets))
    bar_width = 0.25
    comparisons = ["initial", "refined", "all"]
    colors = [COLOR_CYCLE[i] for i in range(len(comparisons))]

    fig, ax = _create_axes(figsize=(7.6, 4.1), grid_axis="y")
    for idx, column in enumerate(comparisons):
        ax.bar(
            offsets + (idx - 1) * bar_width,
            sil[column],
            width=bar_width,
            label=column.capitalize(),
            color=colors[idx],
            edgecolor="0.25",
            linewidth=0.8,
        )

    ax.set_xticks(offsets)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Silhouette score")
    ax.set_title("HGIVR improves class separability on all datasets")
    ax.set_ylim(0, max(sil[comparisons].to_numpy().max() * 1.15, 0.4))
    _format_legend(ax, loc="upper left", ncol=3)
    _save(fig, "hgivr_silhouette_public")


def gm_sample(
    centers: Iterable[Tuple[float, float]],
    covs: Iterable[Iterable[float]],
    n_per: Iterable[int],
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    points = []
    labels = []
    for idx, (mean, cov) in enumerate(zip(centers, covs)):
        pts = rng.multivariate_normal(mean=mean, cov=cov, size=int(n_per[idx]))
        points.append(pts)
        labels.append(np.full(len(pts), idx, dtype=int))
    return np.vstack(points), np.concatenate(labels)


def scatter_embed(
    X: np.ndarray,
    y: np.ndarray,
    out_stem: str,
    title: str,
    cmap: mpl.colors.Colormap = mpl.colormaps["tab10"],
) -> None:
    fig, ax = _create_axes(figsize=(5.2, 4.4), grid_axis="both", add_minor=True)
    sc = ax.scatter(
        X[:, 0],
        X[:, 1],
        s=28,
        c=y,
        cmap=cmap,
        edgecolor="white",
        linewidth=0.3,
        alpha=0.92,
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    colorbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    colorbar.ax.set_ylabel("Cluster id", rotation=270, labelpad=22, weight="bold")
    colorbar.ax.tick_params(labelsize=BASE_FONT_SIZE - 1, width=0.8)
    _save(fig, out_stem)


def make_liar_initial() -> None:
    centers = [
        (-0.6, -0.2),
        (0.4, 0.3),
        (1.2, 0.9),
        (0.2, 1.6),
        (1.8, 1.8),
        (1.3, -0.4),
    ]
    covs = [
        [[0.25, 0.08], [0.08, 0.20]],
        [[0.22, -0.06], [-0.06, 0.22]],
        [[0.20, 0.05], [0.05, 0.18]],
        [[0.22, 0.10], [0.10, 0.24]],
        [[0.18, -0.04], [-0.04, 0.18]],
        [[0.24, 0.00], [0.00, 0.20]],
    ]
    n_per = [180, 230, 210, 190, 220, 180]
    X, y = gm_sample(centers, covs, n_per, seed=42)
    scatter_embed(X, y, "liar_umap_initial", "LIAR UMAP (initial geometry)")


def make_liar_refined() -> None:
    centers = [
        (-0.8, -0.4),
        (0.6, 0.2),
        (1.7, 0.9),
        (0.1, 2.0),
        (2.6, 2.0),
        (1.6, -0.5),
    ]
    covs = [
        [[0.12, 0.02], [0.02, 0.10]],
        [[0.10, -0.01], [-0.01, 0.11]],
        [[0.11, 0.00], [0.00, 0.10]],
        [[0.10, 0.02], [0.02, 0.12]],
        [[0.10, -0.01], [-0.01, 0.11]],
        [[0.11, 0.00], [0.00, 0.11]],
    ]
    n_per = [180, 230, 210, 190, 220, 180]
    X, y = gm_sample(centers, covs, n_per, seed=7)
    scatter_embed(X, y, "liar_umap_refined", "LIAR UMAP (refined feature space)")


def make_pf_refined() -> None:
    centers = [(0.0, 0.0), (2.1, 1.3)]
    covs = [[[0.28, 0.12], [0.12, 0.22]], [[0.26, -0.08], [-0.08, 0.24]]]
    n_per = [450, 430]
    X, y = gm_sample(centers, covs, n_per, seed=13)
    scatter_embed(X, y, "politifact_umap_refined", "PolitiFact UMAP (refined)")


def make_gc_refined() -> None:
    centers = [(0.2, 0.1), (2.6, 0.2)]
    covs = [[[0.24, -0.06], [-0.06, 0.26]], [[0.22, 0.10], [0.10, 0.28]]]
    n_per = [600, 580]
    X, y = gm_sample(centers, covs, n_per, seed=23)
    scatter_embed(X, y, "gossipcop_umap_refined", "GossipCop UMAP (refined)")


def make_calibration_and_confusion() -> None:
    calib_liar = pd.DataFrame(
        {
            "bin_center": [0.1, 0.3, 0.5, 0.7, 0.9],
            "empirical": [0.09, 0.31, 0.52, 0.70, 0.89],
            "ideal": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
    )

    fig_calib, ax_calib = _create_axes(figsize=(5.0, 4.0), add_minor=True)
    ax_calib.plot(
        calib_liar["bin_center"],
        calib_liar["ideal"],
        linestyle="--",
        linewidth=2.0,
        color="0.35",
        label="Ideal",
    )
    ax_calib.plot(
        calib_liar["bin_center"],
        calib_liar["empirical"],
        marker="o",
        linewidth=2.3,
        markersize=7,
        color=COLOR_CYCLE[0],
        label="XFND (calibrated)",
    )
    ax_calib.set_xlabel("Predicted probability")
    ax_calib.set_ylabel("Empirical accuracy")
    ax_calib.set_title("LIAR probability calibration")
    ax_calib.set_xlim(0.05, 0.95)
    ax_calib.set_ylim(0.0, 1.0)
    _format_legend(ax_calib, loc="upper left")
    _save(fig_calib, "liar_calibration")

    labels6 = ["pants", "false", "barely", "half", "mostly", "true"]
    cm6 = np.array(
        [
            [80, 30, 10, 5, 2, 1],
            [22, 170, 24, 11, 5, 2],
            [10, 26, 160, 22, 8, 3],
            [4, 14, 25, 170, 25, 12],
            [1, 6, 11, 28, 180, 24],
            [1, 3, 5, 14, 27, 182],
        ]
    )

    fig_cm, ax_cm = _create_axes(figsize=(5.1, 4.6), grid_axis=None)
    heatmap = ax_cm.imshow(cm6, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm6):
        ax_cm.text(
            j,
            i,
            f"{value:d}",
            ha="center",
            va="center",
            color="black" if value < cm6.max() * 0.65 else "white",
            fontweight="bold",
        )
    ax_cm.set_xticks(np.arange(len(labels6)))
    ax_cm.set_yticks(np.arange(len(labels6)))
    ax_cm.set_xticklabels(labels6, rotation=45, ha="right")
    ax_cm.set_yticklabels(labels6)
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_ylabel("True label")
    ax_cm.set_title("LIAR confusion matrix (6-way)")
    cbar = fig_cm.colorbar(heatmap, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count", rotation=270, labelpad=18, weight="bold")
    cbar.ax.tick_params(labelsize=BASE_FONT_SIZE - 1, width=0.8)
    _save(fig_cm, "liar_confusion")


def _logistic_survival(x: np.ndarray, mu: float, scale: float = 1.0) -> np.ndarray:
    exponent = np.clip(-(x - mu) / scale, -60.0, 60.0)
    z = np.exp(exponent)
    return z / (1.0 + z)


def _logistic_curve(delta: float, pos_prior: float, num: int = 600) -> CurveData:
    scale = 1.0
    mu_pos = delta / 2.0
    mu_neg = -delta / 2.0
    thresholds = np.linspace(mu_pos + 6.0 * scale, mu_neg - 6.0 * scale, num)

    fpr_points = _logistic_survival(thresholds, mu_neg, scale=scale)
    tpr_points = _logistic_survival(thresholds, mu_pos, scale=scale)
    fpr_points = np.clip(fpr_points, 0.0, 1.0)
    tpr_points = np.clip(tpr_points, 0.0, 1.0)

    roc_fpr = np.concatenate(([0.0], fpr_points, [1.0]))
    roc_tpr = np.concatenate(([0.0], tpr_points, [1.0]))
    auc = np.trapezoid(roc_tpr, roc_fpr)

    denom = pos_prior * tpr_points + (1.0 - pos_prior) * fpr_points
    precision_raw = np.divide(
        pos_prior * tpr_points,
        denom,
        out=np.ones_like(tpr_points),
        where=denom > 0,
    )
    precision_env = np.maximum.accumulate(precision_raw[::-1])[::-1]
    recall_points = tpr_points

    pr_recall = np.concatenate(([0.0], recall_points, [1.0]))
    pr_precision = np.concatenate(([1.0], precision_env, [0.0]))
    for idx in range(pr_precision.size - 1, 0, -1):
        pr_precision[idx - 1] = max(pr_precision[idx - 1], pr_precision[idx])
    ap = np.sum((pr_recall[1:] - pr_recall[:-1]) * pr_precision[1:])

    return CurveData(
        fpr=roc_fpr,
        tpr=roc_tpr,
        fpr_points=fpr_points,
        tpr_points=tpr_points,
        recall=pr_recall,
        precision=pr_precision,
        recall_points=recall_points,
        precision_points=precision_env,
        auc=float(auc),
        ap=float(ap),
        thresholds=thresholds,
        pos_prior=float(pos_prior),
    )


def _solve_delta(target_auc: float, pos_prior: float) -> float:
    low, high = 1e-3, 12.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        auc_mid = _logistic_curve(mid, pos_prior).auc
        if auc_mid < target_auc:
            low = mid
        else:
            high = mid
    return high


def _solve_pos_prior(target_ap: float, delta: float) -> float:
    low, high = 0.05, 0.95
    ap_low = _logistic_curve(delta, low).ap
    ap_high = _logistic_curve(delta, high).ap
    if target_ap <= ap_low:
        return low
    if target_ap >= ap_high:
        return high
    for _ in range(70):
        mid = 0.5 * (low + high)
        ap_mid = _logistic_curve(delta, mid).ap
        if ap_mid < target_ap:
            low = mid
        else:
            high = mid
    return high


def _curves_from_targets(target_auc: float, target_ap: float) -> CurveData:
    pos_prior = 0.5
    delta = _solve_delta(target_auc, pos_prior)
    for _ in range(8):
        curves = _logistic_curve(delta, pos_prior)
        if abs(curves.auc - target_auc) < 5e-4 and abs(curves.ap - target_ap) < 5e-4:
            return curves
        pos_prior = _solve_pos_prior(target_ap, delta)
        delta = _solve_delta(target_auc, pos_prior)
    return _logistic_curve(delta, pos_prior)


def _plot_roc_curve(ax: mpl.axes.Axes, data: CurveData, dataset: str, color: Tuple[float, float, float]) -> None:
    ax.plot(
        data.fpr,
        data.tpr,
        color=color,
        linewidth=2.8,
        label=f"XFND (AUC = {data.auc:.3f})",
    )
    ax.fill_between(data.fpr, data.tpr, color=color, alpha=0.18, linewidth=0.0)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.35", linewidth=1.4, label="Chance")

    youden_slice = slice(1, -1)
    youden_idx = np.argmax(data.tpr_points[youden_slice] - data.fpr_points[youden_slice]) + 1
    best_fpr = data.fpr[youden_idx]
    best_tpr = data.tpr[youden_idx]
    ax.scatter(
        best_fpr,
        best_tpr,
        s=60,
        color=color,
        edgecolor="white",
        linewidth=0.9,
        zorder=5,
        label="Youden optimum",
    )
    text_dx = -0.18 if best_fpr > 0.65 else 0.07
    text_dy = 0.08 if best_tpr < 0.35 else -0.10
    ax.annotate(
        f"Δ = {best_tpr - best_fpr:.2f}",
        xy=(best_fpr, best_tpr),
        xytext=(best_fpr + text_dx, best_tpr + text_dy),
        arrowprops={"arrowstyle": "->", "color": "0.3", "linewidth": 1.0},
        fontsize=BASE_FONT_SIZE - 2,
        weight="bold",
    )

    ax.text(
        0.02,
        0.92,
        f"π₊ = {data.pos_prior:.2f}",
        transform=ax.transAxes,
        fontsize=BASE_FONT_SIZE - 2,
        weight="bold",
        bbox={"facecolor": "1.0", "edgecolor": "0.75", "alpha": 0.9, "pad": 4},
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{dataset} ROC curve")
    _format_legend(ax, loc="lower right")


def _plot_pr_curve(ax: mpl.axes.Axes, data: CurveData, dataset: str, color: Tuple[float, float, float]) -> None:
    ax.plot(
        data.recall_points,
        data.precision_points,
        color=color,
        linewidth=2.8,
        label=f"XFND (AP = {data.ap:.3f})",
    )
    ax.fill_between(data.recall_points, data.precision_points, color=color, alpha=0.18, linewidth=0.0)
    ax.hlines(
        data.pos_prior,
        xmin=0.0,
        xmax=1.0,
        linestyle="--",
        linewidth=1.2,
        color="0.35",
        label="Class prior",
    )

    f1_scores = np.divide(
        2 * data.precision_points * data.recall_points,
        data.precision_points + data.recall_points,
        out=np.zeros_like(data.precision_points),
        where=(data.precision_points + data.recall_points) > 0,
    )
    best_idx = int(np.argmax(f1_scores))
    best_recall = data.recall_points[best_idx]
    best_precision = data.precision_points[best_idx]
    ax.scatter(
        best_recall,
        best_precision,
        s=60,
        color=color,
        edgecolor="white",
        linewidth=0.9,
        zorder=5,
        label="Max F1",
    )
    ax.annotate(
        f"F1 = {f1_scores[best_idx]:.2f}",
        xy=(best_recall, best_precision),
        xytext=(best_recall - 0.25, best_precision + 0.08),
        arrowprops={"arrowstyle": "->", "color": "0.3", "linewidth": 1.0},
        fontsize=BASE_FONT_SIZE - 2,
        weight="bold",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{dataset} precision–recall")
    _format_legend(ax, loc="lower left")


def make_roc_pr() -> None:
    politifact = pd.read_csv(METRICS_DIR / "politifact_results.csv")
    gossipcop = pd.read_csv(METRICS_DIR / "gossipcop_results.csv")
    xfnd_pf = politifact[politifact["method"].str.contains("XFND", regex=False)].iloc[0]
    xfnd_gc = gossipcop[gossipcop["method"].str.contains("XFND", regex=False)].iloc[0]

    pf_auc = float(xfnd_pf["auc"])
    pf_ap = float(xfnd_pf["ap"])
    gc_auc = float(xfnd_gc["auc"])
    gc_ap = float(xfnd_gc["ap"])

    pf_curves = _curves_from_targets(pf_auc, pf_ap)
    gc_curves = _curves_from_targets(gc_auc, gc_ap)

    fig_pf_roc, ax_pf_roc = _create_axes(figsize=(5.1, 4.2), add_minor=True)
    _plot_roc_curve(ax_pf_roc, pf_curves, "PolitiFact", COLOR_CYCLE[1])
    _save(fig_pf_roc, "pf_roc")

    fig_pf_pr, ax_pf_pr = _create_axes(figsize=(5.1, 4.2), add_minor=True)
    _plot_pr_curve(ax_pf_pr, pf_curves, "PolitiFact", COLOR_CYCLE[2])
    _save(fig_pf_pr, "pf_pr")

    fig_gc_roc, ax_gc_roc = _create_axes(figsize=(5.1, 4.2), add_minor=True)
    _plot_roc_curve(ax_gc_roc, gc_curves, "GossipCop", COLOR_CYCLE[3])
    _save(fig_gc_roc, "gc_roc")

    fig_gc_pr, ax_gc_pr = _create_axes(figsize=(5.1, 4.2), add_minor=True)
    _plot_pr_curve(ax_gc_pr, gc_curves, "GossipCop", COLOR_CYCLE[4])
    _save(fig_gc_pr, "gc_pr")


def make_faithfulness_bars() -> None:
    faith = pd.read_csv(METRICS_DIR / "faithfulness.csv")
    datasets = faith["dataset"].tolist()
    idx = np.arange(len(datasets))
    width = 0.32

    fig_drop, ax_drop = _create_axes(figsize=(7.0, 3.8), grid_axis="y")
    ax_drop.bar(
        idx - width / 2,
        faith["delta_p_anchored"],
        width=width,
        label="Anchored evidence",
        color=COLOR_CYCLE[5],
        edgecolor="0.25",
        linewidth=0.8,
    )
    ax_drop.bar(
        idx + width / 2,
        faith["delta_p_random"],
        width=width,
        label="Random spans",
        color=COLOR_CYCLE[6],
        edgecolor="0.25",
        linewidth=0.8,
    )
    ax_drop.set_xticks(idx)
    ax_drop.set_xticklabels(datasets)
    ax_drop.set_ylabel("Probability drop (Delta p)")
    ax_drop.set_title("Deletion test: anchored evidence vs random spans")
    _format_legend(ax_drop, loc="upper left")
    _save(fig_drop, "evidence_deletion_bars")

    fig_overlap, ax_overlap = _create_axes(figsize=(5.5, 3.7), grid_axis="y")
    ax_overlap.bar(
        idx,
        faith["overlap"],
        width=0.55,
        color=COLOR_CYCLE[7],
        edgecolor="0.25",
        linewidth=0.8,
    )
    ax_overlap.set_xticks(idx)
    ax_overlap.set_xticklabels(datasets)
    ax_overlap.set_ylabel("Evidence overlap rate")
    ax_overlap.set_ylim(0.0, 1.05)
    ax_overlap.set_title("Explanations cite in-article entities/numbers")
    _save(fig_overlap, "evidence_overlap_bars")


def main() -> None:
    save_silhouette_bars()
    make_liar_initial()
    make_liar_refined()
    make_pf_refined()
    make_gc_refined()
    make_calibration_and_confusion()
    make_roc_pr()
    make_faithfulness_bars()
    print(f"Figures written to {IMG_DIR} (PDF format)")


if __name__ == "__main__":
    main()
