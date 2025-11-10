#!/usr/bin/env python3
"""Train a calibrated OvR classifier and render publication-ready curve plots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize

RANDOM_STATE = 42
N_SAMPLES = 6_000
N_FEATURES = 25
N_CLASSES = 5
CLASS_WEIGHTS = [0.35, 0.25, 0.18, 0.13, 0.09]

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "img"
RESULTS_DIR = ROOT / "results"

IMG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
        "legend.edgecolor": "0.65",
        "legend.facecolor": "1.0",
        "legend.framealpha": 0.9,
        "axes.edgecolor": "0.35",
        "axes.linewidth": 1.0,
        "grid.color": "0.78",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.55,
        "savefig.dpi": 220,
        "svg.fonttype": "none",
    }
)

COLOR_CYCLE = mpl.colormaps["tab10"].colors
MAJOR_GRID_STYLE = {"color": "0.78", "linewidth": 0.85, "alpha": 0.6}
MINOR_GRID_STYLE = {"color": "0.9", "linewidth": 0.6, "alpha": 0.4}


@dataclass(frozen=True)
class MacroCurves:
    fpr: np.ndarray
    tpr: np.ndarray
    recall: np.ndarray
    precision: np.ndarray


def _create_axes(
    figsize: tuple[float, float],
    grid_axis: str = "both",
    add_minor: bool = False,
) -> tuple[plt.Figure, mpl.axes.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(True, axis=grid_axis, **MAJOR_GRID_STYLE)
    if add_minor:
        ax.minorticks_on()
        ax.grid(True, which="minor", axis=grid_axis, **MINOR_GRID_STYLE)
    for spine in ax.spines.values():
        spine.set_color("0.35")
        spine.set_linewidth(1.0)
    return fig, ax


def _format_legend(ax: mpl.axes.Axes, **kwargs) -> mpl.legend.Legend | None:
    legend = ax.legend(**kwargs)
    if legend:
        frame = legend.get_frame()
        frame.set_linewidth(0.9)
        frame.set_edgecolor("0.65")
    return legend


def _save_svg(fig: plt.Figure, stem: str) -> None:
    out_path = IMG_DIR / f"{stem}.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _prepare_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=12,
        n_redundant=5,
        n_classes=N_CLASSES,
        n_clusters_per_class=1,
        weights=CLASS_WEIGHTS,
        flip_y=0.02,
        class_sep=1.15,
        random_state=RANDOM_STATE,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def _fit_classifier(X_train: np.ndarray, y_train: np.ndarray) -> OneVsRestClassifier:
    base_lr = LogisticRegression(
        solver="lbfgs",
        max_iter=2_000,
        C=1.0,
        random_state=RANDOM_STATE,
    )
    calibrated = CalibratedClassifierCV(estimator=base_lr, method="sigmoid", cv=5)
    clf = OneVsRestClassifier(calibrated)
    clf.fit(X_train, y_train)
    return clf


def _interpolate_curve(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    unique_x, idx = np.unique(x, return_index=True)
    unique_y = y[idx]
    return np.interp(grid, unique_x, unique_y)


def _compute_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    classes: Iterable[int],
) -> tuple[MacroCurves, float, Dict[str, object]]:
    y_true_bin = label_binarize(y_true, classes=classes)
    fpr_grid = np.linspace(0.0, 1.0, 1_001)
    recall_grid = np.linspace(0.0, 1.0, 1_001)

    tpr_interp_all = []
    prec_interp_all = []
    per_class_metrics: List[Dict[str, float]] = []

    for idx, class_id in enumerate(classes):
        fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, idx], y_score[:, idx])
        precision_i, recall_i, _ = precision_recall_curve(
            y_true_bin[:, idx],
            y_score[:, idx],
        )
        roc_auc_i = float(np.trapezoid(tpr_i, fpr_i))
        ap_i = float(average_precision_score(y_true_bin[:, idx], y_score[:, idx]))

        tpr_interp_all.append(_interpolate_curve(fpr_i, tpr_i, fpr_grid))
        prec_interp_all.append(_interpolate_curve(recall_i, precision_i, recall_grid))

        per_class_metrics.append(
            {
                "class": f"Class {class_id}",
                "roc_auc": roc_auc_i,
                "avg_precision": ap_i,
                "prevalence": float(y_true_bin[:, idx].mean()),
            }
        )

    tpr_macro = np.mean(np.vstack(tpr_interp_all), axis=0)
    precision_macro = np.mean(np.vstack(prec_interp_all), axis=0)

    roc_auc_macro = float(np.trapezoid(tpr_macro, fpr_grid))
    ap_macro = float(np.trapezoid(precision_macro, recall_grid))
    ap_macro_skl = float(average_precision_score(y_true_bin, y_score, average="macro"))
    ap_micro = float(average_precision_score(y_true_bin, y_score, average="micro"))

    metrics = {
        "random_state": RANDOM_STATE,
        "macro": {
            "roc_auc": roc_auc_macro,
            "ap_trapz": ap_macro,
            "ap_sklearn": ap_macro_skl,
        },
        "micro": {"ap": ap_micro},
        "per_class": per_class_metrics,
    }

    macro_curves = MacroCurves(
        fpr=fpr_grid,
        tpr=tpr_macro,
        recall=recall_grid,
        precision=precision_macro,
    )
    macro_prev = float(np.mean([m["prevalence"] for m in per_class_metrics]))

    return macro_curves, macro_prev, metrics


def _save_curve_data(curves: MacroCurves) -> tuple[Path, Path]:
    roc_path = RESULTS_DIR / "roc_macro_curve.csv"
    pr_path = RESULTS_DIR / "pr_macro_curve.csv"
    pd.DataFrame({"fpr": curves.fpr, "tpr_macro": curves.tpr}).to_csv(
        roc_path,
        index=False,
    )
    pd.DataFrame({"recall": curves.recall, "precision_macro": curves.precision}).to_csv(
        pr_path,
        index=False,
    )
    return roc_path, pr_path


def _plot_macro_roc(curve_path: Path, auc_value: float) -> None:
    curve = pd.read_csv(curve_path)
    fig, ax = _create_axes(figsize=(5.8, 4.4), add_minor=True)
    ax.plot(
        curve["fpr"],
        curve["tpr_macro"],
        color=COLOR_CYCLE[0],
        linewidth=2.8,
        label=f"Macro ROC (AUC = {auc_value:.3f})",
    )
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.2,
        color="0.35",
        label="Chance",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Macro ROC (OvR, calibrated)")
    _format_legend(ax, loc="lower right")
    _save_svg(fig, "roc_curve_macro")


def _plot_macro_pr(
    curve_path: Path,
    ap_value: float,
    baseline: float,
) -> None:
    curve = pd.read_csv(curve_path)
    fig, ax = _create_axes(figsize=(5.8, 4.4), add_minor=True)
    ax.plot(
        curve["recall"],
        curve["precision_macro"],
        color=COLOR_CYCLE[1],
        linewidth=2.8,
        label=f"Macro PR (AP = {ap_value:.3f})",
    )
    ax.hlines(
        baseline,
        xmin=0.0,
        xmax=1.0,
        linestyle="--",
        linewidth=1.2,
        color="0.35",
        label=f"Baseline (π = {baseline:.2f})",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.5, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Macro precision–recall (OvR, calibrated)")
    _format_legend(ax, loc="lower left")
    _save_svg(fig, "pr_curve_macro")


def _plot_confusion_matrix(cm_path: Path, class_names: list[str]) -> None:
    df_cm = pd.read_csv(cm_path, index_col=0)
    fig, ax = _create_axes(figsize=(5.2, 4.8), grid_axis=None)
    heatmap = ax.imshow(df_cm.values, cmap="Blues", interpolation="nearest")
    for (i, j), value in np.ndenumerate(df_cm.values):
        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            color="white" if value > df_cm.values.max() * 0.6 else "black",
        )
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Normalized confusion matrix")
    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.set_ylabel("Probability", rotation=270, labelpad=18, weight="bold")
    _save_svg(fig, "confusion_matrix_macro")


def main() -> None:
    X_train, X_test, y_train, y_test = _prepare_data()
    clf = _fit_classifier(X_train, y_train)
    proba = clf.predict_proba(X_test)
    classes = np.unique(y_train)
    macro_curves, macro_prev, metrics = _compute_curves(
        y_test,
        proba,
        classes,
    )

    roc_path, pr_path = _save_curve_data(macro_curves)
    metrics_path = RESULTS_DIR / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as json_file:
        json.dump(metrics, json_file, indent=2)

    y_pred = np.argmax(proba, axis=1)
    cm = confusion_matrix(y_test, y_pred, labels=classes).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    class_names = [f"Class {c}" for c in classes]
    df_cm = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
    cm_path = RESULTS_DIR / "confusion_matrix.csv"
    df_cm.to_csv(cm_path)

    _plot_macro_roc(roc_path, metrics["macro"]["roc_auc"])
    _plot_macro_pr(pr_path, metrics["macro"]["ap_trapz"], macro_prev)
    _plot_confusion_matrix(cm_path, class_names)

    print(f"Saved metrics to {metrics_path}")
    print(f"Figures written to {IMG_DIR}")


if __name__ == "__main__":
    main()
