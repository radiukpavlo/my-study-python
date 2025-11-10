"""Generate trustworthy macro ROC/PR curves with publication-grade styling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize

ROOT_DIR = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT_DIR / "img"
RESULTS_DIR = ROOT_DIR / "results"

RANDOM_SEED = 42
BASE_FONT_SIZE = mpl.rcParams.get("font.size", 10) + 2
COLOR_CYCLE = tuple(mpl.colormaps["tab10"].colors)

STYLE_PARAMS = {
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
    "grid.linewidth": 0.85,
    "grid.alpha": 0.55,
    "savefig.dpi": 220,
}

_STYLE_INITIALISED = False


def configure_matplotlib() -> None:
    """Apply the shared matplotlib style exactly once."""

    global _STYLE_INITIALISED
    if _STYLE_INITIALISED:
        return
    mpl.rcParams.update(STYLE_PARAMS)
    _STYLE_INITIALISED = True


def ensure_output_directories() -> None:
    """Create folders for numeric results and SVG figures."""

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _bold_ticklabels(ax: plt.Axes) -> None:
    """Set tick labels to bold to keep typography consistent."""

    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")


def _create_axes(
    figsize: tuple[float, float],
    grid_axis: str = "both",
    add_minor: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Prepare an axis with gridlines, subtle spines, and bold ticks."""

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(
            True,
            axis=grid_axis,
            color=STYLE_PARAMS["grid.color"],
            linewidth=STYLE_PARAMS["grid.linewidth"],
            alpha=STYLE_PARAMS["grid.alpha"],
        )
    if add_minor:
        ax.minorticks_on()
        ax.grid(
            True,
            which="minor",
            axis=grid_axis or "both",
            color="0.85",
            linewidth=0.6,
            alpha=0.4,
        )
    for spine in ax.spines.values():
        spine.set_color(STYLE_PARAMS["axes.edgecolor"])
        spine.set_linewidth(STYLE_PARAMS["axes.linewidth"])
    ax.tick_params(axis="both", which="both", width=0.9)
    _bold_ticklabels(ax)
    return fig, ax


def _format_legend(ax: plt.Axes, **kwargs) -> mpl.legend.Legend | None:
    """Attach a lightly framed legend whose text remains bold."""

    legend = ax.legend(**kwargs)
    if legend:
        frame = legend.get_frame()
        frame.set_linewidth(0.8)
        frame.set_edgecolor("0.68")
        for text in legend.get_texts():
            text.set_fontweight("bold")
    return legend


def _save_figure(fig: plt.Figure, stem: str) -> None:
    """Persist a figure in vector form and free resources."""

    out_path = IMG_DIR / f"{stem}.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


@dataclass(frozen=True)
class CurveSummary:
    """Container with everything needed for reporting macro curves."""

    fpr_grid: np.ndarray
    tpr_macro: np.ndarray
    recall_grid: np.ndarray
    precision_macro: np.ndarray
    roc_auc_macro: float
    ap_macro_trapz: float
    ap_macro_sklearn: float
    ap_micro: float
    macro_prevalence: float
    per_class_metrics: list[dict[str, float]]


def _simulate_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate and split a synthetic but challenging classification dataset."""

    features, labels = make_classification(
        n_samples=6000,
        n_features=25,
        n_informative=12,
        n_redundant=5,
        n_classes=5,
        n_clusters_per_class=1,
        weights=[0.35, 0.25, 0.18, 0.13, 0.09],
        flip_y=0.02,
        class_sep=1.15,
        random_state=RANDOM_SEED,
    )
    return train_test_split(
        features,
        labels,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=labels,
    )


def _calibrated_estimator(model: LogisticRegression) -> CalibratedClassifierCV:
    """Create a calibrated wrapper, handling API changes across sklearn versions."""

    try:
        return CalibratedClassifierCV(estimator=model, method="sigmoid", cv=5)
    except TypeError:  # pragma: no cover - backwards compatibility
        return CalibratedClassifierCV(base_estimator=model, method="sigmoid", cv=5)


def _build_classifier() -> OneVsRestClassifier:
    """Return the calibrated One-vs-Rest logistic regression pipeline."""

    base_lr = LogisticRegression(
        solver="lbfgs",
        multi_class="ovr",
        max_iter=2000,
        C=1.0,
        random_state=RANDOM_SEED,
    )
    calibrated = _calibrated_estimator(base_lr)
    return OneVsRestClassifier(calibrated)


def _evaluate_curves(
    proba: np.ndarray,
    y_test: np.ndarray,
    classes: np.ndarray,
) -> CurveSummary:
    """Compute per-class ROC/PR metrics and aggregate them to macro curves."""

    y_test_bin = label_binarize(y_test, classes=classes)
    fpr_grid = np.linspace(0.0, 1.0, 1001)
    recall_grid = np.linspace(0.0, 1.0, 1001)
    interp_tprs: list[np.ndarray] = []
    interp_precisions: list[np.ndarray] = []
    per_class_metrics: list[dict[str, float]] = []

    for idx, class_label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, idx], proba[:, idx])
        precision, recall, _ = precision_recall_curve(
            y_test_bin[:, idx], proba[:, idx]
        )
        interp_tprs.append(np.interp(fpr_grid, fpr, tpr))
        interp_precisions.append(np.interp(recall_grid, recall, precision))
        per_class_metrics.append(
            {
                "class": f"Class {class_label}",
                "roc_auc": float(auc(fpr, tpr)),
                "avg_precision": float(
                    average_precision_score(y_test_bin[:, idx], proba[:, idx])
                ),
                "prevalence": float(y_test_bin[:, idx].mean()),
            }
        )

    tpr_macro = np.mean(np.vstack(interp_tprs), axis=0)
    precision_macro = np.mean(np.vstack(interp_precisions), axis=0)
    roc_auc_macro = auc(fpr_grid, tpr_macro)
    ap_macro_trapz = auc(recall_grid, precision_macro)
    ap_macro_sklearn = average_precision_score(
        y_test_bin, proba, average="macro"
    )
    ap_micro = average_precision_score(y_test_bin, proba, average="micro")
    macro_prevalence = float(
        np.mean([entry["prevalence"] for entry in per_class_metrics])
    )

    return CurveSummary(
        fpr_grid=fpr_grid,
        tpr_macro=tpr_macro,
        recall_grid=recall_grid,
        precision_macro=precision_macro,
        roc_auc_macro=float(roc_auc_macro),
        ap_macro_trapz=float(ap_macro_trapz),
        ap_macro_sklearn=float(ap_macro_sklearn),
        ap_micro=float(ap_micro),
        macro_prevalence=macro_prevalence,
        per_class_metrics=per_class_metrics,
    )


def plot_macro_roc(summary: CurveSummary) -> None:
    """Create the macro ROC figure."""

    fig, ax = _create_axes(figsize=(7.0, 5.0), add_minor=True)
    ax.plot(
        summary.fpr_grid,
        summary.tpr_macro,
        label=f"Macro ROC (AUC = {summary.roc_auc_macro:.3f})",
        linewidth=2.6,
        color=COLOR_CYCLE[0],
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.3,
        color="0.35",
        label="Chance",
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("ROC Curve (Macro OvR)")
    _bold_ticklabels(ax)
    _format_legend(ax, loc="lower right")
    fig.tight_layout()
    _save_figure(fig, "roc_curve_macro")


def plot_macro_pr(summary: CurveSummary) -> None:
    """Create the macro Precision-Recall figure."""

    fig, ax = _create_axes(figsize=(7.0, 5.0), add_minor=True)
    ax.plot(
        summary.recall_grid,
        summary.precision_macro,
        label=f"Macro PR (AUC = {summary.ap_macro_trapz:.3f})",
        linewidth=2.6,
        color=COLOR_CYCLE[1],
    )
    ax.hlines(
        summary.macro_prevalence,
        xmin=0.0,
        xmax=1.0,
        linestyle="--",
        linewidth=1.3,
        color="0.35",
        label=f"Baseline (prevalence = {summary.macro_prevalence:.2f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Precision-Recall Curve (Macro OvR)")
    _bold_ticklabels(ax)
    _format_legend(ax, loc="upper right")
    fig.tight_layout()
    _save_figure(fig, "pr_curve_macro")


def _write_curve_tables(summary: CurveSummary) -> None:
    """Persist the interpolated macro curves for downstream use."""

    pd.DataFrame(
        {"fpr": summary.fpr_grid, "tpr_macro": summary.tpr_macro}
    ).to_csv(RESULTS_DIR / "roc_macro_curve.csv", index=False)
    pd.DataFrame(
        {"recall": summary.recall_grid, "precision_macro": summary.precision_macro}
    ).to_csv(RESULTS_DIR / "pr_macro_curve.csv", index=False)


def _write_metrics(summary: CurveSummary) -> None:
    """Dump summary metrics to JSON for reproducibility."""

    metrics = {
        "random_state": RANDOM_SEED,
        "macro": {
            "roc_auc": summary.roc_auc_macro,
            "ap_trapz": summary.ap_macro_trapz,
            "ap_sklearn": summary.ap_macro_sklearn,
        },
        "micro": {"ap": summary.ap_micro},
        "per_class": summary.per_class_metrics,
    }
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def main() -> None:
    """Entry point that trains the classifier and saves figures + metrics."""

    configure_matplotlib()
    ensure_output_directories()
    X_train, X_test, y_train, y_test = _simulate_dataset()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = _build_classifier()
    classifier.fit(X_train, y_train)
    proba = classifier.predict_proba(X_test)

    classes = np.unique(y_train)
    summary = _evaluate_curves(proba, y_test, classes)
    _write_curve_tables(summary)
    _write_metrics(summary)
    plot_macro_roc(summary)
    plot_macro_pr(summary)
    print(f"Figures written to {IMG_DIR} (SVG format)")


if __name__ == "__main__":
    main()
