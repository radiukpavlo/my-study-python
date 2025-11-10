"""Generates CSVs, LaTeX tables, and professional figures for the paper."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

RNG = np.random.default_rng(42)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figs"
TABLE_DIR = ROOT / "tables"

MODELS = ("Baseline U-Net", "Att-UNet", "U-Net+TAAC", "SKIF-Seg")
STRUCTURES = ("LV", "Myo", "RV")
DATASETS = ("ACDC", "M&Ms-2")
CLASSES = ("NOR", "HCM", "DCM", "MINF", "ARV")

SEGMENTATION_DATA = {
    "ACDC": {
        "Baseline U-Net": {
            "DSC": [94.2, 84.3, 88.2],
            "HD95": [8.4, 10.6, 9.5],
            "ASD": [2.1, 2.7, 2.5],
        },
        "Att-UNet": {
            "DSC": [94.8, 85.2, 89.1],
            "HD95": [7.8, 9.6, 8.8],
            "ASD": [2.0, 2.5, 2.3],
        },
        "U-Net+TAAC": {
            "DSC": [95.2, 86.4, 90.2],
            "HD95": [7.1, 8.1, 7.9],
            "ASD": [1.8, 2.2, 2.1],
        },
        "SKIF-Seg": {
            "DSC": [95.6, 87.1, 91.0],
            "HD95": [6.6, 7.3, 7.0],
            "ASD": [1.7, 2.0, 2.0],
        },
    },
    "M&Ms-2": {
        "Baseline U-Net": {
            "DSC": [92.8, 82.1, 86.4],
            "HD95": [9.6, 11.8, 11.2],
            "ASD": [2.6, 3.2, 3.0],
        },
        "Att-UNet": {
            "DSC": [93.4, 83.4, 87.3],
            "HD95": [9.0, 10.7, 10.1],
            "ASD": [2.5, 2.9, 2.8],
        },
        "U-Net+TAAC": {
            "DSC": [93.8, 84.4, 88.1],
            "HD95": [8.4, 9.2, 9.4],
            "ASD": [2.3, 2.6, 2.6],
        },
        "SKIF-Seg": {
            "DSC": [94.1, 85.3, 89.4],
            "HD95": [7.9, 8.7, 8.8],
            "ASD": [2.1, 2.4, 2.4],
        },
    },
}

TOPOLOGY_DATA = {
    "ACDC": {
        "Baseline U-Net": {
            "TER%": 9.1,
            "RingBreak%": 10.5,
            "LV-RV-Overlap%": 2.6,
            "ClosedRing%": 89.4,
        },
        "SKIF-Seg": {
            "TER%": 3.4,
            "RingBreak%": 4.1,
            "LV-RV-Overlap%": 0.6,
            "ClosedRing%": 96.8,
        },
    },
    "M&Ms-2": {
        "Baseline U-Net": {
            "TER%": 12.7,
            "RingBreak%": 14.9,
            "LV-RV-Overlap%": 3.7,
            "ClosedRing%": 86.2,
        },
        "SKIF-Seg": {
            "TER%": 5.1,
            "RingBreak%": 6.0,
            "LV-RV-Overlap%": 0.9,
            "ClosedRing%": 95.1,
        },
    },
}

NODE_IMPORTANCE = {"LV_ED": 0.23, "LV_ES": 0.29, "Myo": 0.31, "RV": 0.17}

SLUG_OVERRIDES = {"M&Ms-2": "mms2"}

BASE_FONT_SIZE = mpl.rcParams.get("font.size", 10) + 2
COLOR_CYCLE = mpl.colormaps["tab10"].colors
MAJOR_GRID_STYLE = {"color": "0.78", "linewidth": 0.85, "alpha": 0.65}
MINOR_GRID_STYLE = {"color": "0.88", "linewidth": 0.6, "alpha": 0.45}


@dataclass(frozen=True)
class ClassificationCurves:
    roc: dict[str, tuple[np.ndarray, np.ndarray]]
    pr: dict[str, tuple[np.ndarray, np.ndarray]]
    auc_scores: dict[str, float]
    ap_scores: dict[str, float]
    confusion: np.ndarray
    bin_accuracy: np.ndarray
    bin_confidence: np.ndarray
    ece: float
    y_true: np.ndarray
    y_pred: np.ndarray


def ensure_directories() -> None:
    for path in (DATA_DIR, FIG_DIR, TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def configure_matplotlib() -> None:
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
            "legend.framealpha": 0.94,
            "axes.edgecolor": "0.32",
            "axes.linewidth": 1.1,
            "grid.color": "0.7",
            "grid.linewidth": 0.8,
            "savefig.dpi": 220,
        }
    )


def dataset_slug(dataset: str) -> str:
    if dataset in SLUG_OVERRIDES:
        return SLUG_OVERRIDES[dataset]
    return re.sub(r"[^a-z0-9]+", "", dataset.lower())


def create_axes(
    figsize: tuple[float, float] = (6.0, 4.0),
    grid_axis: str | None = "both",
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
        spine.set_linewidth(1.05)
    ax.tick_params(labelsize=BASE_FONT_SIZE - 1, width=1.0)
    for label in chain(ax.get_xticklabels(), ax.get_yticklabels()):
        label.set_fontweight("bold")
    return fig, ax


def format_legend(ax: mpl.axes.Axes, **kwargs: object) -> None:
    legend = ax.legend(**kwargs)
    if legend:
        frame = legend.get_frame()
        frame.set_linewidth(0.8)
        frame.set_edgecolor("0.7")


def save_pdf(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_single_series(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    stem: str,
    *,
    rotation: float = 0.0,
    grid_axis: str | None = "y",
    color_idx: int = 0,
    figsize: tuple[float, float] = (6.0, 3.8),
) -> None:
    fig, ax = create_axes(figsize=figsize, grid_axis=grid_axis)
    idx = np.arange(len(labels))
    ax.bar(
        idx,
        values,
        width=0.65,
        color=COLOR_CYCLE[color_idx % len(COLOR_CYCLE)],
        edgecolor="0.25",
        linewidth=0.9,
    )
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=rotation)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    save_pdf(fig, stem)


def plot_grouped_bars(
    labels: Sequence[str],
    series_values: Sequence[Sequence[float]],
    series_labels: Sequence[str],
    title: str,
    ylabel: str,
    stem: str,
    *,
    rotation: float = 0.0,
    figsize: tuple[float, float] = (6.6, 4.0),
) -> None:
    fig, ax = create_axes(figsize=figsize, grid_axis="y")
    idx = np.arange(len(labels), dtype=float)
    width = 0.8 / max(len(series_values), 1)
    for offset, values in enumerate(series_values):
        color = COLOR_CYCLE[offset % len(COLOR_CYCLE)]
        ax.bar(
            idx + offset * width,
            values,
            width=width,
            label=series_labels[offset],
            color=color,
            edgecolor="0.25",
            linewidth=0.9,
        )
    ax.set_xticks(idx + width * (len(series_values) - 1) / 2)
    ax.set_xticklabels(labels, rotation=rotation)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(series_values) > 1:
        format_legend(ax, loc="upper left")
    fig.tight_layout()
    save_pdf(fig, stem)


def build_per_structure_df() -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for dataset, models in SEGMENTATION_DATA.items():
        for model, metrics in models.items():
            for idx, structure in enumerate(STRUCTURES):
                records.append(
                    {
                        "Dataset": dataset,
                        "Model": model,
                        "Structure": structure,
                        "DSC(%)": metrics["DSC"][idx],
                        "HD95(mm)": metrics["HD95"][idx],
                        "ASD(mm)": metrics["ASD"][idx],
                    }
                )
    return (
        pd.DataFrame(records)
        .sort_values(["Dataset", "Model", "Structure"])
        .reset_index(drop=True)
    )


def build_macro_df(per_structure: pd.DataFrame) -> pd.DataFrame:
    return (
        per_structure.groupby(["Dataset", "Model"])
        .agg(
            Macro_DSC_pct=("DSC(%)", "mean"),
            Macro_HD95_mm=("HD95(mm)", "mean"),
            Macro_ASD_mm=("ASD(mm)", "mean"),
        )
        .reset_index()
    )


def build_domain_gap_df(per_structure: pd.DataFrame) -> pd.DataFrame:
    acdc = per_structure[
        (per_structure["Dataset"] == "ACDC")
        & (per_structure["Model"] == "SKIF-Seg")
    ].set_index("Structure")
    mms = per_structure[
        (per_structure["Dataset"] == "M&Ms-2")
        & (per_structure["Model"] == "SKIF-Seg")
    ].set_index("Structure")
    rows = []
    for structure in STRUCTURES:
        rows.append(
            {
                "Structure": structure,
                "Delta_DSC_pp": acdc.at[structure, "DSC(%)"]
                - mms.at[structure, "DSC(%)"],
                "Delta_HD95_mm": mms.at[structure, "HD95(mm)"]
                - acdc.at[structure, "HD95(mm)"],
            }
        )
    return pd.DataFrame(rows)


def build_ablation_df(per_structure: pd.DataFrame) -> pd.DataFrame:
    baseline = per_structure[
        (per_structure["Dataset"] == "ACDC")
        & (per_structure["Model"] == "Baseline U-Net")
    ].set_index("Structure")
    taac = per_structure[
        (per_structure["Dataset"] == "ACDC")
        & (per_structure["Model"] == "U-Net+TAAC")
    ].set_index("Structure")
    skif = per_structure[
        (per_structure["Dataset"] == "ACDC")
        & (per_structure["Model"] == "SKIF-Seg")
    ].set_index("Structure")
    rows = []
    for structure in STRUCTURES:
        rows.append(
            {
                "Structure": structure,
                "TAAC_Delta_DSC_pp": taac.at[structure, "DSC(%)"]
                - baseline.at[structure, "DSC(%)"],
                "TAAC_Delta_HD95_mm": taac.at[structure, "HD95(mm)"]
                - baseline.at[structure, "HD95(mm)"],
                "EGA_Delta_DSC_pp": skif.at[structure, "DSC(%)"]
                - taac.at[structure, "DSC(%)"],
                "EGA_Delta_HD95_mm": skif.at[structure, "HD95(mm)"]
                - taac.at[structure, "HD95(mm)"],
            }
        )
    return pd.DataFrame(rows)


def build_topology_df() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for dataset, models in TOPOLOGY_DATA.items():
        for model, metrics in models.items():
            rows.append({"Dataset": dataset, "Model": model, **metrics})
    return pd.DataFrame(rows)


def build_node_importance_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"Node": list(NODE_IMPORTANCE.keys()), "Importance": list(NODE_IMPORTANCE.values())}
    )


def simulate_classification_predictions(
    classes: Sequence[str],
    n_samples_per_class: int,
    alpha_low: float,
    alpha_high: float,
) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for class_idx, label in enumerate(classes):
        for sample_idx in range(n_samples_per_class):
            alpha = np.full(len(classes), alpha_low, dtype=float)
            alpha[class_idx] = alpha_high
            probs = RNG.dirichlet(alpha)
            record: dict[str, float | str] = {
                "sample_id": f"{label}_{sample_idx:02d}",
                "true_label": label,
            }
            for cls, prob in zip(classes, probs):
                record[f"prob_{cls}"] = float(prob)
            records.append(record)
    predictions = pd.DataFrame(records)
    prob_cols = [f"prob_{cls}" for cls in classes]
    predictions["pred_label"] = (
        predictions[prob_cols].idxmax(axis=1).str.replace("prob_", "", regex=False)
    )
    return predictions


def build_classification_curves(
    predictions: pd.DataFrame, classes: Sequence[str]
) -> ClassificationCurves:
    prob_cols = [f"prob_{cls}" for cls in classes]
    prob_matrix = predictions[prob_cols].to_numpy()
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_true = predictions["true_label"].map(label_to_idx).to_numpy()
    y_pred = prob_matrix.argmax(axis=1)

    roc_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    pr_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    auc_scores: dict[str, float] = {}
    ap_scores: dict[str, float] = {}
    for idx, label in enumerate(classes):
        y_bin = (y_true == idx).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, prob_matrix[:, idx])
        roc_curves[label] = (fpr, tpr)
        auc_scores[label] = float(auc(fpr, tpr))
        precision, recall, _ = precision_recall_curve(y_bin, prob_matrix[:, idx])
        pr_curves[label] = (recall, precision)
        ap_scores[label] = float(average_precision_score(y_bin, prob_matrix[:, idx]))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    confidence = prob_matrix.max(axis=1)
    correctness = (y_true == y_pred).astype(int)
    bin_edges = np.linspace(0.0, 1.0, 11)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_accuracy = np.full(10, np.nan)
    bin_confidence = np.full(10, np.nan)
    bin_counts = np.zeros(10, dtype=int)
    bin_ids = np.digitize(confidence, bin_edges, right=False) - 1
    for idx in range(10):
        mask = bin_ids == idx
        if np.any(mask):
            bin_accuracy[idx] = correctness[mask].mean()
            bin_confidence[idx] = confidence[mask].mean()
            bin_counts[idx] = mask.sum()
        else:
            bin_confidence[idx] = bin_centers[idx]
    valid = bin_counts > 0
    weights = np.zeros_like(bin_counts, dtype=float)
    weights[valid] = bin_counts[valid] / confidence.size
    ece = float(
        np.sum(np.abs(bin_accuracy[valid] - bin_confidence[valid]) * weights[valid])
    )

    return ClassificationCurves(
        roc=roc_curves,
        pr=pr_curves,
        auc_scores=auc_scores,
        ap_scores=ap_scores,
        confusion=cm,
        bin_accuracy=bin_accuracy,
        bin_confidence=bin_confidence,
        ece=ece,
        y_true=y_true,
        y_pred=y_pred,
    )


def compute_classification_statistics(
    predictions: pd.DataFrame, classes: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = build_classification_curves(predictions, classes)
    accuracy = float(np.mean(curves.y_true == curves.y_pred))

    per_class_f1 = []
    for idx, _ in enumerate(classes):
        tp = np.sum((curves.y_true == idx) & (curves.y_pred == idx))
        fp = np.sum((curves.y_true != idx) & (curves.y_pred == idx))
        fn = np.sum((curves.y_true == idx) & (curves.y_pred != idx))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        per_class_f1.append(f1)
    macro_f1 = float(np.mean(per_class_f1))

    all_fpr = np.unique(np.concatenate([curves.roc[label][0] for label in classes]))
    mean_tpr = np.zeros_like(all_fpr)
    for label in classes:
        fpr, tpr = curves.roc[label]
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= len(classes)
    macro_auc = float(auc(all_fpr, mean_tpr))
    macro_ap = float(np.mean([curves.ap_scores[label] for label in classes]))

    macro_metrics = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Macro-F1",
                "Macro-AUC (ROC)",
                "Macro-AP (PR)",
                "ECE (10 bins)",
            ],
            "Value": [accuracy, macro_f1, macro_auc, macro_ap, curves.ece],
        }
    )

    per_class_table = pd.DataFrame(
        {
            "Class": classes,
            "F1": per_class_f1,
            "AUC": [curves.auc_scores[label] for label in classes],
            "AP": [curves.ap_scores[label] for label in classes],
        }
    )
    return macro_metrics, per_class_table


def dataframe_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    columns = list(df.columns)
    align = "l" + "c" * (len(columns) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(columns) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(row[col]) for col in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, caption: str, label: str, filename: str) -> None:
    (TABLE_DIR / filename).write_text(dataframe_to_latex(df, caption, label))


def write_latex_tables(
    per_structure: pd.DataFrame,
    macro: pd.DataFrame,
    domain_gap: pd.DataFrame,
    topology: pd.DataFrame,
    ablation: pd.DataFrame,
    macro_cls: pd.DataFrame,
    per_class_cls: pd.DataFrame,
) -> None:
    macro_tbl = macro.rename(
        columns={
            "Macro_DSC_pct": "Macro DSC (%)",
            "Macro_HD95_mm": "Macro HD95 (mm)",
            "Macro_ASD_mm": "Macro ASD (mm)",
        }
    ).copy()
    for col in ["Macro DSC (%)", "Macro HD95 (mm)", "Macro ASD (mm)"]:
        macro_tbl[col] = macro_tbl[col].map(lambda val: f"{val:.1f}")
    write_table(
        macro_tbl,
        "Macro performance across LV/Myo/RV for each model and dataset.",
        "tab:macro_seg",
        "tab_macro_seg.tex",
    )

    dataset_tags = (("ACDC", "acdc"), ("M&Ms-2", "mms"))
    for dataset, tag in dataset_tags:
        sub = per_structure[per_structure["Dataset"] == dataset][
            ["Model", "Structure", "DSC(%)", "HD95(mm)", "ASD(mm)"]
        ].copy()
        for col in ["DSC(%)", "HD95(mm)", "ASD(mm)"]:
            sub[col] = sub[col].map(lambda val: f"{val:.1f}")
        write_table(
            sub,
            f"Per-structure segmentation on {dataset}.",
            f"tab:per_structure_{tag}",
            f"tab_per_structure_{tag}.tex",
        )

    topo_tbl = topology.copy()
    for col in ["TER%", "RingBreak%", "LV-RV-Overlap%", "ClosedRing%"]:
        topo_tbl[col] = topo_tbl[col].map(lambda val: f"{val:.1f}")
    write_table(
        topo_tbl,
        "Topology plausibility metrics (lower is better except ClosedRing%).",
        "tab:topology",
        "tab_topology.tex",
    )

    ablation_tbl = ablation.rename(
        columns={
            "TAAC_Delta_DSC_pp": "TAAC Delta DSC (pp)",
            "TAAC_Delta_HD95_mm": "TAAC Delta HD95 (mm)",
            "EGA_Delta_DSC_pp": "EGA Delta DSC (pp)",
            "EGA_Delta_HD95_mm": "EGA Delta HD95 (mm)",
        }
    ).copy()
    for col in ablation_tbl.columns[1:]:
        ablation_tbl[col] = ablation_tbl[col].map(lambda val: f"{val:.1f}")
    write_table(
        ablation_tbl,
        "Ablation on ACDC: gains from TAAC and EGA.",
        "tab:ablation",
        "tab_ablation.tex",
    )

    domain_tbl = domain_gap.rename(
        columns={"Delta_DSC_pp": "Delta DSC (pp)", "Delta_HD95_mm": "Delta HD95 (mm)"}
    ).copy()
    for col in ["Delta DSC (pp)", "Delta HD95 (mm)"]:
        domain_tbl[col] = domain_tbl[col].map(lambda val: f"{val:.1f}")
    write_table(
        domain_tbl,
        "Domain-shift (ACDC -> M\\&Ms-2) for SKIF-Seg "
        "(positive values indicate worse on M\\&Ms-2).",
        "tab:domain_gap",
        "tab_domain_gap.tex",
    )

    macro_cls_tbl = macro_cls.copy()
    formatters = {
        "Accuracy": lambda val: f"{val * 100:.1f}%",
        "Macro-F1": lambda val: f"{val:.3f}",
        "Macro-AUC (ROC)": lambda val: f"{val:.3f}",
        "Macro-AP (PR)": lambda val: f"{val:.3f}",
        "ECE (10 bins)": lambda val: f"{val:.3f}",
    }
    macro_cls_tbl["Value"] = macro_cls_tbl.apply(
        lambda row: formatters[row["Metric"]](row["Value"]), axis=1
    )
    write_table(
        macro_cls_tbl,
        "KI-GCN diagnostic performance on ACDC (5-class).",
        "tab:cls_macro",
        "tab_cls_macro.tex",
    )

    per_class_tbl = per_class_cls.copy()
    for col in ["F1", "AUC", "AP"]:
        per_class_tbl[col] = per_class_tbl[col].map(lambda val: f"{val:.3f}")
    write_table(
        per_class_tbl,
        "Per-class F1, ROC-AUC and AP (AUPRC) for KI-GCN on ACDC.",
        "tab:cls_per_class",
        "tab_cls_per_class.tex",
    )


def plot_relative_hd95(per_structure: pd.DataFrame) -> None:
    baseline = per_structure[
        (per_structure["Dataset"] == "ACDC")
        & (per_structure["Model"] == "Baseline U-Net")
    ].set_index("Structure")
    skif = per_structure[
        (per_structure["Dataset"] == "ACDC")
        & (per_structure["Model"] == "SKIF-Seg")
    ].set_index("Structure")
    reductions = []
    for structure in STRUCTURES:
        base = baseline.at[structure, "HD95(mm)"]
        skif_val = skif.at[structure, "HD95(mm)"]
        reductions.append(100.0 * (base - skif_val) / base)
    plot_single_series(
        STRUCTURES,
        reductions,
        "HD95 relative reduction (ACDC): SKIF-Seg vs Baseline",
        "Reduction (%)",
        "rel_hd95_acdc",
        color_idx=4,
        rotation=0.0,
    )


def plot_domain_gap(domain_gap: pd.DataFrame) -> None:
    row = domain_gap[domain_gap["Structure"] == "Myo"].iloc[0]
    labels = ["Delta DSC (pp)", "Delta HD95 (mm)"]
    values = [row["Delta_DSC_pp"], row["Delta_HD95_mm"]]
    plot_single_series(
        labels,
        values,
        "Domain-shift gap (ACDC -> M&Ms-2) for Myocardium, SKIF-Seg",
        "Gap value",
        "domain_gap_myo",
        color_idx=5,
    )


def plot_topology_bars(topology: pd.DataFrame, dataset: str) -> None:
    metrics = ["TER%", "RingBreak%", "LV-RV-Overlap%", "ClosedRing%"]
    sub = topology[topology["Dataset"] == dataset]
    series_values = []
    for model in ("Baseline U-Net", "SKIF-Seg"):
        series_values.append(
            [float(sub[sub["Model"] == model][metric].iloc[0]) for metric in metrics]
        )
    plot_grouped_bars(
        metrics,
        series_values,
        ["Baseline U-Net", "SKIF-Seg"],
        f"Topology metrics on {dataset} (lower better, except ClosedRing%)",
        "Percentage",
        f"topology_{dataset_slug(dataset)}",
        rotation=12,
    )


def plot_ablation_myo(ablation: pd.DataFrame) -> None:
    row = ablation[ablation["Structure"] == "Myo"].iloc[0]
    labels = ["TAAC Delta DSC (pp)", "EGA Delta DSC (pp)"]
    values = [row["TAAC_Delta_DSC_pp"], row["EGA_Delta_DSC_pp"]]
    plot_single_series(
        labels,
        values,
        "Ablation gains on Myocardium (ACDC)",
        "Dice gain (pp)",
        "ablation_myo_dsc",
        color_idx=6,
    )


def generate_segmentation_figures() -> None:
    macro = pd.read_csv(DATA_DIR / "macro_metrics.csv")
    per_structure = pd.read_csv(DATA_DIR / "per_structure_metrics.csv")
    domain_gap = pd.read_csv(DATA_DIR / "domain_gap_skif.csv")
    topology = pd.read_csv(DATA_DIR / "topology_metrics.csv")
    ablation = pd.read_csv(DATA_DIR / "ablation_acdc.csv")

    for dataset in DATASETS:
        slug = dataset_slug(dataset)
        sub_macro = macro[macro["Dataset"] == dataset]
        plot_single_series(
            list(sub_macro["Model"]),
            list(sub_macro["Macro_DSC_pct"]),
            f"Macro Dice on {dataset}",
            "DSC (%)",
            f"macro_dsc_{slug}",
            rotation=12,
            color_idx=0,
        )
        plot_single_series(
            list(sub_macro["Model"]),
            list(sub_macro["Macro_HD95_mm"]),
            f"Macro HD95 on {dataset}",
            "HD95 (mm)",
            f"macro_hd95_{slug}",
            rotation=12,
            color_idx=1,
        )
        myo = per_structure[
            (per_structure["Dataset"] == dataset)
            & (per_structure["Structure"] == "Myo")
        ]
        plot_single_series(
            list(myo["Model"]),
            list(myo["DSC(%)"]),
            f"Myocardium Dice on {dataset}",
            "DSC (%)",
            f"myo_dsc_{slug}",
            rotation=12,
            color_idx=2,
        )
        plot_single_series(
            list(myo["Model"]),
            list(myo["HD95(mm)"]),
            f"Myocardium HD95 on {dataset}",
            "HD95 (mm)",
            f"myo_hd95_{slug}",
            rotation=12,
            color_idx=3,
        )
        plot_topology_bars(topology, dataset)

    plot_relative_hd95(per_structure)
    plot_domain_gap(domain_gap)
    plot_ablation_myo(ablation)


def plot_classification_roc(
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    auc_scores: dict[str, float],
) -> None:
    fig, ax = create_axes(figsize=(5.8, 4.6), grid_axis="both", add_minor=True)
    for idx, label in enumerate(CLASSES):
        fpr, tpr = roc_curves[label]
        ax.plot(
            fpr,
            tpr,
            color=COLOR_CYCLE[idx % len(COLOR_CYCLE)],
            linewidth=2.4,
            label=f"{label} (AUC = {auc_scores[label]:.3f})",
        )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="0.4",
        linewidth=1.2,
        label="Chance",
    )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("KI-GCN ROC (ACDC, one-vs-rest)")
    format_legend(ax, loc="lower right")
    fig.tight_layout()
    save_pdf(fig, "cls_roc")


def plot_classification_pr(
    pr_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    ap_scores: dict[str, float],
) -> None:
    fig, ax = create_axes(figsize=(5.8, 4.6), grid_axis="both", add_minor=True)
    for idx, label in enumerate(CLASSES):
        recall, precision = pr_curves[label]
        ax.plot(
            recall,
            precision,
            color=COLOR_CYCLE[idx % len(COLOR_CYCLE)],
            linewidth=2.4,
            label=f"{label} (AP = {ap_scores[label]:.3f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("KI-GCN Precision-Recall (ACDC, one-vs-rest)")
    format_legend(ax, loc="lower left")
    fig.tight_layout()
    save_pdf(fig, "cls_pr")


def plot_confusion(cm: np.ndarray) -> None:
    fig, ax = create_axes(figsize=(5.4, 4.8), grid_axis=None)
    heatmap = ax.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        color = "white" if value > cm.max() * 0.6 else "black"
        ax.text(
            j,
            i,
            f"{value:d}",
            ha="center",
            va="center",
            color=color,
            fontweight="bold",
        )
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("KI-GCN Confusion Matrix (ACDC)")
    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count", rotation=270, labelpad=18, weight="bold")
    cbar.ax.tick_params(labelsize=BASE_FONT_SIZE - 1, width=0.8)
    fig.tight_layout()
    save_pdf(fig, "cls_confusion")


def plot_calibration(
    mean_confidence: np.ndarray,
    accuracy: np.ndarray,
    ece: float,
) -> None:
    mask = ~np.isnan(accuracy)
    fig, ax = create_axes(figsize=(5.2, 4.0), add_minor=True)
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.2,
        color="0.5",
        label="Perfect calibration",
    )
    ax.plot(
        mean_confidence[mask],
        accuracy[mask],
        marker="o",
        linewidth=2.2,
        color=COLOR_CYCLE[0],
        label="KI-GCN",
    )
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"KI-GCN Reliability (ECE = {ece:.3f})")
    format_legend(ax, loc="best")
    fig.tight_layout()
    save_pdf(fig, "cls_calibration")


def generate_classification_figures() -> None:
    predictions = pd.read_csv(DATA_DIR / "classification_predictions_acdc.csv")
    curves = build_classification_curves(predictions, CLASSES)
    plot_classification_roc(curves.roc, curves.auc_scores)
    plot_classification_pr(curves.pr, curves.ap_scores)
    plot_confusion(curves.confusion)
    plot_calibration(curves.bin_confidence, curves.bin_accuracy, curves.ece)


def plot_node_importance() -> None:
    node_importance = pd.read_csv(DATA_DIR / "node_importance.csv")
    plot_single_series(
        list(node_importance["Node"]),
        list(node_importance["Importance"]),
        "KI-GCN Node Importance (mean over test)",
        "Importance",
        "kigcn_node_importance",
        color_idx=7,
    )


def main() -> None:
    ensure_directories()
    configure_matplotlib()

    per_structure = build_per_structure_df()
    macro = build_macro_df(per_structure)
    domain_gap = build_domain_gap_df(per_structure)
    ablation = build_ablation_df(per_structure)
    topology = build_topology_df()
    node_importance = build_node_importance_df()
    predictions = simulate_classification_predictions(
        classes=CLASSES,
        n_samples_per_class=20,
        alpha_low=1.3,
        alpha_high=6.0,
    )
    macro_cls, per_class_cls = compute_classification_statistics(predictions, CLASSES)

    csv_data = {
        "per_structure_metrics.csv": per_structure,
        "macro_metrics.csv": macro,
        "domain_gap_skif.csv": domain_gap,
        "ablation_acdc.csv": ablation,
        "topology_metrics.csv": topology,
        "node_importance.csv": node_importance,
        "classification_macro_metrics.csv": macro_cls,
        "classification_per_class.csv": per_class_cls,
    }
    for name, df in csv_data.items():
        df.to_csv(DATA_DIR / name, index=False)
    predictions.to_csv(DATA_DIR / "classification_predictions_acdc.csv", index=False)

    write_latex_tables(
        per_structure,
        macro,
        domain_gap,
        topology,
        ablation,
        macro_cls,
        per_class_cls,
    )

    generate_segmentation_figures()
    generate_classification_figures()
    plot_node_importance()
    print("All data, tables, and figures have been generated.")


if __name__ == "__main__":
    main()
