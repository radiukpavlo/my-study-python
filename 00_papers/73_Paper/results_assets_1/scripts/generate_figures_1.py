import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(".")
fig_dir = root/"figs"
data_dir = root/"data"
fig_dir.mkdir(parents=True, exist_ok=True)

# Load CSVs
seg = pd.read_csv(data_dir/"segmentation_metrics_proposed.csv")
domain = pd.read_csv(data_dir/"domain_shift_dsc.csv")
cm = pd.read_csv(data_dir/"confusion_matrix.csv")
cal = pd.read_csv(data_dir/"calibration_bins.csv")
ep = pd.read_csv(data_dir/"ep_benchmarks.csv")
abl = pd.read_csv(data_dir/"ablation_kigcn.csv")

# 1) Segmentation boxplot (re-simulate per-case samples from mean/sd)
rng = np.random.default_rng(42)
def seg_box():
    fig = plt.figure(figsize=(8,5))
    labels = []
    data = []
    for dataset in ["ACDC", "M&Ms-2"]:
        for struct in ["LV Cavity", "Myocardium", "RV Cavity"]:
            row = seg[(seg.Dataset==dataset) & (seg.Structure==struct)].iloc[0]
            samples = rng.normal(row["DSC_mean"], row["DSC_sd"], size=60)
            samples = np.clip(samples, 0.5, 1.0)
            data.append(samples)
            labels.append(f"{dataset}\n{struct}")
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("Dice Similarity Coefficient")
    plt.title("Segmentation accuracy by dataset and structure (SKIF-Seg)")
    plt.tight_layout()
    fig.savefig(fig_dir/"results_seg_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# 2) Domain shift
def domain_shift():
    fig = plt.figure(figsize=(6,4))
    x = np.arange(len(domain))
    plt.bar(x, domain["Delta"].values)
    plt.xticks(x, domain["Structure"].values)
    plt.ylabel("Δ Dice (M&Ms-2 − ACDC)")
    plt.title("Generalization under domain shift")
    plt.axhline(0.0, linestyle="--")
    plt.tight_layout()
    fig.savefig(fig_dir/"results_domain_shift.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# 3) Confusion matrix
def confusion():
    labels = list(cm.columns[1:])
    mat = cm[labels].values
    mat_norm = mat / mat.sum(axis=1, keepdims=True)
    fig = plt.figure(figsize=(6,5))
    plt.imshow(mat_norm, interpolation="nearest")
    plt.title("KI-GCN diagnostic confusion matrix (normalized)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    for i in range(mat_norm.shape[0]):
        for j in range(mat_norm.shape[1]):
            plt.text(j, i, f"{mat_norm[i,j]:.2f}", ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    fig.savefig(fig_dir/"results_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# 4) Reliability diagram
def reliability():
    bins = cal["bin_center"].values
    pc = cal["pred_conf"].values
    ea = cal["emp_acc"].values
    fig = plt.figure(figsize=(6,5))
    plt.plot([0,1],[0,1])
    w = 0.08
    plt.bar(bins - w/2, ea, width=w, alpha=0.8, label="Observed")
    plt.bar(bins + w/2, pc, width=w, alpha=0.5, label="Predicted")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Reliability diagram")
    plt.legend()
    plt.tight_layout()
    fig.savefig(fig_dir/"results_reliability_diagram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# 5) EP throughput
def throughput():
    fig = plt.figure(figsize=(6,4))
    x = np.arange(len(ep))
    plt.bar(x, ep["Median_seconds_per_volume"].values)
    plt.xticks(x, ep["ExecutionProvider"].values)
    plt.ylabel("Median seconds per volume")
    plt.title("Inference throughput by ONNX Runtime Execution Provider")
    plt.tight_layout()
    fig.savefig(fig_dir/"results_ep_throughput.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# 6) Ablation
def ablation():
    fig = plt.figure(figsize=(7,4))
    x = np.arange(len(abl))
    plt.bar(x, abl["Accuracy_%"].values)
    plt.xticks(x, abl["Model"].values, rotation=20, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title("Ablation study on knowledge integration and distillation")
    plt.tight_layout()
    fig.savefig(fig_dir/"results_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    seg_box()
    domain_shift()
    confusion()
    reliability()
    throughput()
    ablation()
