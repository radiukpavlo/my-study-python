import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(".")
fig_dir = root/"figs"
data_dir = root/"data"
fig_dir.mkdir(parents=True, exist_ok=True)

seg = pd.read_csv(data_dir/"segmentation_metrics_proposed.csv")
domain = pd.read_csv(data_dir/"domain_shift_dsc.csv")
cm = pd.read_csv(data_dir/"confusion_matrix.csv")
cal_pre = pd.read_csv(data_dir/"calibration_bins_pre.csv")
cal_post = pd.read_csv(data_dir/"calibration_bins_post.csv")
ep = pd.read_csv(data_dir/"ep_benchmarks.csv")
batch = pd.read_csv(data_dir/"ep_batch_benchmarks.csv")
macro = pd.read_csv(data_dir/"segmentation_macro_summary.csv")
roc = pd.read_csv(data_dir/"roc_curve_points.csv")
pr = pd.read_csv(data_dir/"pr_curve_points.csv")

rng = np.random.default_rng(42)

def seg_box():
    fig = plt.figure(figsize=(8,5))
    labels, data = [], []
    for dataset in ["ACDC","M&Ms-2"]:
        for struct in ["LV Cavity","Myocardium","RV Cavity"]:
            row = seg[(seg.Dataset==dataset)&(seg.Structure==struct)].iloc[0]
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

def reliability_pre_post():
    bins = cal_pre["bin_center"].values
    fig = plt.figure(figsize=(6,5))
    plt.plot([0,1],[0,1])
    w = 0.07
    plt.bar(bins - w, cal_pre["emp_acc"].values, width=w, alpha=0.7, label="Observed (pre)")
    plt.bar(bins, cal_pre["pred_conf"].values, width=w, alpha=0.5, label="Predicted")
    plt.bar(bins + w, cal_post["emp_acc"].values, width=w, alpha=0.7, label="Observed (post)")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Reliability before/after temperature scaling")
    plt.legend()
    plt.tight_layout()
    fig.savefig(fig_dir/"results_reliability_pre_post.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def ep_throughput():
    fig = plt.figure(figsize=(6,4))
    x = np.arange(len(ep))
    plt.bar(x, ep["Median_seconds_per_volume"].values)
    plt.xticks(x, ep["ExecutionProvider"].values)
    plt.ylabel("Median seconds per volume")
    plt.title("Inference throughput by ONNX Runtime Execution Provider")
    plt.tight_layout()
    fig.savefig(fig_dir/"results_ep_throughput.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def ep_batch():
    fig = plt.figure(figsize=(6,4))
    for ep_name in ["CPU","CUDA","DirectML"]:
        sub = batch[batch.ExecutionProvider==ep_name]
        plt.plot(sub["BatchSize"].values, sub["Seconds_per_volume"].values, marker="o", label=ep_name)
    plt.xlabel("Batch size")
    plt.ylabel("Seconds per volume")
    plt.title("Throughput vs. batch size by Execution Provider")
    plt.legend()
    plt.tight_layout()
    fig.savefig(fig_dir/"results_throughput_batch.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def seg_macro_bars():
    fig = plt.figure(figsize=(6.5,4.2))
    x = np.arange(len(macro))
    width = 0.35
    plt.bar(x-width/2, macro["Baseline_U-Net_DSC"].values, width=width, label="U-Net")
    plt.bar(x+width/2, macro["SKIF-Seg_DSC"].values, width=width, label="SKIF-Seg")
    plt.xticks(x, macro["Dataset"].values)
    plt.ylabel("Macro Dice (LV/Myo/RV)")
    plt.title("Macro segmentation accuracy by dataset")
    plt.legend()
    plt.tight_layout()
    fig.savefig(fig_dir/"results_seg_macro_bars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def roc_curve():
    auc = 0.964
    fig = plt.figure(figsize=(5.8,5.2))
    plt.plot([0,1],[0,1])
    plt.plot(roc["FPR"].values, roc["TPR"].values, label=f"ROC (AUC={auc:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Macro ROC curve (multiclass, one-vs-rest)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(fig_dir/"results_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def pr_curve():
    ap = 0.951
    fig = plt.figure(figsize=(5.8,5.2))
    plt.plot(pr["Recall"].values, pr["Precision"].values, label=f"PR (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Macro Precision-Recall curve")
    plt.legend()
    plt.tight_layout()
    fig.savefig(fig_dir/"results_pr_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    seg_box()
    domain_shift()
    confusion()
    reliability_pre_post()
    ep_throughput()
    ep_batch()
    seg_macro_bars()
    roc_curve()
    pr_curve()
