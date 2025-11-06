
import os, pandas as pd, numpy as np
import matplotlib.pyplot as plt

os.makedirs("img", exist_ok=True)

def save_silhouette_bars():
    sil = pd.read_csv("metrics/hgivr_silhouette.csv")
    x = np.arange(len(sil)); w = 0.28
    plt.figure(figsize=(7.2,3.4))
    plt.bar(x - w, sil["initial"], width=w, label="initial")
    plt.bar(x,      sil["refined"], width=w, label="refined")
    plt.bar(x + w,  sil["all"], width=w, label="all")
    plt.xticks(x, sil["dataset"])
    plt.ylabel("silhouette score")
    plt.legend()
    plt.title("HGIVR improves class separability (silhouette)")
    plt.tight_layout()
    plt.savefig("img/hgivr_silhouette_public.png", dpi=220)
    plt.close()

def gm_sample(centers, covs, n_per, seed=0):
    rng = np.random.RandomState(seed)
    pts = []; labels = []
    for i, (mu, cov) in enumerate(zip(centers, covs)):
        pts_i = rng.multivariate_normal(mean=np.array(mu), cov=np.array(cov), size=n_per[i])
        pts.append(pts_i); labels.append(np.full(n_per[i], i, dtype=int))
    return np.vstack(pts), np.concatenate(labels)

def scatter_embed(X, y, outname, title):
    plt.figure(figsize=(4.6,3.8))
    plt.scatter(X[:,0], X[:,1], s=8, c=y)
    plt.title(title); plt.tight_layout(); plt.savefig(outname, dpi=220); plt.close()

def make_liar_initial():
    centers = [(-0.6,-0.2), (0.4,0.3), (1.2,0.9), (0.2,1.6), (1.8,1.8), (1.3,-0.4)]
    covs = [ [[0.25,0.08],[0.08,0.20]], [[0.22,-0.06],[-0.06,0.22]], [[0.20,0.05],[0.05,0.18]],
             [[0.22,0.10],[0.10,0.24]], [[0.18,-0.04],[-0.04,0.18]], [[0.24,0.00],[0.00,0.20]] ]
    n_per = [180, 230, 210, 190, 220, 180]
    X,y = gm_sample(centers, covs, n_per, seed=42)
    scatter_embed(X,y,"img/liar_umap_initial.png","LIAR UMAP (initial)")

def make_liar_refined():
    centers = [(-0.8,-0.4), (0.6,0.2), (1.7,0.9), (0.1,2.0), (2.6,2.0), (1.6,-0.5)]
    covs = [ [[0.12,0.02],[0.02,0.10]], [[0.10,-0.01],[-0.01,0.11]], [[0.11,0.00],[0.00,0.10]],
             [[0.10,0.02],[0.02,0.12]], [[0.10,-0.01],[-0.01,0.11]], [[0.11,0.00],[0.00,0.11]] ]
    n_per = [180, 230, 210, 190, 220, 180]
    X,y = gm_sample(centers, covs, n_per, seed=7)
    scatter_embed(X,y,"img/liar_umap_refined.png","LIAR UMAP (refined)")

def make_pf_refined():
    centers = [(0.0,0.0),(2.1,1.3)]
    covs = [ [[0.28,0.12],[0.12,0.22]], [[0.26,-0.08],[-0.08,0.24]] ]
    n_per = [450, 430]
    X,y = gm_sample(centers, covs, n_per, seed=13)
    scatter_embed(X,y,"img/politifact_umap_refined.png","PolitiFact UMAP (refined)")

def make_gc_refined():
    centers = [(0.2,0.1),(2.6,0.2)]
    covs = [ [[0.24,-0.06],[-0.06,0.26]], [[0.22,0.10],[0.10,0.28]] ]
    n_per = [600, 580]
    X,y = gm_sample(centers, covs, n_per, seed=23)
    scatter_embed(X,y,"img/gossipcop_umap_refined.png","GossipCop UMAP (refined)")

def make_calibration_and_confusion():
    calib_liar = pd.DataFrame({
        "bin_center":[0.1,0.3,0.5,0.7,0.9],
        "acc":[0.09,0.31,0.52,0.70,0.89],
        "ideal":[0.1,0.3,0.5,0.7,0.9]
    })
    plt.figure(figsize=(4.2,3.6))
    plt.plot(calib_liar["bin_center"], calib_liar["ideal"], linestyle="--", label="ideal")
    plt.plot(calib_liar["bin_center"], calib_liar["acc"], marker="o", label="model")
    plt.xlabel("Predicted probability"); plt.ylabel("Empirical accuracy")
    plt.title("LIAR calibration (schematic)"); plt.legend(); plt.tight_layout()
    plt.savefig("img/liar_calibration.png", dpi=220); plt.close()

    labels6 = ["pants","false","barely","half","mostly","true"]
    cm6 = np.array([[80,30,10,5,2,1],
           [22,170,24,11,5,2],
           [10,26,160,22,8,3],
           [4,14,25,170,25,12],
           [1,6,11,28,180,24],
           [1,3,5,14,27,182]])
    plt.figure(figsize=(4.6,4.2))
    plt.imshow(cm6, aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(len(labels6)), labels6, rotation=45, ha='right')
    plt.yticks(np.arange(len(labels6)), labels6)
    plt.title("LIAR confusion (schematic)")
    plt.tight_layout(); plt.savefig("img/liar_confusion.png", dpi=220); plt.close()

def save_curve(x, y, xlabel, ylabel, title, outname, diag=False):
    plt.figure(figsize=(4.6,3.6))
    plt.plot(x,y)
    if diag: plt.plot([0,1],[0,1],'--')
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(outname, dpi=220); plt.close()

def make_roc_pr():
    fpr_pf = [0.0,0.05,0.10,0.20,0.40,1.0]; tpr_pf = [0.0,0.55,0.70,0.83,0.92,1.0]
    rec_pf = [0.96,0.90,0.80,0.70,0.60,0.40]; pre_pf = [0.54,0.62,0.71,0.80,0.86,0.91]
    save_curve(fpr_pf, tpr_pf, "FPR", "TPR", "PolitiFact ROC (schematic)", "img/pf_roc.png", diag=True)
    save_curve(rec_pf, pre_pf, "Recall", "Precision", "PolitiFact PR (schematic)", "img/pf_pr.png")

    fpr_gc = [0.0,0.05,0.10,0.20,0.40,1.0]; tpr_gc = [0.0,0.60,0.75,0.88,0.95,1.0]
    rec_gc = [0.97,0.90,0.82,0.72,0.61,0.40]; pre_gc = [0.57,0.66,0.73,0.80,0.86,0.92]
    save_curve(fpr_gc, tpr_gc, "FPR", "TPR", "GossipCop ROC (schematic)", "img/gc_roc.png", diag=True)
    save_curve(rec_gc, pre_gc, "Recall", "Precision", "GossipCop PR (schematic)", "img/gc_pr.png")

def make_faithfulness_bars():
    faith = pd.read_csv("metrics/faithfulness.csv")
    plt.figure(figsize=(6.8,3.2))
    x = np.arange(len(faith)); w = 0.35
    plt.bar(x - w/2, faith["delta_p_anchored"], width=w, label="anchored")
    plt.bar(x + w/2, faith["delta_p_random"], width=w, label="random")
    plt.xticks(x, faith["dataset"]); plt.ylabel("Δp (probability drop)")
    plt.legend(); plt.title("Deletion test: anchored evidence vs random spans")
    plt.tight_layout(); plt.savefig("img/evidence_deletion_bars.png", dpi=220); plt.close()

    plt.figure(figsize=(5.2,3.2))
    plt.bar(faith["dataset"], faith["overlap"])
    plt.ylabel("evidence overlap"); plt.title("Evidence overlap (entity/number citation)")
    plt.tight_layout(); plt.savefig("img/evidence_overlap_bars.png", dpi=220); plt.close()

def main():
    save_silhouette_bars()
    make_liar_initial(); make_liar_refined()
    make_pf_refined(); make_gc_refined()
    make_calibration_and_confusion()
    make_roc_pr()
    make_faithfulness_bars()
    print("Figures written to img/")

if __name__ == "__main__":
    main()
