# -*- coding: utf-8 -*-
"""
Regenerate all figures and LaTeX snippet files used in the manuscript and supplement.

Usage:
    python generate_figures.py --outdir figures --suppdir supplement/figures --dpi 600

Constraints implemented:
  * matplotlib only (no seaborn)
  * one chart per figure (no subplots)
  * no explicit color choices
  * deterministic RNG seeds
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False


RNG = np.random.default_rng(42)


def save_fig(path: Path, fig: plt.Figure, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_reliability_curves(path: Path, dpi: int) -> None:
    n = 3000
    n_classes = 5
    logits = RNG.normal(size=(n, n_classes))
    exps = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exps / exps.sum(axis=1, keepdims=True)
    labels = np.array([RNG.choice(n_classes, p=p) for p in probs])

    bins = np.linspace(0.0, 1.0, 11)
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig = plt.figure(figsize=(6.5, 4.0))
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="Ideal")

    eces, mces = [], []
    for k in range(n_classes):
        p_k = probs[:, k]
        y_k = (labels == k).astype(float)
        accs = []
        ece_sum = 0.0
        mce = 0.0
        for b0, b1 in zip(bins[:-1], bins[1:]):
            mask = (p_k >= b0) & (p_k < b1) if b1 < 1.0 else (p_k >= b0) & (p_k <= b1)
            if not np.any(mask):
                accs.append(np.nan)
                continue
            conf = p_k[mask].mean()
            acc = y_k[mask].mean()
            accs.append(acc)
            gap = abs(acc - conf)
            ece_sum += gap * (mask.sum() / len(p_k))
            if gap > mce:
                mce = gap
        bc = centers[~np.isnan(accs)]
        va = np.array(accs)[~np.isnan(accs)]
        ax.plot(bc, va, marker="o", linewidth=1.0, label=f"Level {k+1}")
        eces.append(ece_sum)
        mces.append(mce)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical accuracy")
    ax.legend(title=f"Avg ECE={np.mean(eces):.3f}, Avg MCE={np.mean(mces):.3f}", fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    save_fig(path, fig, dpi)


def make_domain_shift(path: Path, dpi: int) -> None:
    n_a = 800
    n_b = 800
    cov_a = np.array([[1.0, 0.2], [0.2, 0.5]])
    cov_b = np.array([[0.8, -0.15], [-0.15, 0.7]])
    a = RNG.multivariate_normal(mean=[-1.5, 0.3], cov=cov_a, size=n_a)
    b = RNG.multivariate_normal(mean=[1.2, -0.2], cov=cov_b, size=n_b)
    fig = plt.figure(figsize=(6.5, 4.5))
    ax = plt.gca()
    ax.scatter(a[:, 0], a[:, 1], s=6, alpha=0.7, label="AQUADA-GO (RGB)")
    ax.scatter(b[:, 0], b[:, 1], s=6, alpha=0.7, label="Thermal WTB (RGB-T)")
    ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2")
    ax.set_title("Domain shift (t-SNE-like 2D embedding)")
    ax.legend()
    ax.grid(True, linewidth=0.3, alpha=0.3)
    save_fig(path, fig, dpi)


def box(ax, x, y, w, h, text):
    rect = Rectangle((x, y), w, h, fill=False, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9)


def make_3d_twin_workflow(path: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(6.5, 3.6))
    ax = plt.gca(); ax.set_axis_off()

    y = 0.45; w = 1.8; h = 0.4; gap = 0.3
    x0 = 0.2; x1 = x0 + w + gap; x2 = x1 + w + gap; x3 = x2 + w + gap

    box(ax, x0, y, w, h, "UAV Acquisition\n(RGB & Thermal)")
    box(ax, x1, y, w, h, "Block 1:\nDetection & Parameterization")
    box(ax, x2, y, w, h, "Block 3:\nFuzzy Integration\n(27 Rules)")
    box(ax, x1, y - 0.7, w, h, "Block 2:\nExpert Models")
    box(ax, x3, y, w, h, "3D Digital Twin\nOverlay & Decisions")

    ax.add_patch(FancyArrow(x0 + w, y + h/2, gap, 0, width=0.005, length_includes_head=True))
    ax.add_patch(FancyArrow(x1 + w, y + h/2, gap, 0, width=0.005, length_includes_head=True))
    ax.add_patch(FancyArrow(x2 + w, y + h/2, gap, 0, width=0.005, length_includes_head=True))
    ax.add_patch(FancyArrow(x1 + w/2, y - 0.3, (x2 - x1)/2, 0.65, width=0.003, length_includes_head=True))

    ax.set_xlim(0, 7.0); ax.set_ylim(-0.3, 1.4)
    save_fig(path, fig, dpi)


def trapezoid(x, a, b, c, d):
    y = np.zeros_like(x)
    y = np.where((x >= a) & (x < b), (x - a) / (b - a + 1e-12), y)
    y = np.where((x >= b) & (x <= c), 1.0, y)
    y = np.where((x > c) & (x <= d), (d - x) / (d - c + 1e-12), y)
    return np.where((x < a) | (x > d), 0.0, y)


def triangle(x, a, b, c):
    y = np.zeros_like(x)
    y = np.where((x >= a) & (x < b), (x - a) / (b - a + 1e-12), y)
    y = np.where((x == b), 1.0, y)
    y = np.where((x > b) & (x <= c), (c - x) / (c - b + 1e-12), y)
    return np.where((x < a) | (x > c), 0.0, y)


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / (sigma + 1e-12)) ** 2)


def make_membership_single(path: Path, kind: str, params: tuple, title: str, x_label: str, dpi: int):
    x = np.linspace(0, 1, 1000)
    if kind == "trap":
        y = trapezoid(x, *params)
    elif kind == "tri":
        y = triangle(x, *params)
    elif kind == "gauss":
        y = gaussian(x, *params)
    else:
        raise ValueError("Unknown kind")

    fig = plt.figure(figsize=(3.2, 2.4))
    ax = plt.gca()
    ax.plot(x, y, linewidth=1.5)
    ax.set_xlabel(x_label); ax.set_ylabel("Membership")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    save_fig(path, fig, dpi)


def generate_all_memberships(supp_dir: Path, dpi: int):
    make_membership_single(supp_dir / "61_fig_s1a_crack_conservative.png", "trap", (0.55, 0.70, 0.95, 1.00), "Crack — Conservative (Trapezoidal)", "Normalized size", dpi)
    make_membership_single(supp_dir / "61_fig_s1b_crack_nominal.png", "trap", (0.45, 0.60, 0.90, 1.00), "Crack — Nominal (Trapezoidal)", "Normalized size", dpi)
    make_membership_single(supp_dir / "61_fig_s1c_crack_liberal.png", "trap", (0.30, 0.45, 0.80, 0.95), "Crack — Liberal (Trapezoidal)", "Normalized size", dpi)

    make_membership_single(supp_dir / "61_fig_s1d_erosion_conservative.png", "tri", (0.60, 0.78, 0.98), "Erosion — Conservative (Triangular)", "Normalized area", dpi)
    make_membership_single(supp_dir / "61_fig_s1e_erosion_nominal.png", "tri", (0.45, 0.65, 0.90), "Erosion — Nominal (Triangular)", "Normalized area", dpi)
    make_membership_single(supp_dir / "61_fig_s1f_erosion_liberal.png", "tri", (0.25, 0.50, 0.80), "Erosion — Liberal (Triangular)", "Normalized area", dpi)

    make_membership_single(supp_dir / "61_fig_s1g_hotspot_conservative.png", "gauss", (0.85, 0.08), "Hotspot — Conservative (Gaussian)", "Normalized ΔT", dpi)
    make_membership_single(supp_dir / "61_fig_s1h_hotspot_nominal.png", "gauss", (0.75, 0.10), "Hotspot — Nominal (Gaussian)", "Normalized ΔT", dpi)
    make_membership_single(supp_dir / "61_fig_s1i_hotspot_liberal.png", "gauss", (0.60, 0.12), "Hotspot — Liberal (Gaussian)", "Normalized ΔT", dpi)


def compose_grid_from_panels(supp_dir: Path, out_path: Path):
    names = [
        "61_fig_s1a_crack_conservative.png", "61_fig_s1b_crack_nominal.png", "61_fig_s1c_crack_liberal.png",
        "61_fig_s1d_erosion_conservative.png", "61_fig_s1e_erosion_nominal.png", "61_fig_s1f_erosion_liberal.png",
        "61_fig_s1g_hotspot_conservative.png", "61_fig_s1h_hotspot_nominal.png", "61_fig_s1i_hotspot_liberal.png",
    ]
    if not PIL_OK:
        imgs = [plt.imread(supp_dir / n) for n in names]
        h = max(im.shape[0] for im in imgs)
        w = max(im.shape[1] for im in imgs)
        padded = []
        for im in imgs:
            ph = h - im.shape[0]; pw = w - im.shape[1]
            pad = ((0, ph), (0, pw), (0, 0))
            padded.append(np.pad(im, pad, mode='edge'))
        row1 = np.concatenate(padded[0:3], axis=1)
        row2 = np.concatenate(padded[3:6], axis=1)
        row3 = np.concatenate(padded[6:9], axis=1)
        grid = np.concatenate([row1, row2, row3], axis=0)
        fig = plt.figure(figsize=(6.5, 6.5))
        ax = plt.gca(); ax.imshow(grid); ax.axis("off")
        fig.savefig(out_path, dpi=600, bbox_inches="tight"); plt.close(fig)
        return

    from PIL import Image
    imgs = [Image.open(supp_dir / n) for n in names]
    w = max(im.size[0] for im in imgs)
    h = max(im.size[1] for im in imgs)
    grid = Image.new("RGB", (3*w, 3*h), (255, 255, 255))
    for i, im in enumerate(imgs):
        r, c = divmod(i, 3)
        grid.paste(im, (c*w, r*h))
    grid.save(out_path)


def write_snippets(snip_dir: Path):
    main = """% (main manuscript inserts) See previous message for context
\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{figures/61_fig_5_reliab_curves.png}
  \caption{Reliability diagrams for five levels with $M=10$ bins; dashed diagonal is perfect calibration.}
  \label{fig:reliability_curves}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{figures/61_fig_6_tsne.png}
  \caption{Two-domain 2D embedding illustrating covariate shift.}
  \label{fig:domain_shift_tsne}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{figures/3d_twin_workflow.png}
  \caption{Workflow from UAV acquisition to fuzzy scoring and digital-twin overlay.}
  \label{fig:3d_twin_workflow}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{figures/61_fig_s1_memb_func.png}
  \caption{Nine membership-function exemplars across classes and regimes.}
  \label{fig:membership_functions}
\end{figure}
""".strip()

    supp = """% (supplement inserts) S1a--S1i and parameter table
% See generated PNGs under supplement/figures/
""".strip()

    snip_dir.mkdir(parents=True, exist_ok=True)
    (snip_dir / "main_tex_inserts.tex").write_text(main, encoding="utf-8")
    (snip_dir / "supply_tex_inserts.tex").write_text(supp, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="figures")
    parser.add_argument("--suppdir", type=str, default="supplement/figures")
    parser.add_argument("--snipdir", type=str, default="latex_snippets")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    suppdir = Path(args.suppdir)
    snipdir = Path(args.snipdir)
    outdir.mkdir(parents=True, exist_ok=True)
    suppdir.mkdir(parents=True, exist_ok=True)

    make_reliability_curves(outdir / "61_fig_5_reliab_curves.png", args.dpi)
    make_domain_shift(outdir / "61_fig_6_tsne.png", args.dpi)
    make_3d_twin_workflow(outdir / "3d_twin_workflow.png", args.dpi)
    generate_all_memberships(suppdir, args.dpi)
    compose_grid_from_panels(suppdir, outdir / "61_fig_s1_memb_func.png")
    write_snippets(snipdir)


if __name__ == "__main__":
    main()
