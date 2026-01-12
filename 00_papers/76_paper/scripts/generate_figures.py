"""Generate all figures for the CEUR manuscript.

Run:
    python generate_figures.py

It will create PDF figures in ../figs/.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, ArrowStyle

BASE_DIR = Path(__file__).resolve().parent.parent
FIGS_DIR = BASE_DIR / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# === Data values reproduced from the accompanying technical report ===
m2 = {
    "Crack": {"mAP50": 0.93, "Precision": 0.91, "Recall": 0.89},
    "Soiling": {"mAP50": 0.90, "Precision": 0.88, "Recall": 0.86},
    "Delamination": {"mAP50": 0.89, "Precision": 0.87, "Recall": 0.85},
    "Average": {"mAP50": 0.91, "Precision": 0.89, "Recall": 0.87},
}
m3 = {
    "Crack": {"mAP50": 0.87, "Precision": 0.86, "Recall": 0.84},
    "Soiling": {"mAP50": 0.90, "Precision": 0.88, "Recall": 0.86},
    "Delamination": {"mAP50": 0.93, "Precision": 0.90, "Recall": 0.88},
    "Average": {"mAP50": 0.90, "Precision": 0.88, "Recall": 0.86},
}
ensemble_before_after = {
    "Crack": (0.93, 0.96),
    "Soiling": (0.90, 0.90),
    "Delamination": (0.93, 0.95),
}
hardware = [
    ("NVIDIA Jetson Orin Nano", 100, 0.95, 0.93),
    ("Ambarella H2", 60, 0.70, 0.75),
    ("Qualcomm QCS605", 80, 0.85, 0.82),
]
flight_height = [(5, 98, 96), (10, 93, 90), (15, 84, 79)]  # (m, accuracy%, recall%)
flight_speed = [(3, 94, 92), (5, 93, 90), (7, 91, 88), (10, 85, 82)]
time_of_day = [("08:00–10:00", 93, 90), ("12:00–14:00", 96, 94), ("17:00–19:00", 92, 89)]
weather = [("Clear", 92, 89), ("Cloudy", 96, 94)]
sota_map = [
    ("Di Tommaso et al. (IR hotspots) AP@0.5", 0.669),
    ("Dotenco et al. (defect cls) F1", 0.939),
    ("Dey et al. ST-YOLO mAP@0.5", 0.966),
    ("This work (ensemble) mAP@0.5", 0.960),
]

def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()

def box(ax, text, x, y, w=0.26, h=0.08):
    ax.add_patch(Rectangle((x,y), w,h, fill=False, linewidth=1.0))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=9)

def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1,y1),(x2,y2),
        arrowstyle=ArrowStyle("->", head_length=6, head_width=3),
        linewidth=1.2,
        mutation_scale=10
    ))

def fig_cps_architecture():
    fig, ax = plt.subplots(figsize=(10,5))
    ax.axis("off")

    tiers = [
        ("UAV tier\n(sensing + onboard inference)", 0.02, 0.62, 0.30, 0.33),
        ("Edge tier\n(local server at PV plant)", 0.35, 0.62, 0.30, 0.33),
        ("Cloud tier\n(long-term analytics)", 0.68, 0.62, 0.30, 0.33),
        ("Operator & SCADA\n(HMI / alarms / work orders)", 0.35, 0.08, 0.30, 0.33),
    ]
    for label, x, y, w, h in tiers:
        ax.add_patch(Rectangle((x,y), w,h, fill=False, linewidth=1.5))
        ax.text(x+w/2, y+h-0.03, label, ha="center", va="top", fontsize=10)

    box(ax, "RGB camera", 0.04, 0.78)
    box(ax, "Thermal camera", 0.04, 0.69)
    box(ax, "Jetson Orin\n(YOLOv11-seg)", 0.04, 0.60)

    box(ax, "Mission cache\n+ RTK geotagging", 0.37, 0.78)
    box(ax, "Aggregation\n+ de-duplication", 0.37, 0.69)
    box(ax, "SCADA gateway\n(MQTT/OPC-UA)", 0.37, 0.60)

    box(ax, "Object store\n(images + masks)", 0.70, 0.78)
    box(ax, "Analytics\n(dashboards, trends)", 0.70, 0.69)
    box(ax, "Model registry\n+ updates", 0.70, 0.60)

    box(ax, "HMI dashboard", 0.37, 0.24)
    box(ax, "Alarm rules\n+ fire-risk logic", 0.37, 0.15)

    arrow(ax, 0.32, 0.75, 0.35, 0.75)
    arrow(ax, 0.32, 0.66, 0.35, 0.66)
    arrow(ax, 0.65, 0.75, 0.68, 0.75)
    arrow(ax, 0.65, 0.66, 0.68, 0.66)
    arrow(ax, 0.50, 0.62, 0.50, 0.41)

    savefig(FIGS_DIR/"cps_architecture.pdf")

def fig_glare_geometry():
    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot([0.1,0.9],[0.2,0.2], linewidth=2)
    ax.text(0.5,0.14,"PV module plane", ha="center", fontsize=9)
    ax.arrow(0.5,0.2,0,0.5, head_width=0.02, length_includes_head=True)
    ax.text(0.52,0.55,"n", fontsize=10)
    ax.arrow(0.2,0.9,0.25,-0.65, head_width=0.02, length_includes_head=True)
    ax.text(0.18,0.92,"s (sun ray)", fontsize=9)
    ax.arrow(0.45,0.25,0.35,0.55, head_width=0.02, length_includes_head=True)
    ax.text(0.82,0.82,"r (specular)", fontsize=9, ha="right")
    ax.arrow(0.45,0.25,0.45,0.10, head_width=0.02, length_includes_head=True)
    ax.text(0.92,0.35,"v* (camera)\nrotated away", fontsize=9, ha="right")
    savefig(FIGS_DIR/"glare_geometry.pdf")

def plot_map(dataset, title, outfile):
    labels = ["Crack","Soiling","Delamination","Average"]
    vals = [dataset[k]["mAP50"] for k in labels]
    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.bar(labels, vals)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("mAP@0.5")
    ax.set_title(title)
    for i,v in enumerate(vals):
        ax.text(i, v+0.02, f"{v:.2f}", ha="center", fontsize=9)
    savefig(outfile)

def fig_map_charts():
    plot_map(m2, "Thermogram M2 (two-color palette): detection quality", FIGS_DIR/"map_m2.pdf")
    plot_map(m3, "Thermogram M3 (three-color palette): detection quality", FIGS_DIR/"map_m3.pdf")

def fig_ensemble_gain():
    labels = list(ensemble_before_after.keys())
    before = [ensemble_before_after[k][0] for k in labels]
    after = [ensemble_before_after[k][1] for k in labels]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.bar(x - width/2, before, width, label="Before ensemble")
    ax.bar(x + width/2, after, width, label="After ensemble")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Ensemble fusion improves mAP@0.5")
    ax.legend(fontsize=8)
    savefig(FIGS_DIR/"ensemble_gain.pdf")

def fig_hardware_tradeoff():
    fig, ax = plt.subplots(figsize=(6,3.5))
    fps = [h[1] for h in hardware]
    mapv = [h[2] for h in hardware]
    ax.scatter(fps, mapv)
    for name, f, m, p in hardware:
        ax.annotate(name, (f,m), textcoords="offset points", xytext=(5,5), fontsize=8)
    ax.set_xlabel("Throughput (FPS)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Edge hardware tradeoff (thermogram inference)")
    ax.set_xlim(40, 110)
    ax.set_ylim(0.6, 1.0)
    savefig(FIGS_DIR/"hardware_tradeoff.pdf")

def plot_param(data, xlabel, title, outfile):
    if isinstance(data[0][0], str):
        labels = [d[0] for d in data]
        acc = [d[1] for d in data]
        rec = [d[2] for d in data]
        x = np.arange(len(labels))
        width=0.35
        fig, ax = plt.subplots(figsize=(6,3.5))
        ax.bar(x-width/2, acc, width, label="Accuracy (%)")
        ax.bar(x+width/2, rec, width, label="Recall (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
    else:
        xvals = [d[0] for d in data]
        acc = [d[1] for d in data]
        rec = [d[2] for d in data]
        fig, ax = plt.subplots(figsize=(6,3.5))
        ax.plot(xvals, acc, marker="o", label="Accuracy (%)")
        ax.plot(xvals, rec, marker="o", label="Recall (%)")
        ax.set_xticks(xvals)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_ylim(70, 100)
    ax.legend(fontsize=8)
    savefig(outfile)

def fig_flight_params():
    plot_param(flight_height, "Flight altitude (m)", "Effect of altitude on detection quality", FIGS_DIR/"flight_height.pdf")
    plot_param(flight_speed, "Flight speed (m/s)", "Effect of speed on detection quality", FIGS_DIR/"flight_speed.pdf")
    plot_param(time_of_day, "Time window", "Time-of-day sensitivity", FIGS_DIR/"time_of_day.pdf")
    plot_param(weather, "Weather", "Weather sensitivity", FIGS_DIR/"weather.pdf")

def fig_sota_comparison():
    fig, ax = plt.subplots(figsize=(7,3.5))
    labels = [s[0] for s in sota_map]
    vals = [s[1] for s in sota_map]
    ax.barh(labels, vals)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Reported score (mAP@0.5 or F1)")
    ax.set_title("Representative quantitative comparison (reported by authors)")
    for i,v in enumerate(vals):
        ax.text(v+0.01, i, f"{v:.3f}", va="center", fontsize=8)
    savefig(FIGS_DIR/"sota_comparison.pdf")

if __name__ == "__main__":
    fig_cps_architecture()
    fig_glare_geometry()
    fig_map_charts()
    fig_ensemble_gain()
    fig_hardware_tradeoff()
    fig_flight_params()
    fig_sota_comparison()
    print("Done. Figures written to:", FIGS_DIR)
