"""
Create a single comprehensive summary figure for thesis defense presentation.
4-panel layout: pre-typhoon anomaly | dose-response | spatial map | regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import ast, os

DATA = "/Users/helloling/workspace/thesis/data"
OUT  = "/Users/helloling/workspace/thesis"

# Load figures as images for composite
from PIL import Image

imgs = {
    "pretyphoon": Image.open(f"{OUT}/pretyphoon_analysis.png"),
    "scatter":    Image.open(f"{OUT}/polarization_scatter.png"),
    "bars":       Image.open(f"{OUT}/polarization_summary_bars.png"),
    "ts":         Image.open(f"{OUT}/yagiasha_timeseries.png"),
    "recovery":   Image.open(f"{OUT}/recovery_full.png"),
    "map":        Image.open(f"{OUT}/spatial_deviation_map.png"),
    "cat_box":    Image.open(f"{OUT}/spatial_category_boxplot.png"),
    "regression": Image.open(f"{OUT}/spatial_regression.png"),
}

# ── Create thesis presentation figure (4x2 panel) ────────────────────────────
fig = plt.figure(figsize=(20, 22))
gs  = GridSpec(4, 2, figure=fig, hspace=0.08, wspace=0.04)

panels = [
    (gs[0, 0], "pretyphoon", "A. Pre-Typhoon Behavioral Anomaly"),
    (gs[0, 1], "bars",       "B. Dose–Response: Signal Level vs Speed Polarisation"),
    (gs[1, :], "ts",         "C. Yagiasha Network Response Dynamics (Full Signal Sequence)"),
    (gs[2, 0], "scatter",    "D. Baseline Speed vs Deviation (Road-Level)"),
    (gs[2, 1], "cat_box",    "E. Deviation by Road Functional Category"),
    (gs[3, 0], "map",        "F. Geographic Distribution of Speed Deviation"),
    (gs[3, 1], "regression", "G. Spatial Regression: Predictors of Deviation"),
]

for spec, key, title in panels:
    ax = fig.add_subplot(spec)
    ax.imshow(np.array(imgs[key]))
    ax.set_title(title, fontsize=13, fontweight="bold", pad=4, loc="left")
    ax.axis("off")

fig.suptitle(
    "Road-Level Heterogeneity of Traffic Disruption Under Typhoon Forcing in Hong Kong\n"
    "TomTom Floating-Car Data, September–October 2025  |  Three Typhoons, Full Signal 1–10 Spectrum",
    fontsize=18, fontweight="bold", y=0.995,
)

plt.savefig(f"{OUT}/thesis_summary_figure.png", dpi=220, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}/thesis_summary_figure.png")

# ── Create a cleaner dose-response figure for thesis ─────────────────────────
print("Creating dose-response table figure...")
polarity_data = [
    ("Yagiasha", 1,  16213, 32.9, 26.6, -0.103),
    ("Yagiasha", 3,  23633, 31.5, 25.6, -0.112),
    ("Yagiasha", 8,  18457, 45.8, 20.1, -0.278),
    ("Yagiasha", 9,   3142,  7.3,  4.0, -0.189),
    ("Yagiasha", 10, 14691, 45.3, 17.5, -0.387),
    ("Mina",     1,  28391, 29.1, 25.2, -0.051),
    ("Mina",     3,  25086, 30.1, 26.5, -0.079),
    ("Madum",    1,  19740, 26.9, 27.7,  0.004),
    ("Madum",    3,  19461, 28.0, 27.2, -0.008),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Panel 1: % faster/slower by signal for Yagiasha
yagi = [r for r in polarity_data if r[0]=="Yagiasha"]
sigs = [r[1] for r in yagi]
fstr = [r[3] for r in yagi]
slwr = [r[4] for r in yagi]

x = np.arange(len(sigs))
w = 0.35
axes[0].bar(x - w/2, fstr, w, color="#2196F3", alpha=0.85, label="Faster (dev>0.02)")
axes[0].bar(x + w/2, slwr, w, color="#F44336", alpha=0.85, label="Slower (dev<-0.02)")
for i, (f, s) in enumerate(zip(fstr, slwr)):
    axes[0].text(i-w/2, f+0.5, f"{f:.0f}%", ha="center", va="bottom", fontsize=15)
    axes[0].text(i+w/2, s+0.5, f"{s:.0f}%", ha="center", va="bottom", fontsize=15)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"Signal {s}" for s in sigs])
axes[0].set_ylabel("% of Road Segments")
axes[0].set_title("Yagiasha: Proportion of Roads\nFaster vs Slower by Signal Level",
                  fontsize=13, fontweight="bold")
axes[0].legend(fontsize=18)
axes[0].grid(axis="y", alpha=0.3)
axes[0].set_ylim(0, 55)

# Panel 2: Correlation as dose-response
all_yagi = [(r[1], r[5], r[2]) for r in polarity_data if r[0]=="Yagiasha"]
sig_nums = [r[0] for r in all_yagi]
corrs    = [r[1] for r in all_yagi]
ns       = [r[2] for r in all_yagi]

colors_sig = ["#FFF176","#FFE0B2","#FF8A65","#EF5350","#B71C1C"]
bars = axes[1].bar(range(len(sig_nums)), [abs(c) for c in corrs],
                   color=colors_sig, alpha=0.9, edgecolor="gray")
for i, (c, s) in enumerate(zip(corrs, sig_nums)):
    axes[1].text(i, abs(c)+0.005, f"r={c:.3f}", ha="center", va="bottom", fontsize=15)
axes[1].set_xticks(range(len(sig_nums)))
axes[1].set_xticklabels([f"Signal {s}" for s in sig_nums])
axes[1].set_ylabel("|Pearson r| (baseline speed vs deviation)")
axes[1].set_title("Dose–Response: Polarisation Strengthens\nwith Signal Level (Yagiasha)",
                  fontsize=13, fontweight="bold")
axes[1].grid(axis="y", alpha=0.3)
axes[1].set_ylim(0, 0.5)
# Arrow to show trend
axes[1].annotate("", xy=(4, 0.41), xytext=(0, 0.13),
                arrowprops=dict(arrowstyle="->", color="darkred", lw=2))
axes[1].text(2, 0.28, "Stronger\npolarisation\nat higher signals",
             fontsize=18, color="darkred", ha="center")

# Panel 3: Cross-typhoon comparison (max signal period)
comp_data = [
    ("Mina\n(max S3)",   29.1+30.1, 25.2+26.5, "#2196F3"),
    ("Madum\n(max S3)",  26.9+28.0, 27.7+27.2, "#4CAF50"),
    ("Yagiasha\n(max S10)", 45.3, 17.5, "#F44336"),
]
y_pos = range(len(comp_data))
for i, (label, faster, slower, col) in enumerate(comp_data):
    # Normalize to per-event
    f = faster/2 if "Mina" in label or "Madum" in label else faster
    s = slower/2 if "Mina" in label or "Madum" in label else slower
    axes[2].barh(i-0.2, f, 0.35, color="#2196F3", alpha=0.8)
    axes[2].barh(i+0.2, s, 0.35, color="#F44336", alpha=0.8)
    axes[2].text(f+0.3, i-0.2, f"{f:.0f}%", va="center", fontsize=15)
    axes[2].text(s+0.3, i+0.2, f"{s:.0f}%", va="center", fontsize=15)

axes[2].set_yticks(range(3))
axes[2].set_yticklabels(["Mina\n(max S3)","Madum\n(max S3)","Yagiasha\n(max S10)"], fontsize=15)
axes[2].set_xlabel("% of Road Segments")
axes[2].set_title("Cross-Typhoon Comparison\n(% Faster vs Slower at Max Signal)",
                  fontsize=13, fontweight="bold")
from matplotlib.patches import Patch
axes[2].legend(handles=[Patch(color="#2196F3", label="Faster"),
                         Patch(color="#F44336", label="Slower")],
               fontsize=18)
axes[2].grid(axis="x", alpha=0.3)

fig.suptitle("Key Quantitative Findings: Demand Suppression Dominates Road Network Response",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/dose_response_clean.png", dpi=260, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}/dose_response_clean.png")
