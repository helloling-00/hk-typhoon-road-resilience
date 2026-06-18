"""
Research scope diagram: S8→S10→S8 during-typhoon phase.
Same style as 图25d, but highlights the S8-S10-S8 period.
Shows: Ragasa speed-shape with signal bands, focusing on Sep 23 14:20 → Sep 24 20:20.
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime
import warnings; warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 220,
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "lines.linewidth": 2.2,
})

DATA = "/Users/helloling/workspace/thesis/data"
OUT  = "/Users/helloling/workspace/thesis"

# ─── Load pre-computed timeseries ────────────────────────────────────────
print("Loading...", flush=True)
ts = pd.read_parquet(f"{DATA}/yagiasha_road_timeseries.parquet")
ts["dt"] = pd.to_datetime(ts["dt"])

# Filter to Ragasa period, aggregate to city mean per slot
ragasa = ts[(ts["dt"] >= "2025-09-22") & (ts["dt"] <= "2025-09-25")].copy()
city = ragasa.groupby("dt").agg(
    obs=("obs", "mean"),
    bl=("bl", "mean"),
).reset_index().sort_values("dt")

print(f"  {len(city)} slots, obs range: [{city['obs'].min():.3f}, {city['obs'].max():.3f}]")

# ─── Signal definitions ──────────────────────────────────────────────────
SIGNALS = [
    (datetime(2025,9,22,12,20), datetime(2025,9,22,21,40), 1, "S1"),
    (datetime(2025,9,22,21,40), datetime(2025,9,23,14,20), 3, "S3"),
    (datetime(2025,9,23,14,20), datetime(2025,9,24, 1,40), 8, "S8"),
    (datetime(2025,9,24, 1,40), datetime(2025,9,24,13,20),10, "S10"),
    (datetime(2025,9,24,13,20), datetime(2025,9,24,20,20), 8, "S8"),
    (datetime(2025,9,24,20,20), datetime(2025,9,25, 8,20), 3, "S3"),
    (datetime(2025,9,25, 8,20), datetime(2025,9,25,11,20), 1, "S1"),
]
SIG_COLORS = {1: "#aed581", 3: "#ffd54f", 8: "#ef5350", 10: "#b71c1c"}

# ─── Plot ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.subplots_adjust(top=0.86, bottom=0.16, left=0.09, right=0.97)

# Signal shading
for s, e, sig, label in SIGNALS:
    ax.axvspan(s, e, alpha=0.12, color=SIG_COLORS.get(sig, "#ddd"),
               zorder=0, ec="none")
    mid = s + (e - s) / 2
    if sig in (8, 10):
        ax.text(mid, 0.995, label, fontsize=11, ha="center", va="top",
                color="#C62828", fontweight="bold")

# Data gap (Sep 22)
gap_s = datetime(2025, 9, 22, 5, 0)
gap_e = datetime(2025, 9, 22, 21, 0)
ax.axvspan(gap_s, gap_e, alpha=0.18, color="#bbbbbb", zorder=0, ec="none")

# Speed lines
ax.plot(city["dt"], city["bl"], color="#555", lw=1.6, ls="--", alpha=0.7,
        label="Workday baseline (expected)", zorder=3)
ax.plot(city["dt"], city["obs"], color="#E53935", lw=2.6, alpha=0.95,
        label="Ragasa actual speed", zorder=4)

# ─── Highlight S8→S10→S8 zone ───────────────────────────────────────────
zone_s = datetime(2025, 9, 23, 14, 20)
zone_e = datetime(2025, 9, 24, 20, 20)
ax.axvspan(zone_s, zone_e, alpha=0.08, color="#C62828", zorder=1, ec="none")

# Bracket annotation
bracket_y = 0.76
ax.annotate("", xy=(mdates.date2num(zone_s), bracket_y),
            xytext=(mdates.date2num(zone_e), bracket_y),
            arrowprops=dict(arrowstyle="<->", color="#C62828", lw=2.6))
ax.text(zone_s + (zone_e - zone_s) / 2, bracket_y - 0.020,
        "S8 → S10 → S8", fontsize=13, ha="center",
        color="#C62828", fontweight="bold")

# Axes
ax.set_xlim(datetime(2025, 9, 23, 6, 0), datetime(2025, 9, 25, 4, 0))
ax.set_ylim(0.72, 1.02)
ax.set_ylabel("Mean Speed (km/h)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")
ax.grid(alpha=0.2, lw=0.5)
ax.legend(loc="lower right", framealpha=0.9, edgecolor="#cccccc")
ax.set_title("Ragasa  —  S8 → S10 → S8  (Sep 23 14:20 – Sep 24 20:20)",
             fontweight="bold", fontsize=13, loc="left")

out = f"{OUT}/图60_during_typhoon_scope.png"
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")
print("Done.")
