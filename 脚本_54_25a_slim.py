"""
Slim version of figure 25a — diverging share-of-roads bars only.
Width matches 图25d (15"), height ~1/3 of 25d (~2.5").
For side-by-side display with 图25d in PPT slide 1.
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/Users/helloling/workspace/thesis/data"
OUT  = "/Users/helloling/workspace/thesis"
DEV_HI, DEV_LO = 0.03, -0.03
S8_TIME_HHMM = 14 + 20/60

ts = pd.read_parquet(f"{DATA}/yagiasha_road_timeseries.parquet")
ts["ds"] = pd.to_datetime(ts["dt"]).dt.strftime("%Y-%m-%d")
sep23 = ts[ts["ds"]=="2025-09-23"].copy()

agg = sep23.groupby("slot").agg(
    pct_faster=("dev", lambda x: (x>DEV_HI).mean()),
    pct_slower=("dev", lambda x: (x<DEV_LO).mean()),
).reset_index()
agg["hour"] = agg["slot"]*0.5

fig, ax = plt.subplots(figsize=(5, 6))
fig.subplots_adjust(top=0.92, bottom=0.10, left=0.16, right=0.96)

x = agg["hour"].values
w = 0.42
ax.bar(x,  agg["pct_faster"]*100, width=w, color="#2ca02c", alpha=0.88,
       label="Faster (dev > +0.03)")
ax.bar(x, -agg["pct_slower"]*100, width=w, color="#d62728", alpha=0.88,
       label="Slower (dev < −0.03)")
ax.axhline(0, color="black", lw=0.7)

# Phase shading + S8 line
ax.axvspan(8.0, 9.0,   color="gold",       alpha=0.22, zorder=0)
ax.axvspan(12.5, 13.5, color="lightcoral", alpha=0.22, zorder=0)
ax.axvline(S8_TIME_HHMM, color="#C62828", lw=1.6, ls=":", alpha=0.9)

# Shading labels
ax.annotate("Morning peak\n08:30", xy=(8.5, 47), ha="center", va="top",
            fontsize=8.5, color="#8a6d00", fontweight="bold")
ax.annotate("Midday dip\n13:00", xy=(13.0, -47), ha="center", va="bottom",
            fontsize=8.5, color="#8a1f1f", fontweight="bold")
ax.text(S8_TIME_HHMM+0.4, 35, "S8↑", color="#C62828", fontsize=9,
        fontweight="bold", ha="left", va="center")

ax.set_ylim(-50, 50)
yt = [-40,-20,0,20,40]
ax.set_yticks(yt); ax.set_yticklabels([f"{abs(v)}%" for v in yt], fontsize=10)
ax.set_xticks([0,4,8,12,16,20,24])
ax.set_xticklabels(["00","04","08","12","16","20","24"], fontsize=10)
ax.set_xlim(-0.5, 24.5)
ax.set_xlabel("Hour of day (HKT)", fontsize=11)
ax.set_ylabel("Share of roads\n← slower      faster →", fontsize=11)
ax.set_title("Ragasa S3→S8\n% roads faster vs slower",
             fontsize=12, fontweight="bold", pad=8)
ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92,
          edgecolor="#999", ncol=1)
ax.grid(axis="y", alpha=0.25)

out = f"{OUT}/图25a_slim_path占比.png"
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
