"""
Ragasa morning-peak compare: S3 (09-23) vs S10 (09-24) vs workday baseline.
X: morning hours (06:00-11:00). Y: mean relative speed.
"""
import os, glob
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely import wkb as shapely_wkb
from datetime import datetime, time
import warnings; warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 220,
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "lines.linewidth": 2.4,
})

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"
OUT  = "/Users/helloling/workspace/thesis"

MORN_SLOTS = list(range(12, 23))  # 06:00 -> 11:00, slot every 30 min

print("Loading lookups...", flush=True)
bl = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
bl_idx = bl.set_index(["day_type","slot","road_id"])["mean_speed"]
del bl

def get_ep_key(wb):
    try:
        g = shapely_wkb.loads(bytes(wb))
        coords = list(g.coords) if g.geom_type=="LineString" else \
                 [c for line in g.geoms for c in line.coords]
        s = (round(coords[0][0],4), round(coords[0][1],4))
        e = (round(coords[-1][0],4), round(coords[-1][1],4))
        return str((min(s,e), max(s,e)))
    except: return None

def load_slot(day, slot, day_type):
    pat = f"{FLOW}/{day}/traffic_flow_zoom15_{day}_slot{slot:02d}_*.parquet"
    fs = glob.glob(pat)
    if not fs: return None
    df = pd.read_parquet(fs[0],
                         columns=["relative_speed","geometry","road_closure"])
    df = df[df["road_closure"]!=1].dropna(subset=["relative_speed"])
    if len(df) < 50: return None
    df["ep_key"] = df["geometry"].apply(get_ep_key)
    df = df.merge(ep[["ep_key","road_id"]], on="ep_key", how="inner")
    if len(df) < 50: return None
    obs = df.groupby("road_id")["relative_speed"].mean()
    idx = pd.MultiIndex.from_arrays(
        [[day_type]*len(obs), [slot]*len(obs), obs.index],
        names=["day_type","slot","road_id"])
    bl_vals = bl_idx.reindex(idx).values
    valid = ~np.isnan(bl_vals)
    if valid.sum() < 50: return None
    return {"slot": slot,
            "n_roads": int(valid.sum()),
            "mean_speed": obs[valid].mean(),
            "mean_baseline": bl_vals[valid].mean()}

def load_day(day, day_type):
    out = []
    for s in MORN_SLOTS:
        r = load_slot(day, s, day_type)
        if r is not None: out.append(r)
    return pd.DataFrame(out)

print("Loading 09-23 (S3) and 09-24 (S10) morning slots...", flush=True)
s3  = load_day("2025-09-23", "WORKDAY")
s10 = load_day("2025-09-24", "WORKDAY")
print(f"  S3 slots:  {len(s3)} | S10 slots: {len(s10)}")

# Baseline curve from the union of slots present in either day
all_slots = sorted(set(s3["slot"]).union(s10["slot"]))
bl_curve = (bl_idx.loc["WORKDAY"]
            .groupby("slot").mean()
            .reindex(all_slots))

def slot_to_time(s):
    h, m = divmod(s*30, 60)
    return time(h, m)
xticks_slots = MORN_SLOTS
xticks_labels = [slot_to_time(s).strftime("%H:%M") for s in xticks_slots]

fig, ax = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(top=0.92, bottom=0.12, left=0.10, right=0.97)

ax.plot(bl_curve.index, bl_curve.values,
        color="#555", lw=1.8, ls="--", marker="o", ms=5,
        label="Workday baseline (expected)", zorder=3)
ax.plot(s3["slot"], s3["mean_speed"],
        color="#FB8C00", lw=2.4, marker="s", ms=6,
        label="Ragasa S3 actual (09-23)", zorder=4)
ax.plot(s10["slot"], s10["mean_speed"],
        color="#C62828", lw=2.4, marker="^", ms=6,
        label="Ragasa S10 actual (09-24)", zorder=5)

ax.set_xticks(xticks_slots)
ax.set_xticklabels(xticks_labels, rotation=0)
ax.set_xlim(MORN_SLOTS[0]-0.3, MORN_SLOTS[-1]+0.3)
ax.set_xlabel("Time of day (HKT)")
ax.set_ylabel("Mean Relative Speed")
ax.grid(alpha=0.25, lw=0.5)
ax.legend(loc="lower right", framealpha=0.92, edgecolor="#cccccc")
ax.set_title("Ragasa morning peak  —  S3 vs S10 vs workday baseline",
             fontweight="bold", loc="left")

bl_mean = bl_curve.mean()
s3_mean = s3["mean_speed"].mean()
s10_mean = s10["mean_speed"].mean()
stats_txt = (
    "Mean speed (06:00–11:00)\n"
    f"Baseline   {bl_mean:.3f}\n"
    f"S3 (09-23) {s3_mean:.3f}  (+{(s3_mean-bl_mean)/bl_mean*100:.1f}%)\n"
    f"S10 (09-24) {s10_mean:.3f}  (+{(s10_mean-bl_mean)/bl_mean*100:.1f}%)"
)
ax.text(0.015, 0.97, stats_txt, transform=ax.transAxes,
        fontsize=10.5, va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", fc="white",
                  ec="#bbbbbb", lw=0.8, alpha=0.92))

out = f"{OUT}/图25e_Ragasa_morning_S3_S10_baseline.png"
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")
