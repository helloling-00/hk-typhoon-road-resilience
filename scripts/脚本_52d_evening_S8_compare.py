"""
Two S8 evening peaks: rising S8 (09-23, before S10) vs falling S8 (09-24, after S10).
X: evening hours (16:00-22:00). Y: mean relative speed.
"""
import os, glob
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely import wkb as shapely_wkb
from datetime import time
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

EVE_SLOTS = list(range(32, 45))  # 16:00 -> 22:00, 30-min slot

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
    return {"slot": slot, "n_roads": int(valid.sum()),
            "mean_speed": obs[valid].mean(),
            "mean_baseline": bl_vals[valid].mean()}

def load_day(day, day_type, slots):
    out=[]
    for s in slots:
        r=load_slot(day,s,day_type)
        if r is not None: out.append(r)
    return pd.DataFrame(out)

print("Loading 09-23 evening (rising S8) and 09-24 evening (falling S8)...", flush=True)
rs8 = load_day("2025-09-23", "WORKDAY", EVE_SLOTS)   # rising  S8: S3 -> S8 -> S10
fs8 = load_day("2025-09-24", "WORKDAY", EVE_SLOTS)   # falling S8: S10 -> S8 -> S3
print(f"  rising S8 slots:  {len(rs8)} | falling S8 slots: {len(fs8)}")

bl_curve = (bl_idx.loc["WORKDAY"].groupby("slot").mean()
            .reindex(EVE_SLOTS))

def stt(s):
    h,m=divmod(s*30,60); return f"{h:02d}:{m:02d}"
xticks_labels = [stt(s) for s in EVE_SLOTS]

fig, ax = plt.subplots(figsize=(11, 6))
fig.subplots_adjust(top=0.92, bottom=0.12, left=0.09, right=0.97)

# S8 onset / offset markers — 09-23 14:20 (rising S8 starts at slot 28.67)
# rising S8 runs through entire evening; falling S8 ends 20:20 (slot 40.67)
ax.axvspan(EVE_SLOTS[0]-0.5, EVE_SLOTS[-1]+0.5, color="#fff3e0", alpha=0.0, zorder=0)

# Falling S8 ends at 20:20 → S3 zone after that
ax.axvline(40 + 20/60*2, color="#888", lw=1.2, ls=":", alpha=0.6, zorder=2)
ax.text(40 + 20/60*2 + 0.15, 0.99, "falling S8 → S3\n(20:20)",
        fontsize=9, color="#666", ha="left", va="top", transform=ax.get_xaxis_transform())

ax.plot(bl_curve.index, bl_curve.values,
        color="#555", lw=1.8, ls="--", marker="o", ms=5,
        label="Workday baseline (expected)", zorder=3)
ax.plot(rs8["slot"], rs8["mean_speed"],
        color="#1976D2", lw=2.4, marker="s", ms=6,
        label="Rising S8 — 09-23 evening (S3→S8→S10)", zorder=4)
ax.plot(fs8["slot"], fs8["mean_speed"],
        color="#6A1B9A", lw=2.4, marker="^", ms=6,
        label="Falling S8 — 09-24 evening (S10→S8→S3)", zorder=5)

ax.set_xticks(EVE_SLOTS)
ax.set_xticklabels(xticks_labels, rotation=0)
ax.set_xlim(EVE_SLOTS[0]-0.3, EVE_SLOTS[-1]+0.3)
ax.set_xlabel("Time of day (HKT)")
ax.set_ylabel("Mean Relative Speed")
ax.grid(alpha=0.25, lw=0.5)
ax.legend(loc="lower right", framealpha=0.92, edgecolor="#cccccc")
ax.set_title("Ragasa evening peak  —  rising S8 vs falling S8 vs workday baseline",
             fontweight="bold", loc="left")

bl_mean = bl_curve.mean()
r_mean  = rs8["mean_speed"].mean()
f_mean  = fs8["mean_speed"].mean()
stats_txt = (
    "Mean speed (16:00-22:00)\n"
    f"Baseline    {bl_mean:.3f}\n"
    f"Rising  S8  {r_mean:.3f}  ({(r_mean-bl_mean)/bl_mean*100:+.1f}%)\n"
    f"Falling S8  {f_mean:.3f}  ({(f_mean-bl_mean)/bl_mean*100:+.1f}%)"
)
ax.text(0.015, 0.97, stats_txt, transform=ax.transAxes,
        fontsize=10.5, va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", fc="white",
                  ec="#bbbbbb", lw=0.8, alpha=0.92))

out = f"{OUT}/图25f_Ragasa_evening_S8_compare.png"
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")

# ---- numerical table ----
m = pd.DataFrame({"slot": EVE_SLOTS,
                  "time": [stt(s) for s in EVE_SLOTS],
                  "bl":   bl_curve.values})
m = m.merge(rs8[["slot","mean_speed","n_roads"]].rename(columns={"mean_speed":"rS8","n_roads":"n_r"}),on="slot",how="left")
m = m.merge(fs8[["slot","mean_speed","n_roads"]].rename(columns={"mean_speed":"fS8","n_roads":"n_f"}),on="slot",how="left")
m["d_r"]   = m["rS8"] - m["bl"]
m["d_f"]   = m["fS8"] - m["bl"]
m["d_f_r"] = m["fS8"] - m["rS8"]
print("\n=== Evening slot-by-slot ===")
print(m.to_string(index=False,
      formatters={"bl":"{:.3f}".format,"rS8":"{:.3f}".format,"fS8":"{:.3f}".format,
                  "d_r":"{:+.3f}".format,"d_f":"{:+.3f}".format,"d_f_r":"{:+.3f}".format,
                  "n_r":"{:.0f}".format,"n_f":"{:.0f}".format}))

print("\n=== Aggregate (16:00-22:00) ===")
print(f"  Baseline mean : {bl_mean:.3f}   trough={m['bl'].min():.3f} @ {m.loc[m['bl'].idxmin(),'time']}")
print(f"  Rising  S8    : {r_mean:.3f}   trough={m['rS8'].min():.3f} @ {m.loc[m['rS8'].idxmin(),'time']}")
print(f"  Falling S8    : {f_mean:.3f}   trough={m['fS8'].min():.3f} @ {m.loc[m['fS8'].idxmin(),'time']}")
print(f"  d(r-bl) mean  : {m['d_r'].mean():+.3f}")
print(f"  d(f-bl) mean  : {m['d_f'].mean():+.3f}")
print(f"  d(f-r) mean   : {m['d_f_r'].mean():+.3f}")
print(f"  Road coverage avg: rising n={m['n_r'].mean():.0f} | falling n={m['n_f'].mean():.0f}")

m.to_csv(f"{DATA}/../evening_S8_rising_vs_falling.csv", index=False)
print(f"\nCSV saved.")
