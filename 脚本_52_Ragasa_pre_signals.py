"""
Ragasa speed-shape with two pre-S8 signals circled.
Single-panel, minimal annotations.
"""
import os, glob
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Ellipse
from shapely import wkb as shapely_wkb
from datetime import datetime
import warnings; warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 220,
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "lines.linewidth": 2.2,
})

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"
OUT  = "/Users/helloling/workspace/thesis"

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

DAYS = [("2025-09-22","WORKDAY"),("2025-09-23","WORKDAY"),
        ("2025-09-24","WORKDAY"),("2025-09-25","WORKDAY")]
SIGNALS = [
    (datetime(2025,9,22,12,20), datetime(2025,9,22,21,40), 1, "S1"),
    (datetime(2025,9,22,21,40), datetime(2025,9,23,14,20), 3, "S3"),
    (datetime(2025,9,23,14,20), datetime(2025,9,24, 1,40), 8, "S8"),
    # S9 (1h) merged into S10 — too short to distinguish
    (datetime(2025,9,24, 1,40), datetime(2025,9,24,13,20),10, "S10"),
    (datetime(2025,9,24,13,20), datetime(2025,9,24,20,20), 8, "S8"),
    (datetime(2025,9,24,20,20), datetime(2025,9,25, 8,20), 3, "S3"),
    (datetime(2025,9,25, 8,20), datetime(2025,9,25,11,20), 1, "S1"),
]
SIG_COLORS = {1:"#aed581", 3:"#ffd54f", 8:"#ef5350", 9:"#ff7043", 10:"#b71c1c"}

print("Loading Ragasa slots...", flush=True)
records = []
for day, dtype in DAYS:
    folder = f"{FLOW}/{day}"
    if not os.path.exists(folder): continue
    avail = sorted([int(f.split("_slot")[1][:2])
                   for f in os.listdir(folder)
                   if "_slot" in f and f.endswith(".parquet")])
    for slot in avail:
        row = load_slot(day, slot, dtype)
        if row is None: continue
        row["dt"] = datetime.strptime(day, "%Y-%m-%d") + pd.Timedelta(minutes=slot*30)
        records.append(row)
ts = pd.DataFrame(records).sort_values("dt").reset_index(drop=True)
print(f"  {len(ts)} slots loaded")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 6))
fig.subplots_adjust(top=0.90, bottom=0.14, left=0.07, right=0.98)

# Signal shading
for s, e, sig, label in SIGNALS:
    ax.axvspan(s, e, alpha=0.12, color=SIG_COLORS.get(sig, "#ddd"),
               zorder=0, ec="none")
    mid = s + (e-s)/2
    ax.text(mid, 1.01, label, fontsize=10, ha="center", va="bottom",
            color="#C62828" if sig>=8 else "#E65100",
            fontweight="bold" if sig>=8 else "normal")

# Data gap (Sep 22)
gap_s = datetime(2025,9,22,5,0)
gap_e = datetime(2025,9,22,21,0)
ax.axvspan(gap_s, gap_e, alpha=0.18, color="#bbbbbb", zorder=0, ec="none")
ax.text(gap_s + (gap_e-gap_s)/2, 0.94, "Data gap",
        fontsize=10, ha="center", color="#666", style="italic")

# Baseline (dashed) and typhoon (solid)
ax.plot(ts["dt"], ts["mean_baseline"],
        color="#555", lw=1.6, ls="--", alpha=0.7,
        label="Workday baseline (expected)", zorder=3)
ax.plot(ts["dt"], ts["mean_speed"],
        color="#E53935", lw=2.6, alpha=0.95,
        label="Ragasa actual speed", zorder=4)

# ── Two key signal markers ──────────────────────────────────────────────────
def val_at(target):
    diffs = (ts["dt"] - target).abs()
    i = diffs.idxmin()
    return ts.loc[i, "mean_speed"], ts.loc[i, "mean_baseline"]

# Signal 1: morning peak 08:30 — actual ABOVE baseline at baseline's lowest point
sp1, bp1 = val_at(datetime(2025,9,23,8,30))
ax.add_patch(Ellipse((mdates.date2num(datetime(2025,9,23,8,30)),
                      (sp1+bp1)/2),
                     width=0.10, height=0.055,
                     fill=False, edgecolor="#1a6b1a", lw=2.4, zorder=6))
ax.annotate("(1) 08:30  actual > baseline  +0.021",
            xy=(datetime(2025,9,23,8,30), sp1+0.03),
            xytext=(datetime(2025,9,22, 4,0), 0.72),
            fontsize=10, color="#1a6b1a", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="#1a6b1a", lw=1.2))

# Signal 2: midday dip 13:00 — actual BELOW baseline
sp2, bp2 = val_at(datetime(2025,9,23,13,0))
ax.add_patch(Ellipse((mdates.date2num(datetime(2025,9,23,13,0)),
                      (sp2+bp2)/2),
                     width=0.10, height=0.055,
                     fill=False, edgecolor="#8a1f1f", lw=2.4, zorder=6))
ax.annotate("(2) 13:00  actual < baseline  −0.010",
            xy=(datetime(2025,9,23,13,0), sp2-0.025),
            xytext=(datetime(2025,9,23,15,0), 0.72),
            fontsize=10, color="#8a1f1f", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="#8a1f1f", lw=1.2))

# S8 vertical line
ax.axvline(datetime(2025,9,23,14,20), color="#C62828", lw=1.6, ls=":",
           alpha=0.7, zorder=5)

# Axes
ax.set_xlim(datetime(2025,9,22,0,0), datetime(2025,9,25,23,0))
ax.set_ylim(0.70, 1.04)
ax.set_ylabel("Mean Relative Speed")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")
ax.grid(alpha=0.2, lw=0.5)
ax.legend(loc="lower right", framealpha=0.9, edgecolor="#cccccc")
ax.set_title("Ragasa  —  Two pre-S8 early-warning signals during S3",
             fontweight="bold", loc="left")

out = f"{OUT}/图25d_Ragasa_pre_signals圈出.png"
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")
