"""
三台风 during-typhoon speed shape — 参考脚本_07 p5 的风格，clean thesis figures.
纵坐标：mean relative_speed，同时画台风实测 + baseline。
"""
import os, glob, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from shapely import wkb as shapely_wkb
from datetime import datetime
import warnings; warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 260,
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "figure.titlesize": 18,
    "lines.linewidth": 2.4,
})

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"
OUT  = "/Users/helloling/workspace/thesis"

print("Loading lookups...", flush=True)
bl = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
bl_idx = bl.set_index(["day_type","slot","road_id"])["mean_speed"]
ep_lkp = ep.set_index("ep_key")["road_id"]
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

# ── Typhoon definitions ──────────────────────────────────────────────────────
TYPHOONS = {
    "Mitag": {
        "days": [("2025-09-17","WORKDAY"),("2025-09-18","WORKDAY"),
                 ("2025-09-19","WORKDAY"),("2025-09-20","SATURDAY")],
        "signals": [
            (datetime(2025,9,17,21,20), datetime(2025,9,19,9,20),  1, "S1"),
            (datetime(2025,9,19,9,20),  datetime(2025,9,20,9,20),  3, "S3"),
            (datetime(2025,9,20,9,20),  datetime(2025,9,20,10,40), 1, "S1"),
        ],
        "control_days": [("2025-09-16","WORKDAY")],
        "xlim": (datetime(2025,9,17,0,0), datetime(2025,9,20,23,0)),
    },
    "Ragasa": {
        "days": [("2025-09-22","WORKDAY"),("2025-09-23","WORKDAY"),
                 ("2025-09-24","WORKDAY"),("2025-09-25","WORKDAY")],
        "signals": [
            (datetime(2025,9,22,12,20), datetime(2025,9,22,21,40), 1, "S1↑"),
            (datetime(2025,9,22,21,40), datetime(2025,9,23,14,20), 3, "S3↑"),
            (datetime(2025,9,23,14,20), datetime(2025,9,24, 1,40), 8, "S8↑"),
            (datetime(2025,9,24, 1,40), datetime(2025,9,24, 2,40), 9, "S9"),
            (datetime(2025,9,24, 2,40), datetime(2025,9,24,13,20),10, "S10"),
            (datetime(2025,9,24,13,20), datetime(2025,9,24,20,20), 8, "S8↓"),
            (datetime(2025,9,24,20,20), datetime(2025,9,25, 8,20), 3, "S3↓"),
            (datetime(2025,9,25, 8,20), datetime(2025,9,25,11,20), 1, "S1↓"),
        ],
        "control_days": [("2025-09-16","WORKDAY")],
        "xlim": (datetime(2025,9,22,0,0), datetime(2025,9,25,23,0)),
    },
    "Matmo": {
        "days": [("2025-10-03","WORKDAY"),("2025-10-04","SATURDAY"),
                 ("2025-10-05","SUNDAY_HOLIDAY")],
        "signals": [
            (datetime(2025,10,3,19,40), datetime(2025,10,4,12,20), 1, "S1"),
            (datetime(2025,10,4,12,20), datetime(2025,10,5,15,40), 3, "S3"),
            (datetime(2025,10,5,15,40), datetime(2025,10,5,22,20), 1, "S1"),
        ],
        "control_days": [("2025-10-02","WORKDAY")],
        "xlim": (datetime(2025,10,3,0,0), datetime(2025,10,5,23,0)),
    },
}

SIG_COLORS = {1:"#aed581", 3:"#ffd54f", 8:"#ef5350", 9:"#ff7043", 10:"#b71c1c"}

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
    return {
        "slot": slot, "n_roads": valid.sum(),
        "mean_speed": obs[valid].mean(),
        "mean_baseline": bl_vals[valid].mean(),
    }

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading typhoon data...", flush=True)
all_ts = {}
all_ctrl = {}
for tname, tdef in TYPHOONS.items():
    records = []
    for day, dtype in tdef["days"]:
        folder = f"{FLOW}/{day}"
        if not os.path.exists(folder): continue
        avail = sorted([int(f.split("_slot")[1][:2])
                       for f in os.listdir(folder)
                       if "_slot" in f and f.endswith(".parquet")])
        for slot in avail:
            row = load_slot(day, slot, dtype)
            if row is None: continue
            base_dt = datetime.strptime(day, "%Y-%m-%d")
            row["dt"] = base_dt + pd.Timedelta(minutes=slot*30)
            row["day"] = day
            records.append(row)
    all_ts[tname] = pd.DataFrame(records).sort_values("dt").reset_index(drop=True)
    # Control
    ctrl = []
    for day, dtype in tdef["control_days"]:
        folder = f"{FLOW}/{day}"
        if not os.path.exists(folder): continue
        avail = sorted([int(f.split("_slot")[1][:2])
                       for f in os.listdir(folder)
                       if "_slot" in f and f.endswith(".parquet")])
        for slot in avail:
            row = load_slot(day, slot, dtype)
            if row is None: continue
            base_dt = datetime.strptime(day, "%Y-%m-%d")
            row["dt"] = base_dt + pd.Timedelta(minutes=slot*30)
            ctrl.append(row)
    all_ctrl[tname] = pd.DataFrame(ctrl).sort_values("dt").reset_index(drop=True)
    print(f"  {tname}: {len(all_ts[tname])} typhoon slots, {len(all_ctrl[tname])} control slots")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(18, 18), sharex=False)
fig.subplots_adjust(hspace=0.35, top=0.94, bottom=0.05, left=0.06, right=0.98)

for ax, tname in zip(axes, ["Mitag", "Ragasa", "Matmo"]):
    ts = all_ts[tname]
    ctrl = all_ctrl[tname]
    tdef = TYPHOONS[tname]
    signals = tdef["signals"]

    # Shade signal periods
    for s, e, sig, label in signals:
        ax.axvspan(s, e, alpha=0.12, color=SIG_COLORS.get(sig, "#ddd"),
                   zorder=0, ec="none")

    # Control day
    if len(ctrl) > 0:
        ax.plot(ctrl["dt"], ctrl["mean_speed"],
                color="#9E9E9E", lw=2.0, alpha=0.55,
                label="Control workday (mean speed)", zorder=2)

    # Typhoon baseline (dashed)
    ax.plot(ts["dt"], ts["mean_baseline"],
            color="#455A64", lw=1.8, ls="--", alpha=0.6,
            label="Workday baseline (expected)", zorder=3)

    # Typhoon actual (bold solid)
    ax.plot(ts["dt"], ts["mean_speed"],
            color="#E53935", lw=2.8, alpha=0.92,
            label="Typhoon (mean speed)", zorder=4)

    # Zero line
    ax.axhline(1.0, color="black", lw=0.6, ls=":", alpha=0.3, zorder=1)

    # Signal labels
    ybot, ytop = 0.68, 1.00
    for s, e, sig, label in signals:
        mid = s + (e-s)/2
        if sig >= 8:
            ax.annotate(label, xy=(mid, ytop), fontsize=12, ha="center", va="top",
                       color="#C62828", fontweight="bold")
        else:
            ax.annotate(label, xy=(mid, ytop), fontsize=11, ha="center", va="top",
                       color="#E65100")

    # Data gap annotation (Ragasa Sep 22)
    if tname == "Ragasa":
        gap_s = datetime(2025,9,22,5,0)
        gap_e = datetime(2025,9,22,21,0)
        ax.axvspan(gap_s, gap_e, alpha=0.10, color="#757575", zorder=0, ec="none")
        ax.text(gap_s + (gap_e-gap_s)/2, 0.95, "Data gap\n(09-22)", fontsize=11,
                ha="center", va="center", color="#757575", style="italic")

    ax.set_xlim(tdef["xlim"])
    ax.set_ylim(0.70, 1.02)
    ax.set_ylabel("Mean Relative Speed", fontsize=14)
    ax.set_title(tname, fontsize=16, fontweight="bold", loc="left",
                 color="#212121", pad=4)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")
    ax.grid(alpha=0.20, lw=0.5)
    ax.legend(fontsize=11, loc="lower right", framealpha=0.85,
              ncol=2, edgecolor="#cccccc")

fig.suptitle("During-Typhoon Speed Shape  —  Mean Relative Speed vs Baseline\n"
             "Three Hong Kong Typhoons, September–October 2025",
             fontsize=18, fontweight="bold")

out = f"{OUT}/图25_三台风速度形状.png"
fig.savefig(out, dpi=260, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved: {out}")
print("Done.")
