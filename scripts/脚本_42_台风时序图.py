"""
脚本_42_台风时序图.py
Network-Level Speed Deviation across Typhoon Stages (Ragasa, Sep 21-25)
Uses ep_key matching pipeline identical to 脚本_01.
"""

import os
import gc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from shapely import wkb as shapely_wkb
import warnings
warnings.filterwarnings("ignore")

DATA  = "/Users/helloling/workspace/thesis/data"
FLOW  = f"{DATA}/flow_parquet2"
OUT   = "/Users/helloling/workspace/thesis"

# ── Load lookups ─────────────────────────────────────────────────────────────
print("Loading lookups...", flush=True)
bl = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")

bl_idx = bl.set_index(["day_type","slot","road_id"])["mean_speed"]
ep_lookup = ep.set_index("ep_key")["road_id"]

print(f"  baseline: {len(bl):,} rows | ep_to_road: {len(ep):,} rows")

# ── ep_key extraction ────────────────────────────────────────────────────────
def get_ep_key(wb):
    try:
        g = shapely_wkb.loads(bytes(wb))
        coords = list(g.coords) if g.geom_type == "LineString" else \
                 [c for line in g.geoms for c in line.coords]
        s = (round(coords[0][0], 4), round(coords[0][1], 4))
        e = (round(coords[-1][0], 4), round(coords[-1][1], 4))
        return str((min(s, e), max(s, e)))
    except Exception:
        return None

_wkb_ep_cache = {}

def build_wkb_cache(day, sample_slots=(0, 12, 24, 36)):
    folder = f"{FLOW}/{day}"
    if not os.path.exists(folder):
        return {}
    uniq = {}
    for s in sample_slots:
        files = [f for f in os.listdir(folder) if f"_slot{s:02d}_" in f]
        if not files:
            continue
        df = pd.read_parquet(f"{folder}/{files[0]}", columns=["geometry"])
        for g in df["geometry"]:
            if g is not None:
                key = id(bytes(g)[:8])
                if key not in uniq:
                    uniq[key] = g
    ep_map = {}
    for g in uniq.values():
        epk = get_ep_key(g)
        if epk:
            ep_map[bytes(g)] = epk
    _wkb_ep_cache[day] = ep_map
    return ep_map

def compute_slot_deviation(day, slot_num, day_type, wkb_ep):
    folder = f"{FLOW}/{day}"
    if not os.path.exists(folder):
        return None
    files = [f for f in os.listdir(folder) if f"_slot{slot_num:02d}_" in f]
    if not files:
        return None
    try:
        df = pd.read_parquet(f"{folder}/{files[0]}",
                             columns=["relative_speed","geometry","road_closure"])
        df = df[df["road_closure"] != 1].copy()
        if len(df) < 50:
            return None

        def lookup_epk(g):
            if g is None:
                return None
            b = bytes(g)
            if b in wkb_ep:
                return wkb_ep[b]
            epk = get_ep_key(g)
            if epk:
                wkb_ep[b] = epk
            return epk

        df["ep_key"] = df["geometry"].apply(lookup_epk)
        df = df.merge(ep[["ep_key","road_id"]], on="ep_key", how="inner")
        if len(df) < 50:
            return None

        agg = df.groupby("road_id")["relative_speed"].mean().rename("obs").reset_index()
        agg = agg.set_index("road_id")

        idx = pd.MultiIndex.from_arrays(
            [[day_type]*len(agg), [slot_num]*len(agg), agg.index],
            names=["day_type","slot","road_id"])
        agg["baseline"] = bl_idx.reindex(idx).values
        agg = agg.dropna(subset=["baseline"])
        if len(agg) < 100:
            return None
        agg["dev"] = agg["obs"] - agg["baseline"]
        return agg
    except Exception as e:
        print(f"    slot {slot_num} error: {e}")
        return None

# ── Ragasa days & day types ───────────────────────────────────────────────────
DAYS = [
    ("2025-09-21", "SUNDAY_HOLIDAY"),   # Sunday
    ("2025-09-22", "WORKDAY"),           # Monday (S1 12:20, S3 21:40)
    ("2025-09-23", "WORKDAY"),           # Tuesday (S8 14:20)
    ("2025-09-24", "WORKDAY"),           # Wednesday (S10 02:40, S8 13:20, S3 20:20)
    ("2025-09-25", "WORKDAY"),           # Thursday (S1 08:20, clear 11:20)
]

# ── Compute per-slot network mean deviation ───────────────────────────────────
print("\nComputing time series...", flush=True)
records = []

for day, day_type in DAYS:
    folder = f"{FLOW}/{day}"
    if not os.path.exists(folder):
        print(f"  {day}: folder missing, skip")
        continue
    wkb_ep = build_wkb_cache(day)
    all_slots = sorted([int(f.split("_slot")[1][:2])
                        for f in os.listdir(folder)
                        if "_slot" in f and f.endswith(".parquet")])
    print(f"  {day} ({day_type}): {len(all_slots)} slots, {len(wkb_ep)} cached WKBs")

    for s in all_slots:
        res = compute_slot_deviation(day, s, day_type, wkb_ep)
        if res is None:
            continue
        # Datetime: midnight of day + s*30min
        from datetime import datetime, timedelta
        base_dt = datetime.strptime(day, "%Y-%m-%d")
        dt = base_dt + timedelta(minutes=s*30)
        records.append({
            "datetime"  : dt,
            "day"       : day,
            "slot"      : s,
            "mean_dev"  : float(res["dev"].mean()),
            "median_dev": float(res["dev"].median()),
            "p25"       : float(res["dev"].quantile(0.25)),
            "p75"       : float(res["dev"].quantile(0.75)),
            "pct_faster": float((res["dev"] > 0).mean()),
            "n_roads"   : len(res),
        })
    gc.collect()

print(f"\nTotal slots computed: {len(records)}")
if not records:
    print("ERROR: No data computed. Check data paths.")
    exit(1)

ts = pd.DataFrame(records).sort_values("datetime").reset_index(drop=True)
ts["datetime_hour"] = ts["slot"] * 0.5  # fractional hour within day (for ref)
print(ts[["datetime","mean_dev","n_roads"]].to_string())

# ── Save intermediate data ────────────────────────────────────────────────────
ts.to_csv(f"{OUT}/ragasa_timeseries.csv", index=False)
print(f"\nSaved: ragasa_timeseries.csv")

# ── Signal phase definitions (HKT) ───────────────────────────────────────────
from datetime import datetime
SIGNALS = [
    (datetime(2025,9,22,12,20), "S1↑"),
    (datetime(2025,9,22,21,40), "S3↑"),
    (datetime(2025,9,23,14,20), "S8↑"),
    (datetime(2025,9,24, 1,40), "S9↑"),
    (datetime(2025,9,24, 2,40), "S10↑"),
    (datetime(2025,9,24,13,20), "S8↓"),
    (datetime(2025,9,24,20,20), "S3↓"),
    (datetime(2025,9,25, 8,20), "S1↓"),
    (datetime(2025,9,25,11,20), "Clear"),
]

S1_raise  = datetime(2025,9,22,12,20)
S3_raise  = datetime(2025,9,22,21,40)
S8_raise  = datetime(2025,9,23,14,20)
S10_raise = datetime(2025,9,24, 2,40)
S8_lower  = datetime(2025,9,24,13,20)
S3_lower  = datetime(2025,9,24,20,20)
S1_lower  = datetime(2025,9,25, 8,20)
ALL_CLEAR = datetime(2025,9,25,11,20)

# Phase boundaries for shading
PRE_START   = datetime(2025,9,21,18, 0)   # show from Sun evening
EVENT_START = S1_raise
EVENT_END   = ALL_CLEAR
POST_END    = datetime(2025,9,25,23,30)

# ── Plot ─────────────────────────────────────────────────────────────────────
print("\nPlotting...", flush=True)

plt.rcParams.update({
    "figure.dpi": 140,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "font.family": "DejaVu Sans",
})

# Dark theme colors
C_BG     = "#1a1a2e"
C_PANEL  = "#16213e"
C_ACCENT = "#0f8b8d"
C_YELLOW = "#ffc800"
C_WHITE  = "#f5f5f5"
C_GRAY   = "#aaaaaa"
C_RED    = "#e05252"

fig, ax = plt.subplots(figsize=(13, 5), facecolor=C_BG)
ax.set_facecolor(C_PANEL)
for spine in ax.spines.values():
    spine.set_color(C_GRAY)
ax.tick_params(colors=C_WHITE)
ax.xaxis.label.set_color(C_WHITE)
ax.yaxis.label.set_color(C_WHITE)
ax.title.set_color(C_WHITE)

# ── Shade phases ─────────────────────────────────────────────────────────────
xmin = ts["datetime"].min()
xmax = ts["datetime"].max()

# Pre-event shading (light blue)
ax.axvspan(xmin, EVENT_START,
           alpha=0.12, color="#4fc3f7", zorder=0)
# During-event shading (light red/orange)
ax.axvspan(EVENT_START, EVENT_END,
           alpha=0.12, color="#ef5350", zorder=0)
# Post-event shading (light green)
ax.axvspan(EVENT_END, xmax,
           alpha=0.12, color="#66bb6a", zorder=0)

# ── Phase label bands (top strip) ────────────────────────────────────────────
y_top  = ax.get_ylim()[1]  # we'll reposition after plot
PHASE_LABELS = [
    (xmin,        EVENT_START, "Pre-Event",    "#4fc3f7"),
    (EVENT_START, EVENT_END,   "During Event", "#ef5350"),
    (EVENT_END,   xmax,        "Post-Event",   "#66bb6a"),
]

# ── Baseline zero ─────────────────────────────────────────────────────────────
ax.axhline(0, color=C_GRAY, linewidth=0.8, linestyle="--", zorder=1)

# ── Signal vertical lines ─────────────────────────────────────────────────────
sig_colors = {
    "S1↑": "#a5d6a7", "S3↑": "#ffcc02", "S8↑": "#ef5350",
    "S10↑": "#b71c1c", "S8↓": "#ef5350", "S3↓": "#ffcc02",
    "S1↓": "#a5d6a7", "Clear": "#81c784",
}
for dt, label in SIGNALS:
    if dt < xmin or dt > xmax:
        continue
    c = sig_colors.get(label, C_GRAY)
    ax.axvline(dt, color=c, linewidth=0.9, linestyle=":", alpha=0.75, zorder=2)

# ── IQR band ─────────────────────────────────────────────────────────────────
ax.fill_between(ts["datetime"], ts["p25"], ts["p75"],
                alpha=0.20, color=C_ACCENT, zorder=3, label="IQR (25th–75th %ile)")

# ── Main line: mean deviation ─────────────────────────────────────────────────
ts_smooth = ts.set_index("datetime")["mean_dev"].rolling(3, center=True, min_periods=1).mean()
ax.plot(ts["datetime"], ts_smooth, color=C_ACCENT,
        linewidth=2.2, zorder=5, label="Network mean deviation (smoothed)")
ax.plot(ts["datetime"], ts["mean_dev"], color=C_ACCENT,
        linewidth=0.7, alpha=0.35, zorder=4)

# ── Signal phase labels on x-axis ────────────────────────────────────────────
# Show only key signal transitions
key_signals = [
    (S1_raise,  "S1",  "#a5d6a7"),
    (S3_raise,  "S3",  "#ffcc02"),
    (S8_raise,  "S8",  "#ef5350"),
    (S10_raise, "S10", "#b71c1c"),
    (S8_lower,  "S8↓", "#ef5350"),
    (S3_lower,  "S3↓", "#ffcc02"),
    (ALL_CLEAR, "Clear","#81c784"),
]
ymin_ax = ax.get_ylim()[0]
for dt, label, c in key_signals:
    if dt < xmin or dt > xmax:
        continue
    ax.annotate(label, xy=(dt, ax.get_ylim()[0]),
                xytext=(0, 5), textcoords="offset points",
                fontsize=7.5, color=c, ha="center", fontweight="bold",
                rotation=0, zorder=6)

# ── Sep 22 data gap annotation ────────────────────────────────────────────────
gap_start = datetime(2025,9,22, 5, 0)
gap_end   = datetime(2025,9,22,21, 0)
ax.axvspan(gap_start, gap_end, alpha=0.18, color="#888888",
           zorder=1, hatch="///", linewidth=0)
ax.text(gap_start + (gap_end-gap_start)/2,
        (ax.get_ylim()[0] + ax.get_ylim()[1]) * 0.25,
        "Data gap\n(09-22)", color="#aaaaaa",
        fontsize=7, ha="center", va="center", style="italic")

# ── Behavioral annotations ────────────────────────────────────────────────────
# "Anticipatory mobility" — Sep 22 early morning (pre-gap, pre-S1)
ant_x = datetime(2025,9,22, 2, 0)
close = (ts["datetime"] - ant_x).abs()
if close.min().total_seconds() < 7200:
    ant_y = ts.loc[close.idxmin(), "mean_dev"]
    ax.annotate("Anticipatory\nmobility",
                xy=(ant_x, ant_y), xytext=(ant_x, ant_y + 0.018),
                fontsize=8.5, color="#4fc3f7", fontweight="bold",
                ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-|>", color="#4fc3f7", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.2", fc=C_PANEL,
                          ec="#4fc3f7", alpha=0.85))

# "Network clearance" — peak during-event period (Sep 23 17:00)
clear_x = datetime(2025,9,23,17, 0)
close = (ts["datetime"] - clear_x).abs()
if close.min().total_seconds() < 7200:
    clear_y = ts.loc[close.idxmin(), "mean_dev"]
    ax.annotate("Network\nclearance",
                xy=(clear_x, clear_y), xytext=(clear_x, clear_y + 0.02),
                fontsize=8.5, color="#ef5350", fontweight="bold",
                ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-|>", color="#ef5350", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.2", fc=C_PANEL,
                          ec="#ef5350", alpha=0.85))

# "Rapid recovery" — post-clear afternoon (Sep 25 14:00)
rec_x = datetime(2025,9,25,14, 0)
close = (ts["datetime"] - rec_x).abs()
if close.min().total_seconds() < 7200:
    rec_y = ts.loc[close.idxmin(), "mean_dev"]
    ax.annotate("Rapid\nrecovery",
                xy=(rec_x, rec_y), xytext=(rec_x, rec_y + 0.018),
                fontsize=8.5, color="#66bb6a", fontweight="bold",
                ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-|>", color="#66bb6a", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.2", fc=C_PANEL,
                          ec="#66bb6a", alpha=0.85))

# ── Phase strip at top ────────────────────────────────────────────────────────
ylims = ax.get_ylim()
strip_y = ylims[1] * 0.88
for x0, x1, label, c in PHASE_LABELS:
    # Clamp to data range
    x0c = max(x0, ts["datetime"].min())
    x1c = min(x1, ts["datetime"].max())
    if x0c >= x1c:
        continue
    xmid = x0c + (x1c - x0c)/2
    ax.text(xmid, strip_y, label, color=c,
            fontsize=9, fontweight="bold", ha="center", va="center",
            transform=ax.transData, alpha=0.9)

# ── X-axis ticks: one per day + major signal times ───────────────────────────
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
plt.setp(ax.get_xticklabels(), rotation=0, ha="center", color=C_WHITE, fontsize=8.5)

# ── Axes labels ───────────────────────────────────────────────────────────────
ax.set_ylabel("Network Mean Speed Deviation\n(typhoon − baseline, rel. speed)", color=C_WHITE)
ax.set_title("Network-Level Speed Deviation across Typhoon Stages  ·  Ragasa (Sep 2025)",
             color=C_WHITE, fontsize=12, fontweight="bold", pad=8)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    Line2D([0],[0], color=C_ACCENT, lw=2.2, label="Network mean deviation"),
    mpatches.Patch(facecolor=C_ACCENT, alpha=0.20, label="IQR (25–75th %ile)"),
    Line2D([0],[0], color=C_GRAY, lw=0.8, ls="--", label="Baseline (0)"),
    mpatches.Patch(facecolor="#4fc3f7", alpha=0.25, label="Pre-event"),
    mpatches.Patch(facecolor="#ef5350", alpha=0.25, label="During event"),
    mpatches.Patch(facecolor="#66bb6a", alpha=0.25, label="Post-event"),
]
leg = ax.legend(handles=legend_handles, loc="lower left",
                fontsize=8, framealpha=0.7,
                facecolor=C_PANEL, edgecolor=C_GRAY,
                labelcolor=C_WHITE, ncol=3)

plt.tight_layout(pad=1.0)
out = f"{OUT}/图42_台风时序网络偏差.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=C_BG)
print(f"\nSaved: {out}")
plt.close()
