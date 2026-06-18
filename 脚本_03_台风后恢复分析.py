"""
Post-typhoon recovery analysis.
After signal cancellation, how quickly do roads return to baseline?
Look for pent-up demand surge (speeds drop below baseline as traffic rebounds).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")
# ── Readable thesis-figure style ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 260,
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
    "lines.linewidth": 2.4,
})


DATA = "/Users/helloling/workspace/thesis/data"
OUT  = "/Users/helloling/workspace/thesis"

# ── plotting helpers added by ChatGPT ────────────────────────────────────────
def normalize_road_category(x):
    """Normalize OSM-style road_category values so color dictionaries match."""
    if pd.isna(x):
        return "Other"
    s = str(x).strip().lower()
    mapping = {
        "motorway": "Motorway", "motorway_link": "Motorway",
        "trunk": "Trunk", "trunk_link": "Trunk",
        "primary": "Primary", "primary_link": "Primary",
        "secondary": "Secondary", "secondary_link": "Secondary",
        "tertiary": "Tertiary", "tertiary_link": "Tertiary",
        "residential": "Residential", "living_street": "Residential", "unclassified": "Residential",
        "service": "Service", "services": "Service",
    }
    return mapping.get(s, "Other")


def savefig_with_alias(fig, filename, *aliases, dpi=260):
    """Save the same figure under English and thesis Chinese filenames."""
    main_path = f"{OUT}/{filename}"
    fig.savefig(main_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    for alias in aliases:
        fig.savefig(f"{OUT}/{alias}", dpi=dpi, bbox_inches="tight", facecolor="white")
    return main_path

FLOW = f"{DATA}/flow_parquet2"

print("Loading data...")
bl    = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
ep    = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
bl_idx = bl.set_index(["day_type","slot","road_id"])["mean_speed"]
rr    = pd.read_parquet(f"{DATA}/road_registry.parquet")[["ep_key","road_id","road_category"]]
rr["road_category"] = rr["road_category"].apply(normalize_road_category)
road_cat = rr.drop_duplicates("road_id").set_index("road_id")["road_category"]

from shapely import wkb as shapely_wkb

def get_ep_key(wb):
    try:
        g = shapely_wkb.loads(bytes(wb))
        coords = list(g.coords) if g.geom_type == "LineString" else \
                 [c for line in g.geoms for c in line.coords]
        s = (round(coords[0][0], 4), round(coords[0][1], 4))
        e = (round(coords[-1][0], 4), round(coords[-1][1], 4))
        return str((min(s, e), max(s, e)))
    except:
        return None


def load_slot_stats(day, slot_num, day_type):
    """Load one slot → return per-road (road_id, obs, bl, dev, road_category)."""
    folder = f"{FLOW}/{day}"
    files  = [f for f in os.listdir(folder) if f"_slot{slot_num:02d}_" in f]
    if not files:
        return None
    df = pd.read_parquet(f"{folder}/{files[0]}",
                         columns=["relative_speed","geometry","road_closure","road_category"])
    df = df[df["road_closure"] != 1].copy()
    if len(df) < 50:
        return None
    df["ep_key"] = df["geometry"].apply(get_ep_key)
    df = df.merge(ep[["ep_key","road_id"]], on="ep_key", how="inner")
    if len(df) == 0:
        return None
    agg = df.groupby("road_id").agg(obs=("relative_speed","mean"),
                                     road_category=("road_category","first")).reset_index()
    agg = agg.set_index("road_id")
    idx = pd.MultiIndex.from_arrays([[day_type]*len(agg), [slot_num]*len(agg), agg.index],
                                    names=["day_type","slot","road_id"])
    agg["bl"]  = bl_idx.reindex(idx).values
    agg = agg.dropna(subset=["bl"])
    if len(agg) < 50:
        return None
    agg["dev"]  = agg["obs"] - agg["bl"]
    agg["slot"] = slot_num
    agg["ds"]   = day
    return agg


# ── Recovery windows ──────────────────────────────────────────────────────────
# Yagiasha: last signal (S1) ended 09-25 11:20 → analyze 09-25 11:00 to 09-26 23:30
# Mina:     last signal (S1) ended 09-20 10:40 → analyze 09-20 10:00 to 09-21 23:30
# Madum:    last signal (S1) ended 10-05 22:20 → analyze 10-05 22:00 to 10-07 23:30

# Day type helpers
HOLIDAYS = {"2025-10-01", "2025-10-07"}
def get_day_type(ds):
    import datetime
    d = datetime.date.fromisoformat(ds)
    if ds in HOLIDAYS:      return "SUNDAY_HOLIDAY"
    if d.weekday() == 6:    return "SUNDAY_HOLIDAY"
    if d.weekday() == 5:    return "SATURDAY"
    return "WORKDAY"


def build_window_ts(day_slot_pairs):
    """Build network timeseries from list of (day, slot, dt) tuples."""
    rows = []
    for day, slot, dt in day_slot_pairs:
        dtype = get_day_type(day)
        df = load_slot_stats(day, slot, dtype)
        if df is None:
            continue
        rows.append({
            "dt"        : dt,
            "mean_dev"  : df["dev"].mean(),
            "median_dev": df["dev"].median(),
            "p25"       : df["dev"].quantile(0.25),
            "p75"       : df["dev"].quantile(0.75),
            "pct_pos"   : (df["dev"] >  0.02).mean(),
            "pct_neg"   : (df["dev"] < -0.02).mean(),
            "n_roads"   : len(df),
        })
    return pd.DataFrame(rows)


def gen_slot_pairs(start_dt, end_dt):
    """Generate (day, slot, dt) every 30 min from start to end."""
    pairs = []
    cur = start_dt
    while cur <= end_dt:
        pairs.append((cur.strftime("%Y-%m-%d"),
                      int(cur.hour * 2 + cur.minute // 30),
                      cur))
        cur += pd.Timedelta(minutes=30)
    return pairs


# Yagiasha: full event + 2-day recovery
# Signal sequence: 09-22 12:20 → 09-25 11:20
yagi_all_pairs = gen_slot_pairs(
    pd.Timestamp("2025-09-22 12:00"),
    pd.Timestamp("2025-09-27 00:00"),
)
# Mina: full event + 1-day recovery
mina_all_pairs = gen_slot_pairs(
    pd.Timestamp("2025-09-17 21:00"),
    pd.Timestamp("2025-09-22 00:00"),
)
# Madum: full event + 2-day recovery
madum_all_pairs = gen_slot_pairs(
    pd.Timestamp("2025-10-03 19:00"),
    pd.Timestamp("2025-10-08 00:00"),
)

print("Building Yagiasha full-event + recovery time series...")
yagi_ts = build_window_ts(yagi_all_pairs)
yagi_ts = yagi_ts.sort_values("dt")
print(f"  {len(yagi_ts)} time points")

print("Building Mina full-event + recovery time series...")
mina_ts = build_window_ts(mina_all_pairs)
mina_ts = mina_ts.sort_values("dt")
print(f"  {len(mina_ts)} time points")

print("Building Madum full-event + recovery time series...")
madum_ts = build_window_ts(madum_all_pairs)
madum_ts = madum_ts.sort_values("dt")
print(f"  {len(madum_ts)} time points")


# ── Signal annotation helpers ─────────────────────────────────────────────────
YAGIASHA_SIGNALS = [
    (pd.Timestamp("2025-09-22 12:20"), pd.Timestamp("2025-09-22 21:40"), 1),
    (pd.Timestamp("2025-09-22 21:40"), pd.Timestamp("2025-09-23 14:20"), 3),
    (pd.Timestamp("2025-09-23 14:20"), pd.Timestamp("2025-09-24 01:40"), 8),
    (pd.Timestamp("2025-09-24 01:40"), pd.Timestamp("2025-09-24 02:40"), 9),
    (pd.Timestamp("2025-09-24 02:40"), pd.Timestamp("2025-09-24 13:20"), 10),
    (pd.Timestamp("2025-09-24 13:20"), pd.Timestamp("2025-09-24 20:20"), 8),
    (pd.Timestamp("2025-09-24 20:20"), pd.Timestamp("2025-09-25 08:20"), 3),
    (pd.Timestamp("2025-09-25 08:20"), pd.Timestamp("2025-09-25 11:20"), 1),
]
MINA_SIGNALS = [
    (pd.Timestamp("2025-09-17 21:20"), pd.Timestamp("2025-09-19 09:20"), 1),
    (pd.Timestamp("2025-09-19 09:20"), pd.Timestamp("2025-09-20 09:20"), 3),
    (pd.Timestamp("2025-09-20 09:20"), pd.Timestamp("2025-09-20 10:40"), 1),
]
MADUM_SIGNALS = [
    (pd.Timestamp("2025-10-03 19:40"), pd.Timestamp("2025-10-04 12:20"), 1),
    (pd.Timestamp("2025-10-04 12:20"), pd.Timestamp("2025-10-05 15:40"), 3),
    (pd.Timestamp("2025-10-05 15:40"), pd.Timestamp("2025-10-05 22:20"), 1),
]

SIG_COLORS = {1: "#FFF9C4", 3: "#FFE0B2", 8: "#FFCDD2", 9: "#F48FB1", 10: "#EF9A9A"}


def shade_signals(ax, signals, y_label_frac=0.9):
    ymin, ymax = ax.get_ylim()
    y_pos = ymin + (ymax - ymin) * y_label_frac
    for start, end, sig in signals:
        ax.axvspan(start, end, alpha=0.2 if sig < 8 else 0.35,
                   color=SIG_COLORS.get(sig, "#ccc"), zorder=0)
        mid = start + (end - start) / 2
        ax.text(mid, y_pos, f"S{sig}", fontsize=13.5, ha="center",
                color="darkred" if sig >= 8 else "darkorange", fontweight="bold")
    # mark signal end (recovery start)
    last_end = max(end for _, end, _ in signals)
    ax.axvline(last_end, color="darkgreen", lw=2, ls="--", alpha=0.8,
               label="Signals lowered / all clear")


def format_xaxis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=18)


# ── FIGURE 1: Full recovery plots — all 3 typhoons ───────────────────────────
print("Building recovery figures...")
fig, axes = plt.subplots(3, 2, figsize=(16, 14))

for row, (name, ts, signals) in enumerate([
    ("Ragasa (max Signal 10)", yagi_ts, YAGIASHA_SIGNALS),
    ("Mitag (max Signal 3)",   mina_ts, MINA_SIGNALS),
    ("Matmo (max Signal 3)",   madum_ts, MADUM_SIGNALS),
]):
    ax_dev = axes[row][0]
    ax_pct = axes[row][1]

    if ts.empty:
        for ax in [ax_dev, ax_pct]:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        continue

    # Panel A: Mean deviation
    ax_dev.plot(ts["dt"], ts["mean_dev"], color="#7B1FA2", lw=2, label="Network mean dev")
    ax_dev.fill_between(ts["dt"], ts["p25"], ts["p75"], alpha=0.15, color="#7B1FA2")
    ax_dev.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax_dev.set_ylabel("Speed Deviation (ratio)")
    ax_dev.set_title(f"{name}\nNetwork Mean Deviation", fontsize=15, fontweight="bold")
    ax_dev.legend(fontsize=18, loc="upper right")
    ax_dev.grid(alpha=0.3)
    shade_signals(ax_dev, signals)
    format_xaxis(ax_dev)

    # Panel B: % faster vs slower
    ax_pct.plot(ts["dt"], ts["pct_pos"], color="#2196F3", lw=2, label="Faster (dev>0.02)")
    ax_pct.plot(ts["dt"], ts["pct_neg"], color="#F44336", lw=2, label="Slower (dev<-0.02)")
    ax_pct.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.4)
    ax_pct.set_ylabel("Fraction of Roads")
    ax_pct.set_title(f"{name}\nFraction of Roads Faster / Slower", fontsize=15, fontweight="bold")
    ax_pct.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_pct.legend(fontsize=18)
    ax_pct.grid(alpha=0.3)
    shade_signals(ax_pct, signals)
    format_xaxis(ax_pct)

fig.suptitle(
    "Typhoon Road Network Response: Event and Recovery Period\n"
    "(Speed deviation relative to day-type baseline)",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
savefig_with_alias(fig, "recovery_full.png", "图06_三台风恢复动态全图.png")
plt.close()
print(f"Saved: {OUT}/recovery_full.png")


# ── FIGURE 2: Recovery rate analysis — time to return to baseline ─────────────
print("Computing recovery metrics...")

def compute_recovery_time(ts, all_clear_time, window_hours=48, threshold=0.02):
    """
    After all_clear_time, find first continuous 3-hour window where |mean_dev| < threshold.
    Returns hours-to-recovery or NaN.
    """
    post = ts[ts["dt"] >= all_clear_time].sort_values("dt").copy()
    if post.empty:
        return np.nan
    post["abs_dev"] = post["mean_dev"].abs()
    # rolling 6-slot (3h) mean
    post["smooth"] = post["abs_dev"].rolling(6, min_periods=3).mean()
    recovered = post[post["smooth"] < threshold]
    if recovered.empty:
        return np.nan
    first_rec = recovered["dt"].iloc[0]
    return (first_rec - all_clear_time).total_seconds() / 3600


yagi_clear = pd.Timestamp("2025-09-25 11:20")
mina_clear  = pd.Timestamp("2025-09-20 10:40")
madum_clear = pd.Timestamp("2025-10-05 22:20")

yagi_rec_h = compute_recovery_time(yagi_ts, yagi_clear)
mina_rec_h  = compute_recovery_time(mina_ts,  mina_clear)
madum_rec_h = compute_recovery_time(madum_ts, madum_clear)

print(f"Recovery time (hours to |mean_dev|<0.02):")
print(f"  Yagiasha: {yagi_rec_h:.1f}h" if not np.isnan(yagi_rec_h) else "  Yagiasha: not recovered in window")
print(f"  Mina:     {mina_rec_h:.1f}h" if not np.isnan(mina_rec_h) else "  Mina: not recovered in window")
print(f"  Madum:    {madum_rec_h:.1f}h" if not np.isnan(madum_rec_h) else "  Madum: not recovered in window")


# ── FIGURE 3: Focused Yagiasha recovery zoom ─────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

recovery_window = yagi_ts[yagi_ts["dt"] >= pd.Timestamp("2025-09-24 06:00")].copy()

ax1.axhline(0, color="black", lw=1, ls="--", alpha=0.7)
ax1.axvline(yagi_clear, color="darkgreen", lw=2, ls="--", label="All signals cleared (09-25 11:20)")
ax1.plot(recovery_window["dt"], recovery_window["mean_dev"], color="#7B1FA2", lw=2.5)
ax1.fill_between(recovery_window["dt"], recovery_window["p25"], recovery_window["p75"],
                 alpha=0.15, color="#7B1FA2", label="IQR")
# shade signal phases
for start, end, sig in YAGIASHA_SIGNALS:
    if end >= pd.Timestamp("2025-09-24 06:00"):
        ax1.axvspan(max(start, pd.Timestamp("2025-09-24 06:00")), end,
                    alpha=0.2, color=SIG_COLORS.get(sig, "#ccc"), zorder=0)

ax1.set_ylabel("Network Mean Deviation")
ax1.set_title("Ragasa: Signal Phase → Recovery Period\nNetwork-Level Speed Deviation",
              fontsize=18, fontweight="bold")
ax1.legend(fontsize=15)
ax1.grid(alpha=0.3)

ax2.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.4)
ax2.axvline(yagi_clear, color="darkgreen", lw=2, ls="--")
ax2.plot(recovery_window["dt"], recovery_window["pct_pos"], color="#2196F3", lw=2,
         label="Faster than baseline")
ax2.plot(recovery_window["dt"], recovery_window["pct_neg"], color="#F44336", lw=2,
         label="Slower than baseline")
ax2.set_ylabel("Fraction of Roads")
ax2.set_title("Fraction of Roads Faster / Slower Than Baseline", fontsize=13, fontweight="bold")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax2.legend(fontsize=15)
ax2.grid(alpha=0.3)
format_xaxis(ax2)

fig.suptitle("Ragasa Recovery Dynamics: From Peak Signal 10 to Post-Event", fontsize=15, fontweight="bold")
plt.tight_layout()
savefig_with_alias(fig, "recovery_yagiasha_zoom.png", "图07_叶加沙恢复期细节.png")
plt.close()
print(f"Saved: {OUT}/recovery_yagiasha_zoom.png")


# ── FIGURE 4: Road-category recovery profiles (Yagiasha) ─────────────────────
print("Computing road-category recovery profiles...")
# Build category-level timeseries for Yagiasha post-event
yagi_cat_rows = []
recovery_pairs = gen_slot_pairs(
    pd.Timestamp("2025-09-24 00:00"),
    pd.Timestamp("2025-09-27 00:00"),
)

# Load yagiasha timeseries and extract signal info
yagi_full = pd.read_parquet(f"{DATA}/yagiasha_road_timeseries.parquet")
yagi_full["road_category"] = road_cat.reindex(yagi_full["road_id"]).values
yagi_full["road_category"] = yagi_full["road_category"].apply(normalize_road_category)

# For recovery period (09-25 11:20 onward), compute per-category deviations
yagi_rec_period = yagi_full[yagi_full["dt"] >= yagi_clear].copy()

if not yagi_rec_period.empty:
    yagi_cat_ts = yagi_rec_period.groupby(["dt","road_category"]).agg(
        mean_dev=("dev","mean"), n=("dev","count")
    ).reset_index()

    fig, ax = plt.subplots(figsize=(14, 6))
    cat_colors = {
        "Motorway": "#D32F2F", "Trunk": "#F57C00", "Primary": "#FBC02D",
        "Secondary": "#388E3C", "Tertiary": "#1976D2", "Residential": "#7B1FA2",
        "Service": "#5D4037", "Other": "#78909C",
    }
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.7)
    ax.axvline(yagi_clear, color="darkgreen", lw=2, ls="--", label="All signals cleared")

    cats = yagi_cat_ts.groupby("road_category")["n"].mean()
    top_cats = cats[cats >= 100].sort_values(ascending=False).index[:7]
    for cat in top_cats:
        sub = yagi_cat_ts[yagi_cat_ts["road_category"] == cat].sort_values("dt")
        col = cat_colors.get(cat, "#666")
        ax.plot(sub["dt"], sub["mean_dev"], label=cat, color=col, lw=1.8)

    ax.set_title("Post-Ragasa Recovery by Road Category\n(From Signal 1 lowered to full recovery)",
                 fontsize=18, fontweight="bold")
    ax.set_xlabel("Date/Time (HKT)")
    ax.set_ylabel("Mean Speed Deviation from Baseline")
    ax.legend(fontsize=18, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)
    format_xaxis(ax)
    plt.tight_layout()
    savefig_with_alias(fig, "recovery_by_category.png", "图08_道路类别恢复曲线.png")
    plt.close()
    print(f"Saved: {OUT}/recovery_by_category.png")

print("\n=== Recovery Analysis Complete ===")
