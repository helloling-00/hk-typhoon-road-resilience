"""
Polarization / speed stratification analysis.
Core thesis finding: during typhoon, high-speed roads get faster (demand suppression),
low-speed roads get slower (supply disruption) — a divergence from baseline.
"""

import os, gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
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


# ── signal stage mapping (all times in HKT) ──────────────────────────────────
# Yagiasha: full signal sequence
YAGIASHA_SIGNALS = [
    # (start_dt, end_dt, signal)
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

def assign_signal(dt, signal_stages):
    for start, end, sig in signal_stages:
        if start <= dt < end:
            return sig
    return 0  # no signal


# ── load Yagiasha timeseries ─────────────────────────────────────────────────
print("Loading Yagiasha timeseries...")
yagi = pd.read_parquet(f"{DATA}/yagiasha_road_timeseries.parquet")
yagi["signal"] = yagi["dt"].apply(lambda t: assign_signal(t, YAGIASHA_SIGNALS))

# keep only typhoon periods (signal > 0)
yagi_ty = yagi[yagi["signal"] > 0].copy()
print(f"  typhoon obs: {len(yagi_ty):,}")
print("  signal counts:", yagi_ty["signal"].value_counts().sort_index().to_dict())

# ── road-level aggregates by signal level ────────────────────────────────────
# For each road, compute mean baseline and mean deviation per signal level
road_bl = yagi.groupby("road_id")["bl"].mean().rename("mean_bl")  # stable baseline

# Per-signal road stats
def road_stats_for_signal(df, sig):
    sub = df[df["signal"] == sig]
    if len(sub) == 0:
        return pd.DataFrame()
    return sub.groupby("road_id").agg(
        mean_dev=("dev", "mean"),
        n_obs=("dev", "count"),
    ).join(road_bl)

print("Computing road-level stats by signal level...")
road_stats = {}
for sig in [1, 3, 8, 9, 10]:
    rs = road_stats_for_signal(yagi_ty, sig)
    if len(rs) >= 100:
        road_stats[sig] = rs
        print(f"  Signal {sig}: {len(rs):,} roads, mean_dev={rs['mean_dev'].mean():+.4f}")

# ── also load road category from registry ────────────────────────────────────
rr = pd.read_parquet(f"{DATA}/road_registry.parquet")[["ep_key","road_id","road_category"]]
rr["road_category"] = rr["road_category"].apply(normalize_road_category)
road_cat = rr.drop_duplicates("road_id").set_index("road_id")["road_category"]
for sig, rs in road_stats.items():
    road_stats[sig]["road_category"] = road_cat.reindex(rs.index)

# ── build Mina and Madum timeseries on the fly ───────────────────────────────
# We'll build them from flow parquets + baseline
print("\nBuilding Mina and Madum timeseries...")
from shapely import wkb as shapely_wkb
bl    = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
ep    = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
bl_idx = bl.set_index(["day_type","slot","road_id"])["mean_speed"]
ep_lookup = ep.set_index("ep_key")["road_id"]
FLOW  = f"{DATA}/flow_parquet2"

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

_wkb_cache = {}

def load_slot_dev(day, slot_num, day_type):
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
    agg = df.groupby("road_id").agg(obs=("relative_speed","mean")).reset_index()
    agg2 = agg.set_index("road_id")
    idx = pd.MultiIndex.from_arrays([[day_type]*len(agg2), [slot_num]*len(agg2), agg2.index],
                                    names=["day_type","slot","road_id"])
    agg2["bl"]  = bl_idx.reindex(idx).values
    agg2 = agg2.dropna(subset=["bl"])
    if len(agg2) < 50:
        return None
    agg2["dev"] = agg2["obs"] - agg2["bl"]
    agg2["slot"] = slot_num
    agg2["ds"]   = day
    return agg2[["obs","bl","dev","slot","ds"]]


def build_timeseries(signal_stages, day_type_map):
    """Build road-level timeseries for a typhoon given its signal stages."""
    rows = []
    for start, end, sig in signal_stages:
        # iterate over 30-min slots in this window
        cur = start
        while cur < end:
            day = cur.strftime("%Y-%m-%d")
            slot = int(cur.hour * 2 + cur.minute // 30)
            dt = cur
            dtype = day_type_map.get(day, "WORKDAY")
            df = load_slot_dev(day, slot, dtype)
            if df is not None and len(df) > 0:
                df["signal"] = sig
                df["dt"] = dt
                rows.append(df)
            cur += pd.Timedelta(minutes=30)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows)


# Day type for Mina (signal days 09-17 to 09-20)
mina_dtype_map = {
    "2025-09-17": "WORKDAY",   # Wed
    "2025-09-18": "WORKDAY",   # Thu
    "2025-09-19": "WORKDAY",   # Fri
    "2025-09-20": "SATURDAY",  # Sat
}
# Day type for Madum (signal days 10-03 to 10-05)
madum_dtype_map = {
    "2025-10-03": "WORKDAY",   # Fri
    "2025-10-04": "SATURDAY",  # Sat
    "2025-10-05": "SUNDAY_HOLIDAY",  # Sun
}

print("  Building Mina timeseries (Signals 1, 3)...")
mina_ts_all = build_timeseries(MINA_SIGNALS, mina_dtype_map)
print(f"  Mina: {len(mina_ts_all):,} obs")

print("  Building Madum timeseries (Signals 1, 3)...")
madum_ts_all = build_timeseries(MADUM_SIGNALS, madum_dtype_map)
print(f"  Madum: {len(madum_ts_all):,} obs")

# Road stats for Mina/Madum
def road_stats_from_ts(ts_df, sig):
    sub = ts_df[ts_df["signal"] == sig]
    if len(sub) < 100:
        return pd.DataFrame()
    return sub.groupby("road_id").agg(
        mean_dev=("dev","mean"), n_obs=("dev","count"),
    ).join(road_bl, how="left")

mina_road_stats = {}
for sig in [1, 3]:
    rs = road_stats_from_ts(mina_ts_all, sig)
    if len(rs) >= 100:
        mina_road_stats[sig] = rs
        rs["road_category"] = road_cat.reindex(rs.index)
        print(f"  Mina Signal {sig}: {len(rs):,} roads, mean_dev={rs['mean_dev'].mean():+.4f}")

madum_road_stats = {}
for sig in [1, 3]:
    rs = road_stats_from_ts(madum_ts_all, sig)
    if len(rs) >= 100:
        madum_road_stats[sig] = rs
        rs["road_category"] = road_cat.reindex(rs.index)
        print(f"  Madum Signal {sig}: {len(rs):,} roads, mean_dev={rs['mean_dev'].mean():+.4f}")


# ── POLARIZATION ANALYSIS ─────────────────────────────────────────────────────
print("\nComputing polarization statistics...")

def polarization_stats(rs, label=""):
    if rs.empty or "mean_bl" not in rs.columns:
        return {}
    rs_clean = rs.dropna(subset=["mean_bl"])
    pct_pos  = (rs_clean["mean_dev"] > 0.02).mean()
    pct_neg  = (rs_clean["mean_dev"] < -0.02).mean()
    corr     = rs_clean[["mean_bl","mean_dev"]].corr().iloc[0,1]
    # median deviation for high-speed vs low-speed roads
    med_bl   = rs_clean["mean_bl"].median()
    high     = rs_clean[rs_clean["mean_bl"] >  med_bl]["mean_dev"].mean()
    low      = rs_clean[rs_clean["mean_bl"] <= med_bl]["mean_dev"].mean()
    return {
        "label": label,
        "n_roads": len(rs_clean),
        "pct_faster": pct_pos,
        "pct_slower": pct_neg,
        "corr_bl_dev": corr,
        "high_speed_mean_dev": high,
        "low_speed_mean_dev": low,
    }

polarity_table = []
for sig in sorted(road_stats.keys()):
    polarity_table.append(polarization_stats(road_stats[sig], f"Yagiasha S{sig}"))
for sig in sorted(mina_road_stats.keys()):
    polarity_table.append(polarization_stats(mina_road_stats[sig], f"Mina S{sig}"))
for sig in sorted(madum_road_stats.keys()):
    polarity_table.append(polarization_stats(madum_road_stats[sig], f"Madum S{sig}"))

pol_df = pd.DataFrame(polarity_table)
print(pol_df.to_string(index=False))


# ── FIGURE 1: Scatter plot of baseline speed vs deviation (Yagiasha S10) ─────
print("\nBuilding polarization figures...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

CAT_COLORS = {
    "Motorway": "#D32F2F", "Trunk": "#F57C00", "Primary": "#FBC02D",
    "Secondary": "#388E3C", "Tertiary": "#1976D2", "Residential": "#7B1FA2",
    "Service": "#5D4037", "Other": "#78909C",
}

sig_labels = {1: "Signal 1", 3: "Signal 3", 8: "Signal 8", 9: "Signal 9", 10: "Signal 10"}

# Panel layout: Yagiasha S1, S3, S8, S10 | Mina S3 | Madum S3
panel_data = [
    (road_stats.get(1),           "Yagiasha — Signal 1",  axes[0]),
    (road_stats.get(3),           "Yagiasha — Signal 3",  axes[1]),
    (road_stats.get(8),           "Yagiasha — Signal 8",  axes[2]),
    (road_stats.get(10),          "Yagiasha — Signal 10", axes[3]),
    (mina_road_stats.get(3),      "Mina — Signal 3",      axes[4]),
    (madum_road_stats.get(3),     "Madum — Signal 3",     axes[5]),
]

for rs, title, ax in panel_data:
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(0.5, color="gray", lw=0.5, ls=":", alpha=0.4)

    if rs is None or rs.empty or "mean_bl" not in rs.columns:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        ax.set_title(title, fontsize=15, fontweight="bold")
        continue

    rs_plot = rs.dropna(subset=["mean_bl"]).copy()
    if "road_category" in rs_plot.columns:
        rs_plot["road_category"] = rs_plot["road_category"].apply(normalize_road_category)
    # Jitter + downsample for readability
    if len(rs_plot) > 5000:
        rs_plot = rs_plot.sample(5000, random_state=42)

    cats_present = rs_plot["road_category"].dropna().unique()
    for cat in cats_present:
        sub = rs_plot[rs_plot["road_category"] == cat]
        col = CAT_COLORS.get(cat, "#999999")
        ax.scatter(sub["mean_bl"], sub["mean_dev"], color=col, s=2, alpha=0.4,
                   label=cat, rasterized=True)

    # Add regression line
    valid = rs_plot.dropna(subset=["mean_bl","mean_dev"])
    if len(valid) > 10:
        z = np.polyfit(valid["mean_bl"], valid["mean_dev"], 1)
        xr = np.array([valid["mean_bl"].min(), valid["mean_bl"].max()])
        ax.plot(xr, np.polyval(z, xr), color="black", lw=2, ls="-", alpha=0.8,
                label=f"Slope={z[0]:+.3f}")
        corr = valid[["mean_bl","mean_dev"]].corr().iloc[0,1]

    # stats annotation
    pct_pos = (rs_plot["mean_dev"] > 0.02).mean()
    pct_neg = (rs_plot["mean_dev"] < -0.02).mean()
    n = len(rs_plot)
    ax.text(0.03, 0.96,
            f"n={n:,}\nFaster: {pct_pos:.0%}\nSlower: {pct_neg:.0%}\nr={corr:+.3f}",
            transform=ax.transAxes, fontsize=18, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Baseline Speed (rel. speed ratio)")
    ax.set_ylabel("Deviation from Baseline")
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.5, 0.5)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=13, loc="lower right", markerscale=3, ncol=2)

fig.suptitle(
    "Speed Stratification During Typhoon:\nBaseline Speed vs. Observed Deviation by Road Category",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
savefig_with_alias(fig, "polarization_scatter.png", "图04_基线速度与偏差散点图.png")
plt.close()
print(f"Saved: {OUT}/polarization_scatter.png")


# ── FIGURE 2: Deviation distributions by signal level ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (typhoon_name, rs_dict, color_base) in zip(axes, [
    ("Yagiasha", road_stats, "Reds"),
    ("Mina",     mina_road_stats,  "Blues"),
    ("Madum",    madum_road_stats, "Greens"),
]):
    if not rs_dict:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        ax.set_title(f"{typhoon_name}", fontsize=13)
        continue

    sigs = sorted(rs_dict.keys())
    cmap = plt.get_cmap(color_base)
    colors_sig = [cmap(0.3 + 0.7 * i / max(len(sigs)-1, 1)) for i in range(len(sigs))]

    bins = np.linspace(-0.4, 0.4, 60)
    for sig, col in zip(sigs, colors_sig):
        rs = rs_dict[sig].dropna(subset=["mean_dev"])
        devs = rs["mean_dev"].clip(-0.4, 0.4)
        ax.hist(devs, bins=bins, density=True, histtype="step",
                color=col, lw=1.8, label=f"Signal {sig} (n={len(rs):,})")

    ax.axvline(0, color="black", lw=1, ls="--", alpha=0.7)
    ax.set_title(f"Road-Level Deviation Distribution\n{typhoon_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Deviation from Baseline")
    ax.set_ylabel("Density")
    ax.legend(fontsize=18)
    ax.grid(alpha=0.25)

fig.suptitle(
    "Deviation Distribution by Signal Level — Three Typhoons\n"
    "(positive = faster than normal; negative = slower than normal)",
    fontsize=15, fontweight="bold",
)
plt.tight_layout()
savefig_with_alias(fig, "polarization_dist.png", "参考_偏差分布密度图.png")
plt.close()
print(f"Saved: {OUT}/polarization_dist.png")


# ── FIGURE 3: Summary bar chart — % faster vs slower by signal level ─────────
fig, ax = plt.subplots(figsize=(12, 6))

all_rows = []
for sig in sorted(road_stats.keys()):
    rs = road_stats[sig].dropna(subset=["mean_dev"])
    all_rows.append({"Typhoon": "Yagiasha", "Signal": sig,
                     "pct_faster": (rs["mean_dev"] > 0.02).mean(),
                     "pct_slower": (rs["mean_dev"] < -0.02).mean(),
                     "n": len(rs)})
for sig in sorted(mina_road_stats.keys()):
    rs = mina_road_stats[sig].dropna(subset=["mean_dev"])
    all_rows.append({"Typhoon": "Mina", "Signal": sig,
                     "pct_faster": (rs["mean_dev"] > 0.02).mean(),
                     "pct_slower": (rs["mean_dev"] < -0.02).mean(),
                     "n": len(rs)})
for sig in sorted(madum_road_stats.keys()):
    rs = madum_road_stats[sig].dropna(subset=["mean_dev"])
    all_rows.append({"Typhoon": "Madum", "Signal": sig,
                     "pct_faster": (rs["mean_dev"] > 0.02).mean(),
                     "pct_slower": (rs["mean_dev"] < -0.02).mean(),
                     "n": len(rs)})
bar_df = pd.DataFrame(all_rows)
bar_df["label"] = bar_df.apply(lambda r: f"{r['Typhoon']}\nS{r['Signal']}", axis=1)

x = np.arange(len(bar_df))
w = 0.35
bars_f = ax.bar(x - w/2, bar_df["pct_faster"], w, color="#2196F3", alpha=0.8, label="Faster than baseline (dev>0.02)")
bars_s = ax.bar(x + w/2, bar_df["pct_slower"], w, color="#F44336", alpha=0.8, label="Slower than baseline (dev<-0.02)")

for bar, row in zip(bars_f, bar_df.itertuples()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f"{row.pct_faster:.0%}", ha="center", va="bottom", fontsize=18)
for bar, row in zip(bars_s, bar_df.itertuples()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f"{row.pct_slower:.0%}", ha="center", va="bottom", fontsize=18)

ax.set_xticks(x)
ax.set_xticklabels(bar_df["label"], fontsize=15)
ax.set_ylabel("Proportion of Road Segments")
ax.set_title("Proportion of Roads Faster vs. Slower Than Baseline\nby Typhoon and Signal Level",
             fontsize=15, fontweight="bold")
ax.legend(fontsize=15)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.set_ylim(0, max(bar_df["pct_faster"].max(), bar_df["pct_slower"].max()) + 0.1)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
savefig_with_alias(fig, "polarization_summary_bars.png", "图03_信号等级路段极化比例.png")
plt.close()
print(f"Saved: {OUT}/polarization_summary_bars.png")


# ── FIGURE 4: Time series — % roads faster/slower throughout Yagiasha ────────
print("Building time series of polarization over Yagiasha event...")
yagi_ts_pol = yagi_ty.groupby("dt").agg(
    pct_faster=("dev", lambda x: (x > 0.02).mean()),
    pct_slower=("dev", lambda x: (x < -0.02).mean()),
    mean_dev=("dev", "mean"),
    n_roads=("dev", "count"),
).reset_index()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# shade signal periods
sig_colors = {1: "#FFF9C4", 3: "#FFE0B2", 8: "#FFCDD2", 9: "#EF9A9A", 10: "#F44336"}
for start, end, sig in YAGIASHA_SIGNALS:
    for ax in [ax1, ax2]:
        ax.axvspan(start, end, alpha=0.15 if sig < 8 else 0.25, color=sig_colors.get(sig, "#ddd"))
    ax2.annotate(f"S{sig}", xy=((start + (end-start)/2), 0.35),
                 fontsize=18, ha="center", color="darkred" if sig >= 8 else "darkorange")

ax1.plot(yagi_ts_pol["dt"], yagi_ts_pol["pct_faster"], color="#2196F3", lw=2, label="Faster (dev > 0.02)")
ax1.plot(yagi_ts_pol["dt"], yagi_ts_pol["pct_slower"], color="#F44336", lw=2, label="Slower (dev < -0.02)")
ax1.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.4)
ax1.set_ylabel("Fraction of Roads")
ax1.set_title("Yagiasha Typhoon: Fraction of Roads Faster/Slower Than Baseline Over Time",
              fontsize=18, fontweight="bold")
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax1.legend(fontsize=15)
ax1.grid(alpha=0.3)

ax2.plot(yagi_ts_pol["dt"], yagi_ts_pol["mean_dev"], color="#7B1FA2", lw=2, label="Network mean deviation")
ax2.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
ax2.set_ylabel("Mean Dev (ratio units)")
ax2.set_xlabel("Date/Time (HKT)")
ax2.set_title("Network-Level Mean Speed Deviation", fontsize=13, fontweight="bold")
ax2.legend(fontsize=15)
ax2.grid(alpha=0.3)

# Format x-axis
import matplotlib.dates as mdates
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=18)

fig.suptitle("Yagiasha Signal 1–10: Road Network Response Dynamics", fontsize=15, fontweight="bold")
plt.tight_layout()
savefig_with_alias(fig, "yagiasha_timeseries.png", "图02_叶加沙事件时间序列.png")
plt.close()
print(f"Saved: {OUT}/yagiasha_timeseries.png")

print("\n=== Done. Polarization Analysis Complete ===")
print(pol_df.to_string(index=False))
