"""
Pre-typhoon behavioral analysis.
Looks at speed deviations in hours before Signal 1 for all 3 typhoons.
Uses ep_key matching (endpoint clustering) to join flow → road_id.
"""

import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from shapely import wkb as shapely_wkb
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
FLOW = f"{DATA}/flow_parquet2"
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


print("Loading baseline and ep_to_road...")
bl    = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
ep    = pd.read_parquet(f"{DATA}/ep_to_road.parquet")

# Also load road_category from registry (first occurrence per road_id)
rr    = pd.read_parquet(f"{DATA}/road_registry.parquet")[["ep_key","road_id","road_category"]]
rr["road_category"] = rr["road_category"].apply(normalize_road_category)
road_cat_map = rr.drop_duplicates("road_id").set_index("road_id")["road_category"]

# Build baseline lookup: {(day_type, slot, road_id): mean_speed}
bl_idx = bl.set_index(["day_type","slot","road_id"])["mean_speed"]

# ep_key set lookup for fast join
ep_lookup = ep.set_index("ep_key")["road_id"]  # Series: ep_key → road_id


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


# ── day-level cached WKB→ep_key mapping ─────────────────────────────────────
# Build once per day by scanning a few representative slots, then reuse
_wkb_ep_cache = {}

def build_wkb_cache(day, sample_slots=(0, 12, 24, 36)):
    folder = f"{FLOW}/{day}"
    uniq = {}
    for s in sample_slots:
        files = [f for f in os.listdir(folder) if f"_slot{s:02d}_" in f]
        if not files:
            continue
        df = pd.read_parquet(f"{folder}/{files[0]}", columns=["geometry"])
        for g in df["geometry"]:
            if g is not None:
                key = id(bytes(g)[:8])  # cheap dedup proxy
                if key not in uniq:
                    uniq[key] = g
    # compute ep_key for all unique geometries
    ep_map = {}
    for g in uniq.values():
        epk = get_ep_key(g)
        if epk:
            ep_map[bytes(g)] = epk
    _wkb_ep_cache[day] = ep_map
    return ep_map


def load_slot_with_road_id(day, slot_num, wkb_ep=None):
    folder = f"{FLOW}/{day}"
    files  = [f for f in os.listdir(folder) if f"_slot{slot_num:02d}_" in f]
    if not files:
        return None
    df = pd.read_parquet(f"{folder}/{files[0]}",
                         columns=["relative_speed", "geometry", "road_closure", "road_category"])
    df = df[df["road_closure"] != 1].copy()
    if len(df) == 0:
        return None

    # Compute ep_key using cache if available
    if wkb_ep is not None:
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
    else:
        df["ep_key"] = df["geometry"].apply(get_ep_key)

    # join road_id
    df = df.merge(ep[["ep_key","road_id"]], on="ep_key", how="inner")
    if len(df) == 0:
        return None

    # aggregate to road_id level
    df["road_category"] = df["road_category"].apply(normalize_road_category)
    agg = df.groupby("road_id").agg(
        obs=("relative_speed", "mean"),
        road_category=("road_category", "first"),
    ).reset_index()
    return agg


def compute_slot_deviation(day, slot_num, day_type, wkb_ep=None):
    obs = load_slot_with_road_id(day, slot_num, wkb_ep)
    if obs is None or len(obs) < 100:
        return None

    obs = obs.set_index("road_id")
    idx = pd.MultiIndex.from_arrays([[day_type]*len(obs), [slot_num]*len(obs), obs.index],
                                    names=["day_type","slot","road_id"])
    obs["baseline"] = bl_idx.reindex(idx).values
    obs = obs.dropna(subset=["baseline"])
    if len(obs) < 100:
        return None
    obs["dev"] = obs["obs"] - obs["baseline"]
    return obs


def day_timeseries(day, slots, day_type):
    print(f"  {day}: {len(slots)} slots")
    wkb_ep = build_wkb_cache(day)
    rows = []
    for s in slots:
        df = compute_slot_deviation(day, s, day_type, wkb_ep)
        if df is None:
            continue
        rows.append({
            "slot"     : s,
            "hour"     : s * 0.5,
            "mean_dev" : df["dev"].mean(),
            "median_dev": df["dev"].median(),
            "p25"      : df["dev"].quantile(0.25),
            "p75"      : df["dev"].quantile(0.75),
            "pct_pos"  : (df["dev"] >  0.02).mean(),
            "pct_neg"  : (df["dev"] < -0.02).mean(),
            "n_roads"  : len(df),
        })
    return pd.DataFrame(rows)


def day_category_ts(day, slots, day_type):
    wkb_ep = build_wkb_cache(day)
    rows = []
    for s in slots:
        df = compute_slot_deviation(day, s, day_type, wkb_ep)
        if df is None:
            continue
        for cat, grp in df.groupby("road_category"):
            rows.append({"slot": s, "hour": s*0.5,
                         "road_category": cat, "mean_dev": grp["dev"].mean(),
                         "n": len(grp)})
    return pd.DataFrame(rows)


# ── event definitions ────────────────────────────────────────────────────────
# slot number for Signal 1 announcement
# slot N = [N*30, (N+1)*30) minutes from midnight
# Mina:     09-17, Signal 1 @ 21:20 → last analyzable slot = 42 (21:00)
# Yagiasha: 09-22, Signal 1 @ 12:20 → last analyzable slot = 24; data gap after slot 9
# Madum:    10-03, Signal 1 @ 19:40 → last analyzable slot = 39 (19:30)

print("\nComputing pre-typhoon time series...")
mina_ts   = day_timeseries("2025-09-17", range(0, 43),    "WORKDAY")
yagi_ts   = day_timeseries("2025-09-22", range(0, 10),    "WORKDAY")  # data only slots 0-9
madum_ts  = day_timeseries("2025-10-03", range(0, 40),    "WORKDAY")

print("Computing control time series (clean workdays)...")
ctrl1_ts  = day_timeseries("2025-09-16", range(0, 48),    "WORKDAY")  # Tuesday
ctrl2_ts  = day_timeseries("2025-10-02", range(0, 48),    "WORKDAY")  # Thursday

print("Computing road category breakdown (Mina)...")
mina_cat  = day_category_ts("2025-09-17", range(14, 43),  "WORKDAY")
print("Computing road category breakdown (Madum)...")
madum_cat = day_category_ts("2025-10-03", range(14, 40),  "WORKDAY")

# ── average control for plotting ─────────────────────────────────────────────
ctrl_all = []
for df in [ctrl1_ts, ctrl2_ts]:
    if not df.empty:
        ctrl_all.append(df[["hour","mean_dev"]].rename(columns={"mean_dev":"dev"}))
if ctrl_all:
    ctrl_merged = pd.concat(ctrl_all).groupby("hour")["dev"].mean().reset_index()
else:
    ctrl_merged = pd.DataFrame(columns=["hour","dev"])

print("\nBuilding figure...")

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
gs  = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 0])
ax5 = fig.add_subplot(gs[2, 1])

# ── Panel 1: Network-level mean deviation ────────────────────────────────────
ax1.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
ax1.axvspan(6, 9.5,   alpha=0.07, color="orange")
ax1.axvspan(17, 20,   alpha=0.07, color="purple")
ax1.text(7.75, -0.055, "Morning\nrush", fontsize=13.5, color="darkorange", ha="center")
ax1.text(18.5, -0.055, "Evening\nrush", fontsize=13.5, color="purple", ha="center")

if not ctrl_merged.empty:
    ax1.plot(ctrl_merged["hour"], ctrl_merged["dev"], color="#BDBDBD", lw=2,
             ls="--", label="Avg control workday (09-16, 10-02)", zorder=2)

events_plot = [
    ("Mina (09-17, max Signal 3)",   mina_ts,   "#2196F3", 21.33, "S1@21:20"),
    ("Yagiasha (09-22, max Sig 10)", yagi_ts,   "#F44336", 12.33, "S1@12:20\n(data gap)"),
    ("Madum (10-03, max Signal 3)",  madum_ts,  "#4CAF50", 19.67, "S1@19:40"),
]
for label, df, col, s1_h, ann in events_plot:
    if df.empty:
        continue
    ax1.plot(df["hour"], df["mean_dev"], color=col, lw=2.5, label=label, zorder=3)
    ax1.fill_between(df["hour"], df["p25"], df["p75"], color=col, alpha=0.12)
    ax1.axvline(s1_h, color=col, lw=1.5, ls=":", alpha=0.8)
    y_ann = 0.04 if col == "#4CAF50" else 0.06
    ax1.annotate(ann, xy=(s1_h, y_ann), fontsize=13, color=col, ha="center")

ax1.set_title("Network-Level Speed Deviation Before Signal 1 Raised\n"
              "(observed − baseline; positive = roads faster than normal baseline)",
              fontsize=18, fontweight="bold")
ax1.set_xlabel("Hour of Day (HKT)")
ax1.set_ylabel("Mean Speed Deviation (ratio units)")
ax1.set_xlim(0, 22)
ax1.set_xticks(range(0, 23, 2))
ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 23, 2)], fontsize=18)
ax1.legend(fontsize=18, loc="upper left")
ax1.grid(axis="y", alpha=0.3)

# ── Panel 2 & 3: % faster / slower over time ─────────────────────────────────
for ax, metric, title in [
    (ax2, "pct_pos", "Share of Roads Faster Than Baseline (dev > 0.02)"),
    (ax3, "pct_neg", "Share of Roads Slower Than Baseline (dev < -0.02)"),
]:
    ax.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.4)
    ax.axvspan(6, 9.5, alpha=0.06, color="orange")
    ax.axvspan(17, 20, alpha=0.06, color="purple")
    if not mina_ts.empty:
        ax.plot(mina_ts["hour"],  mina_ts[metric],  color="#2196F3", lw=2, label="Mina (09-17)")
    if not madum_ts.empty:
        ax.plot(madum_ts["hour"], madum_ts[metric], color="#4CAF50", lw=2, label="Madum (10-03)")
    ax.axvline(21.33, color="#2196F3", lw=1.5, ls=":", alpha=0.7)
    ax.axvline(19.67, color="#4CAF50", lw=1.5, ls=":", alpha=0.7)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Hour of Day (HKT)")
    ax.set_ylabel("Fraction of Roads")
    ax.set_xlim(0, 22)
    ax.set_xticks(range(0, 23, 4))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 23, 4)], fontsize=18)
    ax.legend(fontsize=18)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

# ── Panel 4 & 5: Road category deviation ─────────────────────────────────────
cat_colors = {
    "Motorway": "#D32F2F",  "Trunk": "#F57C00",     "Primary": "#FBC02D",
    "Secondary": "#388E3C", "Tertiary": "#1976D2",   "Residential": "#7B1FA2",
    "Service": "#5D4037",   "Other": "#78909C",
}

def plot_cat(ax, cat_df, title, s1_h):
    if cat_df.empty:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
        return
    counts = cat_df.groupby("road_category")["n"].mean()
    top    = counts[counts >= 30].sort_values(ascending=False).index[:8]
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.axvspan(6, 9.5, alpha=0.06, color="orange")
    ax.axvspan(17, 20, alpha=0.06, color="purple")
    for cat in top:
        sub = cat_df[cat_df["road_category"] == cat].sort_values("hour")
        col = cat_colors.get(cat, "#666666")
        ax.plot(sub["hour"], sub["mean_dev"], label=cat, color=col, lw=1.8)
    ax.axvline(s1_h, color="red", lw=1.5, ls=":", alpha=0.7)
    ax.text(s1_h + 0.2, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 0.04,
            "Signal 1", fontsize=13, color="red")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Hour of Day (HKT)")
    ax.set_ylabel("Mean Deviation")
    ax.set_xlim(7, 22)
    ax.set_xticks(range(7, 23, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(7, 23, 2)], fontsize=18)
    ax.legend(fontsize=13, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)

plot_cat(ax4, mina_cat,  "Speed Deviation by Road Category — Mina (09-17)", 21.33)
plot_cat(ax5, madum_cat, "Speed Deviation by Road Category — Madum (10-03)", 19.67)

fig.suptitle(
    "Pre-Typhoon Behavioral Anomaly: Speed Deviation Before Warning Signal 1\n"
    "Three Hong Kong Typhoons — September–October 2025",
    fontsize=13, fontweight="bold", y=1.01,
)
savefig_with_alias(fig, "pretyphoon_analysis.png", "图01_台风前行为异常.png")
plt.close()
print(f"\nSaved: {OUT}/pretyphoon_analysis.png")

# ── print summary statistics ─────────────────────────────────────────────────
print("\n=== Pre-typhoon Summary Statistics ===")
for name, df in [("Mina (09-17)", mina_ts), ("Madum (10-03)", madum_ts)]:
    if df.empty:
        print(f"{name}: no data")
        continue
    pm = df[(df["hour"] >= 14) & (df["hour"] <= 21)]
    am = df[(df["hour"] >= 6)  & (df["hour"] <= 12)]
    print(f"\n{name}:")
    if not pm.empty:
        peak_h = pm.loc[pm["mean_dev"].idxmax(), "hour"]
        peak_v = pm["mean_dev"].max()
        print(f"  Peak PM deviation (14-21h): {peak_v:+.4f} at {peak_h:.1f}h")
    if not am.empty:
        print(f"  AM mean deviation (6-12h): {am['mean_dev'].mean():+.4f}")
    full_pm = df[df["hour"] >= 14]
    if not full_pm.empty:
        print(f"  Max pct faster (PM): {full_pm['pct_pos'].max():.1%}")
        print(f"  Max pct slower (PM): {full_pm['pct_neg'].max():.1%}")
    print(f"  Overall mean deviation: {df['mean_dev'].mean():+.4f}")
