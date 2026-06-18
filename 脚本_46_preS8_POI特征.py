"""
Pre-S8 congested roads: POI characteristics analysis.
Compare POI densities around congested roads vs non-congested roads at 13:00 (worst slot).
"""
import pandas as pd, numpy as np
import warnings
warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"

# ── Load regression table (has POI densities per road) ───────────────────────
print("Loading regression table...", flush=True)
rt = pd.read_parquet(f"{DATA}/regression_table.parquet")

# Focus on Yagiasha (Sep 22-25), S3+, MIDDAY (closest to 13:00)
yagi_mid = rt[(rt["typhoon"] == "Ragasa") & (rt["signal_level"] >= 3) &
               (rt["time_group"] == "MIDDAY")].copy()

# Load per-road per-slot deviation at slot 26 (13:00) — from earlier analysis
# We need to re-compute or load saved data. Let's compute directly.
print("Loading slot 26 per-road deviations...", flush=True)

from shapely import wkb as shapely_wkb
import os

FLOW = f"{DATA}/flow_parquet2"
EP = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
BL = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
bl_s26 = BL[(BL.slot == 26) & (BL.day_type == "WORKDAY")].set_index("road_id")["mean_speed"]

def get_ep_key(wb):
    try:
        g = shapely_wkb.loads(bytes(wb))
        coords = list(g.coords) if g.geom_type == "LineString" else \
                 [c for line in g.geoms for c in line.coords]
        s = (round(coords[0][0],4), round(coords[0][1],4))
        e = (round(coords[-1][0],4), round(coords[-1][1],4))
        return str((min(s,e), max(s,e)))
    except: return None

# Load slot 26 for Sep 23
day = "2025-09-23"
folder = f"{FLOW}/{day}"
files = [f for f in os.listdir(folder) if "_slot26_" in f]
if not files:
    print("ERROR: slot 26 file not found")
    exit(1)

df = pd.read_parquet(f"{folder}/{files[0]}",
                     columns=["relative_speed","geometry","road_closure"])
df = df[df["road_closure"] != 1].copy()

# Build WKB cache (quick)
import gc
uniq = {}
for s in [0,12,24,36]:
    fs = [f for f in os.listdir(folder) if f"_slot{s:02d}_" in f]
    if not fs: continue
    d = pd.read_parquet(f"{folder}/{fs[0]}", columns=["geometry"])
    for g in d["geometry"]:
        if g is not None:
            k = id(bytes(g)[:8])
            if k not in uniq: uniq[k] = g
wkb_ep = {}
for g in uniq.values():
    epk = get_ep_key(g)
    if epk: wkb_ep[bytes(g)] = epk

def lookup_epk(g):
    if g is None: return None
    b = bytes(g)
    if b in wkb_ep: return wkb_ep[b]
    epk = get_ep_key(g)
    if epk: wkb_ep[b] = epk
    return epk

df["ep_key"] = df["geometry"].apply(lookup_epk)
df = df.merge(EP[["ep_key","road_id"]], on="ep_key", how="inner")
agg = df.groupby("road_id")["relative_speed"].mean().rename("speed_obs").reset_index()
agg = agg.set_index("road_id")

idx = pd.MultiIndex.from_arrays(
    [["WORKDAY"]*len(agg), [26]*len(agg), agg.index],
    names=["day_type","slot","road_id"])
agg["baseline"] = bl_s26.reindex(agg.index).values
agg = agg.dropna(subset=["baseline"])
agg["deviation"] = agg["speed_obs"] - agg["baseline"]
agg = agg.reset_index()

print(f"  {len(agg)} roads with deviation at slot 26")

# Classify roads
agg["congested"] = agg["deviation"] < -0.03
agg["clearly_fast"] = agg["deviation"] > 0.03
agg["normal"] = agg["deviation"].abs() <= 0.03

print(f"  Congested (<-0.03):  {agg['congested'].sum()} roads ({agg['congested'].mean():.1%})")
print(f"  Clearly fast (>0.03): {agg['clearly_fast'].sum()} roads ({agg['clearly_fast'].mean():.1%})")
print(f"  Normal (±0.03): {agg['normal'].sum()} roads ({agg['normal'].mean():.1%})")

# ── Merge with POI data from regression table ─────────────────────────────────
# Use the MIDDAY data to get POI densities per road
poi_data = yagi_mid[["road_id"] + [c for c in yagi_mid.columns if "_density" in c or "poi" in c.lower()]].drop_duplicates("road_id")

# Actually, the regression table has POI columns per road. Let me find them.
poi_cols = [c for c in yagi_mid.columns if "density" in c.lower() or "poi" in c.lower() or "log_" in c]
print(f"\n  POI-related columns: {poi_cols[:20]}")

# Get one row per road_id (take first occurrence in MIDDAY)
road_poi = yagi_mid.groupby("road_id").first().reset_index()

# POI density columns
POI_CATS = ["work","education","retail","food_drink","recreation",
            "medical","transport","tourism","finance","civic"]
poi_density_cols = [f"{c}_density" for c in POI_CATS]

# Other columns of interest
other_cols = ["population_density_500m","working_pop_ratio_500m",
              "median_income_500m","intersection_degree",
              "dist_to_coast_m","road_length_m","road_broad"]

use_cols = ["road_id"] + poi_density_cols + other_cols
use_cols = [c for c in use_cols if c in road_poi.columns]

merged = agg.merge(road_poi[use_cols], on="road_id", how="inner")
print(f"\n  Merged: {len(merged)} roads with POI data")

# ── Compare congested vs non-congested ────────────────────────────────────────
print("\n" + "="*70)
print("POI DENSITY COMPARISON: Congested vs Normal/Fast roads")
print("="*70)
print(f"{'Variable':<30s} {'Congested':>12s} {'Not Cong.':>12s} {'Ratio':>8s} {'Diff':>10s}")

comparisons = poi_density_cols + [c for c in other_cols if c in merged.columns]

for col in comparisons:
    if col not in merged.columns: continue
    c_mean = merged[merged["congested"]][col].mean()
    n_mean = merged[~merged["congested"]][col].mean()
    ratio = c_mean / n_mean if n_mean > 0 else np.nan
    diff = c_mean - n_mean

    # Mark if significantly different (simple t-test)
    from scipy import stats
    c_vals = merged[merged["congested"]][col].dropna()
    n_vals = merged[~merged["congested"]][col].dropna()
    if len(c_vals) > 30 and len(n_vals) > 30:
        t, p = stats.ttest_ind(c_vals, n_vals, equal_var=False)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    else:
        sig = ""

    print(f"{col:<30s} {c_mean:>12.4f} {n_mean:>12.4f} {ratio:>8.2f} {diff:>+10.4f} {sig}")

# ── Top POIs that distinguish congested roads ──────────────────────────────────
print("\n" + "="*70)
print("TOP DISTINGUISHING FEATURES (ranked by |difference|)")
print("="*70)
diffs = []
for col in comparisons:
    if col not in merged.columns: continue
    c_mean = merged[merged["congested"]][col].mean()
    n_mean = merged[~merged["congested"]][col].mean()
    diffs.append((col, c_mean - n_mean, c_mean / n_mean if n_mean > 0 else np.nan))
diffs.sort(key=lambda x: abs(x[1]), reverse=True)
for col, diff, ratio in diffs[:20]:
    direction = "↑" if diff > 0 else "↓"
    print(f"  {direction} {col:<35s} diff={diff:+.4f}  ratio={ratio:.2f}")

# ── Road category breakdown ────────────────────────────────────────────────────
if "road_broad" in merged.columns:
    print("\n" + "="*70)
    print("ROAD CATEGORY: Congestion rate by type")
    print("="*70)
    for cat, grp in merged.groupby("road_broad"):
        pct = grp["congested"].mean()
        print(f"  {str(cat):<20s}  n={len(grp):5d}  congested={pct:.1%}  mean_dev={grp['deviation'].mean():+.4f}")

print("\nDone.")
