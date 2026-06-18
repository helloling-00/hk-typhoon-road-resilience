"""
Detailed pre-S8 congestion surge analysis for Yagiasha Sep 23.
Per-road per-slot deviations with road categories, distribution analysis.
"""
import os, gc, pandas as pd, numpy as np
from shapely import wkb as shapely_wkb
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"

print("Loading lookups...", flush=True)
bl = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
bl_idx = bl.set_index(["day_type","slot","road_id"])["mean_speed"]

# Road category lookup
rr = pd.read_parquet(f"{DATA}/road_registry.parquet")[["ep_key","road_id","road_category"]]
rr["road_category"] = rr["road_category"].fillna("other")
# normalize
mapping = {
    "motorway":"Motorway","motorway_link":"Motorway",
    "trunk":"Trunk","trunk_link":"Trunk",
    "primary":"Primary","primary_link":"Primary",
    "secondary":"Secondary","secondary_link":"Secondary",
    "tertiary":"Tertiary","tertiary_link":"Tertiary",
    "residential":"Residential","living_street":"Residential","unclassified":"Residential",
    "service":"Service","services":"Service",
}
rr["cat"] = rr["road_category"].str.strip().str.lower().map(
    lambda x: mapping.get(x, "Other"))
road_cat = rr.drop_duplicates("road_id").set_index("road_id")["cat"]

def get_ep_key(wb):
    try:
        g = shapely_wkb.loads(bytes(wb))
        coords = list(g.coords) if g.geom_type == "LineString" else \
                 [c for line in g.geoms for c in line.coords]
        s = (round(coords[0][0],4), round(coords[0][1],4))
        e = (round(coords[-1][0],4), round(coords[-1][1],4))
        return str((min(s,e), max(s,e)))
    except: return None

# Build WKB cache for Sep 23
day = "2025-09-23"
folder = f"{FLOW}/{day}"
print(f"Building WKB cache for {day}...", flush=True)
uniq = {}
for s in [0, 6, 12, 18, 24, 30, 36, 42]:
    files = [f for f in os.listdir(folder) if f"_slot{s:02d}_" in f]
    if not files: continue
    df = pd.read_parquet(f"{folder}/{files[0]}", columns=["geometry"])
    for g in df["geometry"]:
        if g is not None:
            key = id(bytes(g)[:8])
            if key not in uniq: uniq[key] = g
wkb_ep = {}
for g in uniq.values():
    epk = get_ep_key(g)
    if epk: wkb_ep[bytes(g)] = epk
print(f"  {len(wkb_ep)} unique geometries cached")

# Compute per-road deviations for each slot
print("Computing per-slot per-road deviations...", flush=True)
all_slots = sorted([int(f.split("_slot")[1][:2])
                    for f in os.listdir(folder)
                    if "_slot" in f and f.endswith(".parquet")])

slot_summaries = []
all_road_records = []  # per-road per-slot deviations for selected slots

def lookup_epk(g):
    if g is None: return None
    b = bytes(g)
    if b in wkb_ep: return wkb_ep[b]
    epk = get_ep_key(g)
    if epk: wkb_ep[b] = epk
    return epk

for s in all_slots:
    files = [f for f in os.listdir(folder) if f"_slot{s:02d}_" in f]
    if not files: continue
    try:
        df = pd.read_parquet(f"{folder}/{files[0]}",
                             columns=["relative_speed","geometry","road_closure"])
        df = df[df["road_closure"] != 1].copy()
        if len(df) < 50: continue

        df["ep_key"] = df["geometry"].apply(lookup_epk)
        df = df.merge(ep[["ep_key","road_id"]], on="ep_key", how="inner")
        if len(df) < 50: continue

        agg = df.groupby("road_id")["relative_speed"].mean().rename("obs")
        agg = agg.reset_index().set_index("road_id")

        idx = pd.MultiIndex.from_arrays(
            [["WORKDAY"]*len(agg), [s]*len(agg), agg.index],
            names=["day_type","slot","road_id"])
        agg["baseline"] = bl_idx.reindex(idx).values
        agg = agg.dropna(subset=["baseline"])
        if len(agg) < 100: continue
        agg["dev"] = agg["obs"] - agg["baseline"]

        # Road category
        agg["road_cat"] = agg.index.map(road_cat).fillna("Other")

        hour = s * 0.5
        # Summary stats
        slot_summaries.append({
            "slot": s, "hour": hour,
            "n_roads": len(agg),
            "mean_dev": float(agg["dev"].mean()),
            "median_dev": float(agg["dev"].median()),
            "std_dev": float(agg["dev"].std()),
            "p10": float(agg["dev"].quantile(0.10)),
            "p25": float(agg["dev"].quantile(0.25)),
            "p75": float(agg["dev"].quantile(0.75)),
            "p90": float(agg["dev"].quantile(0.90)),
            "pct_faster": float((agg["dev"] > 0).mean()),
            "pct_better_005": float((agg["dev"] > 0.05).mean()),
            "pct_worse_005": float((agg["dev"] < -0.05).mean()),
            "pct_better_003": float((agg["dev"] > 0.03).mean()),
            "pct_worse_003": float((agg["dev"] < -0.03).mean()),
            "iqr": float(agg["dev"].quantile(0.75) - agg["dev"].quantile(0.25)),
        })
        # Per-category means
        for cat, grp in agg.groupby("road_cat"):
            slot_summaries[-1][f"mean_{cat}"] = float(grp["dev"].mean())
            slot_summaries[-1][f"n_{cat}"] = len(grp)
            slot_summaries[-1][f"pct_worse_{cat}"] = float((grp["dev"] < -0.03).mean())

        # Save per-road for key slots (pre-S8 hours)
        if 10 <= s <= 29:  # 05:00-14:30
            agg_reset = agg.reset_index()
            agg_reset["slot"] = s
            agg_reset["hour"] = hour
            all_road_records.append(agg_reset[["road_id","slot","hour","dev","road_cat"]])

    except Exception as e:
        pass
    gc.collect()

ss = pd.DataFrame(slot_summaries).sort_values("slot").reset_index(drop=True)
road_devs = pd.concat(all_road_records, ignore_index=True) if all_road_records else None

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
S8_SLOT = 28  # 14:00, S8 at 14:20 is between slot 28 and 29

print("\n" + "="*70)
print("HOUR-BY-HOUR BREAKDOWN (Pre-S8, Sep 23)")
print("="*70)
for _, row in ss[ss.slot <= 29].iterrows():
    print(f"  {row['hour']:5.1f}h  slot{int(row['slot']):02d}  "
          f"mean={row['mean_dev']:+.4f}  median={row['median_dev']:+.4f}  "
          f"n={int(row['n_roads']):,}  "
          f"faster={row['pct_faster']:.1%}  worse_005={row['pct_worse_005']:.1%}  "
          f"better_005={row['pct_better_005']:.1%}")

print("\n" + "="*70)
print("PRE-S8 vs POST-S8 COMPARISON")
print("="*70)
pre_s8 = ss[ss.slot <= S8_SLOT]
post_s8_first2h = ss[(ss.slot > S8_SLOT) & (ss.slot <= S8_SLOT+4)]
post_s8_eve = ss[(ss.slot > S8_SLOT) & (ss.slot <= 40)]

for label, subset in [("Pre-S8 (00:00-14:00)", pre_s8),
                       ("Post-S8 first 2h (14:30-16:30)", post_s8_first2h),
                       ("Post-S8 evening (14:30-20:00)", post_s8_eve)]:
    print(f"\n  {label}:")
    print(f"    mean_dev = {subset['mean_dev'].mean():+.4f}")
    print(f"    pct_worse_005 = {subset['pct_worse_005'].mean():.1%}")
    print(f"    pct_better_005 = {subset['pct_better_005'].mean():.1%}")
    print(f"    median IQR = {subset['iqr'].median():.4f}")

print("\n" + "="*70)
print("ROAD CATEGORY BREAKDOWN (Pre-S8 dip hours: 11:00-14:00)")
print("="*70)
dip_slots = ss[(ss.slot >= 22) & (ss.slot <= 28)]
cats = ["Motorway","Trunk","Primary","Secondary","Tertiary","Residential","Service","Other"]
for cat in cats:
    col_mean = f"mean_{cat}"
    col_worse = f"pct_worse_{cat}"
    if col_mean in dip_slots.columns:
        print(f"  {cat:15s}  mean_dev={dip_slots[col_mean].mean():+.4f}  "
              f"pct_worse_003={dip_slots[col_worse].mean():.1%}"
              if col_worse in dip_slots.columns else f"  {cat:15s}  mean_dev={dip_slots[col_mean].mean():+.4f}")

print("\n" + "="*70)
print("WORST SLOT (max congestion)")
print("="*70)
worst = ss.loc[ss["mean_dev"].idxmin()]
print(f"  Slot {int(worst['slot'])} ({worst['hour']:.1f}h)")
print(f"  mean_dev = {worst['mean_dev']:+.4f}")
print(f"  pct_worse_005 = {worst['pct_worse_005']:.1%}")
print(f"  pct_worse_003 = {worst['pct_worse_003']:.1%}")
print(f"  IQR = {worst['iqr']:.4f}")
print(f"  p10 = {worst['p10']:+.4f}  p90 = {worst['p90']:+.4f}")

# Distribution of per-road deviations at worst slot
if road_devs is not None:
    worst_slot = int(worst["slot"])
    worst_devs = road_devs[road_devs.slot == worst_slot]
    print(f"\n  Per-road at worst slot ({len(worst_devs)} roads):")
    for cat in cats:
        sub = worst_devs[worst_devs.road_cat == cat]
        if len(sub) > 10:
            print(f"    {cat:15s}  n={len(sub):5d}  mean={sub['dev'].mean():+.4f}  "
                  f"pct_worse003={(sub['dev'] < -0.03).mean():.1%}  "
                  f"pct_better003={(sub['dev'] > 0.03).mean():.1%}")

# Transition: when does the flip happen?
print("\n" + "="*70)
print("TRANSITION TIMELINE (hour by hour around S8)")
print("="*70)
for _, row in ss[(ss.slot >= 24) & (ss.slot <= 34)].iterrows():
    arrow = " ← S8" if row["slot"] == 28 else ""
    print(f"  {row['hour']:5.1f}h  mean={row['mean_dev']:+.4f}  "
          f"faster={row['pct_faster']:.1%}  worse_005={row['pct_worse_005']:.1%}  "
          f"p25={row['p25']:+.4f}  p75={row['p75']:+.4f}{arrow}")

print("\nDone.")
