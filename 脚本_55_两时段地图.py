"""
Two standalone maps with basemap + HK zoom:
  Fig 50d — 08:30 Morning peak: which roads cleared (faster) vs which slowed
  Fig 50e — 13:00 Midday dip: bidirectional churn

Length-weighted shares already computed (see 脚本_53):
  08:30  Faster 45.6%  Slower 25.2%
  13:00  Faster 33.9%  Slower 35.8%
"""
import os, gc
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely import wkb as shapely_wkb
import geopandas as gpd
import contextily as cx
import warnings
warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"
OUT  = "/Users/helloling/workspace/thesis"
DEV_HI, DEV_LO = 0.03, -0.03

# HK urban core zoom — Kowloon + HK Island + west NT
HK_BBOX = (113.82, 22.15, 114.45, 22.60)  # (xmin, ymin, xmax, ymax)

def get_ep_key(wb):
    try:
        g = shapely_wkb.loads(bytes(wb))
        coords = list(g.coords) if g.geom_type=="LineString" else \
                 [c for line in g.geoms for c in line.coords]
        s = (round(coords[0][0],4), round(coords[0][1],4))
        e = (round(coords[-1][0],4), round(coords[-1][1],4))
        return str((min(s,e), max(s,e)))
    except: return None

# ─── Load Sep 23 deviations and link to geometries ──────────────────────────
print("Loading deviations...", flush=True)
ts = pd.read_parquet(f"{DATA}/yagiasha_road_timeseries.parquet")
ts["ds"] = pd.to_datetime(ts["dt"]).dt.strftime("%Y-%m-%d")
sep23 = ts[ts["ds"]=="2025-09-23"].copy()

print("Building geometry cache from Sep 23 flow files...", flush=True)
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
ep_to_rid = ep.set_index("ep_key")["road_id"].to_dict()

folder = f"{FLOW}/2025-09-23"
geom_per_rid = {}
# Load multiple slots to maximize geometry coverage
for s in [10, 14, 17, 20, 22, 26, 30, 36]:
    files = [f for f in os.listdir(folder) if f"_slot{s:02d}_" in f]
    if not files: continue
    df = pd.read_parquet(f"{folder}/{files[0]}", columns=["geometry"])
    for g in df["geometry"]:
        if g is None: continue
        epk = get_ep_key(g)
        if epk and epk in ep_to_rid:
            rid = ep_to_rid[epk]
            if rid not in geom_per_rid:
                try:
                    geom_per_rid[rid] = shapely_wkb.loads(bytes(g))
                except: pass
print(f"  {len(geom_per_rid):,} road geometries cached")

def cohort_for_slot(slot):
    sub = sep23[sep23["slot"]==slot]
    faster_ids = sub[sub["dev"] >  DEV_HI]["road_id"].tolist()
    slower_ids = sub[sub["dev"] < DEV_LO]["road_id"].tolist()
    return faster_ids, slower_ids

def build_gdf(road_ids, label):
    rows = []
    for rid in road_ids:
        g = geom_per_rid.get(rid)
        if g is None: continue
        rows.append({"road_id": rid, "geometry": g, "cls": label})
    if not rows: return gpd.GeoDataFrame(columns=["road_id","geometry","cls"])
    g = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    return g

def make_map(slot, hour_label, faster_pct, slower_pct, title, out_path):
    fast_ids, slow_ids = cohort_for_slot(slot)
    print(f"\n  slot {slot} ({hour_label}): {len(fast_ids)} faster, {len(slow_ids)} slower")

    gdf_fast = build_gdf(fast_ids, "Faster")
    gdf_slow = build_gdf(slow_ids, "Slower")

    # Reproject to Web Mercator for basemap
    if len(gdf_fast):
        gdf_fast_m = gdf_fast.to_crs("EPSG:3857")
    if len(gdf_slow):
        gdf_slow_m = gdf_slow.to_crs("EPSG:3857")

    fig, ax = plt.subplots(figsize=(11, 8))

    # Plot slower first (so faster overlays on top — both colors visible)
    if len(gdf_slow):
        gdf_slow_m.plot(ax=ax, color="#d62728", linewidth=1.0, alpha=0.85,
                        zorder=4, label=f"Slower (dev < −0.03)  —  {slower_pct:.1f}% by length")
    if len(gdf_fast):
        gdf_fast_m.plot(ax=ax, color="#2ca02c", linewidth=1.0, alpha=0.85,
                        zorder=5, label=f"Faster (dev > +0.03)  —  {faster_pct:.1f}% by length")

    # Set zoom (in lon/lat then convert)
    bbox_4326 = gpd.GeoSeries([
        gpd.points_from_xy([HK_BBOX[0], HK_BBOX[2]],
                           [HK_BBOX[1], HK_BBOX[3]])[0],
        gpd.points_from_xy([HK_BBOX[0], HK_BBOX[2]],
                           [HK_BBOX[1], HK_BBOX[3]])[1]
    ], crs="EPSG:4326").to_crs("EPSG:3857").total_bounds
    ax.set_xlim(bbox_4326[0], bbox_4326[2])
    ax.set_ylim(bbox_4326[1], bbox_4326[3])

    # Basemap
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                       attribution_size=7, zorder=0)
    except Exception as e:
        print(f"  basemap warning: {e}")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=10, framealpha=0.92,
              edgecolor="#999")
    ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved -> {out_path}")

# ─── Two figures ─────────────────────────────────────────────────────────────
make_map(17, "08:30", 45.6, 25.2,
         "Pre-Event Morning Clearance under Signal 3  (Ragasa, Sep 23, 08:30)\n"
         "45.6% of road-km cleared (faster)  ·  25.2% slower",
         f"{OUT}/图50d_pre_event_morning_clearance.png")

make_map(26, "13:00", 33.9, 35.8,
         "Pre-Event Midday Bidirectional Churn under Signal 3  (Ragasa, Sep 23, 13:00)\n"
         "33.9% of road-km faster  ·  35.8% slower",
         f"{OUT}/图50e_pre_event_midday_churn.png")

print("\nDone.")
