"""
F/S/N maps for normal workday morning vs Sep 23 morning (08:30, slot 17).
Two maps side by side with same F/S color scheme.
"""
import os
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

CTRL_DATES = ["2025-09-16", "2025-09-26", "2025-09-29", "2025-09-30",
              "2025-10-02", "2025-10-06", "2025-10-08", "2025-10-09"]

HK_BBOX = (113.82, 22.15, 114.45, 22.60)

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

def classify(d):
    if d > DEV_HI: return "F"
    if d < DEV_LO: return "S"
    return "N"

# ─── Load geometry cache ────────────────────────────────────────────────
print("Building geometry cache...", flush=True)
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
ep_to_rid = ep.set_index("ep_key")["road_id"].to_dict()

geom_per_rid = {}
folder = f"{FLOW}/2025-09-23"
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
print(f"  {len(geom_per_rid):,} geometries cached")

# ─── Load data ───────────────────────────────────────────────────────────
print("Loading...", flush=True)
ts = pd.read_parquet(f"{DATA}/yagiasha_road_timeseries.parquet")
ts["ds"] = pd.to_datetime(ts["dt"]).dt.strftime("%Y-%m-%d")

# Control workday mean deviation
ctrl = ts[(ts["ds"].isin(CTRL_DATES)) & (ts["slot"] == 17)].copy()
ctrl_agg = ctrl.groupby("road_id")["dev"].mean().reset_index()
ctrl_agg["state"] = ctrl_agg["dev"].apply(classify)

# Sep 23 deviation
sep23 = ts[(ts["ds"] == "2025-09-23") & (ts["slot"] == 17)].copy()

def build_gdf(data, state_col):
    rows = []
    for _, row in data.iterrows():
        g = geom_per_rid.get(row["road_id"])
        if g is None: continue
        rows.append({"road_id": row["road_id"], "geometry": g, "state": row[state_col]})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")

ctrl_gdf = build_gdf(ctrl_agg, "state")
sep23["state"] = sep23["dev"].apply(classify)
sep23_gdf = build_gdf(sep23, "state")

# ─── Plot side by side ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(22, 9))

state_colors = {"F": "#2ca02c", "S": "#d62728", "N": "#7f7f7f"}
state_labels = {"F": "Faster", "S": "Slower", "N": "Near baseline"}
state_alpha = {"F": 0.85, "S": 0.85, "N": 0.35}
state_lw =    {"F": 0.8, "S": 0.8, "N": 0.3}
state_zorder = {"F": 5, "S": 4, "N": 2}

titles = [
    (ctrl_gdf, "Normal Workday Morning Peak (08:30)\n"
               f"Mean deviation across 8 control workdays"),
    (sep23_gdf, "Pre-Event Morning Clearance under S3 (08:30)\n"
                "Ragasa, Sep 23 — 45.6% faster, 25.2% slower by road-km"),
]

for ax, (gdf, title) in zip(axes, titles):
    for state in ["N", "S", "F"]:
        sub = gdf[gdf["state"] == state]
        if len(sub) == 0: continue
        sub_m = sub.to_crs("EPSG:3857")
        cnt = len(sub)
        sub_m.plot(ax=ax, color=state_colors[state], linewidth=state_lw[state],
                   alpha=state_alpha[state], zorder=state_zorder[state],
                   label=f"{state_labels[state]} ({state}) — {cnt:,} roads")

    bbox_4326 = gpd.GeoSeries([
        gpd.points_from_xy([HK_BBOX[0], HK_BBOX[2]], [HK_BBOX[1], HK_BBOX[3]])[0],
        gpd.points_from_xy([HK_BBOX[0], HK_BBOX[2]], [HK_BBOX[1], HK_BBOX[3]])[1]
    ], crs="EPSG:4326").to_crs("EPSG:3857").total_bounds
    ax.set_xlim(bbox_4326[0], bbox_4326[2])
    ax.set_ylim(bbox_4326[1], bbox_4326[3])

    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                       attribution_size=7, zorder=0)
    except: pass

    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92, edgecolor="#999")
    ax.set_aspect("equal")

plt.tight_layout()
out_path = f"{OUT}/图59d_morning_FS_comparison.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  saved -> {out_path}")
print("Done.")
