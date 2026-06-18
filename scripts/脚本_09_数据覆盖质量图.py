"""
Single map: road coverage completeness for typhoon-vs-baseline comparison.
Categories:
  Gold:   all 3 day types (WD/SAT/SUN) + typhoon observation  -> 完整
  Green:  WD+SAT or WD+SUN + typhoon                          -> 较完整
  Blue:   WD baseline only + typhoon                          -> 仅工作日基线
  Purple: typhoon observed but NO workday baseline             -> 台风有数据但无基线
  Gray:   baseline exists but NEVER observed in typhoon        -> 基线存在但台风无观测
"""
import ast, pickle, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely import wkb as shapely_wkb
import geopandas as gpd
import contextily as ctx
import warnings; warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
OUT  = "/Users/helloling/workspace/thesis"

# ── load road geometries ──────────────────────────────────────────────────────
print("Loading geometries...")
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
with open(f"{DATA}/osm_cache/road_wkb_store.pkl", "rb") as f:
    wkb_store = pickle.load(f)

def build_geom(row):
    rid = row["road_id"]
    if rid in wkb_store:
        try: return shapely_wkb.loads(wkb_store[rid])
        except: pass
    try:
        pts = ast.literal_eval(row["ep_key"])
        from shapely.geometry import LineString
        return LineString([pts[0], pts[1]])
    except: return None

ep["geometry"] = ep.apply(build_geom, axis=1)
ep = ep.dropna(subset=["geometry"])

# ── baseline day_type sets per road ──────────────────────────────────────────
print("Computing coverage categories...")
bl = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
road_daytypes = bl.groupby("road_id")["day_type"].apply(set)

has_WD  = set(road_daytypes[road_daytypes.apply(lambda s: "WORKDAY" in s)].index)
has_SAT = set(road_daytypes[road_daytypes.apply(lambda s: "SATURDAY" in s)].index)
has_SUN = set(road_daytypes[road_daytypes.apply(lambda s: "SUNDAY_HOLIDAY" in s)].index)

# typhoon roads (pre-computed from script above - re-derive from baseline proxy)
# Use: roads that appear in typhoon flow data = roads with n_obs implying typhoon presence
# Faster: any road_id in typhoon_roads set; re-scan quickly using signal-period parquets
import os
from shapely import wkb as _swkb

FLOW = f"{DATA}/flow_parquet2"
ep_lkp = ep.set_index("ep_key")["road_id"]

def get_ep_key(wb):
    try:
        g = _swkb.loads(bytes(wb))
        coords = list(g.coords) if g.geom_type=="LineString" else \
                 [c for line in g.geoms for c in line.coords]
        s=(round(coords[0][0],4),round(coords[0][1],4))
        e=(round(coords[-1][0],4),round(coords[-1][1],4))
        return str((min(s,e),max(s,e)))
    except: return None

typhoon_days = {
    "2025-09-17","2025-09-18","2025-09-19","2025-09-20",
    "2025-09-22","2025-09-23","2025-09-24","2025-09-25",
    "2025-10-03","2025-10-04","2025-10-05"
}
typhoon_roads = set()
for day in sorted(typhoon_days):
    folder = f"{FLOW}/{day}"
    if not os.path.exists(folder): continue
    for fname in os.listdir(folder):
        try:
            df = pd.read_parquet(f"{folder}/{fname}", columns=["geometry","road_closure"])
        except: continue
        df = df[df["road_closure"] != 1]
        for wb in df["geometry"]:
            if wb is None: continue
            try:
                epk = get_ep_key(wb)
                if epk and epk in ep_lkp.index:
                    typhoon_roads.add(int(ep_lkp[epk]))
            except: pass

print(f"  Typhoon roads: {len(typhoon_roads):,}")

# ── assign category ───────────────────────────────────────────────────────────
def categorize(road_id):
    has_wd  = road_id in has_WD
    has_sat = road_id in has_SAT
    has_sun = road_id in has_SUN
    has_typh = road_id in typhoon_roads

    if has_wd and has_sat and has_sun and has_typh:
        return "complete"          # all 3 + typhoon
    if has_wd and (has_sat or has_sun) and has_typh:
        return "wd_partial"        # WD + 1 weekend type + typhoon
    if has_wd and has_typh:
        return "wd_only"           # WD baseline only + typhoon
    if has_typh and not has_wd:
        return "typhoon_no_bl"     # typhoon but no WD baseline
    if (has_wd or has_sat or has_sun) and not has_typh:
        return "baseline_only"     # baseline but no typhoon obs
    return "other"

ep["category"] = ep["road_id"].apply(categorize)
print("Category counts:")
print(ep["category"].value_counts())

# ── OSM background ────────────────────────────────────────────────────────────
print("Loading OSM...")
gdf_osm = gpd.read_file(f"{DATA}/osm_cache/hk_roads.gpkg").to_crs("EPSG:3857")

# ── build GeoDataFrames per category ─────────────────────────────────────────
gdf = gpd.GeoDataFrame(ep, geometry="geometry", crs="EPSG:4326").to_crs("EPSG:3857")

cat_cfg = [
    # (category,          color,     label,                                    lw,   zorder)
    ("complete",       "#FFD600", "Complete: all 3 day types + typhoon",      0.13,  1),
    ("wd_partial",     "#00E5FF", "WD+SAT or WD+SUN + typhoon",               0.15,  2),
    ("wd_only",        "#FF6F00", "WD baseline + typhoon (SAT/SUN missing)",  0.15,  3),
    ("typhoon_no_bl",  "#E040FB", "Typhoon observed, no WD baseline",         0.18,  4),
    ("baseline_only",  "#546E7A", "Baseline only (no typhoon observation)",   0.13,  5),
    ("other",          "#1a1a1a", "No usable data",                           0.10,  0),
]

# ── plot ──────────────────────────────────────────────────────────────────────
print("Plotting...")
fig, ax = plt.subplots(figsize=(16, 14))
fig.patch.set_facecolor("#0d0d0d")
ax.set_facecolor("#0d0d0d")

# OSM background
gdf_osm.plot(ax=ax, color="#2a2a2a", linewidth=0.10, alpha=0.7)

# TomTom roads by category (draw less important first)
for cat, color, label, lw, zo in cat_cfg:
    grp = gdf[gdf["category"] == cat]
    if len(grp) == 0: continue
    grp.plot(ax=ax, color=color, linewidth=lw, alpha=0.9, zorder=zo)

try:
    ctx.add_basemap(ax, crs="EPSG:3857",
                    source=ctx.providers.CartoDB.DarkMatter,
                    zoom=12, alpha=0.18)
except Exception as e:
    print(f"  Basemap: {e}")

ax.set_axis_off()
ax.set_title(
    "Road Segment Data Completeness\n"
    "Typhoon vs. Baseline Coverage by Day Type (WORKDAY / SATURDAY / SUNDAY-HOLIDAY)",
    color="white", fontsize=14, pad=14, fontweight="bold")

# legend (skip "other" = hidden roads)
handles = [
    Line2D([0],[0], color=color, lw=2.5,
           label=f"{label}  ({len(gdf[gdf['category']==cat]):,})")
    for cat, color, label, lw, zo in cat_cfg
    if cat != "other"
]
handles.append(
    Line2D([0],[0], color="#2a2a2a", lw=1.5,
           label=f"OSM only / not in TomTom  ({len(gdf_osm):,} edges)"))
ax.legend(handles=handles, loc="lower left", framealpha=0.5,
          facecolor="#0d0d0d", edgecolor="#444444",
          labelcolor="white", fontsize=11, title="Data Completeness",
          title_fontsize=11)

n_core = len(gdf[gdf["category"].isin(["complete","wd_partial","wd_only"])])
fig.text(0.5, 0.015,
    f"Roads usable for typhoon-baseline comparison: {n_core:,}  "
    f"·  Study area: Hong Kong SAR  ·  Data: TomTom floating-car, Sep–Oct 2025",
    ha="center", color="#777777", fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 1])
out = f"{OUT}/图17_数据覆盖完整度图.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="#0d0d0d")
plt.close()
print(f"Saved: {out}")
