"""
Road network coverage visualization (high-res, real WKB geometries):
  Left: OSM full HK road network (gray) + TomTom roads (colored by category)
  Right: TomTom roads colored by mean baseline speed
"""
import ast, pickle, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely import wkb as shapely_wkb
from shapely.geometry import LineString
import geopandas as gpd
import contextily as ctx
import warnings; warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
OUT  = "/Users/helloling/workspace/thesis"
OSM_CACHE    = f"{DATA}/osm_cache/hk_roads.gpkg"
WKB_CACHE    = f"{DATA}/osm_cache/road_wkb_store.pkl"

# ── load TomTom data ──────────────────────────────────────────────────────────
print("Loading data...")
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
rr = pd.read_parquet(f"{DATA}/road_registry.parquet")[
        ["road_id","road_category"]].drop_duplicates("road_id")
bl = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
road_bl = bl.groupby("road_id")["mean_speed"].mean().rename("mean_baseline")

with open(WKB_CACHE, "rb") as f:
    wkb_store = pickle.load(f)
print(f"  WKB store: {len(wkb_store):,} roads with real geometry")

# build geometry: real WKB where available, endpoint line as fallback
def build_geom(row):
    rid = row["road_id"]
    if rid in wkb_store:
        try:
            return shapely_wkb.loads(wkb_store[rid])
        except:
            pass
    try:
        pts = ast.literal_eval(row["ep_key"])
        return LineString([pts[0], pts[1]])
    except:
        return None

print("Building geometries (real WKB + fallback)...")
ep["geometry"] = ep.apply(build_geom, axis=1)
ep = ep.dropna(subset=["geometry"])
ep = ep.merge(rr, on="road_id", how="left")
ep = ep.merge(road_bl, on="road_id", how="left")
ep["geom_type"] = ep["road_id"].apply(
    lambda r: "wkb" if r in wkb_store else "fallback")
print(f"  Real WKB: {(ep['geom_type']=='wkb').sum():,}  "
      f"Fallback lines: {(ep['geom_type']=='fallback').sum():,}")

gdf_tt = gpd.GeoDataFrame(ep, geometry="geometry", crs="EPSG:4326").to_crs("EPSG:3857")

# ── OSM reference network ─────────────────────────────────────────────────────
print("Loading OSM motorized roads only...")
gdf_osm_all = gpd.read_file(OSM_CACHE)
_motorized = {"motorway","motorway_link","trunk","trunk_link","primary","primary_link",
              "secondary","secondary_link","tertiary","tertiary_link","residential",
              "unclassified","living_street","service","road"}
def _is_motor(hw):
    if pd.isna(hw): return False
    import ast as _ast
    if str(hw).startswith("["):
        tags = _ast.literal_eval(hw)
        return any(t in _motorized for t in tags)
    return hw in _motorized
gdf_osm = gdf_osm_all[gdf_osm_all["highway"].apply(_is_motor)].to_crs("EPSG:3857")
print(f"  OSM motorized edges: {len(gdf_osm):,} (of {len(gdf_osm_all):,} total)")

# ── color maps ────────────────────────────────────────────────────────────────
cat_color = {
    "motorway": "#FF4444", "motorway_link": "#FF8888",
    "trunk":    "#FF9900", "trunk_link":    "#FFCC66",
    "primary":  "#FFE033", "primary_link":  "#FFF59D",
    "secondary":"#44BB44", "secondary_link":"#99DD99",
    "tertiary": "#44AAFF", "tertiary_link": "#AADDFF",
    "street":   "#CC66FF",
    "service":  "#888888",
}

def get_color(cat):
    if pd.isna(cat):
        return "#888888"
    return cat_color.get(str(cat).lower(), "#888888")

gdf_tt["color"] = gdf_tt["road_category"].apply(get_color)

# ── PLOT ──────────────────────────────────────────────────────────────────────
print("Plotting (high-res)...")
fig, axes = plt.subplots(1, 2, figsize=(26, 15))
fig.patch.set_facecolor("#0d0d0d")

# same thin weight as OSM gray background
def lw_for_cat(cat):
    if pd.isna(cat): return 0.12
    c = str(cat).lower()
    if "motorway" in c: return 0.22
    if "trunk" in c:    return 0.18
    if "primary" in c:  return 0.15
    return 0.12

# ── Left: coverage map ────────────────────────────────────────────────────────
ax1 = axes[0]
ax1.set_facecolor("#0d0d0d")

# OSM background (thin gray)
gdf_osm.plot(ax=ax1, color="#404040", linewidth=0.12, alpha=0.7)

# TomTom roads, finer roads drawn first (so major roads appear on top)
cat_order = ["service","street","tertiary_link","tertiary",
             "secondary_link","secondary","primary_link","primary",
             "trunk_link","trunk","motorway_link","motorway"]
for cat in cat_order:
    grp = gdf_tt[gdf_tt["road_category"] == cat]
    if len(grp) == 0: continue
    grp.plot(ax=ax1, color=get_color(cat), linewidth=lw_for_cat(cat), alpha=0.9)

try:
    ctx.add_basemap(ax1, crs="EPSG:3857",
                    source=ctx.providers.CartoDB.DarkMatter,
                    zoom=12, alpha=0.18)
except Exception as e:
    print(f"  Basemap: {e}")

n_osm = len(gdf_osm); n_tt = len(gdf_tt)
ax1.set_title(
    f"Road Network Coverage\n"
    f"Gray = OSM motorized roads ({n_osm:,} edges)  ·  Colored = TomTom segments ({n_tt:,})",
    color="white", fontsize=13, pad=12)
ax1.set_axis_off()

legend_items = [
    ("Motorway",         "#FF4444"),
    ("Trunk",            "#FF9900"),
    ("Primary",          "#FFE033"),
    ("Secondary",        "#44BB44"),
    ("Tertiary",         "#44AAFF"),
    ("Street",           "#CC66FF"),
    ("Service / Other",  "#888888"),
    ("OSM motorized road, no TomTom data\n  (mainly service/residential, low traffic)", "#404040"),
]
handles = [Line2D([0],[0], color=c, lw=2.2, label=l) for l, c in legend_items]
ax1.legend(handles=handles, loc="lower left", framealpha=0.45,
           facecolor="#0d0d0d", edgecolor="#555555",
           labelcolor="white", fontsize=10.5)

# ── Right: baseline speed distribution ───────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor("#0d0d0d")

gdf_osm.plot(ax=ax2, color="#282828", linewidth=0.10, alpha=0.6)

bins   = [0.0, 0.35, 0.50, 0.65, 0.80, 1.01]
blabels= ["< 0.35 (congested)", "0.35 - 0.50", "0.50 - 0.65",
          "0.65 - 0.80", "> 0.80 (free-flow)"]
bcolors= ["#D50000", "#FF6D00", "#FFD600", "#00C853", "#0091EA"]

gdf_tt["speed_bin"] = pd.cut(gdf_tt["mean_baseline"], bins=bins,
                              labels=blabels, right=False)
for lbl, col in zip(blabels, bcolors):
    grp = gdf_tt[gdf_tt["speed_bin"] == lbl]
    if len(grp) == 0: continue
    lw = 0.14
    grp.plot(ax=ax2, color=col, linewidth=lw, alpha=0.88)
    print(f"  {lbl}: {len(grp):,}")

try:
    ctx.add_basemap(ax2, crs="EPSG:3857",
                    source=ctx.providers.CartoDB.DarkMatter,
                    zoom=12, alpha=0.18)
except:
    pass

ax2.set_title(
    "Baseline Speed Distribution\n"
    "(mean relative speed vs free-flow, averaged across all times of day)",
    color="white", fontsize=13, pad=12)
ax2.set_axis_off()

handles2 = [Line2D([0],[0], color=c, lw=2.2, label=l)
            for l, c in zip(blabels, bcolors)]
ax2.legend(handles=handles2, loc="lower left", framealpha=0.45,
           facecolor="#0d0d0d", edgecolor="#555555",
           labelcolor="white", fontsize=10.5)

fig.text(
    0.5, 0.005,
    f"TomTom floating-car data: {n_tt:,} road segments  ·  "
    f"OSM reference network: {n_osm:,} edges  ·  "
    f"Coverage: 113.82°–114.44°E, 22.19°–22.58°N (Hong Kong SAR)",
    ha="center", color="#777777", fontsize=10)

plt.tight_layout(rect=[0, 0.02, 1, 1])
out_path = f"{OUT}/图16_路网覆盖对比图.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#0d0d0d")
plt.close()
print(f"\nSaved: {out_path}")
