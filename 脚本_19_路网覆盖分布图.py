"""
图19：TomTom浮动车数据对香港机动车路网的覆盖验证
单图，重点显示 TomTom 路段（按道路类别着色）对 OSM 机动车参考路网的覆盖情况。
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
WKB_CACHE = f"{DATA}/osm_cache/road_wkb_store.pkl"

# ── load TomTom data ──────────────────────────────────────────────────────────
print("Loading data...")
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
rr = pd.read_parquet(f"{DATA}/road_registry.parquet")[
        ["road_id","road_category"]].drop_duplicates("road_id")

with open(WKB_CACHE, "rb") as f:
    wkb_store = pickle.load(f)
print(f"  WKB store: {len(wkb_store):,} roads")

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

print("Building geometries...")
ep["geometry"] = ep.apply(build_geom, axis=1)
ep = ep.dropna(subset=["geometry"])
ep = ep.merge(rr, on="road_id", how="left")
gdf_tt = gpd.GeoDataFrame(ep, geometry="geometry", crs="EPSG:4326").to_crs("EPSG:3857")

# ── OSM motorized roads only ──────────────────────────────────────────────────
print("Loading OSM motorized roads...")
gdf_osm_all = gpd.read_file(f"{DATA}/osm_cache/hk_roads.gpkg")
_motorized = {"motorway","motorway_link","trunk","trunk_link","primary","primary_link",
              "secondary","secondary_link","tertiary","tertiary_link","residential",
              "unclassified","living_street","service","road"}
def _is_motor(hw):
    if pd.isna(hw): return False
    if str(hw).startswith("["):
        import ast as _a
        tags = _a.literal_eval(hw)
        return any(t in _motorized for t in tags)
    return hw in _motorized
gdf_osm = gdf_osm_all[gdf_osm_all["highway"].apply(_is_motor)].to_crs("EPSG:3857")
print(f"  OSM motorized edges: {len(gdf_osm):,}")

# ── color scheme ──────────────────────────────────────────────────────────────
cat_color = {
    "motorway":      "#FF4444",
    "motorway_link": "#FF8888",
    "trunk":         "#FF9900",
    "trunk_link":    "#FFCC66",
    "primary":       "#FFE033",
    "primary_link":  "#FFF59D",
    "secondary":     "#44BB44",
    "secondary_link":"#99DD99",
    "tertiary":      "#44AAFF",
    "tertiary_link": "#AADDFF",
    "street":        "#CC66FF",
    # service: warm sand-gray, clearly distinct from the OSM cool-dark background
    "service":       "#B0A090",
}

def get_color(cat):
    if pd.isna(cat): return "#B0A090"
    return cat_color.get(str(cat).lower(), "#B0A090")

gdf_tt["color"] = gdf_tt["road_category"].apply(get_color)

def lw_for_cat(cat):
    if pd.isna(cat): return 0.17
    c = str(cat).lower()
    if "motorway" in c: return 0.28
    if "trunk" in c:    return 0.24
    if "primary" in c:  return 0.20
    return 0.17

cat_order = ["service","street","tertiary_link","tertiary",
             "secondary_link","secondary","primary_link","primary",
             "trunk_link","trunk","motorway_link","motorway"]

# ── plot ──────────────────────────────────────────────────────────────────────
print("Plotting...")
fig, ax = plt.subplots(figsize=(16, 14))
fig.patch.set_facecolor("#080c14")   # deep navy-black for whole figure
ax.set_facecolor("#080c14")

# OSM reference network — cool dark navy-gray so land reads warm-dark
# and sea (from basemap) reads cool-dark → visible contrast
gdf_osm.plot(ax=ax, color="#5a7a96", linewidth=0.25, alpha=0.75)

# TomTom roads, finer categories first so major roads draw on top
for cat in cat_order:
    grp = gdf_tt[gdf_tt["road_category"] == cat]
    if len(grp) == 0: continue
    grp.plot(ax=ax, color=get_color(cat), linewidth=lw_for_cat(cat), alpha=0.92)

# Basemap with higher alpha to let land/sea distinction show through
try:
    ctx.add_basemap(ax, crs="EPSG:3857",
                    source=ctx.providers.CartoDB.DarkMatter,
                    zoom=12, alpha=0.38)
except Exception as e:
    print(f"  Basemap: {e}")

ax.set_axis_off()

n_tt = len(gdf_tt)
n_osm = len(gdf_osm)
ax.set_title(
    "TomTom Floating-Car Data: Road Network Coverage in Hong Kong SAR\n"
    "Spatial verification against OSM motorized road network — coloured by TomTom road category",
    color="white", fontsize=13, pad=14, fontweight="bold")

# ── legend ────────────────────────────────────────────────────────────────────
legend_items = [
    ("Motorway  (coverage 100%)",          "#FF4444"),
    ("Trunk  (99% covered, category differs from OSM)", "#FF9900"),
    ("Primary  (99.6%)",                   "#FFE033"),
    ("Secondary  (99.9%)",                 "#44BB44"),
    ("Tertiary  (99.5%)",                  "#44AAFF"),
    ("Street / residential / unclassified  (64–80%)", "#CC66FF"),
    ("Service / other  (59.7%)",           "#B0A090"),
    ("OSM motorized roads not in TomTom\n"
     "  (low-flow residential, service dead-ends)", "#5a7a96"),
]
handles = [Line2D([0],[0], color=c, lw=2.2, label=l) for l, c in legend_items]
leg = ax.legend(handles=handles, loc="lower left", framealpha=0.55,
                facecolor="#0d1420", edgecolor="#445566",
                labelcolor="white", fontsize=10.5,
                title="Road Category  (coverage vs OSM reference)",
                title_fontsize=11)
leg.get_title().set_color("white")

fig.text(
    0.5, 0.012,
    f"TomTom floating-car data: {n_tt:,} road segments  ·  "
    f"OSM motorized reference: {n_osm:,} edges  ·  "
    f"Coverage verification: 50 m spatial matching  ·  "
    f"Study area: Hong Kong SAR (113.82°–114.44°E, 22.19°–22.58°N)",
    ha="center", color="#7799aa", fontsize=9.5)

plt.tight_layout(rect=[0, 0.03, 1, 1])
out_path = f"{OUT}/图19_路网覆盖分布图.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#080c14")
plt.close()
print(f"\nSaved: {out_path}")
