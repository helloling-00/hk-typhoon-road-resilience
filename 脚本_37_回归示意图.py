"""
脚本_37_回归示意图.py
以湾仔一条真实primary道路（road_id=28755）为例，
展示回归模型中用到的所有X变量的空间含义
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch
from shapely.geometry import LineString, Point, box
from shapely import wkb
import osmium
import ast
import warnings
warnings.filterwarnings("ignore")

ROAD_ID   = 28755
PBF       = "/Users/helloling/workspace/thesis/hong-kong-260502.osm.pbf"
DATA      = "/Users/helloling/workspace/thesis/data"
BUFFER_M  = 500
CRS_GEO   = "EPSG:4326"
CRS_PROJ  = "EPSG:32650"   # UTM zone 50N (Hong Kong)

# ── 1. 道路几何 ───────────────────────────────────────────────────────────────
print("Loading road geometry...", flush=True)
road_line = LineString([
    (114.183985, 22.282276), (114.183572, 22.282115),
    (114.183234, 22.281981), (114.182896, 22.281854),
    (114.182684, 22.281772), (114.182316, 22.281636),
    (114.18229,  22.281623), (114.182048, 22.281537),
    (114.18193,  22.281484), (114.181758, 22.281467),
    (114.181512, 22.28138),  (114.181394, 22.281343),
    (114.180973, 22.281184), (114.18094,  22.281177),
    (114.180916, 22.281167), (114.180777, 22.281127),
    (114.1806,   22.281117), (114.180184, 22.281008),
    (114.180004, 22.280956), (114.17987,  22.280906),
    (114.179808, 22.280879)
])

gdf_road = gpd.GeoDataFrame({"road_id": [ROAD_ID]},
                              geometry=[road_line], crs=CRS_GEO)
gdf_road_proj = gdf_road.to_crs(CRS_PROJ)
buffer_proj = gdf_road_proj.geometry.iloc[0].buffer(BUFFER_M)
buffer_geo  = gpd.GeoSeries([buffer_proj], crs=CRS_PROJ).to_crs(CRS_GEO).iloc[0]

bx_tuple = buffer_geo.bounds   # (minx, miny, maxx, maxy)
print(f"Buffer bounds: {bx_tuple}")

class _BX:
    def __init__(self, t): self.minx,self.miny,self.maxx,self.maxy = t
bx = _BX(bx_tuple)

# ── 2. 提取缓冲区内POI ────────────────────────────────────────────────────────
print("Extracting POI from PBF...", flush=True)

POI_CAT_COLORS = {
    "work":        ("#4e79a7", "Work"),
    "education":   ("#f28e2b", "Education"),
    "retail":      ("#e15759", "Retail"),
    "food_drink":  ("#76b7b2", "Food & Drink"),
    "recreation":  ("#59a14f", "Recreation"),
    "medical":     ("#edc948", "Medical"),
    "transport":   ("#b07aa1", "Transport"),
    "tourism":     ("#ff9da7", "Tourism"),
    "finance":     ("#9c755f", "Finance"),
    "civic":       ("#bab0ac", "Civic"),
}

def classify_node(tags):
    amenity = tags.get("amenity","")
    shop    = tags.get("shop","")
    tourism = tags.get("tourism","")
    office  = tags.get("office","")
    leisure = tags.get("leisure","")
    railway = tags.get("railway","")
    pt      = tags.get("public_transport","")
    landuse = tags.get("landuse","")

    if railway in ("station","subway_entrance","tram_stop","halt") or \
       pt in ("station","stop_position","platform") or \
       amenity in ("bus_station","ferry_terminal","taxi","parking"):
        return "transport"
    if amenity in ("hospital","clinic","pharmacy","doctors","dentist","veterinary"):
        return "medical"
    if amenity in ("school","university","college","kindergarten","library") or \
       landuse == "education":
        return "education"
    if amenity in ("restaurant","cafe","fast_food","bar","pub","food_court","ice_cream"):
        return "food_drink"
    if shop in ("supermarket","convenience","mall","department_store","clothes",
                "electronics","hardware","furniture","books","bakery") or \
       amenity == "marketplace":
        return "retail"
    if tourism in ("hotel","hostel","guest_house","attraction","museum","viewpoint",
                   "zoo","aquarium","theme_park","gallery") or \
       amenity in ("arts_centre","cinema","theatre"):
        return "tourism"
    if amenity in ("bank","atm") or office == "financial":
        return "finance"
    if leisure in ("park","sports_centre","fitness_centre","swimming_pool",
                   "playground","pitch","golf_course") or \
       amenity in ("gym","sports_hall"):
        return "recreation"
    if office in ("company","commercial","it","consulting","architect",
                  "engineer","insurance","lawyer","ngo") or \
       landuse in ("commercial","office","industrial"):
        return "work"
    if amenity in ("police","post_office","fire_station","courthouse",
                   "townhall","community_centre","social_facility") or \
       office == "government":
        return "civic"
    return None

class POIHandler(osmium.SimpleHandler):
    def __init__(self, bbox):
        super().__init__()
        self.bbox = bbox  # (minx, miny, maxx, maxy)
        self.features = []

    def node(self, n):
        if not n.location.valid():
            return
        lon, lat = n.location.lon, n.location.lat
        bx = self.bbox
        if not (bx[0] <= lon <= bx[2] and bx[1] <= lat <= bx[3]):
            return
        tags = {t.k: t.v for t in n.tags}
        cat = classify_node(tags)
        if cat:
            self.features.append({"cat": cat, "lon": lon, "lat": lat})

# 用稍大范围提取（buffer_geo的bbox）
bbox = (bx.minx - 0.002, bx.miny - 0.002, bx.maxx + 0.002, bx.maxy + 0.002)
handler = POIHandler(bbox)
handler.apply_file(PBF, locations=True)
poi_df = pd.DataFrame(handler.features)
print(f"  Raw POI in area: {len(poi_df)}")

if len(poi_df) > 0:
    gdf_poi = gpd.GeoDataFrame(
        poi_df,
        geometry=[Point(r.lon, r.lat) for r in poi_df.itertuples()],
        crs=CRS_GEO
    )
    # 只保留buffer内
    gdf_poi = gdf_poi[gdf_poi.geometry.within(buffer_geo)].copy()
    print(f"  POI within buffer: {len(gdf_poi)}")
    print(gdf_poi["cat"].value_counts().to_string())

# ── 3. 屋苑坐标 ───────────────────────────────────────────────────────────────
est = pd.read_parquet(f"{DATA}/estate_features.parquet")
gdf_est = gpd.GeoDataFrame(
    est, geometry=gpd.points_from_xy(est.lon, est.lat), crs=CRS_GEO
)
gdf_est_nearby = gdf_est[gdf_est.geometry.within(buffer_geo)].copy()
print(f"Estates in buffer: {len(gdf_est_nearby)}")
if len(gdf_est_nearby) > 0:
    print(gdf_est_nearby[["estate","total_pop","median_income"]].to_string())

# ── 4. 回归变量实际值 ─────────────────────────────────────────────────────────
rt = pd.read_parquet(f"{DATA}/regression_table.parquet")
row = rt[(rt.road_id == ROAD_ID) & (rt.typhoon == "Ragasa") &
         (rt.signal_level >= 3) & (rt.time_group == "MIDDAY")].iloc[0]

# ── 5. 作图 ───────────────────────────────────────────────────────────────────
print("Plotting...", flush=True)

import contextily as ctx

fig, axes = plt.subplots(1, 2, figsize=(16, 9),
                          gridspec_kw={"width_ratios": [1.6, 1]})
ax_map, ax_tab = axes

# ── 5a. 地图 ─────────────────────────────────────────────────────────────────
# 转投影后画图
gdf_road_plot  = gdf_road.to_crs(epsg=3857)
buffer_plot    = gpd.GeoDataFrame(geometry=[buffer_geo], crs=CRS_GEO).to_crs(epsg=3857)
gdf_poi_plot   = gdf_poi.to_crs(epsg=3857) if len(gdf_poi) > 0 else None
gdf_est_plot   = gdf_est_nearby.to_crs(epsg=3857) if len(gdf_est_nearby) > 0 else None

# Buffer
buffer_plot.plot(ax=ax_map, color="#4682B4", alpha=0.10, zorder=1)
buffer_plot.boundary.plot(ax=ax_map, color="#4682B4", linewidth=1.2,
                           linestyle="--", alpha=0.6, zorder=2)

# POI散点
if gdf_poi_plot is not None and len(gdf_poi_plot) > 0:
    for cat, (color, label) in POI_CAT_COLORS.items():
        sub = gdf_poi_plot[gdf_poi_plot["cat"] == cat]
        if len(sub) > 0:
            sub.plot(ax=ax_map, color=color, markersize=18, alpha=0.75,
                     zorder=3, marker="o")

# 屋苑
if gdf_est_plot is not None and len(gdf_est_plot) > 0:
    gdf_est_plot.plot(ax=ax_map, color="gold", markersize=120, marker="*",
                      zorder=5, edgecolor="black", linewidth=0.5)
    for _, r in gdf_est_plot.iterrows():
        x, y = r.geometry.x, r.geometry.y
        ax_map.annotate(r["estate"], xy=(x, y), xytext=(6, 6),
                        textcoords="offset points",
                        fontsize=7, color="black",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

# 道路（加粗红色）
gdf_road_plot.plot(ax=ax_map, color="crimson", linewidth=4, zorder=6)

# basemap
ctx.add_basemap(ax_map, crs="EPSG:3857",
                source=ctx.providers.CartoDB.Positron, zoom=15)

# 标注关键地点
road_x = gdf_road_plot.geometry.iloc[0].centroid.x
road_y = gdf_road_plot.geometry.iloc[0].centroid.y
ax_map.annotate("Study Road\n(road_id=28755, primary, ~496m)",
                xy=(road_x, road_y), xytext=(road_x + 300, road_y + 350),
                fontsize=8, color="crimson", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2))

# 500m缓冲区标注
buf_center = buffer_plot.geometry.iloc[0].centroid
ax_map.annotate("500 m line buffer\n(capsule shape)",
                xy=(buf_center.x - 550, buf_center.y + 450),
                fontsize=8, color="#4682B4", style="italic",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

ax_map.set_axis_off()
ax_map.set_title("Regression Variables: Wan Chai Primary Road (road_id 28755)",
                  fontsize=11, fontweight="bold", pad=8)

# POI图例
poi_handles = [
    mlines.Line2D([], [], color=color, marker="o", markersize=6,
                  linestyle="None", label=f"{label} (POI)")
    for cat, (color, label) in POI_CAT_COLORS.items()
    if gdf_poi_plot is not None and (gdf_poi_plot["cat"] == cat).sum() > 0
]
road_handle  = mlines.Line2D([], [], color="crimson", linewidth=2.5,
                               linestyle="-", label="Study road (primary)")
buf_handle   = mpatches.Patch(facecolor="#4682B4", alpha=0.2,
                               edgecolor="#4682B4", linestyle="--",
                               label="500 m buffer")
est_handle   = mlines.Line2D([], [], color="gold", marker="*", markersize=10,
                               linestyle="None", markeredgecolor="black",
                               label="Residential estate (census)")
ax_map.legend(handles=[road_handle, buf_handle, est_handle] + poi_handles,
              loc="lower left", fontsize=7, framealpha=0.85,
              title="Legend", title_fontsize=8)

# ── 5b. 变量值表格 ────────────────────────────────────────────────────────────
ax_tab.set_axis_off()

# POI counts in buffer
poi_counts = {cat: int((gdf_poi["cat"] == cat).sum()) for cat in POI_CAT_COLORS} if len(gdf_poi) > 0 else {cat: 0 for cat in POI_CAT_COLORS}
poi_dens   = {cat: round(row.get(f"{cat}_density", 0), 1) for cat in POI_CAT_COLORS}

table_data = [
    ["Variable", "Group", "Value", "Unit"],
    # Y
    ["mean_deviation (Y)", "Outcome", f"{row['mean_deviation']:+.3f}", "rel. speed"],
    ["baseline_speed",     "Outcome", f"{row['baseline_tg_speed']:.3f}", "rel. speed"],
    # Road structure
    ["road_length",        "Road structure", f"{row['road_length_m']:.0f}", "m"],
    ["road_category",      "Road structure", str(row['road_broad']), "category"],
    ["intersection_degree","Road structure", f"{row['intersection_degree']:.0f}", "count"],
    ["dist_to_coast",      "Road structure", f"{row['dist_to_coast_m']:.0f}", "m"],
    # POI (counts + density)
    ["work_density",       "POI demand", f"{poi_counts['work']} pts / {poi_dens['work']}", "n / per km²"],
    ["education_density",  "POI demand", f"{poi_counts['education']} pts / {poi_dens['education']}", "n / per km²"],
    ["retail_density",     "POI demand", f"{poi_counts['retail']} pts / {poi_dens['retail']}", "n / per km²"],
    ["food_drink_density", "POI demand", f"{poi_counts['food_drink']} pts / {poi_dens['food_drink']}", "n / per km²"],
    ["recreation_density", "POI demand", f"{poi_counts['recreation']} pts / {poi_dens['recreation']}", "n / per km²"],
    ["medical_density",    "POI demand", f"{poi_counts['medical']} pts / {poi_dens['medical']}", "n / per km²"],
    ["transport_density",  "POI demand", f"{poi_counts['transport']} pts / {poi_dens['transport']}", "n / per km²"],
    ["tourism_density",    "POI demand", f"{poi_counts['tourism']} pts / {poi_dens['tourism']}", "n / per km²"],
    ["finance_density",    "POI demand", f"{poi_counts['finance']} pts / {poi_dens['finance']}", "n / per km²"],
    ["civic_density",      "POI demand", f"{poi_counts['civic']} pts / {poi_dens['civic']}", "n / per km²"],
    # Demographics
    ["population_density", "Demographics", f"{row['population_density_500m']:,.0f}", "persons/km²"],
    ["median_income",      "Demographics", f"{row['median_income_500m']:,.0f}", "HKD/month"],
    ["working_pop_ratio",  "Demographics", f"{row['working_pop_ratio_500m']:.3f}", "fraction"],
    ["ratio_age_65+",      "Demographics", f"{row['ratio_age_65plus_500m']:.3f}", "fraction"],
    # Signal
    ["signal_group",       "Typhoon control", str(row['signal_group']), "S3/S8/S10"],
]

group_colors = {
    "Outcome":         "#fce8e8",
    "Road structure":  "#e8f0fe",
    "POI demand":      "#e8fce8",
    "Demographics":    "#fff8e1",
    "Typhoon control": "#f3e8fc",
}

y0, dy = 0.98, 0.046
for i, row_data in enumerate(table_data):
    grp = row_data[1]
    bg = group_colors.get(grp, "white") if grp != "Group" else "#eeeeee"
    if i == 0:
        bg = "#333333"
        fc = "white"
        fw = "bold"
    else:
        fc = "black"
        fw = "normal" if grp not in ("Outcome",) else "bold"

    y = y0 - i * dy
    ax_tab.add_patch(mpatches.FancyBboxPatch(
        (0.0, y - dy * 0.85), 1.0, dy * 0.88,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=bg, transform=ax_tab.transAxes, clip_on=False
    ))
    xs = [0.02, 0.34, 0.62, 0.82]
    for xi, txt in zip(xs, row_data):
        ax_tab.text(xi, y - dy * 0.35, txt,
                    transform=ax_tab.transAxes,
                    fontsize=7.2, va="center", color=fc,
                    fontweight=fw if xi == xs[0] else "normal")

# group color legend
legend_x, legend_y = 0.02, y - dy * 1.5
for grp, col in group_colors.items():
    ax_tab.add_patch(mpatches.Rectangle(
        (legend_x, legend_y), 0.025, 0.018,
        transform=ax_tab.transAxes, color=col,
        linewidth=0.5, edgecolor="gray"
    ))
    ax_tab.text(legend_x + 0.03, legend_y + 0.009, grp,
                transform=ax_tab.transAxes, fontsize=6.5, va="center")
    legend_x += 0.21

ax_tab.set_title("X and Y Variable Values for This Road\n(Typhoon Ragasa, Signal 3, Midday)",
                  fontsize=9.5, fontweight="bold", pad=8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = "/Users/helloling/workspace/thesis/图37_回归示意图.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()
