"""
脚本_38_湾仔速度图.py
铜锣湾/湾仔区域所有道路：台风前基线速度 vs 台风期Y值(mean_deviation)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from shapely.geometry import LineString
from shapely import wkb
import ast, os
import warnings
warnings.filterwarnings("ignore")

DATA     = "/Users/helloling/workspace/thesis/data"
CRS_GEO  = "EPSG:4326"
AREA_LON = (114.172, 114.200)
AREA_LAT = (114.172, 114.200)

# ── 1. 找区域路段 + ep_key ────────────────────────────────────────────────────
print("Loading road registry...", flush=True)
rr = pd.read_parquet(f"{DATA}/road_registry.parquet")
rr_u = rr.drop_duplicates("road_id").copy()

def parse_ep(s):
    try:
        pts = ast.literal_eval(s)
        return pts[0], pts[1]
    except:
        return None, None

parsed = rr_u["ep_key"].apply(parse_ep)
rr_u["pt1"] = parsed.apply(lambda x: x[0])
rr_u["pt2"] = parsed.apply(lambda x: x[1])
rr_u["cx"]  = parsed.apply(lambda x: (x[0][0]+x[1][0])/2 if x[0] else None)
rr_u["cy"]  = parsed.apply(lambda x: (x[0][1]+x[1][1])/2 if x[0] else None)

mask = (
    (rr_u.cx > 114.172) & (rr_u.cx < 114.200) &
    (rr_u.cy > 22.272)  & (rr_u.cy < 22.294)
)
area = rr_u[mask].copy()
print(f"  Area road_ids: {len(area)}")

# ── 2. 取回归表 ───────────────────────────────────────────────────────────────
print("Loading regression table...", flush=True)
rt = pd.read_parquet(f"{DATA}/regression_table.parquet")

# 台风期 Ragasa S3+ MIDDAY
typhoon_mid = (
    rt[(rt.typhoon=="Ragasa") & (rt.signal_level>=3) & (rt.time_group=="MIDDAY")]
    .groupby("road_id")
    .agg(
        mean_deviation    = ("mean_deviation",    "mean"),
        baseline_tg_speed = ("baseline_tg_speed", "mean"),
        road_broad        = ("road_broad",         "first"),
        road_length_m     = ("road_length_m",      "first"),
    )
    .reset_index()
)

# 台风前基线（用baseline_avg_speed，即全天均值）
baseline = (
    rt[(rt.typhoon=="Ragasa")]
    .groupby("road_id")
    .agg(baseline_avg_speed = ("baseline_avg_speed","mean"))
    .reset_index()
)

# 合并 area roads + 回归结果
df = area[["road_id","pt1","pt2","road_category"]].merge(
    typhoon_mid, on="road_id", how="inner"
).merge(baseline, on="road_id", how="left")

print(f"  Roads with regression data: {len(df)}")
print(f"  mean_deviation range: {df.mean_deviation.min():.3f} to {df.mean_deviation.max():.3f}")

# ── 3. 提取真实几何（从一个台风日slot读取）──────────────────────────────────
print("Extracting real geometries from flow parquet...", flush=True)
flow_dir = f"{DATA}/flow_parquet2/2025-09-19"
slot_files = sorted(os.listdir(flow_dir))
mid_slots = [f for f in slot_files if any(
    x in f for x in ["slot16","slot18","slot20","slot22","slot24","slot26","slot28","slot30"]
)]

target_ep_keys = set(rr_u[rr_u.road_id.isin(df.road_id)]["ep_key"].tolist())

ep_to_geom = {}
for fname in mid_slots[:6]:
    fp = os.path.join(flow_dir, fname)
    flow = pd.read_parquet(fp)
    for _, row in flow.iterrows():
        try:
            g = wkb.loads(row["geometry"])
            coords = (list(g.geoms[0].coords) if g.geom_type == "MultiLineString"
                      else list(g.coords))
            if len(coords) < 2:
                continue
            lon1,lat1 = round(coords[0][0],4), round(coords[0][1],4)
            lon2,lat2 = round(coords[-1][0],4), round(coords[-1][1],4)
            ep = str(((min(lon1,lon2),min(lat1,lat2)),(max(lon1,lon2),max(lat1,lat2))))
            if ep in target_ep_keys and ep not in ep_to_geom:
                ep_to_geom[ep] = g
        except:
            pass
    print(f"  {fname}: {len(ep_to_geom)} geometries found")
    if len(ep_to_geom) >= len(target_ep_keys) * 0.8:
        break

# 用真实几何优先，否则端点直线
def get_geom(row_):
    ep = rr_u[rr_u.road_id == row_["road_id"]]["ep_key"].iloc[0] if len(rr_u[rr_u.road_id==row_["road_id"]])>0 else None
    if ep and ep in ep_to_geom:
        return ep_to_geom[ep], True   # real geometry
    # 退路：端点连线，但只用当两端点距离与road_length接近的时候
    if row_["pt1"] and row_["pt2"]:
        line = LineString([row_["pt1"], row_["pt2"]])
        # 端点距离（度→米，粗估：1度≈111km）
        dx = (row_["pt2"][0]-row_["pt1"][0]) * 111000 * np.cos(np.radians(22.28))
        dy = (row_["pt2"][1]-row_["pt1"][1]) * 111000
        ep_dist = np.sqrt(dx**2 + dy**2)
        road_len = row_.get("road_length_m", 9999)
        if pd.notna(road_len) and ep_dist < road_len * 1.5:
            return line, False
    return None, False

geom_real = df.apply(get_geom, axis=1)
df["geometry"] = geom_real.apply(lambda x: x[0])
df["real_geom"] = geom_real.apply(lambda x: x[1])
# 只保留真实几何，剔除端点直线（避免穿越海湾的假线段）
df = df[df["geometry"].notna() & df["real_geom"]].copy()
print(f"  Using real geometry only: {len(df)} roads")
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=CRS_GEO)
print(f"  GDF: {len(gdf)} roads with geometry")

# ── 4. 作图（两列：左=基线速度，右=台风偏差） ─────────────────────────────────
print("Plotting...", flush=True)
import contextily as ctx

gdf_plot = gdf.to_crs(epsg=3857)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# 颜色方案
# 左图：baseline speed（蓝低→黄高）
# 右图：mean_deviation（红负→白0→绿正）

lw_map = {"motorway": 3.5, "trunk": 3.0, "primary": 2.5,
          "secondary": 2.0, "tertiary": 1.5, "other": 1.0}

# ── 左图：基线速度 ────────────────────────────────────────────────────────────
ax = axes[0]
norm_base = mcolors.Normalize(vmin=0.3, vmax=1.0)
cmap_base = cm.YlOrRd_r

for _, row in gdf_plot.iterrows():
    val = row["baseline_tg_speed"]
    if pd.isna(val): continue
    color = cmap_base(norm_base(val))
    lw    = lw_map.get(str(row.get("road_broad","")), 1.2)
    g = row.geometry
    lines = list(g.geoms) if g.geom_type == "MultiLineString" else [g]
    for line in lines:
        x, y = line.xy
        ax.plot(x, y, color=color, linewidth=lw, solid_capstyle="round")

ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron, zoom=15)

sm1 = cm.ScalarMappable(norm=norm_base, cmap=cmap_base)
sm1.set_array([])
cb1 = plt.colorbar(sm1, ax=ax, shrink=0.6, pad=0.02)
cb1.set_label("Baseline Relative Speed\n(pre-typhoon MIDDAY)", fontsize=9)
ax.set_axis_off()
ax.set_title("Baseline Speed (Ragasa Pre-Typhoon, Midday)", fontsize=11, fontweight="bold")

# 标记研究路段
road28755 = gdf_plot[gdf_plot.road_id == 28755]
if len(road28755) > 0:
    road28755.plot(ax=ax, color="blue", linewidth=3.5, zorder=10)
    cx = road28755.geometry.iloc[0].centroid.x
    cy = road28755.geometry.iloc[0].centroid.y
    ax.annotate("Study road\n(road_id=28755)", xy=(cx, cy),
                xytext=(cx+350, cy+350), fontsize=7.5, color="blue",
                arrowprops=dict(arrowstyle="->", color="blue", lw=1))

# ── 右图：台风偏差 ────────────────────────────────────────────────────────────
ax = axes[1]
dev_abs = gdf_plot["mean_deviation"].abs().quantile(0.95)
norm_dev = mcolors.TwoSlopeNorm(vmin=-dev_abs, vcenter=0, vmax=dev_abs)
cmap_dev = cm.RdYlGn   # 红=变慢，绿=变快

for _, row in gdf_plot.iterrows():
    val = row["mean_deviation"]
    if pd.isna(val): continue
    color = cmap_dev(norm_dev(val))
    lw    = lw_map.get(str(row.get("road_broad","")), 1.2)
    g = row.geometry
    lines = list(g.geoms) if g.geom_type == "MultiLineString" else [g]
    for line in lines:
        x, y = line.xy
        ax.plot(x, y, color=color, linewidth=lw, solid_capstyle="round")

ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron, zoom=15)

sm2 = cm.ScalarMappable(norm=norm_dev, cmap=cmap_dev)
sm2.set_array([])
cb2 = plt.colorbar(sm2, ax=ax, shrink=0.6, pad=0.02)
cb2.set_label("Speed Deviation During Typhoon (Y)\n(typhoon − baseline, MIDDAY, Ragasa S3+)", fontsize=9)
ax.set_axis_off()
ax.set_title("Speed Deviation During Typhoon (Y = mean_deviation)", fontsize=11, fontweight="bold")

if len(road28755) > 0:
    road28755.plot(ax=ax, color="blue", linewidth=3.5, zorder=10)
    cx = road28755.geometry.iloc[0].centroid.x
    cy = road28755.geometry.iloc[0].centroid.y
    ax.annotate("Study road\n(road_id=28755)", xy=(cx, cy),
                xytext=(cx+350, cy+350), fontsize=7.5, color="blue",
                arrowprops=dict(arrowstyle="->", color="blue", lw=1))

# 图例：道路宽度说明
legend_lines = [
    Line2D([0],[0], color="gray", linewidth=3.5, label="Motorway"),
    Line2D([0],[0], color="gray", linewidth=2.5, label="Primary"),
    Line2D([0],[0], color="gray", linewidth=2.0, label="Secondary"),
    Line2D([0],[0], color="gray", linewidth=1.5, label="Tertiary"),
    Line2D([0],[0], color="gray", linewidth=1.0, label="Other"),
    Line2D([0],[0], color="blue", linewidth=2.5, label="Study road (id=28755)"),
]
axes[1].legend(handles=legend_lines, loc="lower right", fontsize=7.5,
               title="Road type (line width)", title_fontsize=8, framealpha=0.85)

fig.suptitle("Wan Chai / Causeway Bay — Pre-Typhoon Speed vs Typhoon Speed Deviation",
             fontsize=13, fontweight="bold", y=1.01)

plt.tight_layout()
out = "/Users/helloling/workspace/thesis/图38_湾仔速度对比图.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()
