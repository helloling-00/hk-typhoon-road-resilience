"""
匹配验证可视化：网格 / 道路 / POI 三层叠加
确认 road → grid、road_feature → road 的空间匹配是否正确
"""
import pandas as pd, numpy as np
import geopandas as gpd, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from shapely.geometry import box, Point, LineString
from shapely import wkb as swkb
import ast, pickle, warnings
warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"

# ── 0. 加载回归宽表 ───────────────────────────────────────────────────────────
print("Loading regression table...", flush=True)
reg = pd.read_parquet(f"{DATA}/regression_table.parquet")
rag = reg[(reg["typhoon"] == "Ragasa") &
          (reg["signal_level"] >= 3) &
          (reg["road_length_m"] >= 100)].copy()

# 按路段聚合（5个时段取均值）
road_y = (rag.groupby("road_id")
             .agg(mean_deviation=("mean_deviation","mean"),
                  pct_pos=("mean_deviation", lambda x:(x>0).mean()))
             .reset_index())
road_feat = (rag.drop_duplicates("road_id")
               [["road_id","work_density","education_density","retail_density",
                 "transport_density","tourism_density","civic_density",
                 "population_density_500m","median_income_500m",
                 "intersection_degree","dist_to_coast_m"]])
road_df = road_y.merge(road_feat, on="road_id", how="left")
print(f"  road_df: {len(road_df):,} roads")

# ── 1. 建路段几何 GeoDataFrame ──────────────────────────────────────────────
print("Building road geometries...", flush=True)
ep_df = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
with open(f"{DATA}/osm_cache/road_wkb_store.pkl","rb") as f:
    wkb_store = pickle.load(f)

def build_geom(row):
    rid = row["road_id"]
    if rid in wkb_store:
        try: return swkb.loads(wkb_store[rid])
        except: pass
    try:
        pts = ast.literal_eval(row["ep_key"])
        return LineString([pts[0], pts[1]])
    except: return None

ep_sub = ep_df[ep_df["road_id"].isin(road_df["road_id"])].copy()
ep_sub["geometry"] = ep_sub.apply(build_geom, axis=1)
ep_sub = ep_sub.dropna(subset=["geometry"])
gdf_roads = (gpd.GeoDataFrame(ep_sub[["road_id","geometry"]],
                               geometry="geometry", crs="EPSG:4326")
               .to_crs("EPSG:3857")
               .drop_duplicates("road_id"))
gdf_roads = gdf_roads.merge(road_df, on="road_id", how="left")
gdf_roads["cx"] = gdf_roads.geometry.centroid.x
gdf_roads["cy"] = gdf_roads.geometry.centroid.y
print(f"  {len(gdf_roads):,} roads with geometry")

# ── 2. 加载 POI ───────────────────────────────────────────────────────────────
print("Loading POI...", flush=True)
lu = pd.read_parquet(f"{DATA}/road_landuse_features.parquet")
# POI 点位从 OSM 重建（快速近似：用道路特征文件 + 已知 buf_area 反算）
# 改为直接用关键 POI 特征展示（无需重解析 pbf）

# ── 3. 定义研究区域（选3个代表性区域） ────────────────────────────────────────
# 使用 EPSG:3857 坐标
AREAS = {
    "旺角/油尖旺（高密度商业）": (12711500, 2558000, 12715500, 12711500+4000, 2558000+4000),
    "中环/上环（商业金融）":      (12706000, 2554000, 12706000+4000, 2554000+4000),
    "沙田（郊区住宅）":           (12726000, 2563000, 12726000+4000, 2563000+4000),
}

# 重新整理坐标格式
AREAS = {
    "Mong Kok / Yau Tsim": {"xmin":12708500,"xmax":12713000,"ymin":2558500,"ymax":2563000},
    "Central / Sheung Wan": {"xmin":12704500,"xmax":12709000,"ymin":2553500,"ymax":2558000},
    "Sha Tin":              {"xmin":12722500,"xmax":12727000,"ymin":2562500,"ymax":2567000},
}

# ── 4. 建 500m 网格 ───────────────────────────────────────────────────────────
GRID_SIZE = 500
x0 = np.floor(gdf_roads["cx"].min() / GRID_SIZE)*GRID_SIZE - GRID_SIZE
x1 = np.ceil( gdf_roads["cx"].max() / GRID_SIZE)*GRID_SIZE + GRID_SIZE
y0 = np.floor(gdf_roads["cy"].min() / GRID_SIZE)*GRID_SIZE - GRID_SIZE
y1 = np.ceil( gdf_roads["cy"].max() / GRID_SIZE)*GRID_SIZE + GRID_SIZE

xs = np.arange(x0, x1, GRID_SIZE)
ys = np.arange(y0, y1, GRID_SIZE)
cells = [{"grid_id":f"{i}_{j}",
          "geometry":box(x, y, x+GRID_SIZE, y+GRID_SIZE)}
         for i,x in enumerate(xs) for j,y in enumerate(ys)]
gdf_grid = gpd.GeoDataFrame(cells, crs="EPSG:3857")

# 道路→网格
gdf_pts = gpd.GeoDataFrame(
    gdf_roads[["road_id","mean_deviation","cx","cy"]],
    geometry=[Point(x,y) for x,y in zip(gdf_roads["cx"], gdf_roads["cy"])],
    crs="EPSG:3857"
)
road_grid_join = gpd.sjoin(
    gdf_pts[["road_id","mean_deviation","geometry"]],
    gdf_grid[["grid_id","geometry"]],
    how="left", predicate="within"
)[["road_id","grid_id","mean_deviation"]].dropna(subset=["grid_id"])

grid_y = (road_grid_join.groupby("grid_id")
          .agg(mean_deviation=("mean_deviation","mean"),
               n_roads=("road_id","nunique"))
          .reset_index()
          .query("n_roads >= 3"))
gdf_grid_y = gdf_grid.merge(grid_y, on="grid_id", how="inner")
print(f"  {len(gdf_grid_y):,} grids with ≥3 roads")

# ── 5. 全香港概览图 ───────────────────────────────────────────────────────────
print("Plotting overview...", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.patch.set_facecolor("#1a1a2e")

vmin, vmax = -0.05, 0.10
cmap = plt.cm.RdYlGn

# 图1：网格 Y 值（mean_deviation，所有时段均值）
ax = axes[0]
ax.set_facecolor("#16213e")
gdf_grid_y.plot(column="mean_deviation", cmap=cmap, vmin=vmin, vmax=vmax,
                ax=ax, linewidth=0.3, edgecolor="#333355", alpha=0.85)
sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
cb.set_label("Speed deviation (km/h)", color="white", fontsize=9)
cb.ax.yaxis.set_tick_params(color="white")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
ax.set_title("Grid mean speed deviation\nRagasa Signal 3+ (red=slower, green=faster)",
             color="white", fontsize=11, pad=10)
ax.set_xlabel(""), ax.set_ylabel("")
ax.tick_params(colors="white", labelsize=7)
for sp in ax.spines.values(): sp.set_edgecolor("#444")

# 矩形标注三个区域
colors_area = ["cyan","yellow","orange"]
for (name, bbox), c in zip(AREAS.items(), colors_area):
    rect = mpatches.Rectangle(
        (bbox["xmin"], bbox["ymin"]),
        bbox["xmax"]-bbox["xmin"], bbox["ymax"]-bbox["ymin"],
        linewidth=2, edgecolor=c, facecolor="none"
    )
    ax.add_patch(rect)
    ax.text(bbox["xmin"], bbox["ymax"]+200, name, color=c, fontsize=7)

# 图2 & 图3：两个放大区域（旺角 + 中环）对比
for idx, (name, bbox) in enumerate(list(AREAS.items())[:2]):
    ax = axes[idx+1]
    ax.set_facecolor("#16213e")

    xlim = (bbox["xmin"], bbox["xmax"])
    ylim = (bbox["ymin"], bbox["ymax"])

    # 该区域的网格
    clip = box(bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"])
    grids_sub = gdf_grid_y[gdf_grid_y.geometry.intersects(clip)]
    roads_sub  = gdf_roads[
        (gdf_roads["cx"] >= bbox["xmin"]) & (gdf_roads["cx"] <= bbox["xmax"]) &
        (gdf_roads["cy"] >= bbox["ymin"]) & (gdf_roads["cy"] <= bbox["ymax"])
    ]

    # 网格底色
    grids_sub.plot(column="mean_deviation", cmap=cmap, vmin=vmin, vmax=vmax,
                   ax=ax, linewidth=0.5, edgecolor="#445566", alpha=0.6)
    # 道路线（按 mean_deviation 着色）
    if len(roads_sub) > 0:
        roads_sub.plot(column="mean_deviation", cmap=cmap, vmin=vmin, vmax=vmax,
                       ax=ax, linewidth=1.5, alpha=0.9)

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title(f"{name}\nGrid (translucent) + Road lines (solid)",
                 color="white", fontsize=10, pad=8)
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    n_r = len(roads_sub)
    n_g = len(grids_sub)
    ax.text(0.02, 0.04, f"Grids={n_g}  Roads={n_r}\nGreen=faster  Red=slower",
            transform=ax.transAxes, color="white", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#223", alpha=0.7))

plt.tight_layout()
out1 = "图31a_网格匹配概览.png"
plt.savefig(out1, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {out1}")

# ── 6. 单个网格放大验证 ─────────────────────────────────────────────────────
# 选两个典型网格：Y最高（最变快）、Y最低（最变慢），展示内部路段特征
print("Plotting single-grid zoom...", flush=True)

# 取高值/低值各3个网格（排除边缘，要求n_roads>=5）
top_grids = (gdf_grid_y.query("n_roads>=5")
             .nlargest(3, "mean_deviation"))
bot_grids = (gdf_grid_y.query("n_roads>=5")
             .nsmallest(3, "mean_deviation"))
showcase = pd.concat([top_grids, bot_grids])

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.patch.set_facecolor("#1a1a2e")
axes = axes.flatten()

for ax_i, (_, grid_row) in enumerate(showcase.iterrows()):
    ax = axes[ax_i]
    ax.set_facecolor("#16213e")

    gx = grid_row.geometry
    cx, cy = gx.centroid.x, gx.centroid.y
    pad = 800
    xlim = (cx - pad, cx + pad)
    ylim = (cy - pad, cy + pad)

    # 周围网格
    clip = box(xlim[0], ylim[0], xlim[1], ylim[1])
    grids_sub = gdf_grid_y[gdf_grid_y.geometry.intersects(clip)]
    roads_sub  = gdf_roads[
        (gdf_roads["cx"] >= xlim[0]) & (gdf_roads["cx"] <= xlim[1]) &
        (gdf_roads["cy"] >= ylim[0]) & (gdf_roads["cy"] <= ylim[1])
    ]

    grids_sub.plot(column="mean_deviation", cmap=cmap, vmin=-0.08, vmax=0.15,
                   ax=ax, linewidth=0.8, edgecolor="#336688", alpha=0.5)
    if len(roads_sub) > 0:
        roads_sub.plot(column="mean_deviation", cmap=cmap, vmin=-0.08, vmax=0.15,
                       ax=ax, linewidth=2.2, alpha=0.95)

    # 高亮目标网格
    gpd.GeoDataFrame([{"geometry": gx}], crs="EPSG:3857").plot(
        ax=ax, facecolor="none", edgecolor="white", linewidth=2.5)

    # 目标网格内路段
    roads_in = road_grid_join[road_grid_join["grid_id"] == grid_row["grid_id"]]
    roads_in_geo = gdf_roads[gdf_roads["road_id"].isin(roads_in["road_id"])]
    if len(roads_in_geo) > 0:
        roads_in_geo.plot(column="mean_deviation", cmap=cmap, vmin=-0.08, vmax=0.15,
                          ax=ax, linewidth=3.5, alpha=1.0)

    ax.set_xlim(xlim); ax.set_ylim(ylim)

    dev = grid_row["mean_deviation"]
    n_r = int(grid_row["n_roads"])
    label = "FASTER" if dev > 0 else "SLOWER"
    color = "lime" if ax_i < 3 else "tomato"

    feat_roads = gdf_roads[gdf_roads["road_id"].isin(roads_in["road_id"])]
    if len(feat_roads) > 0:
        tm  = feat_roads["tourism_density"].mean()
        tr  = feat_roads["transport_density"].mean()
        wk  = feat_roads["work_density"].mean()
        edu = feat_roads["education_density"].mean()
        civ = feat_roads["civic_density"].mean()
        pop = feat_roads["population_density_500m"].mean()
        feat_str = (f"Tourism={tm:.1f}  Transit={tr:.1f}  Work={wk:.1f}\n"
                    f"Edu={edu:.1f}  Civic={civ:.1f}  Pop.dens={pop:.0f}")
    else:
        feat_str = ""

    ax.set_title(f"{label}  dev={dev:+.4f}  roads={n_r}",
                 color=color, fontsize=10, pad=6)
    ax.text(0.02, 0.02, feat_str, transform=ax.transAxes,
            color="white", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#223344", alpha=0.8))
    ax.tick_params(colors="white", labelsize=6)
    for sp in ax.spines.values(): sp.set_edgecolor("#445")
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle("Single-grid zoom: white box = target grid, thick lines = roads inside grid\n"
             "Top row = most improved  |  Bottom row = most degraded  (Ragasa S3+, all periods avg)",
             color="white", fontsize=11, y=1.01)
plt.tight_layout()
out2 = "图31b_单格放大验证.png"
plt.savefig(out2, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {out2}")

# ── 7. 关键变量散点图（X vs Y） ───────────────────────────────────────────────
print("Plotting scatter checks...", flush=True)

# 取日间时段做散点
grid_mid = pd.read_parquet(f"{DATA}/grid_regression_data.parquet")
grid_mid = grid_mid[grid_mid["time_group"] == "MIDDAY"].copy()

plot_vars = [
    ("log_tourism",      "log(Tourism density)"),
    ("log_transport",    "log(Transit hub density)"),
    ("log_education",    "log(Education density)"),
    ("log_intersection", "log(Intersection degree)"),
    ("log_civic",        "log(Civic density)"),
    ("log_pop_density",  "log(Population density)"),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.patch.set_facecolor("#1a1a2e")
axes = axes.flatten()

for i, (xvar, xlabel) in enumerate(plot_vars):
    ax = axes[i]
    ax.set_facecolor("#16213e")
    x = grid_mid[xvar].values
    y = grid_mid["mean_deviation"].values
    mask = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[mask], y[mask], alpha=0.3, s=8, color="#4db8ff")
    # 趋势线
    if mask.sum() > 10:
        z = np.polyfit(x[mask], y[mask], 1)
        p = np.poly1d(z)
        xl = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(xl, p(xl), color="orange", linewidth=1.5)
        corr = np.corrcoef(x[mask], y[mask])[0,1]
        ax.text(0.05, 0.92, f"r={corr:.3f}", transform=ax.transAxes,
                color="orange", fontsize=9)
    ax.axhline(0, color="#666", linewidth=0.8, linestyle="--")
    ax.set_xlabel(xlabel, color="white", fontsize=9)
    ax.set_ylabel("mean_deviation" if i%3==0 else "", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#445")

plt.suptitle("MIDDAY period: grid-level X variables vs. speed deviation\n"
             "Orange line = OLS trend,  r = Pearson correlation",
             color="white", fontsize=11, y=1.02)
plt.tight_layout()
out3 = "图31c_X变量散点验证.png"
plt.savefig(out3, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {out3}")

print("\nDone.")
