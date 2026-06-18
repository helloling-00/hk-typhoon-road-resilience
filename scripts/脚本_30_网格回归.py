"""
网格回归：叶加莎 Ragasa Signal 3+
500×500m 网格聚合 → OLS（5个时段）
Y = 格内道路 mean_deviation 均值（台风速度 − 同槽正常基线）
X = 格内道路特征均值
"""
import pandas as pd, numpy as np
import geopandas as gpd
from shapely.geometry import box, Point, LineString
from shapely import wkb as swkb
import statsmodels.formula.api as smf
import ast, pickle, warnings
warnings.filterwarnings("ignore")

DATA  = "/Users/helloling/workspace/thesis/data"
GRID_SIZE = 500  # meters

# ── 1. 加载 Ragasa Signal≥3 ─────────────────────────────────────────────────
print("Loading data...", flush=True)
reg = pd.read_parquet(f"{DATA}/regression_table.parquet")
rag = reg[(reg["typhoon"]  == "Ragasa") &
          (reg["signal_level"] >= 3)    &
          (reg["road_length_m"] >= 100)].copy()
print(f"  Ragasa S3+, len≥100m: {len(rag):,} rows  {rag['road_id'].nunique():,} roads")
print(f"  signal分布: {rag['signal_level'].value_counts().sort_index().to_dict()}")

# ── 2. 道路中心点 ─────────────────────────────────────────────────────────────
print("\nBuilding road centroids...", flush=True)
ep_df = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
with open(f"{DATA}/osm_cache/road_wkb_store.pkl", "rb") as f:
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

ep_df["geometry"] = ep_df.apply(build_geom, axis=1)
ep_df = ep_df.dropna(subset=["geometry"])
gdf_r = gpd.GeoDataFrame(ep_df[["road_id","geometry"]],
                          geometry="geometry", crs="EPSG:4326").to_crs("EPSG:3857")
gdf_r["cx"] = gdf_r.geometry.centroid.x
gdf_r["cy"] = gdf_r.geometry.centroid.y
road_xy = gdf_r[["road_id","cx","cy"]].drop_duplicates("road_id")

# ── 3. 创建 500m 网格 ─────────────────────────────────────────────────────────
print("Creating grid...", flush=True)
sample_xy = road_xy[road_xy["road_id"].isin(rag["road_id"])]
x0 = np.floor(sample_xy["cx"].min() / GRID_SIZE) * GRID_SIZE - GRID_SIZE
x1 = np.ceil( sample_xy["cx"].max() / GRID_SIZE) * GRID_SIZE + GRID_SIZE
y0 = np.floor(sample_xy["cy"].min() / GRID_SIZE) * GRID_SIZE - GRID_SIZE
y1 = np.ceil( sample_xy["cy"].max() / GRID_SIZE) * GRID_SIZE + GRID_SIZE

xs = np.arange(x0, x1, GRID_SIZE)
ys = np.arange(y0, y1, GRID_SIZE)
cells = [{"grid_id": f"{i}_{j}",
          "grid_cx": x + GRID_SIZE/2,
          "grid_cy": y + GRID_SIZE/2,
          "geometry": box(x, y, x+GRID_SIZE, y+GRID_SIZE)}
         for i, x in enumerate(xs) for j, y in enumerate(ys)]
gdf_grid = gpd.GeoDataFrame(cells, crs="EPSG:3857")
print(f"  {len(xs)}×{len(ys)} = {len(gdf_grid):,} cells")

# ── 4. 道路→网格 空间连接 ────────────────────────────────────────────────────
gdf_pts = gpd.GeoDataFrame(
    road_xy,
    geometry=[Point(x, y) for x, y in zip(road_xy["cx"], road_xy["cy"])],
    crs="EPSG:3857"
)
road_grid = gpd.sjoin(
    gdf_pts[["road_id","geometry"]],
    gdf_grid[["grid_id","grid_cx","grid_cy","geometry"]],
    how="left", predicate="within"
)[["road_id","grid_id","grid_cx","grid_cy"]].dropna(subset=["grid_id"])
print(f"  {road_grid['road_id'].nunique():,} roads assigned to {road_grid['grid_id'].nunique():,} grids")

# ── 5. 合并网格 ID ────────────────────────────────────────────────────────────
rag = rag.merge(road_grid, on="road_id", how="inner")

# ── 6. 特征工程（路段级） ──────────────────────────────────────────────────────
LU_CATS = ["work","education","retail","food_drink","recreation",
           "medical","transport","tourism","finance","civic"]
for cat in LU_CATS:
    rag[f"log_{cat}"] = np.log1p(rag[f"{cat}_density"].fillna(0))

for col in ["employed_ratio_500m","student_ratio_500m",
            "retiree_ratio_500m","elderly_ratio_500m"]:
    rag[col] = rag[col].fillna(0)

for col in ["population_density_500m","median_income_500m",
            "intersection_degree","dist_to_coast_m","road_length_m"]:
    rag[col] = rag[col].fillna(rag[col].median())

# ── 7. 聚合到 网格 × 时段 ────────────────────────────────────────────────────
FEAT_COLS = ([f"log_{c}" for c in LU_CATS] +
             ["population_density_500m","median_income_500m",
              "employed_ratio_500m","student_ratio_500m",
              "retiree_ratio_500m","elderly_ratio_500m",
              "intersection_degree","dist_to_coast_m","road_length_m"])

PERIODS = [
    ("NIGHT",   "凌晨  NIGHT"),
    ("AM_PEAK", "早高峰 AM_PEAK"),
    ("MIDDAY",  "日间  MIDDAY"),
    ("PM_PEAK", "晚高峰 PM_PEAK"),
    ("EVENING", "夜间  EVENING"),
]

def make_grid_df(sub, min_roads=3):
    y_agg = sub.groupby("grid_id").agg(
        mean_deviation = ("mean_deviation", "mean"),
        pct_pos        = ("mean_deviation", lambda x: (x > 0).mean()),
        n_roads        = ("road_id", "nunique"),
        n_obs          = ("mean_deviation", "count"),
    ).reset_index()

    x_agg = sub.groupby("grid_id")[FEAT_COLS].mean().reset_index()
    coord  = sub[["grid_id","grid_cx","grid_cy"]].drop_duplicates("grid_id")

    g = (y_agg.merge(x_agg, on="grid_id")
              .merge(coord, on="grid_id")
              .query(f"n_roads >= {min_roads}"))

    # 对非比例 X 做 log
    g["log_pop_density"]  = np.log1p(g["population_density_500m"])
    g["log_income"]       = np.log1p(g["median_income_500m"])
    g["log_intersection"] = np.log1p(g["intersection_degree"])
    g["log_dist_coast"]   = np.log1p(g["dist_to_coast_m"])
    return g

grid_data = {}
for tg, label in PERIODS:
    sub = rag[rag["time_group"] == tg]
    g = make_grid_df(sub)
    grid_data[tg] = g
    print(f"  {label}: {len(g):,} 格  "
          f"median roads/grid={g['n_roads'].median():.0f}  "
          f"Y均值={g['mean_deviation'].mean():+.4f}  "
          f"变快格占比={(g['mean_deviation']>0).mean():.1%}")

# ── 8. 回归 ──────────────────────────────────────────────────────────────────
LU_TERMS    = " + ".join(f"log_{c}" for c in LU_CATS)
DEMO_TERMS  = ("log_pop_density + log_income + "
               "employed_ratio_500m + student_ratio_500m + "
               "retiree_ratio_500m + elderly_ratio_500m")
STRUCT_TERMS = "log_intersection + log_dist_coast"

FORMULA = f"mean_deviation ~ {LU_TERMS} + {DEMO_TERMS} + {STRUCT_TERMS}"

print("\n\n" + "═"*72)
print("  网格级回归结果（Ragasa Signal 3+，≥3条路/格）")
print("═"*72)

reg_results = {}
KEY_VARS = [f"log_{c}" for c in LU_CATS] + [
    "log_pop_density","log_income",
    "employed_ratio_500m","student_ratio_500m",
    "retiree_ratio_500m","elderly_ratio_500m",
    "log_intersection","log_dist_coast"
]

for tg, label in PERIODS:
    g = grid_data[tg]
    if len(g) < 50:
        print(f"\n{label}: 样本太少({len(g)})，跳过")
        continue

    try:
        res = smf.ols(FORMULA, data=g).fit()
    except Exception as e:
        print(f"\n{label}: 失败 ({e})")
        continue

    tbl = pd.DataFrame({
        "coef": res.params, "se": res.bse,
        "t": res.tvalues,   "p": res.pvalues,
    })
    tbl["sig"] = tbl["p"].apply(
        lambda p: "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "."
    )
    reg_results[tg] = (res, tbl)

    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"  N={len(g):,} 格  R²={res.rsquared:.4f}  Adj-R²={res.rsquared_adj:.4f}")
    print(f"  Y均值={g['mean_deviation'].mean():+.4f}  "
          f"变快格占比={(g['mean_deviation']>0).mean():.1%}")
    print(f"{'='*72}")
    print(tbl[["coef","se","t","p","sig"]].round(4).to_string())

# ── 9. 汇总表 ──────────────────────────────────────────────────────────────────
print("\n\n" + "═"*90)
print("  各变量系数汇总（网格级，Ragasa S3+）")
print("═"*90)
rows = []
for v in KEY_VARS:
    row = {"变量": v}
    for tg, _ in PERIODS:
        if tg in reg_results:
            _, tbl = reg_results[tg]
            if v in tbl.index:
                row[tg] = f"{tbl.loc[v,'coef']:+.4f}{tbl.loc[v,'sig']}"
            else:
                row[tg] = "—"
        else:
            row[tg] = "—"
    rows.append(row)
sumtbl = pd.DataFrame(rows).set_index("变量")
print(sumtbl.to_string())

# ── 10. 保存网格数据（供可视化用） ────────────────────────────────────────────
print("\nSaving grid data...", flush=True)
all_grid = []
for tg, g in grid_data.items():
    g2 = g.copy()
    g2["time_group"] = tg
    all_grid.append(g2)
all_grid_df = pd.concat(all_grid, ignore_index=True)
all_grid_df.to_parquet(f"{DATA}/grid_regression_data.parquet", index=False)
print(f"  Saved: data/grid_regression_data.parquet  ({len(all_grid_df):,} grid×period rows)")

try:
    with pd.ExcelWriter(f"{DATA}/grid_regression_results.xlsx") as w:
        sumtbl.to_excel(w, sheet_name="汇总")
        for tg, _ in PERIODS:
            if tg in reg_results:
                reg_results[tg][1].round(4).to_excel(w, sheet_name=tg)
    print(f"  Saved: data/grid_regression_results.xlsx")
except Exception as e:
    print(f"  Excel保存失败: {e}")
