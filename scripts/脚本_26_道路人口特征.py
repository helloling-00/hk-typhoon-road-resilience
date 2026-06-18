"""
对每条道路计算 500m 路段线 buffer 内屋苑人口特征加权均值
（线 buffer = 路段两侧500m，不是中点圆）

population_density_500m  — 总人口 / 实际buffer面积（人/km²）
median_income_500m       — 人口加权收入中位数
elderly/employed/student/retiree_ratio_500m — 人口加权比例

输出：data/road_demo_features.parquet
"""
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import warnings; warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"

print("Loading estates...", flush=True)
est = pd.read_parquet(f"{DATA}/estate_features.parquet")
est = est.dropna(subset=["lat","lon","population_total"])
est["working_pop_ratio"] = est["working_pop"] / est["population_total"]
gdf_est = gpd.GeoDataFrame(
    est,
    geometry=[Point(lon, lat) for lat, lon in zip(est["lat"], est["lon"])],
    crs="EPSG:4326"
).to_crs("EPSG:3857")
print(f"  {len(gdf_est)} estates")

print("Loading road geometries...", flush=True)
import ast, pickle
from shapely import wkb as swkb
from shapely.geometry import LineString

ep_df = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
rr    = pd.read_parquet(f"{DATA}/road_registry.parquet")[
            ["road_id","road_category"]].drop_duplicates("road_id")
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

ep_df["geometry"] = ep_df.apply(build_geom, axis=1)
ep_df = ep_df.dropna(subset=["geometry"])
ep_df = ep_df.merge(rr, on="road_id", how="left")
gdf_roads = gpd.GeoDataFrame(ep_df, geometry="geometry",
                              crs="EPSG:4326").to_crs("EPSG:3857")
print(f"  {len(gdf_roads):,} roads")

# ── 500m 线段 buffer 空间连接 ────────────────────────────────────────────────
print("Building 500m LINE buffers and spatial join...", flush=True)
BUFFER_M = 500
gdf_buf = gdf_roads[["road_id"]].copy()
gdf_buf["geometry"] = gdf_roads.geometry.buffer(BUFFER_M)   # 线段两侧
gdf_buf = gdf_buf.set_geometry("geometry")
gdf_buf["buf_area_km2"] = gdf_buf.geometry.area / 1e6

# sjoin：找出每个 buffer 内的屋苑
joined = gpd.sjoin(gdf_buf, gdf_est[["geometry","population_total",
                                       "median_income","elderly_ratio",
                                       "working_pop_ratio","student_ratio",
                                       "retiree_ratio","employed_ratio"]],
                   how="left", predicate="contains")
print(f"  {joined['index_right'].notna().sum():,} road-estate pairs within 500m")

# ── 按路段聚合 ────────────────────────────────────────────────────────────────
print("Aggregating per road...", flush=True)

def wavg(group, val_col, wt_col="population_total"):
    """人口加权均值"""
    mask = group[val_col].notna() & group[wt_col].notna()
    v = group.loc[mask, val_col].values
    w = group.loc[mask, wt_col].values
    if len(v) == 0 or w.sum() == 0: return np.nan
    return float(np.average(v, weights=w))

buf_area_lkp = gdf_buf.set_index("road_id")["buf_area_km2"]
records = []

for road_id, grp in joined.groupby("road_id"):
    valid = grp.dropna(subset=["population_total"])
    pop_sum = valid["population_total"].sum()
    buf_area_km2 = buf_area_lkp.get(road_id, np.pi*(BUFFER_M/1000)**2)
    records.append({
        "road_id":                road_id,
        "estate_count_500m":      len(valid),
        "population_total_500m":  pop_sum,
        "population_density_500m": pop_sum / buf_area_km2,
        "median_income_500m":     wavg(valid, "median_income"),
        "elderly_ratio_500m":     wavg(valid, "elderly_ratio"),
        "working_pop_ratio_500m": wavg(valid, "working_pop_ratio"),
        "employed_ratio_500m":    wavg(valid, "employed_ratio"),
        "student_ratio_500m":     wavg(valid, "student_ratio"),
        "retiree_ratio_500m":     wavg(valid, "retiree_ratio"),
    })

road_demo = pd.DataFrame(records)
print(f"  {len(road_demo):,} roads with demographic features")
print(f"  Roads with ≥1 estate in 500m: {(road_demo['estate_count_500m']>0).sum():,}")
print(f"  Roads with 0 estates:          {(road_demo['estate_count_500m']==0).sum():,}")

# 0个屋苑的路设为 NaN（主要是郊区/山区道路）
for col in ["population_density_500m","median_income_500m","elderly_ratio_500m",
            "working_pop_ratio_500m","employed_ratio_500m",
            "student_ratio_500m","retiree_ratio_500m"]:
    road_demo.loc[road_demo["estate_count_500m"]==0, col] = np.nan

print(f"\n统计摘要:")
print(road_demo[["population_density_500m","median_income_500m","elderly_ratio_500m"]].describe().round(2))

out = f"{DATA}/road_demo_features.parquet"
road_demo.to_parquet(out, index=False)
print(f"\nSaved: {out}")
