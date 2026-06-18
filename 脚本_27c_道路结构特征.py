"""
计算道路结构特征（不需要空间连接）：
  intersection_degree  — 路段端点平均接入路数（连通度）
  dist_to_coast_m      — 路段中点到最近海岸线的距离（米）

输出：data/road_structural_features.parquet
"""
import ast, pickle, osmium
import pandas as pd, numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from shapely import wkb as swkb
from collections import defaultdict
import warnings; warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
PBF  = "/Users/helloling/workspace/thesis/hong-kong-260502.osm.pbf"

# ── 1. 路口连通度（端点接入路数） ─────────────────────────────────────────────
print("Computing intersection degree...", flush=True)
ep_df = pd.read_parquet(f"{DATA}/ep_to_road.parquet")

# 从 ep_key 拆出两个端点
def parse_ep(key):
    try:
        pts = ast.literal_eval(key)
        return pts[0], pts[1]
    except: return None, None

ep_df[["pt_a","pt_b"]] = ep_df["ep_key"].apply(
    lambda k: pd.Series(parse_ep(k)))
ep_df = ep_df.dropna(subset=["pt_a","pt_b"])

# 统计每个端点被多少条路段共享
degree = defaultdict(int)
for _, row in ep_df.iterrows():
    degree[row["pt_a"]] += 1
    degree[row["pt_b"]] += 1

# 每条路段的连通度 = 两个端点的平均接入路数
ep_df["deg_a"] = ep_df["pt_a"].map(degree)
ep_df["deg_b"] = ep_df["pt_b"].map(degree)
ep_df["intersection_degree"] = (ep_df["deg_a"] + ep_df["deg_b"]) / 2

road_deg = (ep_df.groupby("road_id")["intersection_degree"]
            .mean().reset_index())
print(f"  intersection_degree: mean={road_deg['intersection_degree'].mean():.2f}  "
      f"median={road_deg['intersection_degree'].median():.2f}")

# ── 2. 海岸线距离 ─────────────────────────────────────────────────────────────
print("Extracting coastline from .pbf...", flush=True)

class CoastHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def way(self, w):
        if w.tags.get("natural") == "coastline":
            try:
                coords = [(nd.location.lon, nd.location.lat)
                          for nd in w.nodes if nd.location.valid()]
                if len(coords) >= 2:
                    self.lines.append(LineString(coords))
            except: pass

ch = CoastHandler()
ch.apply_file(PBF, locations=True)
print(f"  {len(ch.lines)} coastline way segments")

coast_gdf = gpd.GeoDataFrame(geometry=ch.lines, crs="EPSG:4326").to_crs("EPSG:3857")
coast_union = coast_gdf.geometry.unary_union  # 合并为一个几何

# ── 3. 路段中点到海岸线距离 ───────────────────────────────────────────────────
print("Computing distance to coast...", flush=True)
with open(f"{DATA}/osm_cache/road_wkb_store.pkl","rb") as f:
    wkb_store = pickle.load(f)

def get_centroid(row):
    rid = row["road_id"]
    if rid in wkb_store:
        try:
            g = swkb.loads(wkb_store[rid])
            return g.centroid
        except: pass
    try:
        pts = ast.literal_eval(row["ep_key"])
        return LineString([pts[0], pts[1]]).centroid
    except: return None

ep_unique = ep_df.drop_duplicates("road_id")[["road_id","ep_key"]].copy()
ep_unique["centroid"] = ep_unique.apply(get_centroid, axis=1)
ep_unique = ep_unique.dropna(subset=["centroid"])

gdf_cent = gpd.GeoDataFrame(ep_unique[["road_id"]],
                             geometry=ep_unique["centroid"].values,
                             crs="EPSG:4326").to_crs("EPSG:3857")

# 批量计算距离（每10000条一批）
print(f"  Computing distance for {len(gdf_cent):,} roads...", flush=True)
dist_list = []
batch = 5000
for i in range(0, len(gdf_cent), batch):
    chunk = gdf_cent.iloc[i:i+batch]
    dists = chunk.geometry.distance(coast_union)
    dist_list.append(dists)
    if (i // batch) % 5 == 0:
        print(f"    {i+batch:,}/{len(gdf_cent):,}", flush=True)

gdf_cent["dist_to_coast_m"] = pd.concat(dist_list).values
road_dist = gdf_cent[["road_id","dist_to_coast_m"]]

print(f"  dist_to_coast_m: mean={road_dist['dist_to_coast_m'].mean():.0f}m  "
      f"median={road_dist['dist_to_coast_m'].median():.0f}m")

# ── 4. 合并保存 ────────────────────────────────────────────────────────────────
struct = road_deg.merge(road_dist, on="road_id", how="outer")
out = f"{DATA}/road_structural_features.parquet"
struct.to_parquet(out, index=False)
print(f"Saved: {out}  ({len(struct):,} roads)")
