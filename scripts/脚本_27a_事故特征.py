"""
计算每条道路 500m buffer 内台风期间事故数
输出：data/road_incident_features.parquet
"""
import glob, pandas as pd, numpy as np
import geopandas as gpd
from shapely import wkb as swkb
import warnings; warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"

# ── 台风日期范围 ──────────────────────────────────────────────────────────────
TYPHOON_DATES = {
    "Mitag":  ("2025-09-17", "2025-09-20"),
    "Ragasa": ("2025-09-22", "2025-09-25"),
    "Matmo":  ("2025-10-03", "2025-10-05"),
}

# ── 读所有 incident parquet ───────────────────────────────────────────────────
print("Loading incidents...", flush=True)
files = glob.glob(f"{DATA}/incident_parquet/**/*.parquet", recursive=True)
chunks = []
for f in files:
    df = pd.read_parquet(f, columns=["ts","geometry_wkb","magnitude_of_delay","closed"])
    chunks.append(df)
inc = pd.concat(chunks, ignore_index=True)
inc["ts"] = pd.to_datetime(inc["ts"])
inc["date"] = inc["ts"].dt.strftime("%Y-%m-%d")
print(f"  {len(inc):,} incident records total")

# 只保留台风期事故
typhoon_dates = set()
for t, (s, e) in TYPHOON_DATES.items():
    for d in pd.date_range(s, e):
        typhoon_dates.add(d.strftime("%Y-%m-%d"))
inc_ty = inc[inc["date"].isin(typhoon_dates)].copy()
print(f"  {len(inc_ty):,} during typhoon periods")

# ── 解析 incident 几何 ────────────────────────────────────────────────────────
def parse_geom(wb):
    try: return swkb.loads(wb)
    except: return None

inc_ty["geometry"] = inc_ty["geometry_wkb"].apply(parse_geom)
inc_ty = inc_ty.dropna(subset=["geometry"])
gdf_inc = gpd.GeoDataFrame(inc_ty, geometry="geometry",
                            crs="EPSG:4326").to_crs("EPSG:3857")
print(f"  {len(gdf_inc):,} incidents with valid geometry")

# ── 加载道路中心点并建 500m buffer ────────────────────────────────────────────
print("Loading road centroids...", flush=True)
import ast, pickle
from shapely import wkb as shapely_wkb
from shapely.geometry import LineString, Point

ep_df = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
with open(f"{DATA}/osm_cache/road_wkb_store.pkl","rb") as f:
    wkb_store = pickle.load(f)

def build_geom(row):
    rid = row["road_id"]
    if rid in wkb_store:
        try: return shapely_wkb.loads(wkb_store[rid])
        except: pass
    try:
        pts = ast.literal_eval(row["ep_key"])
        return LineString([pts[0], pts[1]])
    except: return None

ep_df["geometry"] = ep_df.apply(build_geom, axis=1)
ep_df = ep_df.dropna(subset=["geometry"])
gdf_roads = gpd.GeoDataFrame(ep_df[["road_id","geometry"]],
                              geometry="geometry", crs="EPSG:4326").to_crs("EPSG:3857")

# 路段线 500m buffer（胶囊形，不是中点圆）
gdf_buf = gdf_roads[["road_id"]].copy()
gdf_buf["geometry"] = gdf_roads.geometry.buffer(500)
gdf_buf = gdf_buf.set_geometry("geometry")

# ── 空间连接：每个 buffer 内的事故 ──────────────────────────────────────────
print("Spatial joining incidents to road buffers...", flush=True)
joined = gpd.sjoin(gdf_buf, gdf_inc[["geometry","magnitude_of_delay","closed"]],
                   how="left", predicate="contains")

# ── 聚合：每路段事故总数、严重事故数、道路封闭数 ─────────────────────────────
print("Aggregating...", flush=True)
agg = joined.groupby("road_id").agg(
    incident_count_500m   = ("index_right", lambda x: x.notna().sum()),
    severe_incident_500m  = ("magnitude_of_delay",
                             lambda x: (pd.to_numeric(x, errors="coerce") >= 3).sum()),
    closure_nearby_500m   = ("closed",
                             lambda x: (x == True).sum()),
).reset_index()

print(f"\n统计摘要:")
print(agg.describe().round(1))
print(f"\n有事故的路段: {(agg['incident_count_500m']>0).sum():,} / {len(agg):,}")

out = f"{DATA}/road_incident_features.parquet"
agg.to_parquet(out, index=False)
print(f"Saved: {out}")
