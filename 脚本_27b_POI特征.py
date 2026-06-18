"""
从 hong-kong-260502.osm.pbf 提取 10 类土地利用 POI（nodes + ways）
buffer: 路段线段两侧 500m（不是中点圆）
输出密度: 个/km²

10 类别（对应出行目的）：
  work        办公/商业  → 通勤
  education   教育      → 上学接送
  retail      零售购物   → 购物
  food_drink  餐饮      → 用餐
  recreation  休闲娱乐   → 休闲
  medical     医疗      → 就医
  transport   交通枢纽   → 换乘
  tourism     旅游酒店   → 旅游
  finance     金融      → 银行 ATM
  civic       政务公共   → 公共服务

输出：data/road_landuse_features.parquet
  列：road_id, buf_area_km2, {cat}_count, {cat}_density  （每类两列）
"""
import osmium, pandas as pd, numpy as np
import geopandas as gpd
from shapely.geometry import Point
import ast, pickle
from shapely import wkb as swkb
from shapely.geometry import LineString
import warnings; warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
PBF  = "/Users/helloling/workspace/thesis/hong-kong-260502.osm.pbf"
CATS = ["work","education","retail","food_drink","recreation",
        "medical","transport","tourism","finance","civic"]

# ── 分类规则 ──────────────────────────────────────────────────────────────────
def categorize(tags):
    office  = tags.get("office","")
    amenity = tags.get("amenity","")
    shop    = tags.get("shop","")
    tourism = tags.get("tourism","")
    railway = tags.get("railway","")
    pt      = tags.get("public_transport","")
    leisure = tags.get("leisure","")
    bldg    = tags.get("building","")

    if office or bldg in ("office","commercial","industrial"):
        return "work"
    if amenity in ("school","university","college","kindergarten","language_school","training"):
        return "education"
    if (shop in ("mall","supermarket","department_store","shopping_centre",
                 "convenience","clothes","electronics","furniture","hardware",
                 "sports","books","gifts","jewelry","shoes") or
            amenity in ("marketplace","supermarket")):
        return "retail"
    if amenity in ("restaurant","fast_food","cafe","bar","pub","food_court",
                   "ice_cream","biergarten","bbq"):
        return "food_drink"
    if (leisure in ("park","sports_centre","fitness_centre","swimming_pool",
                    "stadium","golf_course","tennis_court","pitch") or
            amenity in ("cinema","theatre","arts_centre","nightclub","gambling","casino")):
        return "recreation"
    if amenity in ("hospital","clinic","pharmacy","dentist","doctors",
                   "nursing_home","veterinary"):
        return "medical"
    if (railway in ("station","subway_entrance","tram_stop","halt") or
            pt in ("station","stop_position","platform") or
            amenity in ("bus_station","ferry_terminal","taxi","parking")):
        return "transport"
    if (tourism in ("hotel","guest_house","hostel","attraction","museum",
                    "viewpoint","zoo","aquarium","theme_park","gallery") or
            amenity in ("casino",)):
        return "tourism"
    if amenity in ("bank","atm","bureau_de_change","money_transfer"):
        return "finance"
    if amenity in ("police","post_office","fire_station","courthouse",
                   "townhall","embassy","social_facility","community_centre",
                   "library","place_of_worship"):
        return "civic"
    return None

# ── OSM handler（nodes + ways） ───────────────────────────────────────────────
class LandUseHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.features = []

    def node(self, n):
        cat = categorize(n.tags)
        if cat and n.location.valid():
            self.features.append({"lon": n.location.lon,
                                  "lat": n.location.lat, "cat": cat})

    def way(self, w):
        cat = categorize(w.tags)
        if not cat: return
        try:
            lons = [nd.location.lon for nd in w.nodes if nd.location.valid()]
            lats = [nd.location.lat for nd in w.nodes if nd.location.valid()]
            if lons:
                self.features.append({"lon": float(np.mean(lons)),
                                      "lat": float(np.mean(lats)), "cat": cat})
        except: pass

print("Parsing POI from .pbf (nodes + ways)...", flush=True)
h = LandUseHandler()
h.apply_file(PBF, locations=True)
poi_df = pd.DataFrame(h.features)
print(f"  {len(poi_df):,} features extracted")
print(poi_df["cat"].value_counts().to_string())

gdf_poi = gpd.GeoDataFrame(
    poi_df,
    geometry=[Point(r.lon, r.lat) for r in poi_df.itertuples()],
    crs="EPSG:4326"
).to_crs("EPSG:3857")

# ── 道路线段 buffer（500m 两侧，不是中点圆） ───────────────────────────────────
print("\nBuilding road LINE buffers (500m)...", flush=True)
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

ep_df["geometry"] = ep_df.apply(build_geom, axis=1)
ep_df = ep_df.dropna(subset=["geometry"])
gdf_roads = gpd.GeoDataFrame(ep_df[["road_id","geometry"]],
                              geometry="geometry", crs="EPSG:4326").to_crs("EPSG:3857")

# 线段 buffer（胶囊形）
gdf_buf = gdf_roads[["road_id"]].copy()
gdf_buf["geometry"] = gdf_roads.geometry.buffer(500)
gdf_buf = gdf_buf.set_geometry("geometry")
gdf_buf["buf_area_km2"] = gdf_buf.geometry.area / 1e6
print(f"  {len(gdf_buf):,} road buffers  "
      f"(median area={gdf_buf['buf_area_km2'].median():.3f} km²)")

# ── 空间连接 ──────────────────────────────────────────────────────────────────
print("Spatial join...", flush=True)
joined = gpd.sjoin(gdf_buf[["road_id","buf_area_km2","geometry"]],
                   gdf_poi[["geometry","cat"]],
                   how="left", predicate="contains")
print(f"  {joined['index_right'].notna().sum():,} road-POI pairs")

# ── 聚合：每路段 × 每类别 count + density ────────────────────────────────────
print("Aggregating...", flush=True)
buf_area = gdf_buf.set_index("road_id")["buf_area_km2"]

count_wide = (joined.groupby(["road_id","cat"])
              .size()
              .unstack(fill_value=0)
              .reindex(columns=CATS, fill_value=0))

# 加上面积列
count_wide = count_wide.join(buf_area)

# 计算密度
for cat in CATS:
    count_wide[f"{cat}_density"] = count_wide[cat] / count_wide["buf_area_km2"]
count_wide = count_wide.rename(columns={c: f"{c}_count" for c in CATS})
count_wide = count_wide.reset_index()

print(f"\n统计摘要（密度，个/km²）：")
dens_cols = [f"{c}_density" for c in CATS]
print(count_wide[dens_cols].describe().round(1).to_string())
print(f"\n各类有覆盖路段（count>0）：")
for cat in CATS:
    n = (count_wide[f"{cat}_count"] > 0).sum()
    print(f"  {cat:<12}: {n:,} roads")

out = f"{DATA}/road_landuse_features.parquet"
count_wide.to_parquet(out, index=False)
print(f"\nSaved: {out}")
