"""
网格特征直接在格心计算（不再从路段聚合）
- 250m 网格
- POI → 格心 500m 缓冲 → 密度
- 人口 → 格心 500m 缓冲 → 加权特征
- Y：格内路段 mean_deviation 均值（Ragasa S3+）
"""
import pandas as pd, numpy as np
import geopandas as gpd, osmium
from shapely.geometry import box, Point, LineString
from shapely import wkb as swkb
import statsmodels.formula.api as smf
import ast, pickle, warnings, time
warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
PBF  = "/Users/helloling/workspace/thesis/hong-kong-260502.osm.pbf"
GRID_SIZE = 250
BUFFER_M  = 500

# ══════════════════════════════════════════════════════════════════════════════
# 1. POI 分类提取
# ══════════════════════════════════════════════════════════════════════════════
CATS = ["work","education","retail","food_drink","recreation",
        "medical","transport","tourism","finance","civic"]

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
    if amenity in ("restaurant","fast_food","cafe","bar","pub","food_court","ice_cream","biergarten","bbq"):
        return "food_drink"
    if (leisure in ("park","sports_centre","fitness_centre","swimming_pool","stadium","golf_course",
                    "tennis_court","pitch") or
            amenity in ("cinema","theatre","arts_centre","nightclub","gambling","casino")):
        return "recreation"
    if amenity in ("hospital","clinic","pharmacy","dentist","doctors","nursing_home","veterinary"):
        return "medical"
    if (railway in ("station","subway_entrance","tram_stop","halt") or
            pt in ("station","stop_position","platform") or
            amenity in ("bus_station","ferry_terminal","taxi","parking")):
        return "transport"
    if (tourism in ("hotel","guest_house","hostel","attraction","museum","viewpoint","zoo",
                    "aquarium","theme_park","gallery") or amenity in ("casino",)):
        return "tourism"
    if amenity in ("bank","atm","bureau_de_change","money_transfer"):
        return "finance"
    if amenity in ("police","post_office","fire_station","courthouse","townhall","embassy",
                   "social_facility","community_centre","library","place_of_worship"):
        return "civic"
    return None

class POIHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.features = []
    def node(self, n):
        cat = categorize(n.tags)
        if cat and n.location.valid():
            self.features.append({"lon": n.location.lon, "lat": n.location.lat, "cat": cat})
    def way(self, w):
        cat = categorize(w.tags)
        if not cat: return
        lons = [nd.location.lon for nd in w.nodes if nd.location.valid()]
        lats = [nd.location.lat for nd in w.nodes if nd.location.valid()]
        if lons:
            self.features.append({"lon": float(np.mean(lons)),
                                  "lat": float(np.mean(lats)), "cat": cat})

print("1. Extracting POI from OSM...", flush=True)
t0 = time.time()
h = POIHandler()
h.apply_file(PBF, locations=True)
poi_df = pd.DataFrame(h.features)
print(f"   {len(poi_df):,} POI extracted ({time.time()-t0:.0f}s)")
gdf_poi = gpd.GeoDataFrame(
    poi_df,
    geometry=[Point(r.lon, r.lat) for _, r in poi_df.iterrows()],
    crs="EPSG:4326"
).to_crs("EPSG:3857")

# ══════════════════════════════════════════════════════════════════════════════
# 2. 加载路段数据，建网格
# ══════════════════════════════════════════════════════════════════════════════
print("2. Loading roads & building grid...", flush=True)
reg = pd.read_parquet(f"{DATA}/regression_table.parquet")
rag = reg[(reg["typhoon"]=="Ragasa") & (reg["signal_level"]>=3) & (reg["road_length_m"]>=100)].copy()

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

ep_sub = ep_df[ep_df["road_id"].isin(rag["road_id"])].copy()
ep_sub["geometry"] = ep_sub.apply(build_geom, axis=1)
ep_sub = ep_sub.dropna(subset=["geometry"])
gdf_roads_all = gpd.GeoDataFrame(ep_sub[["road_id","geometry"]],
                                  geometry="geometry", crs="EPSG:4326").to_crs("EPSG:3857")
gdf_roads_all["cx"] = gdf_roads_all.geometry.centroid.x
gdf_roads_all["cy"] = gdf_roads_all.geometry.centroid.y
road_xy = gdf_roads_all[["road_id","cx","cy"]].drop_duplicates("road_id")

# 网格
x0 = np.floor(road_xy["cx"].min() / GRID_SIZE) * GRID_SIZE - GRID_SIZE
x1 = np.ceil( road_xy["cx"].max() / GRID_SIZE) * GRID_SIZE + GRID_SIZE
y0 = np.floor(road_xy["cy"].min() / GRID_SIZE) * GRID_SIZE - GRID_SIZE
y1 = np.ceil( road_xy["cy"].max() / GRID_SIZE) * GRID_SIZE + GRID_SIZE

xs = np.arange(x0, x1, GRID_SIZE)
ys = np.arange(y0, y1, GRID_SIZE)
cells = [{"grid_id": f"{i}_{j}",
          "geometry": box(x, y, x+GRID_SIZE, y+GRID_SIZE),
          "cx": x + GRID_SIZE/2, "cy": y + GRID_SIZE/2}
         for i, x in enumerate(xs) for j, y in enumerate(ys)]
gdf_grid = gpd.GeoDataFrame(cells, crs="EPSG:3857")
print(f"   {len(xs)}x{len(ys)} = {len(gdf_grid):,} grids")

# ══════════════════════════════════════════════════════════════════════════════
# 3. 格心 500m 缓冲 → POI 密度（直接在格级算！）
# ══════════════════════════════════════════════════════════════════════════════
print("3. Computing grid-level POI density...", flush=True)
t0 = time.time()

# 格心点
gdf_centroids = gpd.GeoDataFrame(
    gdf_grid[["grid_id"]],
    geometry=[Point(cx, cy) for cx, cy in zip(gdf_grid["cx"], gdf_grid["cy"])],
    crs="EPSG:3857"
)
# 格心缓冲
gdf_centroids["geometry"] = gdf_centroids.geometry.buffer(BUFFER_M)
gdf_centroids = gdf_centroids.set_geometry("geometry")
gdf_centroids["buf_area_km2"] = gdf_centroids.geometry.area / 1e6

# 空间连接
joined = gpd.sjoin(gdf_centroids, gdf_poi[["geometry","cat"]],
                   how="left", predicate="contains")
print(f"   {joined['index_right'].notna().sum():,} grid-POI pairs ({time.time()-t0:.0f}s)")

# 聚合
buf_area = gdf_centroids.set_index("grid_id")["buf_area_km2"]
count_wide = (joined.groupby(["grid_id","cat"])
              .size()
              .unstack(fill_value=0)
              .reindex(columns=CATS, fill_value=0))
count_wide = count_wide.join(buf_area)

# 保存 count + density（reindex 确保所有网格都有行）
all_grid_ids = gdf_grid["grid_id"].values
count_wide = count_wide.reindex(all_grid_ids, fill_value=0)
buf_area_full = buf_area.reindex(all_grid_ids, fill_value=np.pi*(BUFFER_M/1000)**2)

grid_poi = gdf_grid[["grid_id","cx","cy"]].copy()
for cat in CATS:
    grid_poi[f"{cat}_count"] = count_wide[cat].values
    grid_poi[f"{cat}_density"] = count_wide[cat].values / buf_area_full.values
grid_poi["buf_area_km2"] = buf_area_full.values
print(f"   Median buffer area: {buf_area.median():.3f} km²")

# ══════════════════════════════════════════════════════════════════════════════
# 4. 格心缓冲 → 人口特征
# ══════════════════════════════════════════════════════════════════════════════
print("4. Computing grid-level demographic features...", flush=True)
est = pd.read_parquet(f"{DATA}/estate_features.parquet")
est = est.dropna(subset=["lat","lon","population_total"])
gdf_est = gpd.GeoDataFrame(
    est,
    geometry=[Point(lon, lat) for lat, lon in zip(est["lat"], est["lon"])],
    crs="EPSG:4326"
).to_crs("EPSG:3857")

j_demo = gpd.sjoin(gdf_centroids,
                   gdf_est[["geometry","population_total","median_income",
                            "elderly_ratio","employed_ratio","student_ratio","retiree_ratio"]],
                   how="left", predicate="contains")

def wavg(group, val_col, wt_col="population_total"):
    mask = group[val_col].notna() & group[wt_col].notna()
    v = group.loc[mask, val_col].values
    w = group.loc[mask, wt_col].values
    if len(v)==0 or w.sum()==0: return np.nan
    return float(np.average(v, weights=w))

demo_records = []
for gid, grp in j_demo.groupby("grid_id"):
    valid = grp.dropna(subset=["population_total"])
    pop_sum = valid["population_total"].sum()
    ba = buf_area.get(gid, np.pi*(BUFFER_M/1000)**2)
    demo_records.append({
        "grid_id": gid,
        "estate_count": len(valid),
        "population_total": pop_sum,
        "population_density": pop_sum / ba if ba > 0 else 0,
        "median_income": wavg(valid, "median_income"),
        "elderly_ratio": wavg(valid, "elderly_ratio"),
        "employed_ratio": wavg(valid, "employed_ratio"),
        "student_ratio": wavg(valid, "student_ratio"),
        "retiree_ratio": wavg(valid, "retiree_ratio"),
    })
grid_demo = pd.DataFrame(demo_records)
grid_demo = pd.merge(
    gdf_grid[["grid_id","cx","cy"]],
    grid_demo, on="grid_id", how="left"
)
for c in ["estate_count","population_total","population_density"]:
    grid_demo[c] = grid_demo[c].fillna(0)
print(f"   {len(grid_demo):,} grids  estates>0: {(grid_demo['estate_count']>0).sum():,}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. 路段 → 网格分配（Y）
# ══════════════════════════════════════════════════════════════════════════════
print("5. Assigning roads to grids for Y...", flush=True)
rag_xy = rag.merge(road_xy, on="road_id", how="inner")
rag_xy["grid_id"] = np.where(
    (rag_xy["cx"] >= x0) & (rag_xy["cx"] < x1) & (rag_xy["cy"] >= y0) & (rag_xy["cy"] < y1),
    ((rag_xy["cx"] - x0) / GRID_SIZE).astype(int).astype(str) + "_" +
    ((rag_xy["cy"] - y0) / GRID_SIZE).astype(int).astype(str),
    np.nan
)
rag_xy = rag_xy.dropna(subset=["grid_id"])
print(f"   {rag_xy['road_id'].nunique():,} roads assigned")

# ══════════════════════════════════════════════════════════════════════════════
# 6. 特征工程 + 合并
# ══════════════════════════════════════════════════════════════════════════════
print("6. Feature engineering & merging...", flush=True)

# 网格级 X（从 POI 和 demo 直算的）
grid_x = grid_poi[["grid_id","buf_area_km2"] + [f"{c}_density" for c in CATS]].copy()
grid_x = grid_x.merge(
    grid_demo[["grid_id","population_density","median_income",
               "employed_ratio","student_ratio","retiree_ratio","elderly_ratio"]],
    on="grid_id", how="left"
)

# 填充缺失
for col in ["employed_ratio","student_ratio","retiree_ratio","elderly_ratio",
            "population_density","median_income"]:
    grid_x[col] = grid_x[col].fillna(0)

# log 变换
for cat in CATS:
    grid_x[f"log_{cat}"] = np.log1p(grid_x[f"{cat}_density"].fillna(0))
grid_x["log_pop_density"] = np.log1p(grid_x["population_density"])
grid_x["log_income"]      = np.log1p(grid_x["median_income"])

# 格内道路条数（取所有时段均值）作为路口度代理
grid_road_count = rag_xy.groupby("grid_id")["road_id"].nunique().rename("n_roads_in_grid").reset_index()
grid_x = grid_x.merge(grid_road_count, on="grid_id", how="left")
grid_x["log_n_roads"] = np.log1p(grid_x["n_roads_in_grid"])

# ══════════════════════════════════════════════════════════════════════════════
# 7. 构建 时段 × 网格 面板
# ══════════════════════════════════════════════════════════════════════════════
PERIODS = [
    ("NIGHT","NIGHT"),
    ("AM_PEAK","AM_PEAK"),
    ("MIDDAY","MIDDAY"),
    ("PM_PEAK","PM_PEAK"),
    ("EVENING","EVENING"),
]

records = []
for tg, label in PERIODS:
    sub = rag_xy[rag_xy["time_group"]==tg]
    y_agg = sub.groupby("grid_id")["mean_deviation"].agg(
        mean_deviation="mean", pct_pos=lambda x:(x>0).mean(), n_obs="count"
    ).reset_index()
    y_agg["time_group"] = tg
    records.append(y_agg)
grid_y = pd.concat(records, ignore_index=True)

# 合并 X + Y
data = grid_y.merge(grid_x, on="grid_id", how="inner")
data = data[data["n_obs"] >= 3].copy()
print(f"   Final: {len(data):,} grid×period rows, {data['grid_id'].nunique():,} unique grids")

# ══════════════════════════════════════════════════════════════════════════════
# 8. 回归
# ══════════════════════════════════════════════════════════════════════════════
LU_TERMS    = " + ".join(f"log_{c}" for c in CATS)
DEMO_TERMS  = ("log_pop_density + log_income + "
               "employed_ratio + student_ratio + retiree_ratio + elderly_ratio")
STRUCT_TERMS = "log_n_roads"
FORMULA = f"mean_deviation ~ {LU_TERMS} + {DEMO_TERMS} + {STRUCT_TERMS}"

KEY_VARS = [f"log_{c}" for c in CATS] + [
    "log_pop_density", "log_income",
    "employed_ratio", "student_ratio", "retiree_ratio", "elderly_ratio",
    "log_n_roads"
]

print("\n" + "="*72)
print(f"  网格级回归结果（250m 格，X 直接在格心算，Ragasa S3+）")
print("="*72)

results = {}
for tg, label in PERIODS:
    sub = data[data["time_group"]==tg]
    if len(sub) < 50:
        print(f"\n{label}: too few ({len(sub)}), skip")
        continue

    try:
        res = smf.ols(FORMULA, data=sub).fit()
    except Exception as e:
        print(f"\n{label}: FAILED ({e})")
        continue

    tbl = pd.DataFrame({
        "coef": res.params, "se": res.bse,
        "t": res.tvalues,   "p": res.pvalues,
    })
    tbl["sig"] = tbl["p"].apply(
        lambda p: "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "."
    )
    results[tg] = (res, tbl)

    print(f"\n{'='*72}")
    print(f"  {label}  N={len(sub):,}  R²={res.rsquared:.4f}  Adj-R²={res.rsquared_adj:.4f}")
    print(f"  Y mean={sub['mean_deviation'].mean():+.4f}  pct_pos={sub['pct_pos'].mean():.1%}")
    print(f"{'='*72}")
    print(tbl[["coef","se","t","p","sig"]].round(4).to_string())

# 汇总表
print("\n\n" + "="*90)
print("  Grid-level coefficient summary (250m, direct X computation)")
print("="*90)
rows = []
for v in KEY_VARS:
    row = {"Variable": v}
    for tg, _ in PERIODS:
        if tg in results:
            _, tbl = results[tg]
            if v in tbl.index:
                row[tg] = f"{tbl.loc[v,'coef']:+.4f}{tbl.loc[v,'sig']}"
            else:
                row[tg] = "--"
        else:
            row[tg] = "--"
    rows.append(row)
sumtbl = pd.DataFrame(rows).set_index("Variable")
print(sumtbl.to_string())

# 保存
print("\nSaving...", flush=True)
data.to_parquet(f"{DATA}/grid_direct_features.parquet", index=False)
print(f"  Saved: data/grid_direct_features.parquet ({len(data):,} rows)")

try:
    with pd.ExcelWriter(f"{DATA}/grid_direct_regression.xlsx") as w:
        sumtbl.to_excel(w, sheet_name="Summary")
        for tg, _ in PERIODS:
            if tg in results:
                results[tg][1].round(4).to_excel(w, sheet_name=tg)
    print(f"  Saved: data/grid_direct_regression.xlsx")
except Exception as e:
    print(f"  Excel: {e}")

print("\nDone.")
