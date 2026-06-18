"""
从 OSM 提取建筑 footprint + 土地利用 polygon → 网格面积占比
替代 POI 点计数，面积占比更精确反映土地利用强度
"""
import osmium, pandas as pd, numpy as np
import geopandas as gpd
from shapely.geometry import box, Point, Polygon
from shapely import wkb as swkb
import warnings, time
warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
PBF  = "/Users/helloling/workspace/thesis/hong-kong-260502.osm.pbf"

# ══════════════════════════════════════════════════════════════════════════════
# 1. 分类规则
# ══════════════════════════════════════════════════════════════════════════════

# 建筑分类（按出行目的）
def classify_building(tags):
    b = tags.get("building", "")
    a = tags.get("amenity", "")
    o = tags.get("office", "")

    if b in ("office",) or o:
        return "work"
    if b in ("school", "university", "college", "dormitory") or a in ("school", "university", "college"):
        return "education"
    if b in ("retail", "shop", "supermarket", "mall") or a in ("marketplace",):
        return "retail"
    if b in ("commercial",):
        return "commercial"
    if b in ("hotel",) or tags.get("tourism") == "hotel":
        return "hotel"
    if b in ("hospital",) or a in ("hospital", "clinic"):
        return "medical"
    if b in ("industrial", "warehouse", "factory"):
        return "industrial"
    if b in ("civic", "public", "government", "train_station", "transportation"):
        return "civic"
    if b in ("house", "apartments", "residential", "semidetached_house", "terrace", "detached"):
        return "residential"
    if b in ("ruins", "construction", "roof", "shed", "hut", "garage", "parking",
             "storage_tank", "toilets", "service", "church", "temple", "religious"):
        return "other"
    if b == "yes":
        return "untyped"
    return "other"

# 土地利用分类
def classify_landuse(tags):
    lu = tags.get("landuse", "")
    if lu in ("residential",):
        return "residential"
    if lu in ("industrial",):
        return "industrial"
    if lu in ("commercial", "retail"):
        return "commercial"
    if lu in ("institutional", "education", "religious"):
        return "institutional"
    if lu in ("farmland", "grass", "forest", "recreation_ground", "greenfield",
              "orchard", "greenhouse_horticulture", "plant_nursery", "flowerbed"):
        return "green"
    if lu in ("construction", "brownfield"):
        return "construction"
    if lu in ("cemetery", "military", "highway", "railway", "garages"):
        return "other"
    return "other"

# ══════════════════════════════════════════════════════════════════════════════
# 2. OSM 面状要素提取
# ══════════════════════════════════════════════════════════════════════════════
class AreaHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.buildings = []
        self.landuses  = []

    def way(self, w):
        tags = dict(w.tags)

        # 只处理闭合 way（第一个和最后一个 node 相同）
        refs = [n.ref for n in w.nodes]
        if len(refs) < 4 or refs[0] != refs[-1]:
            return

        coords = [(nd.location.lon, nd.location.lat) for nd in w.nodes
                  if nd.location.valid()]
        if len(coords) < 3:
            return
        # 去重——闭合 way 首尾相同，shapely 会自动闭合
        poly = Polygon(coords)
        if not poly.is_valid or poly.area <= 0:
            return

        # 建筑
        bcat = classify_building(tags)
        if bcat:
            self.buildings.append({"cat": bcat, "geometry": poly, "area_m2": poly.area})

        # 土地利用
        lcat = classify_landuse(tags)
        if lcat:
            self.landuses.append({"cat": lcat, "geometry": poly, "area_m2": poly.area})

    def area(self, a):
        # 处理 multipolygon relations（少数）
        tags = dict(a.tags)
        try:
            rings = [a.outer_ring] + list(a.inner_rings)
            coords = []
            for ring in rings:
                coords.append([(nd.lon, nd.lat) for nd in ring if nd.location.valid()])
            if len(coords[0]) >= 3:
                poly = Polygon(coords[0], coords[1:]) if len(coords) > 1 else Polygon(coords[0])
                if poly.is_valid and poly.area > 0:
                    bcat = classify_building(tags)
                    if bcat:
                        self.buildings.append({"cat": bcat, "geometry": poly, "area_m2": poly.area})
                    lcat = classify_landuse(tags)
                    if lcat:
                        self.landuses.append({"cat": lcat, "geometry": poly, "area_m2": poly.area})
        except:
            pass

print("1. Extracting OSM areas...", flush=True)
t0 = time.time()
h = AreaHandler()
h.apply_file(PBF, locations=True)
print(f"   {len(h.buildings):,} buildings  {len(h.landuses):,} landuse areas ({time.time()-t0:.0f}s)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. 建筑 → GeoDataFrame → 网格面积占比
# ══════════════════════════════════════════════════════════════════════════════
print("2. Computing building area proportions...", flush=True)
t0 = time.time()

bldg_df = pd.DataFrame(h.buildings)
bldg_df = bldg_df[bldg_df["area_m2"] > 0].copy()

# 面积近似（lat/lon → 粗略的 m²，用 cos(lat) 缩放经度）
# 或者直接用 GeoDataFrame + .to_crs("EPSG:3857") 再算面积
bldg_gdf = gpd.GeoDataFrame(bldg_df, geometry="geometry", crs="EPSG:4326")
bldg_gdf = bldg_gdf.to_crs("EPSG:3857")
bldg_gdf["area_m2"] = bldg_gdf.geometry.area
bldg_gdf["cx"] = bldg_gdf.geometry.centroid.x
bldg_gdf["cy"] = bldg_gdf.geometry.centroid.y

# 同样处理土地利用
lu_df = pd.DataFrame(h.landuses)
lu_df = lu_df[lu_df["area_m2"] > 0].copy()
lu_gdf = gpd.GeoDataFrame(lu_df, geometry="geometry", crs="EPSG:4326")
lu_gdf = lu_gdf.to_crs("EPSG:3857")
lu_gdf["area_m2"] = lu_gdf.geometry.area
lu_gdf["cx"] = lu_gdf.geometry.centroid.x
lu_gdf["cy"] = lu_gdf.geometry.centroid.y

print(f"   EPSG:3857 converted ({time.time()-t0:.0f}s)")

# ══════════════════════════════════════════════════════════════════════════════
# 4. 建网格
# ══════════════════════════════════════════════════════════════════════════════
print("3. Building grid...", flush=True)
GRID_SIZE = 500

# 用道路数据定范围
ep_df = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
import ast, pickle, warnings
from shapely.geometry import LineString

reg = pd.read_parquet(f"{DATA}/regression_table.parquet")
rag = reg[(reg["typhoon"]=="Ragasa") & (reg["signal_level"]>=3) & (reg["road_length_m"]>=200)].copy()

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
gdf_r = gpd.GeoDataFrame(ep_sub[["road_id","geometry"]],
                          geometry="geometry", crs="EPSG:4326").to_crs("EPSG:3857")
gdf_r["cx"] = gdf_r.geometry.centroid.x
gdf_r["cy"] = gdf_r.geometry.centroid.y

x0 = np.floor(gdf_r["cx"].min()/GRID_SIZE)*GRID_SIZE - GRID_SIZE
x1 = np.ceil( gdf_r["cx"].max()/GRID_SIZE)*GRID_SIZE + GRID_SIZE
y0 = np.floor(gdf_r["cy"].min()/GRID_SIZE)*GRID_SIZE - GRID_SIZE
y1 = np.ceil( gdf_r["cy"].max()/GRID_SIZE)*GRID_SIZE + GRID_SIZE

xs = np.arange(x0, x1, GRID_SIZE)
ys = np.arange(y0, y1, GRID_SIZE)
cells = [{"grid_id": f"{i}_{j}",
          "geometry": box(x, y, x+GRID_SIZE, y+GRID_SIZE)}
         for i,x in enumerate(xs) for j,y in enumerate(ys)]
gdf_grid = gpd.GeoDataFrame(cells, crs="EPSG:3857")
gdf_grid["grid_area_m2"] = gdf_grid.geometry.area
print(f"   {len(gdf_grid):,} grids")

# ══════════════════════════════════════════════════════════════════════════════
# 5. 建筑 → 网格（通过 centroid）→ 面积汇总
# ══════════════════════════════════════════════════════════════════════════════
print("4. Assigning buildings to grids...", flush=True)
t0 = time.time()

bldg_pts = gpd.GeoDataFrame(
    bldg_gdf[["cat","area_m2"]],
    geometry=[Point(x,y) for x,y in zip(bldg_gdf["cx"], bldg_gdf["cy"])],
    crs="EPSG:3857"
)
bldg_grid = gpd.sjoin(bldg_pts, gdf_grid[["grid_id","grid_area_m2","geometry"]],
                      how="left", predicate="within")
bldg_grid = bldg_grid.dropna(subset=["grid_id"])

bldg_cats = ["work","education","retail","commercial","hotel","medical",
             "industrial","civic","residential","untyped","other"]

# 每格各类建筑面积占比
grid_bldg = (bldg_grid.groupby(["grid_id","cat"])["area_m2"]
              .sum()
              .unstack(fill_value=0)
              .reindex(columns=bldg_cats, fill_value=0))
grid_area = gdf_grid.set_index("grid_id")["grid_area_m2"]
grid_bldg = grid_bldg.join(grid_area)

for c in bldg_cats:
    grid_bldg[f"bldg_{c}_pct"] = grid_bldg[c] / grid_bldg["grid_area_m2"] * 100

print(f"   {grid_bldg['bldg_residential_pct'].median():.1f}% median residential building coverage")
print(f"   {grid_bldg['bldg_work_pct'].median():.1f}% median office building coverage")

# ══════════════════════════════════════════════════════════════════════════════
# 6. 土地利用 → 网格
# ══════════════════════════════════════════════════════════════════════════════
print("5. Assigning landuse to grids...", flush=True)
lu_pts = gpd.GeoDataFrame(
    lu_gdf[["cat","area_m2"]],
    geometry=[Point(x,y) for x,y in zip(lu_gdf["cx"], lu_gdf["cy"])],
    crs="EPSG:3857"
)
lu_grid = gpd.sjoin(lu_pts, gdf_grid[["grid_id","geometry"]],
                    how="left", predicate="within")
lu_grid = lu_grid.dropna(subset=["grid_id"])

lu_cats = ["residential","industrial","commercial","institutional","green","construction","other"]
grid_lu = (lu_grid.groupby(["grid_id","cat"])["area_m2"]
            .sum()
            .unstack(fill_value=0)
            .reindex(columns=lu_cats, fill_value=0))
grid_lu = grid_lu.join(grid_area)

for c in lu_cats:
    grid_lu[f"lu_{c}_pct"] = grid_lu[c] / grid_lu["grid_area_m2"] * 100

# ══════════════════════════════════════════════════════════════════════════════
# 7. 合并面积特征 + Y 变量 + 人口
# ══════════════════════════════════════════════════════════════════════════════
print("6. Merging with Y and demographics...", flush=True)

# 建筑占比
bldg_pct = grid_bldg[[f"bldg_{c}_pct" for c in bldg_cats]].reset_index()
# 土地利用占比
lu_pct = grid_lu[[f"lu_{c}_pct" for c in lu_cats]].reset_index()
# 合并
grid_feat = bldg_pct.merge(lu_pct, on="grid_id", how="outer").fillna(0)

# 加人口
demo = pd.read_parquet(f"{DATA}/road_demo_features.parquet")
# 先 road→grid，再取均值
road_xy = gdf_r[["road_id","cx","cy"]].drop_duplicates("road_id")
road_xy["grid_id"] = np.where(
    (road_xy["cx"]>=x0) & (road_xy["cx"]<x1) & (road_xy["cy"]>=y0) & (road_xy["cy"]<y1),
    ((road_xy["cx"]-x0)/GRID_SIZE).astype(int).astype(str)+"_"+
    ((road_xy["cy"]-y0)/GRID_SIZE).astype(int).astype(str),
    np.nan
)
road_xy = road_xy.dropna(subset=["grid_id"])
rg_demo = road_xy.merge(demo, on="road_id", how="left")

for col in ["population_density_500m","median_income_500m",
            "employed_ratio_500m","student_ratio_500m",
            "retiree_ratio_500m","elderly_ratio_500m"]:
    rg_demo[col] = rg_demo[col].fillna(0)

grid_demo = rg_demo.groupby("grid_id").agg(
    pop_density=("population_density_500m","mean"),
    median_income=("median_income_500m","mean"),
    employed_ratio=("employed_ratio_500m","mean"),
    student_ratio=("student_ratio_500m","mean"),
    retiree_ratio=("retiree_ratio_500m","mean"),
    elderly_ratio=("elderly_ratio_500m","mean"),
).reset_index()

grid_feat = grid_feat.merge(grid_demo, on="grid_id", how="left")

# 加 n_roads
grid_n_roads = road_xy.groupby("grid_id")["road_id"].nunique().rename("n_roads").reset_index()
grid_feat = grid_feat.merge(grid_n_roads, on="grid_id", how="left")

# ══════════════════════════════════════════════════════════════════════════════
# 8. 加 Y（各时段 mean_deviation）
# ══════════════════════════════════════════════════════════════════════════════
rag_xy = rag.merge(road_xy[["road_id","grid_id"]], on="road_id", how="inner")

PERIODS = [("NIGHT","NIGHT"),("AM_PEAK","AM_PEAK"),("MIDDAY","MIDDAY"),
           ("PM_PEAK","PM_PEAK"),("EVENING","EVENING")]

records = []
for tg, label in PERIODS:
    sub = rag_xy[rag_xy["time_group"]==tg]
    y_agg = sub.groupby("grid_id")["mean_deviation"].agg(
        mean_deviation="mean", pct_pos=lambda x:(x>0).mean()
    ).reset_index()
    y_agg["time_group"] = tg
    records.append(y_agg)
grid_y = pd.concat(records, ignore_index=True)

data = grid_y.merge(grid_feat, on="grid_id", how="inner")
data = data[data["n_roads"] >= 3].copy()
print(f"   {len(data):,} grid×period rows, {data['grid_id'].nunique():,} grids")

# ══════════════════════════════════════════════════════════════════════════════
# 9. 特征工程
# ══════════════════════════════════════════════════════════════════════════════
# 建筑占比（已经 %，用 log1p）
BLDG_USE = ["work","education","retail","commercial","hotel","medical",
            "industrial","civic","residential"]
for c in BLDG_USE:
    data[f"log_bldg_{c}"] = np.log1p(data[f"bldg_{c}_pct"])

# 土地利用占比
LU_USE = ["residential","industrial","commercial","institutional","green"]
for c in LU_USE:
    data[f"log_lu_{c}"] = np.log1p(data[f"lu_{c}_pct"])

data["log_pop_density"] = np.log1p(data["pop_density"])
data["log_income"]      = np.log1p(data["median_income"])
data["log_n_roads"]     = np.log1p(data["n_roads"])

# ══════════════════════════════════════════════════════════════════════════════
# 10. 回归
# ══════════════════════════════════════════════════════════════════════════════
import statsmodels.formula.api as smf

BLDG_TERMS = " + ".join(f"log_bldg_{c}" for c in BLDG_USE)
LU_TERMS   = " + ".join(f"log_lu_{c}" for c in LU_USE)
DEMO_TERMS = ("log_pop_density + log_income + "
              "employed_ratio + student_ratio + retiree_ratio + elderly_ratio")
STRUCT     = "log_n_roads"

FORMULA = f"mean_deviation ~ {BLDG_TERMS} + {LU_TERMS} + {DEMO_TERMS} + {STRUCT}"

KEY_VARS = ([f"log_bldg_{c}" for c in BLDG_USE] +
            [f"log_lu_{c}" for c in LU_USE] +
            ["log_pop_density","log_income",
             "employed_ratio","student_ratio","retiree_ratio","elderly_ratio",
             "log_n_roads"])

print("\n" + "="*72)
print("  Grid regression w/ OSM AREA proportions (500m, building + landuse)")
print("="*72)

results = {}
for tg, label in PERIODS:
    sub = data[data["time_group"]==tg]
    if len(sub) < 50:
        print(f"\n{label}: too few")
        continue
    res = smf.ols(FORMULA, data=sub).fit()
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
    print(f"  Y mean={sub['mean_deviation'].mean():+.4f}  "
          f"pct_pos={(sub['mean_deviation']>0).mean():.1%}")
    print(f"{'='*72}")
    # 只显示前20行（关键的）
    print(tbl.head(25)[["coef","se","t","p","sig"]].round(4).to_string())

# 汇总
print("\n\n" + "="*90)
print("  Area-based coefficient summary")
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

# R² 对比
print("\n\n" + "="*72)
print("  R² Comparison: POI-point vs OSM-area")
print("="*72)
prev_poi_grid = {"AM_PEAK":0.0464, "MIDDAY":0.1062, "PM_PEAK":0.1181, "NIGHT":0.0481, "EVENING":0.0898}
prev_poi_road = {"AM_PEAK":0.0385, "MIDDAY":0.0342, "PM_PEAK":0.0549, "NIGHT":0.0099, "EVENING":0.0183}
print(f"{'Period':<12} {'POI point (road)':>16} {'POI point (grid)':>16} {'OSM area (grid)':>16}")
print("-"*65)
for tg, _ in PERIODS:
    r2_new = f"{results[tg][0].rsquared:.4f}" if tg in results else "--"
    r2_poig = f"{prev_poi_grid.get(tg,0):.4f}"
    r2_poir = f"{prev_poi_road.get(tg,0):.4f}"
    print(f"{tg:<12} {r2_poir:>16} {r2_poig:>16} {r2_new:>16}")

# 保存
print("\nSaving...")
data.to_parquet(f"{DATA}/grid_area_features.parquet", index=False)
try:
    with pd.ExcelWriter(f"{DATA}/grid_area_regression.xlsx") as w:
        sumtbl.to_excel(w, sheet_name="Summary")
        for tg, _ in PERIODS:
            if tg in results:
                results[tg][1].round(4).to_excel(w, sheet_name=tg)
    print("  Saved: data/grid_area_regression.xlsx")
except Exception as e:
    print(f"  Excel: {e}")

print("Done.")
