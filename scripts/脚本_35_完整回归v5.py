"""
v5 完整回归：重建 census 经济特征 + 10 类 POI + 收入交互项
"""
import pandas as pd, numpy as np
import geopandas as gpd, statsmodels.formula.api as smf
from shapely.geometry import Point, LineString
from shapely import wkb as swkb
import ast, pickle, warnings, time
warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
CENSUS = "/Users/helloling/Desktop/人口普查"

print("=" * 72)
print("  v5: Census economic detail + 10 POI + income interactions")
print("=" * 72)

# ═══════════════════════════════════════════════════════════════════════════
# Part A: Rebuild estate_features with proper economic activity ratios
# ═══════════════════════════════════════════════════════════════════════════
print("\nA. Rebuilding estate_features from raw census...", flush=True)

# A1. 经济活动身份
eco = pd.read_excel(f"{CENSUS}/经济活动身份.xlsx", header=None)
eco = eco.iloc[6:].copy()  # skip metadata rows + column labels
eco.columns = ["estate", "activity", "count"]
# Forward-fill estate name (merged cells in excel)
eco["estate"] = eco["estate"].fillna(method="ffill")
eco = eco.dropna(subset=["count"])
eco["count"] = pd.to_numeric(eco["count"], errors="coerce")
# 过滤小计/总计
eco = eco[~eco["activity"].str.contains("小计|总计", na=False)].copy()
eco = eco.dropna(subset=["count"])

# pivot: estate × activity
eco_wide = eco.pivot_table(index="estate", columns="activity", values="count", aggfunc="sum")
eco_wide.columns = [c.strip() for c in eco_wide.columns]
print(f"  Eco activities: {eco_wide.columns.tolist()}")
eco_wide["eco_total"] = eco_wide.sum(axis=1)

# 计算各类比例
activity_cols = eco_wide.columns.tolist()
for c in activity_cols:
    if c == "eco_total": continue
    safe_name = c.replace("(", "").replace(")", "").replace(" ", "_").replace("（","").replace("）","")
    eco_wide[f"ratio_{safe_name}"] = eco_wide[c] / eco_wide["eco_total"]

# A2. 人口及工作人口
pop = pd.read_excel(f"{CENSUS}/人口及工作人口.xlsx", header=None)
pop = pop.iloc[5:].copy()  # skip metadata + column labels
pop.columns = ["year", "estate", "total_pop", "working_pop"]
pop["estate"] = pop["estate"].fillna(method="ffill")
pop = pop.dropna(subset=["total_pop","estate"])
for c in ["total_pop", "working_pop"]:
    pop[c] = pd.to_numeric(pop[c], errors="coerce")
pop = pop.dropna(subset=["total_pop"])
pop["working_pop_ratio"] = pop["working_pop"] / pop["total_pop"]

# A3. 年龄
age = pd.read_excel(f"{CENSUS}/年龄.xlsx", header=None)
age = age.iloc[5:].copy()
age.columns = ["year", "estate", "age_group", "count"]
age["estate"] = age["estate"].fillna(method="ffill")
age = age.dropna(subset=["count","estate"])
age = age[~age["age_group"].str.contains("小计|总计", na=False)]
age["count"] = pd.to_numeric(age["count"], errors="coerce")
age_wide = age.pivot_table(index="estate", columns="age_group", values="count", aggfunc="sum")
age_wide["age_total"] = age_wide.sum(axis=1)
# elderly: use "65+" column if available
if "65+" in age_wide.columns:
    age_wide["elderly_count"] = age_wide["65+"]
else:
    elderly_cols = [c for c in age_wide.columns if "65" in str(c)]
    age_wide["elderly_count"] = age_wide[elderly_cols].sum(axis=1) if elderly_cols else 0
age_wide["elderly_ratio"] = age_wide["elderly_count"] / age_wide["age_total"]

# A4. 收入
inc = pd.read_excel(f"{CENSUS}/家庭住户每月收入中位数.xlsx", header=None)
inc = inc.iloc[5:].copy()  # skip header rows including column labels
inc.columns = ["year", "estate", "median_income"]
inc["estate"] = inc["estate"].fillna(method="ffill")
inc = inc.dropna(subset=["median_income","estate"])
inc["median_income"] = pd.to_numeric(inc["median_income"], errors="coerce")

# A5. 坐标
coord = pd.read_excel(f"{CENSUS}/经纬度.xlsx")
coord = coord.rename(columns={"EstateName": "estate"})
coord = coord[["estate", "lat", "lon"]].dropna(subset=["lat","lon"]).drop_duplicates("estate")

# A6. 合并所有 census 表
est = coord.copy()
for df_merge in [eco_wide, pop[["estate","total_pop","working_pop","working_pop_ratio"]],
                 age_wide[["elderly_ratio"]], inc[["estate","median_income"]]]:
    est = est.merge(df_merge, on="estate", how="left")

# 保留有足够数据的屋苑
est = est.dropna(subset=["total_pop", "lat", "lon"])
est["population_total"] = est["total_pop"]
print(f"  Final estates: {len(est):,}")

# 保存（覆盖）
est.to_parquet(f"{DATA}/estate_features.parquet", index=False)
print(f"  Saved: data/estate_features.parquet")

# ═══════════════════════════════════════════════════════════════════════════
# Part B: Rebuild road_demo_features (line buffer spatial join)
# ═══════════════════════════════════════════════════════════════════════════
print("\nB. Rebuilding road_demo_features...", flush=True)

est = pd.read_parquet(f"{DATA}/estate_features.parquet").dropna(subset=["lat","lon","population_total"])
gdf_est = gpd.GeoDataFrame(
    est, geometry=[Point(lon, lat) for lat, lon in zip(est["lat"], est["lon"])],
    crs="EPSG:4326"
).to_crs("EPSG:3857")

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
gdf_roads = gpd.GeoDataFrame(ep_df, geometry="geometry", crs="EPSG:4326").to_crs("EPSG:3857")

# Only roads appearing in regression_table (dramatically reduces memory)
reg_all = pd.read_parquet(f"{DATA}/regression_table.parquet", columns=["road_id"])
reg_road_ids = set(reg_all["road_id"].unique())
gdf_roads = gdf_roads[gdf_roads["road_id"].isin(reg_road_ids)]
print(f"  Roads in regression sample: {len(gdf_roads):,}")

# 500m line buffer
BUFFER_M = 500
gdf_buf = gdf_roads[["road_id"]].copy()
gdf_buf["geometry"] = gdf_roads.geometry.buffer(BUFFER_M)
gdf_buf = gdf_buf.set_geometry("geometry")
gdf_buf["buf_area_km2"] = gdf_buf.geometry.area / 1e6

# Spatial join
# Select columns from est: all ratio columns + lat/lon-derived ones
est_cols = ["population_total","median_income","elderly_ratio","working_pop","working_pop_ratio"]
# Add new economic activity ratio columns
ratio_cols = [c for c in est.columns if c.startswith("ratio_")]
est_cols += ratio_cols

joined = gpd.sjoin(gdf_buf, gdf_est[["geometry"] + est_cols],
                   how="left", predicate="contains")
print(f"  {joined['index_right'].notna().sum():,} road-estate pairs")

def wavg(group, val_col, wt_col="population_total"):
    mask = group[val_col].notna() & group[wt_col].notna()
    v = group.loc[mask, val_col].values
    w = group.loc[mask, wt_col].values
    if len(v)==0 or w.sum()==0: return np.nan
    return float(np.average(v, weights=w))

buf_area_lkp = gdf_buf.set_index("road_id")["buf_area_km2"]
records = []
for road_id, grp in joined.groupby("road_id"):
    valid = grp.dropna(subset=["population_total"])
    pop_sum = valid["population_total"].sum()
    buf_area_km2 = buf_area_lkp.get(road_id, np.pi*(BUFFER_M/1000)**2)
    rec = {
        "road_id": road_id,
        "estate_count_500m": len(valid),
        "population_total_500m": pop_sum,
        "population_density_500m": pop_sum / buf_area_km2 if buf_area_km2>0 else 0,
    }
    # weighted averages for all columns
    for col in est_cols:
        if col == "population_total": continue
        rec[f"{col}_500m"] = wavg(valid, col)
    records.append(rec)

road_demo = pd.DataFrame(records)
# 0 estates → NaN
for c in road_demo.columns:
    if c not in ["road_id","estate_count_500m"]:
        road_demo.loc[road_demo["estate_count_500m"]==0, c] = np.nan

road_demo.to_parquet(f"{DATA}/road_demo_features.parquet", index=False)
print(f"  {len(road_demo):,} roads  estates>0: {(road_demo['estate_count_500m']>0).sum():,}")
print(f"  New ratio columns: {[c for c in road_demo.columns if 'ratio_' in c]}")

# ═══════════════════════════════════════════════════════════════════════════
# Part C: Rebuild regression_table with new demo features
# ═══════════════════════════════════════════════════════════════════════════
print("\nC. Updating regression_table...", flush=True)

reg = pd.read_parquet(f"{DATA}/regression_table.parquet")

# Drop old demo columns
old_demo = ["population_density_500m","median_income_500m",
            "employed_ratio_500m","student_ratio_500m",
            "retiree_ratio_500m","elderly_ratio_500m",
            "working_pop_ratio_500m","estate_count_500m","population_total_500m"]
old_found = [c for c in old_demo if c in reg.columns]
reg = reg.drop(columns=old_found)

# Merge new
demo = pd.read_parquet(f"{DATA}/road_demo_features.parquet")
reg = reg.merge(demo, on="road_id", how="left")

# Also refresh landuse features (10 POI densities)
old_lu = [c for c in reg.columns if c.endswith("_density") or c.endswith("_count")]
reg = reg.drop(columns=[c for c in old_lu if c in reg.columns and "population" not in c and "incident" not in c])

lu = pd.read_parquet(f"{DATA}/road_landuse_features.parquet")
lu_keep = ["road_id"] + [c for c in lu.columns if c.endswith("_density")]
reg = reg.merge(lu[lu_keep], on="road_id", how="left")

# Refresh structural
old_struct = ["intersection_degree","dist_to_coast_m"]
reg = reg.drop(columns=[c for c in old_struct if c in reg.columns], errors="ignore")
struct = pd.read_parquet(f"{DATA}/road_structural_features.parquet")
reg = reg.merge(struct, on="road_id", how="left")

# Refresh incidents
old_inc = ["incident_count_500m","severe_incident_500m","closure_nearby_500m"]
reg = reg.drop(columns=[c for c in old_inc if c in reg.columns], errors="ignore")
inc_feat = pd.read_parquet(f"{DATA}/road_incident_features.parquet")
reg = reg.merge(inc_feat, on="road_id", how="left")

print(f"  Columns: {reg.columns.tolist()}")
print(f"  Rows: {len(reg):,}")

reg.to_parquet(f"{DATA}/regression_table.parquet", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# Part D: Feature engineering + regression
# ═══════════════════════════════════════════════════════════════════════════
print("\nD. Feature engineering...", flush=True)

# Filter: Ragasa S3+, road length >= 200m
rag = reg[(reg["typhoon"]=="Ragasa") & (reg["signal_level"]>=3) & (reg["road_length_m"]>=200)].copy()
print(f"  Ragasa S3+ len>=200: {len(rag):,} rows  {rag['road_id'].nunique():,} roads")

# POI 10 categories
LU_CATS = ["work","education","retail","food_drink","recreation",
           "medical","transport","tourism","finance","civic"]
for cat in LU_CATS:
    rag[f"log_{cat}"] = np.log1p(rag[f"{cat}_density"].fillna(0))

# Road structure
rag["log_road_length"]  = np.log1p(rag["road_length_m"])
rag["log_intersection"] = np.log1p(rag["intersection_degree"].fillna(0))
rag["log_dist_coast"]   = np.log1p(rag["dist_to_coast_m"].fillna(0))
rag["log_incidents"]    = np.log1p(rag["incident_count_500m"].fillna(0))

# Demographics — fill NaN (roads with no estates) with 0 for ratios, median for continuous
demo_cols_500m = [c for c in rag.columns if c.endswith("_500m") and "ratio" not in c]
for col in demo_cols_500m:
    rag[col] = rag[col].fillna(rag[col].median())

ratio_cols_500m = [c for c in rag.columns if c.endswith("_500m") and ("ratio" in c or "elderly" in c)]
for col in ratio_cols_500m:
    rag[col] = rag[col].fillna(0)

rag["log_pop_density"]  = np.log1p(rag["population_density_500m"])
rag["log_income"]       = np.log1p(rag["median_income_500m"])

# Signal
rag["sig_num"]   = rag["signal_level"].replace({9:10})
rag["is_sig10"]  = (rag["signal_level"]==10).astype(float)

# Key interaction: income × work_density (high-income office areas = WFH capability)
rag["log_income_x_log_work"] = rag["log_income"] * rag["log_work"]

# Identify the new ratio columns
new_ratio_cols = [c for c in rag.columns if c.startswith("ratio_") and c.endswith("_500m")]
print(f"\n  Demographics in regression:")
for c in new_ratio_cols + ["population_density_500m","median_income_500m",
                            "elderly_ratio_500m","working_pop_ratio_500m"]:
    if c in rag.columns:
        non_zero = (rag[c] > 0).mean()
        print(f"    {c}: non-zero={non_zero:.1%}  median={rag[c].median():.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# Part E: Regression — 5 periods, road-level with cluster SE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("  E. Regression: 10 POI categories + detailed census + interactions")
print("="*72)

# Build formula
LU_TERMS    = " + ".join(f"log_{c}" for c in LU_CATS)

# Demographics: use the best available columns
available_demo = []
for c in ["log_pop_density","log_income",
          "elderly_ratio_500m","working_pop_ratio_500m",
          "log_income_x_log_work"]:
    if c in rag.columns:
        available_demo.append(c)
# Add any new ratio columns
for c in new_ratio_cols:
    if c in rag.columns and rag[c].notna().sum() > 100:
        available_demo.append(c)

DEMO_TERMS = " + ".join(available_demo)
STRUCT_TERMS = "log_road_length + log_intersection + log_dist_coast + log_incidents"
CTRL_TERMS   = "sig_num + is_sig10"

FORMULA = f"mean_deviation ~ {LU_TERMS} + {DEMO_TERMS} + {STRUCT_TERMS} + {CTRL_TERMS}"
print(f"  Formula length: {len(FORMULA)} chars")

PERIODS = [
    ("NIGHT","NIGHT"),
    ("AM_PEAK","AM_PEAK"),
    ("MIDDAY","MIDDAY"),
    ("PM_PEAK","PM_PEAK"),
    ("EVENING","EVENING"),
]

def run(df, label):
    if len(df) < 200: return None, None
    try:
        res = smf.ols(FORMULA, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["road_id"].values})
    except Exception as e:
        print(f"\n  {label}: FAILED ({e})")
        return None, None
    tbl = pd.DataFrame({
        "coef": res.params, "se": res.bse,
        "t": res.tvalues,   "p": res.pvalues,
    })
    tbl["sig"] = tbl["p"].apply(
        lambda p: "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ".")
    print(f"\n{'='*72}")
    print(f"  {label}  N={len(df):,}  roads={df['road_id'].nunique():,}  "
          f"R²={res.rsquared:.4f}  Adj-R²={res.rsquared_adj:.4f}")
    print(f"  Y mean={df['mean_deviation'].mean():+.4f}  "
          f"pct_pos={(df['mean_deviation']>0).mean():.1%}")
    print(f"{'='*72}")
    # Show POI + some demo
    print(tbl[["coef","se","t","p","sig"]].round(4).to_string())
    return res, tbl

results = {}
for tg, label in PERIODS:
    sub = rag[rag["time_group"]==tg]
    res, tbl = run(sub, label)
    if tbl is not None:
        results[tg] = (res, tbl)

# ═══════════════════════════════════════════════════════════════════════════
# Part F: Summary
# ═══════════════════════════════════════════════════════════════════════════
KEY_VARS = [f"log_{c}" for c in LU_CATS] + [
    "log_pop_density","log_income","log_income_x_log_work",
    "elderly_ratio_500m","working_pop_ratio_500m",
    "log_intersection","log_dist_coast","log_incidents","log_road_length",
] + [c for c in new_ratio_cols if any(c in (results[tg][1].index if tg in results else []) for tg, _ in PERIODS)]

print("\n\n" + "="*90)
print("  v5 Coefficient Summary (POI + census detail + income interactions)")
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

# R² comparison
print("\n\nR² Comparison:")
prev_v1 = {"AM_PEAK":0.0385,"MIDDAY":0.0342,"PM_PEAK":0.0549,"NIGHT":0.0099,"EVENING":0.0183}
prev_grid = {"AM_PEAK":0.0464,"MIDDAY":0.1062,"PM_PEAK":0.1181,"NIGHT":0.0481,"EVENING":0.0898}
prev_phys = {"AM_PEAK":0.0238,"MIDDAY":0.0370,"PM_PEAK":0.0441,"NIGHT":0.0111,"EVENING":0.0232}
print(f"{'Period':<12} {'v1 (3台风)':>12} {'v3 (len>=200)':>14} {'500m grid':>12} {'v5 (census+int)':>16}")
print("-"*70)
for tg, _ in PERIODS:
    r2_v5 = f"{results[tg][0].rsquared:.4f}" if tg in results else "--"
    print(f"{tg:<12} {prev_v1[tg]:>12.4f} {prev_phys[tg]:>14.4f} {prev_grid[tg]:>12.4f} {r2_v5:>16}")

# Save
try:
    with pd.ExcelWriter(f"{DATA}/regression_v5_results.xlsx") as w:
        sumtbl.to_excel(w, sheet_name="Summary")
        for tg, _ in PERIODS:
            if tg in results:
                results[tg][1].round(4).to_excel(w, sheet_name=tg)
    print(f"\nSaved: data/regression_v5_results.xlsx")
except Exception as e:
    print(f"\nExcel: {e}")

print("\nDone.")
