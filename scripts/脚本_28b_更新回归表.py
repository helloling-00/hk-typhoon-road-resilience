"""
在 regression_table.parquet 基础上合并所有特征：
  - 土地利用密度（10类，个/km²）
  - 人口特征（完整版含 employed/student/retiree ratio）
  - 道路结构特征（路口度、海岸距离）
  - baseline_tg_speed（本路本时段正常速度）
"""
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"

print("Loading regression table...", flush=True)
reg = pd.read_parquet(f"{DATA}/regression_table.parquet")
print(f"  {len(reg):,} rows")

# ── 移除所有旧特征列 ─────────────────────────────────────────────────────────
old = [c for c in reg.columns if c not in
       ["road_id","typhoon","signal_level","time_group","mean_speed",
        "mean_deviation","n_slots","road_length_m","road_category",
        "baseline_avg_speed","incident_count_500m","severe_incident_500m",
        "closure_nearby_500m","signal_group","road_broad"]]
reg = reg.drop(columns=old)
print(f"  Kept base columns: {reg.columns.tolist()}")

# ── 土地利用密度（10类） ──────────────────────────────────────────────────────
lu = pd.read_parquet(f"{DATA}/road_landuse_features.parquet")
# 只保留密度列和 buf_area
lu_keep = ["road_id","buf_area_km2"] + [c for c in lu.columns if c.endswith("_density")]
reg = reg.merge(lu[lu_keep], on="road_id", how="left")
# 无匹配（极少郊区路）填 0
for c in lu_keep[2:]:
    reg[c] = reg[c].fillna(0)

# ── 人口特征 ─────────────────────────────────────────────────────────────────
demo = pd.read_parquet(f"{DATA}/road_demo_features.parquet")
reg = reg.merge(demo, on="road_id", how="left")

# ── 道路结构特征 ──────────────────────────────────────────────────────────────
struct = pd.read_parquet(f"{DATA}/road_structural_features.parquet")
reg = reg.merge(struct, on="road_id", how="left")

# ── baseline_tg_speed ────────────────────────────────────────────────────────
print("Computing baseline_tg_speed...", flush=True)
bl = pd.read_parquet(f"{DATA}/baseline_speed.parquet",
                     columns=["road_id","day_type","slot","mean_speed"])

def tg(slot):
    if  0 <= slot <= 13: return "NIGHT"
    if 14 <= slot <= 18: return "AM_PEAK"
    if 19 <= slot <= 33: return "MIDDAY"
    if 34 <= slot <= 38: return "PM_PEAK"
    return "EVENING"

bl["time_group"] = bl["slot"].apply(tg)
bl_tg = (bl[bl["day_type"]=="WORKDAY"]
         .groupby(["road_id","time_group"])["mean_speed"].mean()
         .rename("baseline_tg_speed").reset_index())
reg = reg.merge(bl_tg, on=["road_id","time_group"], how="left")
reg["baseline_tg_speed"] = reg["baseline_tg_speed"].fillna(reg["baseline_avg_speed"])

# ── 保存 ─────────────────────────────────────────────────────────────────────
out = f"{DATA}/regression_table.parquet"
reg.to_parquet(out, index=False)
print(f"\n最终列 ({len(reg.columns)}):")
for c in reg.columns:
    nn = reg[c].notna().sum()
    print(f"  {c:<35} notna={nn:,}")
print(f"\nSaved: {out}  ({len(reg):,} rows)")
