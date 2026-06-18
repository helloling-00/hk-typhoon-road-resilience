"""
完整重建 estate_features + road_demo_features
包含：所有年龄段、经济活动、收入、工作人口
"""
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings("ignore")

CENSUS = "/Users/helloling/Desktop/人口普查"
DATA   = "/Users/helloling/workspace/thesis/data"

# ══════════════════════════════════════════════════════════════════════════
# 1. 经济活动身份
# ══════════════════════════════════════════════════════════════════════════
eco = pd.read_excel(f"{CENSUS}/经济活动身份.xlsx", header=None)
eco = eco.iloc[6:].copy()
eco.columns = ["estate", "activity", "count"]
eco["estate"] = eco["estate"].ffill()
eco = eco.dropna(subset=["count"])
eco = eco[~eco["activity"].str.contains("小计|总计", na=False)]
eco["count"] = pd.to_numeric(eco["count"], errors="coerce")

eco_wide = eco.pivot_table(index="estate", columns="activity", values="count", aggfunc="sum")
# 更安全的列名
eco_wide.columns = [str(c).replace("(","").replace(")","").replace(" ","_").strip() for c in eco_wide.columns]
eco_wide["eco_total"] = eco_wide.sum(axis=1)

for c in eco_wide.columns:
    if c == "eco_total": continue
    eco_wide[f"ratio_{c}"] = eco_wide[c] / eco_wide["eco_total"]

# 重命名几个关键的
rename_map = {}
for c in eco_wide.columns:
    if "雇员" in c and "ratio" in c:
        rename_map[c] = "ratio_employee"
    elif "雇主" in c and "ratio" in c:
        rename_map[c] = "ratio_employer"
    elif "自营" in c and "ratio" in c:
        rename_map[c] = "ratio_self_employed"
    elif "学生" in c and "ratio" in c:
        rename_map[c] = "ratio_student"
    elif "退休" in c and "ratio" in c:
        rename_map[c] = "ratio_retiree"
    elif "料理家务" in c and "ratio" in c:
        rename_map[c] = "ratio_homemaker"
    elif "无酬照顾" in c and "ratio" in c:
        rename_map[c] = "ratio_carer"
    elif "无酬家庭" in c and "ratio" in c:
        rename_map[c] = "ratio_unpaid_family"
    elif "其他" in c and "ratio" in c:
        rename_map[c] = "ratio_other_activity"
eco_wide = eco_wide.rename(columns=rename_map)
print(f"1. Eco activities: {[c for c in eco_wide.columns if c.startswith('ratio_')]}")

# ══════════════════════════════════════════════════════════════════════════
# 2. 年龄（5段 + elderly_ratio）
# ══════════════════════════════════════════════════════════════════════════
age = pd.read_excel(f"{CENSUS}/年龄.xlsx", header=None)
age = age.iloc[5:].copy()
age.columns = ["year", "estate", "age_group", "count"]
age["estate"] = age["estate"].ffill()
age = age.dropna(subset=["count","estate"])
age = age[~age["age_group"].str.contains("小计|总计", na=False)]
age["count"] = pd.to_numeric(age["count"], errors="coerce")

age_wide = age.pivot_table(index="estate", columns="age_group", values="count", aggfunc="sum")
age_wide["age_total"] = age_wide.sum(axis=1)

# 各年龄段比例
age_map = {"0 - 14": "age_0_14", "15 - 24": "age_15_24",
           "25 - 44": "age_25_44", "45 - 64": "age_45_64", "65+": "age_65plus"}
for orig, new in age_map.items():
    if orig in age_wide.columns:
        age_wide[f"ratio_{new}"] = age_wide[orig] / age_wide["age_total"]
        age_wide[new] = age_wide[orig]

age_wide["elderly_ratio"] = age_wide["ratio_age_65plus"]
print(f"2. Age groups: {[c for c in age_wide.columns if 'ratio' in c]}")

# ══════════════════════════════════════════════════════════════════════════
# 3. 人口及工作人口
# ══════════════════════════════════════════════════════════════════════════
pop = pd.read_excel(f"{CENSUS}/人口及工作人口.xlsx", header=None)
pop = pop.iloc[5:].copy()
pop.columns = ["year", "estate", "total_pop", "working_pop"]
pop["estate"] = pop["estate"].ffill()
pop = pop.dropna(subset=["total_pop","estate"])
for c in ["total_pop", "working_pop"]:
    pop[c] = pd.to_numeric(pop[c], errors="coerce")
pop["working_pop_ratio"] = pop["working_pop"] / pop["total_pop"]
print(f"3. Population: {len(pop)} estates")

# ══════════════════════════════════════════════════════════════════════════
# 4. 收入
# ══════════════════════════════════════════════════════════════════════════
inc = pd.read_excel(f"{CENSUS}/家庭住户每月收入中位数.xlsx", header=None)
inc = inc.iloc[5:].copy()
inc.columns = ["year", "estate", "median_income"]
inc["estate"] = inc["estate"].ffill()
inc = inc.dropna(subset=["median_income","estate"])
inc["median_income"] = pd.to_numeric(inc["median_income"], errors="coerce")
print(f"4. Income: {len(inc)} estates")

# ══════════════════════════════════════════════════════════════════════════
# 5. 坐标
# ══════════════════════════════════════════════════════════════════════════
coord = pd.read_excel(f"{CENSUS}/经纬度.xlsx")
coord = coord.rename(columns={"EstateName": "estate"})
coord = coord[["estate", "lat", "lon"]].dropna(subset=["lat","lon"]).drop_duplicates("estate")
print(f"5. Coordinates: {len(coord)} estates")

# ══════════════════════════════════════════════════════════════════════════
# 6. 合并
# ══════════════════════════════════════════════════════════════════════════
est = coord.copy()
for df_merge in [eco_wide, age_wide, pop[["estate","total_pop","working_pop","working_pop_ratio"]],
                 inc[["estate","median_income"]]]:
    est = est.merge(df_merge, on="estate", how="left")

# 只保留有坐标+总人口的
est = est.dropna(subset=["total_pop", "lat", "lon"])
est["population_total"] = est["total_pop"]

print(f"\n6. Final: {len(est)} estates, {len(est.columns)} columns")
ratio_cols = [c for c in est.columns if c.startswith("ratio_")]
print(f"   Ratio columns ({len(ratio_cols)}): {ratio_cols}")

# 数值检查
for c in ["total_pop","working_pop","median_income","elderly_ratio"]:
    print(f"   {c}: {est[c].min():.0f} - {est[c].max():.0f}  (missing: {est[c].isna().sum()})")

est.to_parquet(f"{DATA}/estate_features.parquet", index=False)
est.to_excel(f"{DATA}/estate_features.xlsx", index=False)
print(f"\nSaved: estate_features.parquet + .xlsx")
