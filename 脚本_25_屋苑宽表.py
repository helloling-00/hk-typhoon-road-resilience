"""
整理屋苑人口宽表，输出：
  data/estate_features.parquet
  每行 = 一个屋苑，包含经纬度 + 人口 + 收入 + 年龄结构
"""
import pandas as pd
import numpy as np
from pathlib import Path

CENSUS = "/Users/helloling/Desktop/人口普查"
OUT    = "/Users/helloling/workspace/thesis/data"

# ── 1. 经纬度（来自 CSV） ────────────────────────────────────────────────────
print("Loading coordinates...", flush=True)
coords = pd.read_csv(f"{CENSUS}/estates_with_coords (3).csv",
                     usecols=["EstateName","lat","lon"])
coords = coords.dropna(subset=["lat","lon","EstateName"])
coords = coords.rename(columns={"EstateName":"estate"})
coords = coords.drop_duplicates("estate")
print(f"  {len(coords)} estates with coordinates")

# ── 2. 总人口 + 工作人口 ────────────────────────────────────────────────────
print("Loading population...", flush=True)
pop_raw = pd.read_excel(f"{CENSUS}/人口及工作人口.xlsx", header=None)
# 数据从第6行(index 5)开始，列: 年, 屋苑, 人口, 工作人口
pop = pop_raw.iloc[5:].copy()
pop.columns = ["年","estate","population_total","working_pop"]
pop["estate"] = pop["estate"].ffill()   # 合并格里屋苑名向下填充
pop["年"] = pop["年"].ffill()
pop = pop[pop["年"].astype(str).str.contains("2021")]
pop = pop[["estate","population_total","working_pop"]].dropna(subset=["estate"])
pop = pop[~pop["estate"].astype(str).str.contains("备注|来源|统计|查询|注：|符号|格式|标记|XLSX|CSV|下载", na=True)]
pop["population_total"] = pd.to_numeric(pop["population_total"], errors="coerce")
pop["working_pop"]      = pd.to_numeric(pop["working_pop"],      errors="coerce")
pop = pop.dropna(subset=["population_total"])
print(f"  {len(pop)} estates, population range {pop['population_total'].min():.0f}–{pop['population_total'].max():.0f}")

# ── 3. 家庭月收入中位数 ─────────────────────────────────────────────────────
print("Loading income...", flush=True)
inc_raw = pd.read_excel(f"{CENSUS}/家庭住户每月收入中位数.xlsx", header=None)
inc = inc_raw.iloc[5:].copy()
inc.columns = ["年","estate","median_income"]
inc["estate"] = inc["estate"].ffill()
inc["年"] = inc["年"].ffill()
inc = inc[inc["年"].astype(str).str.contains("2021")]
inc = inc[["estate","median_income"]].dropna(subset=["estate"])
inc = inc[~inc["estate"].astype(str).str.contains("备注|来源|统计|查询|注：|符号|格式|标记", na=True)]
inc["median_income"] = pd.to_numeric(inc["median_income"], errors="coerce")
inc = inc.dropna(subset=["median_income"])
print(f"  {len(inc)} estates, income range {inc['median_income'].min():.0f}–{inc['median_income'].max():.0f} HKD")

# ── 4. 年龄结构 ──────────────────────────────────────────────────────────────
print("Loading age structure...", flush=True)
age_raw = pd.read_excel(f"{CENSUS}/年龄.xlsx", header=None)
age = age_raw.iloc[5:].copy()
age.columns = ["年","estate","age_group","count"]
age["estate"] = age["estate"].ffill()
age["年"] = age["年"].ffill()
age = age[age["年"].astype(str).str.contains("2021")]
age = age[["estate","age_group","count"]].dropna(subset=["estate","age_group"])
age = age[~age["estate"].astype(str).str.contains("备注|来源|统计|查询|注：|符号|格式|标记", na=True)]
age["count"] = pd.to_numeric(age["count"], errors="coerce")
age = age.dropna(subset=["count"])

# 计算 65+ 人数和比例
age65 = age[age["age_group"].astype(str).str.contains("65")].groupby("estate")["count"].sum().rename("pop_65plus")
age_total = age.groupby("estate")["count"].sum().rename("pop_age_total")
age_df = pd.concat([age65, age_total], axis=1).reset_index()
age_df["elderly_ratio"] = age_df["pop_65plus"] / age_df["pop_age_total"]
print(f"  {len(age_df)} estates with age data")
print(f"  elderly_ratio range: {age_df['elderly_ratio'].min():.3f}–{age_df['elderly_ratio'].max():.3f}")

# ── 4b. 经济活动身份：学生 + 退休人士 ──────────────────────────────────────
print("Loading economic activity status...", flush=True)
eco_raw = pd.read_excel(f"{CENSUS}/经济活动身份.xlsx", header=None)
eco = eco_raw.iloc[5:].copy()
eco.columns = ["estate", "activity", "count"]   # 3列：屋苑/年份合并列, 身份, 人数
eco["estate"] = eco["estate"].ffill()
eco = eco[["estate", "activity", "count"]].dropna(subset=["estate", "activity"])
eco = eco[~eco["estate"].astype(str).str.contains("备注|来源|统计|查询|注：|符号|格式|标记", na=True)]
eco["count"] = pd.to_numeric(eco["count"], errors="coerce")
eco = eco.dropna(subset=["count"])

# 各身份汇总到屋苑
def eco_sum(keyword):
    return (eco[eco["activity"].astype(str).str.contains(keyword)]
            .groupby("estate")["count"].sum())

# 排除合计行（小计），只保留明细身份行
eco_detail = eco[~eco["activity"].astype(str).str.contains("小计|合计|总计", na=False)]
student_s  = eco_sum("学生").rename("student_count")
retiree_s  = eco_sum("退休").rename("retiree_count")
worker_s   = eco_sum("雇员|雇主|自营").rename("employed_count")
eco_total  = eco_detail.groupby("estate")["count"].sum().rename("eco_total")

eco_df = pd.concat([student_s, retiree_s, worker_s, eco_total], axis=1).reset_index()
# 用 eco_total（屋苑总人口，来自经济活动表）做分母
eco_df["student_ratio"]  = eco_df["student_count"]  / eco_df["eco_total"]
eco_df["retiree_ratio"]  = eco_df["retiree_count"]  / eco_df["eco_total"]
eco_df["employed_ratio"] = eco_df["employed_count"] / eco_df["eco_total"]
print(f"  {len(eco_df)} estates with economic activity data")
print(f"  student_ratio:  {eco_df['student_ratio'].mean():.3f}")
print(f"  retiree_ratio:  {eco_df['retiree_ratio'].mean():.3f}")
print(f"  employed_ratio: {eco_df['employed_ratio'].mean():.3f}")

# ── 5. 合并宽表 ──────────────────────────────────────────────────────────────
print("Merging...", flush=True)
df = (coords
      .merge(pop,    on="estate", how="left")
      .merge(inc,    on="estate", how="left")
      .merge(age_df, on="estate", how="left")
      .merge(eco_df[["estate","student_ratio","retiree_ratio","employed_ratio"]],
             on="estate", how="left"))

print(f"\n宽表概况:")
print(f"  总行数: {len(df)}")
print(f"  有人口数据: {df['population_total'].notna().sum()}")
print(f"  有收入数据: {df['median_income'].notna().sum()}")
print(f"  有老龄比例: {df['elderly_ratio'].notna().sum()}")
print(f"\n宽表概况:")
print(f"  总行数: {len(df)}")
for col in ["population_total","working_pop","median_income","elderly_ratio",
            "student_ratio","retiree_ratio","employed_ratio"]:
    print(f"  有 {col}: {df[col].notna().sum()}")
missing = df[df["population_total"].isna()]["estate"].tolist()
print(f"\n缺失情况: {len(missing)} 个屋苑名匹配失败: {missing[:10]}")

# ── 6. 保存 ─────────────────────────────────────────────────────────────────
out_path = f"{OUT}/estate_features.parquet"
df.to_parquet(out_path, index=False)
print(f"\nSaved: {out_path}")
print(df[["estate","lat","lon","population_total","median_income","elderly_ratio"]].head(10).to_string())
