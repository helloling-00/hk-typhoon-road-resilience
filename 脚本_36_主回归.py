"""
脚本_36_主回归.py
物理道路级OLS回归：台风Ragasa S3+期间速度偏差的决定因素
两个规格：
  Model 1 — 路结构 + POI（全样本）
  Model 2 — 路结构 + POI + 人口/年龄/经济活动（完整案例）
分5个time_group分别跑，聚类SE by road_id
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
OUT  = "/Users/helloling/workspace/thesis/data"

# ── 1. 载入数据 ───────────────────────────────────────────────────────────────
print("Loading regression table...", flush=True)
rt = pd.read_parquet(f"{DATA}/regression_table.parquet")

# 过滤：Ragasa S3+，路长>=200m
df = rt[
    (rt["typhoon"] == "Ragasa") &
    (rt["signal_level"] >= 3) &
    (rt["road_length_m"] >= 200)
].copy()
print(f"  过滤后行数: {len(df):,}  road_id数: {df['road_id'].nunique():,}")

# ── 2. 构建变量 ───────────────────────────────────────────────────────────────
POI_CATS = ["work","education","retail","food_drink","recreation",
            "medical","transport","tourism","finance","civic"]
for cat in POI_CATS:
    col = f"{cat}_density"
    df[f"log_{cat}"] = np.log1p(df[col].fillna(0))

df["log_road_length"]   = np.log(df["road_length_m"])
df["log_pop_density"]   = np.log1p(df["population_density_500m"])
df["log_income"]        = np.log(df["median_income_500m"].clip(lower=1))
df["log_dist_coast"]    = np.log1p(df["dist_to_coast_m"])

# 信号虚拟变量（S3为参照）
df["S8"]  = (df["signal_group"] == "S8").astype(int)
df["S10"] = (df["signal_group"] == "S10").astype(int)

# road_broad 哑变量（other为参照）
df["road_broad"] = df["road_broad"].fillna("other")
for rb in ["motorway","trunk","primary","secondary","tertiary"]:
    df[f"rb_{rb}"] = (df["road_broad"] == rb).astype(int)

# 年龄比例（删掉 ratio_age_25_44 作参照，避免完全共线）
age_vars = ["ratio_age_0_14_500m","ratio_age_15_24_500m",
            "ratio_age_45_64_500m","ratio_age_65plus_500m"]
for v in age_vars:
    df[v] = df[v].astype(float)

# 经济活动（删掉 ratio_其他职业 作参照）
eco_vars = ["ratio_雇员_500m","ratio_退休人士_500m","ratio_学生_500m",
            "ratio_料理家务者_500m","ratio_自营作业者_500m","ratio_雇主_500m",
            "ratio_无酬家庭从业员_500m","ratio_无酬照顾者_500m"]
for v in eco_vars:
    df[v] = df[v].astype(float)

# ── 3. 公式 ───────────────────────────────────────────────────────────────────
poi_terms  = " + ".join([f"log_{c}" for c in POI_CATS])
road_terms = "log_road_length + intersection_degree + log_dist_coast"
road_type  = "rb_motorway + rb_trunk + rb_primary + rb_secondary + rb_tertiary"
signal_fe  = "S8 + S10"
incident   = "incident_count_500m + severe_incident_500m + closure_nearby_500m"

age_terms = " + ".join(age_vars)
eco_terms = " + ".join([f"Q('{v}')" for v in eco_vars])
FORMULA_M2 = (
    f"mean_deviation ~ {poi_terms} + {road_terms} + {road_type}"
    f" + log_pop_density + log_income + working_pop_ratio_500m"
    f" + {age_terms} + {eco_terms}"
    f" + {signal_fe} + {incident}"
)

TIME_GROUPS = ["NIGHT","AM_PEAK","MIDDAY","PM_PEAK","EVENING"]

# ── 4. 跑回归（只跑全变量模型，完整案例） ────────────────────────────────────
results = {}
demo_cols = ["population_density_500m","median_income_500m"] + age_vars + eco_vars

for tg in TIME_GROUPS:
    sub = df[df["time_group"] == tg].copy()
    sub2 = sub.dropna(subset=demo_cols)
    print(f"\n{'='*60}")
    print(f"TIME GROUP: {tg}  总样本={len(sub):,}  完整案例={len(sub2):,} ({len(sub2)/len(sub)*100:.1f}%)  roads={sub2['road_id'].nunique():,}")

    if len(sub2) > 200:
        try:
            m = smf.ols(FORMULA_M2, data=sub2).fit(
                cov_type="cluster", cov_kwds={"groups": sub2["road_id"]}
            )
            results[tg] = m
            print(f"  R²={m.rsquared:.4f}  adj-R²={m.rsquared_adj:.4f}  n={int(m.nobs):,}")
        except Exception as e:
            print(f"  Failed: {e}")

# ── 6. 汇总系数表 ─────────────────────────────────────────────────────────────
print("\n\n" + "="*80)
print("COEFFICIENT SUMMARY")
print("="*80)

rows = []
for tg, m in results.items():
    for param in m.params.index:
        if param == "Intercept": continue
        rows.append({
            "time_group": tg,
            "variable": param,
            "coef": m.params[param],
            "se": m.bse[param],
            "pval": m.pvalues[param],
            "sig": "***" if m.pvalues[param]<0.001 else
                   "**"  if m.pvalues[param]<0.01  else
                   "*"   if m.pvalues[param]<0.05  else
                   "."   if m.pvalues[param]<0.1   else ""
        })
coef_df = pd.DataFrame(rows)

pivot_coef = coef_df.pivot_table(index="variable", columns="time_group",
                                  values="coef", aggfunc="first")
pivot_sig  = coef_df.pivot_table(index="variable", columns="time_group",
                                  values="sig", aggfunc="first")
summary = pd.DataFrame(index=pivot_coef.index)
for tg in TIME_GROUPS:
    if tg in pivot_coef.columns:
        summary[tg] = (
            pivot_coef[tg].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "") +
            pivot_sig[tg].fillna("")
        )
print(summary.to_string())

# R² 汇总
print("\n\nR² SUMMARY")
print(f"{'Time Group':<12} {'R2':>8} {'adj-R2':>10} {'n':>8} {'roads':>8}")
for tg in TIME_GROUPS:
    m = results.get(tg)
    if m:
        n_roads = coef_df[coef_df["time_group"]==tg]["time_group"].count()
        print(f"{tg:<12} {m.rsquared:>8.4f} {m.rsquared_adj:>10.4f} {int(m.nobs):>8,} {m.model.data.frame['road_id'].nunique():>8,}")

# ── 7. 保存Excel ──────────────────────────────────────────────────────────────
out_rows = []
for tg, m in results.items():
    for param in m.params.index:
        out_rows.append({
            "time_group": tg,
            "variable": param,
            "coef": m.params[param],
            "se": m.bse[param],
            "tstat": m.tvalues[param],
            "pval": m.pvalues[param],
            "sig": "***" if m.pvalues[param]<0.001 else
                   "**"  if m.pvalues[param]<0.01  else
                   "*"   if m.pvalues[param]<0.05  else
                   "."   if m.pvalues[param]<0.1   else "",
            "R2": m.rsquared,
            "adj_R2": m.rsquared_adj,
            "n": int(m.nobs),
            "n_roads": m.model.data.frame["road_id"].nunique()
        })

out_df = pd.DataFrame(out_rows)
out_df.to_excel(f"{OUT}/回归结果_主回归.xlsx", index=False)
print(f"\n结果已保存: {OUT}/回归结果_主回归.xlsx")
