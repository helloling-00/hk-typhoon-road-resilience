"""
回归分析 v4：台风期间道路变快/变慢的特征（完整版）
Y = mean_deviation（台风速度 − 同槽正常基线）
5 时段分别跑 OLS，聚类 SE：road_id

X 变量：
  土地利用密度（10类，log1p，个/km²）
  人口结构比例（employed/student/retiree/elderly ratio）
  人口密度 + 收入（log）
  道路结构（路口度、海岸距离）
  信号强度 + 台风 FE + 道路等级 FE（控制变量）
"""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
import warnings; warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"

print("Loading...", flush=True)
reg = pd.read_parquet(f"{DATA}/regression_table.parquet")
reg = reg[reg["n_slots"] >= 2].copy()
print(f"  {len(reg):,} 行  {reg['road_id'].nunique():,} 条路")

# ── 特征工程 ─────────────────────────────────────────────────────────────────
LU_CATS = ["work","education","retail","food_drink","recreation",
           "medical","transport","tourism","finance","civic"]

for cat in LU_CATS:
    reg[f"log_{cat}"] = np.log1p(reg[f"{cat}_density"].fillna(0))

reg["log_road_length"]  = np.log1p(reg["road_length_m"].fillna(0))
reg["log_pop_density"]  = np.log1p(reg["population_density_500m"].fillna(0))
reg["log_income"]       = np.log1p(reg["median_income_500m"].fillna(0))
reg["log_incidents"]    = np.log1p(reg["incident_count_500m"].fillna(0))
reg["log_dist_coast"]   = np.log1p(reg["dist_to_coast_m"].fillna(0))
reg["log_intersection"] = np.log1p(reg["intersection_degree"].fillna(0))

# 人口比例（缺失=郊区路，填 0）
for col in ["employed_ratio_500m","student_ratio_500m",
            "retiree_ratio_500m","elderly_ratio_500m"]:
    reg[col] = reg[col].fillna(0)

# 信号 + 台风
reg["sig_num"]    = reg["signal_level"].replace({9: 10})
reg["is_ragasa"]  = (reg["typhoon"] == "Ragasa").astype(float)
reg["is_matmo"]   = (reg["typhoon"] == "Matmo").astype(float)

# 道路等级基准：tertiary
present_cats = [c for c in ["tertiary","secondary","primary","trunk","motorway","other"]
                if c in reg["road_broad"].values]
reg["road_broad"] = pd.Categorical(reg["road_broad"], categories=present_cats)

# ── 描述统计 ─────────────────────────────────────────────────────────────────
print("\n各时段样本量与 Y 均值：")
print(reg.groupby("time_group")["mean_deviation"].agg(
    N="count", mean="mean", pct_pos=lambda x: (x>0).mean()
).round(3).to_string())

# ── 回归公式 ─────────────────────────────────────────────────────────────────
LU_TERMS   = " + ".join(f"log_{c}" for c in LU_CATS)
DEMO_TERMS = ("log_pop_density + log_income + "
              "employed_ratio_500m + student_ratio_500m + "
              "retiree_ratio_500m + elderly_ratio_500m")
STRUCT_TERMS = "log_road_length + log_intersection + log_dist_coast + log_incidents"
CTRL_TERMS  = ("sig_num + is_ragasa + is_matmo + "
               "C(road_broad, Treatment('tertiary'))")

FORMULA = (f"mean_deviation ~ "
           f"{LU_TERMS} + {DEMO_TERMS} + {STRUCT_TERMS} + {CTRL_TERMS}")

# ── 回归函数 ──────────────────────────────────────────────────────────────────
def run(df, label):
    if len(df) < 200:
        print(f"\n  {label}: 样本过少({len(df)})，跳过")
        return None, None
    try:
        res = smf.ols(FORMULA, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["road_id"].values})
    except Exception as e:
        print(f"\n  {label}: 失败 ({e})")
        return None, None

    tbl = pd.DataFrame({
        "coef": res.params, "se": res.bse,
        "t": res.tvalues, "p": res.pvalues,
    })
    tbl["sig"] = tbl["p"].apply(
        lambda p: "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ".")

    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"  N={len(df):,}  roads={df['road_id'].nunique():,}"
          f"  R²={res.rsquared:.4f}  Adj-R²={res.rsquared_adj:.4f}")
    print(f"  Y均值={df['mean_deviation'].mean():+.4f}"
          f"  变快占比={(df['mean_deviation']>0).mean():.1%}")
    print(f"{'='*72}")
    print(tbl[["coef","se","t","p","sig"]].round(4).to_string())
    return res, tbl

PERIODS = [
    ("NIGHT",   "凌晨 NIGHT (00:00–07:00)"),
    ("AM_PEAK", "早高峰 AM_PEAK (07:00–09:30)"),
    ("MIDDAY",  "日间 MIDDAY (09:30–17:00)"),
    ("PM_PEAK", "晚高峰 PM_PEAK (17:00–19:30)"),
    ("EVENING", "夜间 EVENING (19:30–24:00)"),
]

results = {}
for tg, label in PERIODS:
    sub = reg[reg["time_group"] == tg]
    res, tbl = run(sub, label)
    if tbl is not None:
        results[tg] = (res, tbl)

# ── 汇总表 ────────────────────────────────────────────────────────────────────
KEY_VARS = (
    [f"log_{c}" for c in LU_CATS] +
    ["log_pop_density","log_income",
     "employed_ratio_500m","student_ratio_500m",
     "retiree_ratio_500m","elderly_ratio_500m",
     "log_intersection","log_dist_coast","log_incidents"]
)

print("\n\n" + "═"*90)
print("  各变量系数汇总（+正=台风期变快，-负=变慢）")
print("═"*90)
rows = []
for v in KEY_VARS:
    row = {"变量": v}
    for tg, _ in PERIODS:
        if tg in results:
            _, tbl = results[tg]
            if v in tbl.index:
                row[tg] = f"{tbl.loc[v,'coef']:+.4f}{tbl.loc[v,'sig']}"
            else:
                row[tg] = "—"
        else:
            row[tg] = "—"
    rows.append(row)
sumtbl = pd.DataFrame(rows).set_index("变量")
print(sumtbl.to_string())

# ── 保存 ─────────────────────────────────────────────────────────────────────
try:
    with pd.ExcelWriter(f"{DATA}/regression_results.xlsx") as w:
        sumtbl.to_excel(w, sheet_name="汇总")
        for tg, _ in PERIODS:
            if tg in results:
                results[tg][1].round(4).to_excel(w, sheet_name=tg)
    print(f"\nSaved: {DATA}/regression_results.xlsx")
except Exception as e:
    for tg, _ in PERIODS:
        if tg in results:
            results[tg][1].round(4).to_csv(f"{DATA}/reg_{tg}.csv")
    print(f"CSV saved ({e})")
