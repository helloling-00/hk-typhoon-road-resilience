"""
物理道路回归：
1. 过滤 < 200m 路段
2. POI 按出行必要性分 3 组（刚性/弹性/中转）
3. 路段级 OLS，cluster SE by road_id
"""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
import warnings; warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"

# ── 1. 加载 & 过滤 ────────────────────────────────────────────────────────────
print("Loading...", flush=True)
reg = pd.read_parquet(f"{DATA}/regression_table.parquet")
rag = reg[(reg["typhoon"] == "Ragasa") &
          (reg["signal_level"] >= 3) &
          (reg["road_length_m"] >= 200)].copy()  # 过滤短路段
print(f"  {len(rag):,} rows  {rag['road_id'].nunique():,} roads")
print(f"  road length: min={rag['road_length_m'].min():.0f}m  "
      f"median={rag['road_length_m'].median():.0f}m  "
      f"max={rag['road_length_m'].max():.0f}m")
print(f"  过滤前 15,752 roads → 过滤后 {rag['road_id'].nunique():,} roads")

# ── 2. POI 出行必要性分组 ────────────────────────────────────────────────────
# 刚性（necessary）— 台风期间仍可能出行
RIGID = ["medical", "civic"]
# 弹性（discretionary）— 台风期间可取消出行
ELASTIC = ["work", "education", "retail", "food_drink", "recreation", "tourism", "finance"]
# 中转（transit）— 枢纽，公交替代效应
TRANSIT = ["transport"]

# 合并为一个变量（log 密度之和 = log 乘积，就是总 POI 密度）
for group_name, members in [("rigid", RIGID), ("elastic", ELASTIC), ("transit", TRANSIT)]:
    # 原始密度之和，再 log1p
    rag[f"{group_name}_density"] = 0.0
    for cat in members:
        rag[f"{group_name}_density"] += rag[f"{cat}_density"].fillna(0)
    rag[f"log_{group_name}"] = np.log1p(rag[f"{group_name}_density"])

print(f"\n  Rigid POI (medical+civic):     median density={rag['rigid_density'].median():.1f}/km²")
print(f"  Elastic POI (work+edu+...):     median density={rag['elastic_density'].median():.1f}/km²")
print(f"  Transit POI (transport):        median density={rag['transit_density'].median():.1f}/km²")

# ── 3. 其余特征 ────────────────────────────────────────────────────────────────
rag["log_pop_density"] = np.log1p(rag["population_density_500m"].fillna(0))
rag["log_income"]      = np.log1p(rag["median_income_500m"].fillna(0))
rag["log_road_length"]  = np.log1p(rag["road_length_m"])
rag["log_intersection"] = np.log1p(rag["intersection_degree"].fillna(0))
rag["log_dist_coast"]   = np.log1p(rag["dist_to_coast_m"].fillna(0))
rag["log_incidents"]    = np.log1p(rag["incident_count_500m"].fillna(0))

for col in ["employed_ratio_500m","student_ratio_500m",
            "retiree_ratio_500m","elderly_ratio_500m"]:
    rag[col] = rag[col].fillna(0)

rag["sig_num"]   = rag["signal_level"].replace({9: 10})
rag["is_sig10"]  = (rag["signal_level"] == 10).astype(float)

# ── 4. 回归公式 ────────────────────────────────────────────────────────────────
# 核心公式：3 组 POI + 人口 + 道路结构 + 控制
FORMULA = ("mean_deviation ~ "
           "log_rigid + log_elastic + log_transit + "    # POI 3 组
           "log_pop_density + log_income + "
           "employed_ratio_500m + student_ratio_500m + "
           "retiree_ratio_500m + elderly_ratio_500m + "  # 人口
           "log_road_length + log_intersection + "
           "log_dist_coast + log_incidents + "            # 道路结构
           "sig_num + is_sig10")                          # 控制

# ── 5. 回归 ──────────────────────────────────────────────────────────────────
def run(df, label):
    if len(df) < 200:
        print(f"\n  {label}: too few ({len(df)}), skip")
        return None, None
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
        lambda p: "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "."
    )
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"  N={len(df):,}  roads={df['road_id'].nunique():,}  "
          f"R²={res.rsquared:.4f}  Adj-R²={res.rsquared_adj:.4f}")
    print(f"  Y mean={df['mean_deviation'].mean():+.4f}  "
          f"pct_pos={(df['mean_deviation']>0).mean():.1%}")
    print(f"{'='*72}")
    print(tbl[["coef","se","t","p","sig"]].round(4).to_string())
    return res, tbl

PERIODS = [
    ("NIGHT",   "NIGHT (00:00-07:00)"),
    ("AM_PEAK", "AM_PEAK (07:00-09:30)"),
    ("MIDDAY",  "MIDDAY (09:30-17:00)"),
    ("PM_PEAK", "PM_PEAK (17:00-19:30)"),
    ("EVENING", "EVENING (19:30-24:00)"),
]

print("\n" + "=" * 72)
print("  Physical road regression (len>=200m, POI grouped by necessity)")
print("=" * 72)

results = {}
for tg, label in PERIODS:
    sub = rag[rag["time_group"] == tg]
    res, tbl = run(sub, label)
    if tbl is not None:
        results[tg] = (res, tbl)

# ── 6. 汇总图 ──────────────────────────────────────────────────────────────────
KEY_VARS = [
    "log_rigid", "log_elastic", "log_transit",
    "log_pop_density", "log_income",
    "employed_ratio_500m", "student_ratio_500m",
    "retiree_ratio_500m", "elderly_ratio_500m",
    "log_road_length", "log_intersection", "log_dist_coast", "log_incidents",
]

print("\n\n" + "=" * 90)
print("  Coefficient Summary (len>=200m, POI necessity groups)")
print("  + = faster during typhoon  |  - = slower during typhoon")
print("=" * 90)
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

# ── 7. 对比之前的 R² ──────────────────────────────────────────────────────────
print("\n\n" + "=" * 72)
print("  R² Comparison Across Approaches")
print("=" * 72)
# 之前路段级（三台风，全部路段）
prev_road = {"AM_PEAK": 0.0385, "MIDDAY": 0.0342, "PM_PEAK": 0.0549, "NIGHT": 0.0099, "EVENING": 0.0183}
# 之前网格级 500m
prev_grid = {"AM_PEAK": 0.0464, "MIDDAY": 0.1062, "PM_PEAK": 0.1181, "NIGHT": 0.0481, "EVENING": 0.0898}

print(f"{'Period':<12} {'3-typhoon road':>15} {'500m grid':>12} {'len>=200 necessity':>18}")
print("-"*60)
for tg, _ in PERIODS:
    nw = results.get(tg)
    r2_new = f"{nw[0].rsquared:.4f}" if nw else "--"
    r2_old = f"{prev_road.get(tg,0):.4f}"
    r2_grd = f"{prev_grid.get(tg,0):.4f}"
    print(f"{tg:<12} {r2_old:>15} {r2_grd:>12} {r2_new:>18}")

# ── 保存 ─────────────────────────────────────────────────────────────────────
try:
    with pd.ExcelWriter(f"{DATA}/regression_physical_road.xlsx") as w:
        sumtbl.to_excel(w, sheet_name="Summary")
        for tg, _ in PERIODS:
            if tg in results:
                results[tg][1].round(4).to_excel(w, sheet_name=tg)
    print(f"\nSaved: data/regression_physical_road.xlsx")
except Exception as e:
    print(f"\nExcel save: {e}")

print("\nDone.")
