"""
脚本_39_网格回归v2.py
研究单元：500m×500m 网格
研究问题：哪些网格在台风期间整体变快/变慢？由什么决定？

V1: 简单均值聚合（基准，与旧脚本_30对比）
V2: 路段长度加权聚合 + 完整变量集 + 信号控制（主回归）

过滤：Ragasa S3+，网格内 ≥3 条路且总长度 ≥500m
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"

# ── 1. 载入数据 ───────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
rt = pd.read_parquet(f"{DATA}/regression_table.parquet")

# Ragasa S3+
rag = rt[(rt["typhoon"] == "Ragasa") & (rt["signal_level"] >= 3)].copy()
print(f"  Ragasa S3+: {len(rag):,} rows, {rag['road_id'].nunique():,} roads")

# 500m 网格坐标（从 grid_regression_data 取 grid_id 映射）
grid_ref = pd.read_parquet(f"{DATA}/grid_regression_data.parquet")[
    ["grid_id", "grid_cx", "grid_cy"]
].drop_duplicates("grid_id")

# 把 road → grid 的映射从旧数据里拿（或重建）
# 直接用 grid_regression_data 里已有的 grid_id 作为索引
old_grid = pd.read_parquet(f"{DATA}/grid_regression_data.parquet")

# 从 regression_table 提取每条路的 grid_id（通过 road 坐标分配）
# 方案：把 road 坐标按 500m 格划分
# grid_cx / grid_cy 是格中心，用 floor 对齐

# 先从旧 grid_regression_data 还原 road → grid 映射
# 旧数据按 time_group 分了，用 MIDDAY 取一份做映射基准
old_midday = old_grid[old_grid["time_group"] == "MIDDAY"][["grid_id","grid_cx","grid_cy"]]

# 实际上 regression_table 没有 grid_id 列，需要重新分配
# 用 road 的 ep_key 中心点做网格分配
import ast

rr = pd.read_parquet(f"{DATA}/road_registry.parquet").drop_duplicates("road_id")
def parse_center(s):
    try:
        (lon1,lat1),(lon2,lat2) = ast.literal_eval(s)
        return (lon1+lon2)/2, (lat1+lat2)/2
    except:
        return None, None
centers = rr["ep_key"].apply(parse_center)
rr["cx"] = centers.apply(lambda x: x[0])
rr["cy"] = centers.apply(lambda x: x[1])
road_centers = rr[["road_id","cx","cy"]].dropna()

# 500m 格 (lat: 1° ≈ 111km → 500m ≈ 0.0045°; lon at 22.3°N: ≈ 0.00493°)
CELL_LAT = 500 / 111000
CELL_LON = 500 / (111000 * np.cos(np.radians(22.3)))

road_centers["grid_lat"] = (road_centers["cy"] // CELL_LAT) * CELL_LAT + CELL_LAT / 2
road_centers["grid_lon"] = (road_centers["cx"] // CELL_LON) * CELL_LON + CELL_LON / 2
road_centers["grid_id"]  = (road_centers["grid_lat"].round(6).astype(str) + "_" +
                             road_centers["grid_lon"].round(6).astype(str))

# 合并 road → grid_id 到 regression_table
rag = rag.merge(road_centers[["road_id","grid_id","grid_lat","grid_lon"]],
                on="road_id", how="left")
rag = rag.dropna(subset=["grid_id"])
print(f"  Matched to grid: {len(rag):,} rows, {rag['grid_id'].nunique():,} grids")

# ── 2. 变量准备 ───────────────────────────────────────────────────────────────
POI_CATS = ["work","education","retail","food_drink","recreation",
            "medical","transport","tourism","finance","civic"]
for cat in POI_CATS:
    rag[f"log_{cat}"] = np.log1p(rag[f"{cat}_density"].fillna(0))

rag["log_pop_density"] = np.log1p(rag["population_density_500m"])
rag["log_income"]      = np.log(rag["median_income_500m"].clip(lower=1))
rag["log_dist_coast"]  = np.log1p(rag["dist_to_coast_m"])

# 信号哑变量（S3为参照）
rag["S8"]  = (rag["signal_group"] == "S8").astype(float)
rag["S10"] = (rag["signal_group"] == "S10").astype(float)

# 年龄（删 25-44 参照）
AGE_VARS = ["ratio_age_0_14_500m","ratio_age_15_24_500m",
            "ratio_age_45_64_500m","ratio_age_65plus_500m"]

# 经济活动（删 其他职业 参照）
ECO_VARS = ["ratio_雇员_500m","ratio_退休人士_500m","ratio_学生_500m",
            "ratio_料理家务者_500m","ratio_自营作业者_500m","ratio_雇主_500m"]

# ── 3. 聚合函数 ───────────────────────────────────────────────────────────────
POI_COLS    = [f"log_{c}" for c in POI_CATS]
STRUCT_COLS = ["intersection_degree","dist_to_coast_m","log_dist_coast"]
DEMO_COLS   = (["log_pop_density","log_income","working_pop_ratio_500m"] +
               AGE_VARS + ECO_VARS)
SIGNAL_COLS = ["S8","S10"]

def make_grid(sub, weighted=False, min_roads=3, min_total_len=500):
    """
    weighted=False → 简单均值 (V1)
    weighted=True  → 路段长度加权均值 (V2)
    """
    L = "road_length_m"

    def wavg(grp, col):
        if weighted:
            w = grp[L]
            return (grp[col] * w).sum() / w.sum() if w.sum() > 0 else np.nan
        else:
            return grp[col].mean()

    records = []
    for gid, grp in sub.groupby("grid_id"):
        n_roads   = grp["road_id"].nunique()
        total_len = grp[L].sum()
        if n_roads < min_roads or total_len < min_total_len:
            continue

        row = {
            "grid_id":         gid,
            "grid_lat":        grp["grid_lat"].iloc[0],
            "grid_lon":        grp["grid_lon"].iloc[0],
            "n_roads":         n_roads,
            "total_length_m":  total_len,
            "mean_deviation":  wavg(grp, "mean_deviation"),
            "pct_faster":      (grp["mean_deviation"] > 0).mean(),  # 路段中变快比例（未加权，用于描述）
        }

        for col in POI_COLS + STRUCT_COLS + DEMO_COLS + SIGNAL_COLS:
            if col in grp.columns:
                row[col] = wavg(grp, col)

        # road_broad 取最长路段的类型
        road_type = grp.loc[grp[L].idxmax(), "road_broad"] if L in grp.columns else "other"
        row["dominant_road_type"] = road_type

        records.append(row)

    g = pd.DataFrame(records)
    # log 转换（在聚合后对聚合值再取 log 效果更稳定）
    g["log_intersection"] = np.log1p(g["intersection_degree"])
    g["log_dist_coast"]   = np.log1p(g["dist_to_coast_m"])
    return g

TIME_GROUPS = ["NIGHT","AM_PEAK","MIDDAY","PM_PEAK","EVENING"]

# ── 4. 回归公式 ───────────────────────────────────────────────────────────────
poi_terms   = " + ".join(POI_COLS)
struct_terms = "log_intersection + log_dist_coast"
demo_terms  = ("log_pop_density + log_income + working_pop_ratio_500m + " +
               " + ".join(AGE_VARS) + " + " +
               " + ".join([f"Q('{v}')" for v in ECO_VARS]))
signal_terms = "S8 + S10"

FORMULA = f"mean_deviation ~ {poi_terms} + {struct_terms} + {demo_terms} + {signal_terms}"

# ── 5. 跑两个版本 ─────────────────────────────────────────────────────────────
all_results = {}

for version, weighted in [("V1_simple", False), ("V2_weighted", True)]:
    print(f"\n{'='*70}")
    print(f"  {version}  ({'长度加权' if weighted else '简单均值'})")
    print(f"{'='*70}")

    version_results = {}
    version_data    = {}

    for tg in TIME_GROUPS:
        sub = rag[rag["time_group"] == tg].copy()
        g = make_grid(sub, weighted=weighted)

        # 有完整人口数据的格子（listwise deletion）
        demo_need = (["log_pop_density","log_income","working_pop_ratio_500m"] +
                     AGE_VARS + ECO_VARS)
        demo_need_avail = [c for c in demo_need if c in g.columns]
        g2 = g.dropna(subset=demo_need_avail).copy()

        print(f"\n  {tg}: 总格数={len(g):,}  完整案例={len(g2):,}  "
              f"变快格={( g2['mean_deviation']>0).mean():.1%}  "
              f"Y均值={g2['mean_deviation'].mean():+.4f}")

        if len(g2) < 50:
            print(f"    样本太少，跳过")
            continue

        # 处理缺失的 eco 列
        for v in ECO_VARS:
            if v not in g2.columns:
                g2[v] = 0.0

        try:
            formula_use = FORMULA
            # 检查哪些列存在
            avail_eco = [v for v in ECO_VARS if v in g2.columns and g2[v].notna().any()]
            eco_str = " + ".join([f"Q('{v}')" for v in avail_eco])
            formula_use = (f"mean_deviation ~ {poi_terms} + {struct_terms} + "
                           f"log_pop_density + log_income + working_pop_ratio_500m + "
                           f"{' + '.join(AGE_VARS)} + {eco_str} + {signal_terms}")

            m = smf.ols(formula_use, data=g2).fit()
            version_results[tg] = m
            version_data[tg]    = g2
            print(f"    R²={m.rsquared:.4f}  adj-R²={m.rsquared_adj:.4f}  n={int(m.nobs):,}")
        except Exception as e:
            print(f"    回归失败: {e}")

    all_results[version] = version_results

    # 保存数据
    combined = pd.concat([d.assign(time_group=tg)
                           for tg, d in version_data.items()], ignore_index=True)
    combined.to_parquet(f"{DATA}/grid_v2_{version}.parquet", index=False)

# ── 6. R² 对比汇总 ───────────────────────────────────────────────────────────
print("\n\n" + "="*60)
print("R² COMPARISON: V1 (simple mean) vs V2 (length-weighted)")
print("="*60)
print(f"{'Time Group':<12} {'V1_R2':>8} {'V1_adjR2':>10} {'V2_R2':>8} {'V2_adjR2':>10}")
for tg in TIME_GROUPS:
    m1 = all_results["V1_simple"].get(tg)
    m2 = all_results["V2_weighted"].get(tg)
    r1 = f"{m1.rsquared:.4f}" if m1 else "  N/A"
    a1 = f"{m1.rsquared_adj:.4f}" if m1 else "  N/A"
    r2 = f"{m2.rsquared:.4f}" if m2 else "  N/A"
    a2 = f"{m2.rsquared_adj:.4f}" if m2 else "  N/A"
    print(f"{tg:<12} {r1:>8} {a1:>10} {r2:>8} {a2:>10}")

# ── 7. V2 系数汇总 ────────────────────────────────────────────────────────────
print("\n\nCOEFFICIENT SUMMARY — V2 (length-weighted)")
print("="*80)
rows = []
for tg, m in all_results["V2_weighted"].items():
    for param in m.params.index:
        if param == "Intercept": continue
        rows.append({
            "time_group": tg, "variable": param,
            "coef": m.params[param], "pval": m.pvalues[param],
            "sig": ("***" if m.pvalues[param]<0.001 else
                    "**"  if m.pvalues[param]<0.01  else
                    "*"   if m.pvalues[param]<0.05  else
                    "."   if m.pvalues[param]<0.1   else "")
        })
coef_df = pd.DataFrame(rows)

pivot_c = coef_df.pivot_table(index="variable", columns="time_group",
                               values="coef", aggfunc="first")
pivot_s = coef_df.pivot_table(index="variable", columns="time_group",
                               values="sig", aggfunc="first")
summary = pd.DataFrame(index=pivot_c.index)
for tg in TIME_GROUPS:
    if tg in pivot_c.columns:
        summary[tg] = (pivot_c[tg].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "") +
                       pivot_s[tg].fillna(""))
print(summary.to_string())

# ── 8. 保存结果 ───────────────────────────────────────────────────────────────
out_rows = []
for version, vres in all_results.items():
    for tg, m in vres.items():
        for param in m.params.index:
            out_rows.append({
                "version": version, "time_group": tg, "variable": param,
                "coef": m.params[param], "se": m.bse[param],
                "tstat": m.tvalues[param], "pval": m.pvalues[param],
                "sig": ("***" if m.pvalues[param]<0.001 else
                        "**"  if m.pvalues[param]<0.01  else
                        "*"   if m.pvalues[param]<0.05  else
                        "."   if m.pvalues[param]<0.1   else ""),
                "R2": m.rsquared, "adj_R2": m.rsquared_adj, "n": int(m.nobs),
            })

pd.DataFrame(out_rows).to_excel(f"{DATA}/回归结果_网格v2.xlsx", index=False)
print(f"\n结果已保存: {DATA}/回归结果_网格v2.xlsx")
