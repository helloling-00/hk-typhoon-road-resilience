"""
脚本_40_网格比例回归.py
Y = 长度加权的"明显变好路段"占比（deviation > 0.05）
研究问题：什么特征的网格在台风期有更大比例路段明显变顺？
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import ast
import warnings
warnings.filterwarnings("ignore")

DATA  = "/Users/helloling/workspace/thesis/data"
DELTA = 0.05   # "明显变好"门槛

# ── 1. 数据 ───────────────────────────────────────────────────────────────────
print("Loading...", flush=True)
rt = pd.read_parquet(f"{DATA}/regression_table.parquet")
rag = rt[(rt["typhoon"] == "Ragasa") & (rt["signal_level"] >= 3)].copy()

# 路段级标签
rag["clearly_better"] = (rag["mean_deviation"] >  DELTA).astype(float)
rag["clearly_worse"]  = (rag["mean_deviation"] < -DELTA).astype(float)

# road → grid 分配
rr = pd.read_parquet(f"{DATA}/road_registry.parquet").drop_duplicates("road_id")
def parse_center(s):
    try:
        (lon1,lat1),(lon2,lat2) = ast.literal_eval(s)
        return (lon1+lon2)/2, (lat1+lat2)/2
    except:
        return None, None
centers = rr["ep_key"].apply(parse_center)
rr = rr.copy()
rr["cx"] = centers.apply(lambda x: x[0])
rr["cy"] = centers.apply(lambda x: x[1])

CELL_LAT = 500 / 111000
CELL_LON = 500 / (111000 * np.cos(np.radians(22.3)))
rr["grid_lat"] = (rr["cy"] // CELL_LAT) * CELL_LAT + CELL_LAT / 2
rr["grid_lon"] = (rr["cx"] // CELL_LON) * CELL_LON + CELL_LON / 2
rr["grid_id"]  = (rr["grid_lat"].round(6).astype(str) + "_" +
                  rr["grid_lon"].round(6).astype(str))

rag = rag.merge(rr[["road_id","grid_id","grid_lat","grid_lon"]], on="road_id", how="left")
rag = rag.dropna(subset=["grid_id"])

# ── 2. 变量 ───────────────────────────────────────────────────────────────────
POI_CATS = ["work","education","retail","food_drink","recreation",
            "medical","transport","tourism","finance","civic"]
for cat in POI_CATS:
    rag[f"log_{cat}"] = np.log1p(rag[f"{cat}_density"].fillna(0))

rag["log_pop_density"] = np.log1p(rag["population_density_500m"])
rag["log_income"]      = np.log(rag["median_income_500m"].clip(lower=1))
rag["log_dist_coast"]  = np.log1p(rag["dist_to_coast_m"])
rag["S8"]  = (rag["signal_group"] == "S8").astype(float)
rag["S10"] = (rag["signal_group"] == "S10").astype(float)

AGE_VARS = ["ratio_age_0_14_500m","ratio_age_15_24_500m",
            "ratio_age_45_64_500m","ratio_age_65plus_500m"]
ECO_VARS = ["ratio_雇员_500m","ratio_退休人士_500m","ratio_学生_500m",
            "ratio_料理家务者_500m","ratio_自营作业者_500m","ratio_雇主_500m"]

CONT_COLS = ([f"log_{c}" for c in POI_CATS] +
             ["log_pop_density","log_income","working_pop_ratio_500m",
              "intersection_degree","dist_to_coast_m",
              "S8","S10"] + AGE_VARS + ECO_VARS)

# ── 3. 网格聚合 ───────────────────────────────────────────────────────────────
def make_grid(sub, min_roads=3, min_len=500):
    L = "road_length_m"
    records = []
    for gid, g in sub.groupby("grid_id"):
        n_roads   = g["road_id"].nunique()
        total_len = g[L].sum()
        if n_roads < min_roads or total_len < min_len:
            continue
        w = g[L]
        W = w.sum()

        def wavg(col):
            if col not in g.columns: return np.nan
            return (g[col] * w).sum() / W

        row = {
            "grid_id":       gid,
            "grid_lat":      g["grid_lat"].iloc[0],
            "grid_lon":      g["grid_lon"].iloc[0],
            "n_roads":       n_roads,
            "total_length_m": total_len,
            # Y 变量
            "pct_better":    wavg("clearly_better"),   # 主Y
            "pct_worse":     wavg("clearly_worse"),
            "mean_dev":      wavg("mean_deviation"),   # 对照
        }
        for col in CONT_COLS:
            row[col] = wavg(col)

        # 路网密度：格内总路长 / 格面积
        cell_area_km2 = (CELL_LAT * 111) * (CELL_LON * 111 * np.cos(np.radians(22.3)))
        row["road_density"] = total_len / 1000 / cell_area_km2  # km/km²

        records.append(row)
    gdf = pd.DataFrame(records)
    gdf["log_intersection"] = np.log1p(gdf["intersection_degree"])
    gdf["log_dist_coast2"]  = np.log1p(gdf["dist_to_coast_m"])
    gdf["log_road_density"] = np.log1p(gdf["road_density"])
    return gdf

# ── 4. 回归公式 ───────────────────────────────────────────────────────────────
POI_TERMS    = " + ".join(f"log_{c}" for c in POI_CATS)
STRUCT_TERMS = "log_intersection + log_dist_coast2 + log_road_density"
DEMO_TERMS   = ("log_pop_density + log_income + working_pop_ratio_500m + " +
                " + ".join(AGE_VARS) + " + " +
                " + ".join(f"Q('{v}')" for v in ECO_VARS))
SIG_TERMS    = "S8 + S10"

FORMULA = f"pct_better ~ {POI_TERMS} + {STRUCT_TERMS} + {DEMO_TERMS} + {SIG_TERMS}"

TIME_GROUPS = ["NIGHT","AM_PEAK","MIDDAY","PM_PEAK","EVENING"]

# ── 5. 跑回归 ─────────────────────────────────────────────────────────────────
results = {}
grid_data = {}

print("\n" + "="*70)
print(f"  网格比例回归  Y = 长度加权明显变好比例（deviation > {DELTA}）")
print("="*70)

for tg in TIME_GROUPS:
    sub = rag[rag["time_group"] == tg].copy()
    g   = make_grid(sub)

    # listwise deletion on demo vars
    need = (["log_pop_density","log_income","working_pop_ratio_500m"] +
            AGE_VARS + ECO_VARS)
    avail = [c for c in need if c in g.columns]
    g2 = g.dropna(subset=avail).copy()
    for v in ECO_VARS:
        if v not in g2.columns: g2[v] = 0.0

    print(f"\n  {tg}: 总格={len(g):,}  完整案例={len(g2):,}  "
          f"Y均值={g2['pct_better'].mean():.3f}  "
          f"Y>0.5格占比={(g2['pct_better']>0.5).mean():.1%}")

    if len(g2) < 50:
        print("    样本不足，跳过")
        continue

    try:
        m = smf.ols(FORMULA, data=g2).fit()
        results[tg] = m
        grid_data[tg] = g2
        print(f"    R²={m.rsquared:.4f}  adj-R²={m.rsquared_adj:.4f}  n={int(m.nobs):,}")
    except Exception as e:
        print(f"    失败: {e}")

# ── 6. 系数汇总 ───────────────────────────────────────────────────────────────
print("\n\n" + "="*80)
print("COEFFICIENT SUMMARY — Y = pct_clearly_better (deviation > 0.05)")
print("="*80)

rows = []
for tg, m in results.items():
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

# R² 汇总
print("\n\nR² SUMMARY")
print(f"{'Time Group':<12} {'R2':>8} {'adj-R2':>10} {'n_grids':>8} "
      f"{'Y_mean':>8} {'pct_Y>0.5':>10}")
for tg in TIME_GROUPS:
    m  = results.get(tg)
    g2 = grid_data.get(tg)
    if m and g2 is not None:
        print(f"{tg:<12} {m.rsquared:>8.4f} {m.rsquared_adj:>10.4f} "
              f"{int(m.nobs):>8,} {g2['pct_better'].mean():>8.3f} "
              f"{(g2['pct_better']>0.5).mean():>10.1%}")

# ── 7. 保存 ───────────────────────────────────────────────────────────────────
out_rows = []
for tg, m in results.items():
    for param in m.params.index:
        out_rows.append({
            "time_group": tg, "variable": param,
            "coef": m.params[param], "se": m.bse[param],
            "tstat": m.tvalues[param], "pval": m.pvalues[param],
            "sig": ("***" if m.pvalues[param]<0.001 else
                    "**"  if m.pvalues[param]<0.01  else
                    "*"   if m.pvalues[param]<0.05  else
                    "."   if m.pvalues[param]<0.1   else ""),
            "R2": m.rsquared, "adj_R2": m.rsquared_adj, "n": int(m.nobs),
        })
pd.DataFrame(out_rows).to_excel(f"{DATA}/回归结果_网格比例.xlsx", index=False)

# 保存网格数据供可视化
all_grid = pd.concat([d.assign(time_group=tg)
                       for tg, d in grid_data.items()], ignore_index=True)
all_grid.to_parquet(f"{DATA}/grid_pct_better.parquet", index=False)
print(f"\n已保存: 回归结果_网格比例.xlsx  /  grid_pct_better.parquet")
