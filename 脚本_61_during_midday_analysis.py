"""
9月24日 during-typhoon 中段（早高峰结束 ~ 晚高峰开始前）非高峰分析
时段: 09:30-16:30 (slot 19-33, S10 + S8 mix), 工作日
研究: 速度变好/变差路段, baseline 拥堵分位异质性, 商业/休闲 POI 关系
"""
import os, glob
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely import wkb as shapely_wkb
import warnings; warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 220,
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
})

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"
OUT  = "/Users/helloling/workspace/thesis"

MID_SLOTS = list(range(19, 34))   # 09:30 -> 16:30
DAY  = "2025-09-24"
DTYP = "WORKDAY"

print("Loading lookups...", flush=True)
bl = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
rr = pd.read_parquet(f"{DATA}/road_registry.parquet")[
        ["road_id","road_category"]].drop_duplicates("road_id")
lu = pd.read_parquet(f"{DATA}/road_landuse_features.parquet")
poi = pd.read_parquet(f"{DATA}/road_poi_features.parquet")
bl_idx = bl.set_index(["day_type","slot","road_id"])["mean_speed"]

def get_ep_key(wb):
    try:
        g = shapely_wkb.loads(bytes(wb))
        c = list(g.coords) if g.geom_type=="LineString" else \
            [c for line in g.geoms for c in line.coords]
        s = (round(c[0][0],4), round(c[0][1],4))
        e = (round(c[-1][0],4), round(c[-1][1],4))
        return str((min(s,e), max(s,e)))
    except: return None

def load_slot(slot):
    pat = f"{FLOW}/{DAY}/traffic_flow_zoom15_{DAY}_slot{slot:02d}_*.parquet"
    fs = glob.glob(pat)
    if not fs: return None
    df = pd.read_parquet(fs[0],
        columns=["relative_speed","geometry","road_closure"])
    df = df[df["road_closure"]!=1].dropna(subset=["relative_speed"])
    if len(df) < 50: return None
    df["ep_key"] = df["geometry"].apply(get_ep_key)
    df = df.merge(ep[["ep_key","road_id"]], on="ep_key", how="inner")
    obs = df.groupby("road_id")["relative_speed"].mean().reset_index()
    obs["slot"] = slot
    return obs

print(f"Loading 09-24 slots {MID_SLOTS[0]}–{MID_SLOTS[-1]}...", flush=True)
parts = [load_slot(s) for s in MID_SLOTS]
parts = [p for p in parts if p is not None]
obs = pd.concat(parts, ignore_index=True)
print(f"  {len(obs):,} (road,slot) observations across {len(parts)} slots")

# 每条路: typhoon 平均速度 + baseline 同时段平均速度 (用观测到的 slot 集合)
def baseline_lookup(row):
    key = (DTYP, row["slot"], row["road_id"])
    return bl_idx.get(key, np.nan)
obs["bl"] = obs.apply(baseline_lookup, axis=1)
obs = obs.dropna(subset=["bl"])

agg = obs.groupby("road_id").agg(
    typh_speed=("relative_speed","mean"),
    bl_speed  =("bl","mean"),
    n_slots   =("slot","nunique")
).reset_index()
agg = agg[agg["n_slots"] >= 8]   # 至少 8 个 slot 观测，剔除偶发样本
agg["dev"] = agg["typh_speed"] - agg["bl_speed"]
agg = agg.merge(rr, on="road_id", how="left")
print(f"  {len(agg):,} roads after n_slots>=8 filter")

KEEP = {"motorway","motorway_link","trunk","trunk_link",
        "primary","primary_link","secondary","secondary_link",
        "tertiary","tertiary_link"}
agg_main = agg[agg["road_category"].isin(KEEP)].copy()
print(f"  {len(agg_main):,} main-road segments")

# ── 1. 总体分布 ──────────────────────────────────────────────────────────────
n_better = (agg_main["dev"] >  0.10).sum()
n_worse  = (agg_main["dev"] < -0.10).sum()
n_neutral= len(agg_main) - n_better - n_worse
print("\n=== Deviation distribution (main roads, mid-day 09-24) ===")
print(f"  total roads          : {len(agg_main):,}")
print(f"  faster (>+0.10)      : {n_better:,}  ({n_better/len(agg_main)*100:.1f}%)")
print(f"  slower (<-0.10)      : {n_worse:,}   ({n_worse/len(agg_main)*100:.1f}%)")
print(f"  neutral (|dev|<=0.10): {n_neutral:,} ({n_neutral/len(agg_main)*100:.1f}%)")
print(f"  mean dev             : {agg_main['dev'].mean():+.3f}")
print(f"  median dev           : {agg_main['dev'].median():+.3f}")

# ── 2. baseline 五分位 vs deviation ──────────────────────────────────────────
agg_main["bl_quintile"] = pd.qcut(agg_main["bl_speed"].rank(method="first"), 5,
                                  labels=["Q1 slow","Q2","Q3","Q4","Q5 fast"])
g = agg_main.groupby("bl_quintile").agg(
    n=("road_id","count"),
    bl_mean=("bl_speed","mean"),
    typh_mean=("typh_speed","mean"),
    dev_mean=("dev","mean"),
    pct_better=("dev", lambda x: (x>0.10).mean()*100),
    pct_worse =("dev", lambda x: (x<-0.10).mean()*100),
).reset_index()
print("\n=== Heterogeneity by baseline congestion quintile ===")
print(g.to_string(index=False, formatters={
    "bl_mean":"{:.3f}".format, "typh_mean":"{:.3f}".format,
    "dev_mean":"{:+.3f}".format,
    "pct_better":"{:.1f}".format, "pct_worse":"{:.1f}".format,
}))

# ── 3. Top 变好 / 变坏路 ─────────────────────────────────────────────────────
top_better = agg_main.nlargest(20, "dev")[
    ["road_id","road_category","bl_speed","typh_speed","dev","n_slots"]]
top_worse  = agg_main.nsmallest(20, "dev")[
    ["road_id","road_category","bl_speed","typh_speed","dev","n_slots"]]
print("\n=== Top-20 SPEED IMPROVED roads ===")
print(top_better.to_string(index=False, formatters={
    "bl_speed":"{:.3f}".format,"typh_speed":"{:.3f}".format,"dev":"{:+.3f}".format}))
print("\n=== Top-20 SPEED DEGRADED roads ===")
print(top_worse.to_string(index=False, formatters={
    "bl_speed":"{:.3f}".format,"typh_speed":"{:.3f}".format,"dev":"{:+.3f}".format}))

# ── 4. POI / landuse 异质性 ──────────────────────────────────────────────────
m = agg_main.merge(lu, on="road_id", how="left").merge(poi, on="road_id", how="left")
poi_cats = ["retail","food_drink","recreation","tourism","finance","civic","work","education"]

print("\n=== POI density quintile vs mean dev (main roads) ===")
print(f"{'Category':<14}{'Q1(low)':>10}{'Q2':>10}{'Q3':>10}{'Q4':>10}{'Q5(high)':>10}{'Q5−Q1':>10}")
het_rows = []
for cat in poi_cats:
    col = f"{cat}_density"
    if col not in m.columns: continue
    sub = m[m[col].notna()].copy()
    if len(sub) < 100: continue
    sub["q"] = pd.qcut(sub[col].rank(method="first"), 5, labels=False)+1
    qm = sub.groupby("q")["dev"].mean()
    q1, q5 = qm.iloc[0], qm.iloc[-1]
    print(f"{cat:<14}" + "".join(f"{qm[i]:>+10.3f}" for i in range(1,6)) + f"{q5-q1:>+10.3f}")
    het_rows.append({"cat":cat,**{f"q{i}":qm[i] for i in range(1,6)},"q5_minus_q1":q5-q1})

# 商业区聚合 (retail+food+recreation+finance) 高密度 vs 低密度路段对比
m["commercial_score"] = (m["retail_density"].fillna(0) +
                         m["food_drink_density"].fillna(0) +
                         m["recreation_density"].fillna(0) +
                         m["finance_density"].fillna(0))
m["work_score"] = m["work_density"].fillna(0) + m["finance_density"].fillna(0)

m["comm_q"] = pd.qcut(m["commercial_score"].rank(method="first"), 5,
                      labels=["low","Q2","Q3","Q4","high"])
print("\n=== Commercial-zone score (retail+food+recreation+finance) ===")
gg = m.groupby("comm_q").agg(
    n=("road_id","count"),
    bl_mean=("bl_speed","mean"),
    dev_mean=("dev","mean"),
    pct_better=("dev", lambda x: (x>0.10).mean()*100),
    pct_worse =("dev", lambda x: (x<-0.10).mean()*100),
).reset_index()
print(gg.to_string(index=False, formatters={
    "bl_mean":"{:.3f}".format,"dev_mean":"{:+.3f}".format,
    "pct_better":"{:.1f}".format,"pct_worse":"{:.1f}".format}))

# 控制 baseline_speed 的回归: dev ~ commercial + work + bl_speed
import statsmodels.api as sm
reg = m.dropna(subset=["dev","bl_speed"]).copy()
for c in ["retail_density","food_drink_density","recreation_density",
          "finance_density","work_density","tourism_density"]:
    if c in reg.columns: reg[c] = reg[c].fillna(0)
X = reg[["bl_speed","retail_density","food_drink_density",
         "recreation_density","finance_density","tourism_density","work_density"]]
X = (X - X.mean()) / X.std()
X = sm.add_constant(X)
y = reg["dev"]
res = sm.OLS(y, X).fit(cov_type="HC3")
print("\n=== OLS: dev = f(bl_speed, POI densities)  [standardized X] ===")
print(res.summary().tables[1])

# ── 5. 图: heterogeneity bar + commercial zone bar ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# (a) baseline 五分位 → mean dev
ax = axes[0]
xs = np.arange(len(g))
bars = ax.bar(xs, g["dev_mean"], color="#1976D2", alpha=0.85, edgecolor="white")
for b, v, p in zip(bars, g["dev_mean"], g["pct_better"]):
    ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:+.3f}\n({p:.0f}% faster)",
            ha="center", va="bottom", fontsize=10)
ax.set_xticks(xs)
ax.set_xticklabels([f"{q}\nbl={b:.2f}" for q, b in zip(g["bl_quintile"], g["bl_mean"])])
ax.set_ylabel("Mean speed deviation (typhoon − baseline)")
ax.set_title("Baseline-congestion heterogeneity\n(more congested → more relief)",
             fontweight="bold", loc="left")
ax.axhline(0, color="#444", lw=0.6)
ax.grid(axis="y", alpha=0.25)

# (b) commercial-zone 五分位 → mean dev
ax = axes[1]
xs = np.arange(len(gg))
bars = ax.bar(xs, gg["dev_mean"], color="#E65100", alpha=0.85, edgecolor="white")
for b, v, p in zip(bars, gg["dev_mean"], gg["pct_better"]):
    ax.text(b.get_x()+b.get_width()/2, v+0.003, f"{v:+.3f}\n({p:.0f}% faster)",
            ha="center", va="bottom", fontsize=10)
ax.set_xticks(xs)
ax.set_xticklabels(gg["comm_q"])
ax.set_ylabel("Mean speed deviation (typhoon − baseline)")
ax.set_title("Commercial-density quintile\n(retail+food+recreation+finance)",
             fontweight="bold", loc="left")
ax.axhline(0, color="#444", lw=0.6)
ax.grid(axis="y", alpha=0.25)

fig.suptitle("Sep 24 mid-day (09:30–16:30, S10/S8) — Speed change vs baseline",
             fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96])
out_fig = f"{OUT}/图25g_midday_heterogeneity.png"
fig.savefig(out_fig, dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved figure: {out_fig}")

# ── 6. 保存 CSV ───────────────────────────────────────────────────────────────
agg_main.to_csv(f"{OUT}/midday_road_dev.csv", index=False)
top_better.to_csv(f"{OUT}/midday_top_better.csv", index=False)
top_worse.to_csv(f"{OUT}/midday_top_worse.csv", index=False)
gg.to_csv(f"{OUT}/midday_commercial_quintile.csv", index=False)
g.to_csv(f"{OUT}/midday_baseline_quintile.csv", index=False)
print("CSVs saved.")
