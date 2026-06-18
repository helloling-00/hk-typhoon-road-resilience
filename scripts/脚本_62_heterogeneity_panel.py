"""
异质性扩展：跨时段 + 跨台风阶段
看"拥堵越严重 / 商业越密集 → 改善越大"是否在台风全过程稳定。

时段（HKT, 30min slot）:
  morning_peak  07:00–09:30  slot 14-18
  mid_morning   09:30–12:00  slot 19-23
  lunch         12:00–13:30  slot 24-26
  afternoon     13:30–16:30  slot 27-32
  evening_peak  17:00–20:00  slot 34-39
  late_evening  20:00–23:00  slot 40-45

日期（工作日，避免 B/C 类基线干扰）:
  09-22 PRE    Ragasa 进入前/S1（数据缺失多）
  09-23 RISE   S3 → S8 升级
  09-24 PEAK   S10 + S8 双重抑制
  09-25 FALL   S3 → S1 → 解除
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
    "font.size": 12, "axes.titlesize": 13.5, "axes.labelsize": 12,
})

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"
OUT  = "/Users/helloling/workspace/thesis"

WINDOWS = [
    ("morning_peak",  "07:00–09:30", list(range(14,19))),
    ("mid_morning",   "09:30–12:00", list(range(19,24))),
    ("lunch",         "12:00–13:30", list(range(24,27))),
    ("afternoon",     "13:30–16:30", list(range(27,33))),
    ("evening_peak",  "17:00–20:00", list(range(34,40))),
    ("late_evening",  "20:00–23:00", list(range(40,46))),
]
DAYS = [
    ("2025-09-23", "RISE  S3→S8"),
    ("2025-09-24", "PEAK  S10+S8"),
    ("2025-09-25", "FALL  S3→S1"),
]
DTYP = "WORKDAY"
KEEP = {"motorway","motorway_link","trunk","trunk_link",
        "primary","primary_link","secondary","secondary_link",
        "tertiary","tertiary_link"}

print("Loading lookups...", flush=True)
bl = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
rr = pd.read_parquet(f"{DATA}/road_registry.parquet")[
        ["road_id","road_category"]].drop_duplicates("road_id")
lu = pd.read_parquet(f"{DATA}/road_landuse_features.parquet")
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

def load_slot(day, slot):
    pat = f"{FLOW}/{day}/traffic_flow_zoom15_{day}_slot{slot:02d}_*.parquet"
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

# 一次性加载每天所有相关 slot
all_slots_needed = sorted(set().union(*[set(w[2]) for w in WINDOWS]))
print(f"  loading {len(all_slots_needed)} slots × {len(DAYS)} days...", flush=True)

day_obs = {}
for day, label in DAYS:
    parts = []
    for s in all_slots_needed:
        r = load_slot(day, s)
        if r is not None:
            r["day"] = day
            parts.append(r)
    if parts:
        df = pd.concat(parts, ignore_index=True)
        df["bl"] = [bl_idx.get((DTYP, s, rid), np.nan)
                    for s, rid in zip(df["slot"], df["road_id"])]
        df = df.dropna(subset=["bl"])
        day_obs[day] = df
        print(f"    {day}: {len(df):,} obs across {df['slot'].nunique()} slots")

# ── 主分析: 每 (day,window) 计算异质性 ────────────────────────────────────────
rows = []
quint_rows = []   # 每 (day,window,quintile_type,quintile) 的 dev mean

for day, day_label in DAYS:
    if day not in day_obs: continue
    obs = day_obs[day]
    for wname, wlabel, wslots in WINDOWS:
        sub = obs[obs["slot"].isin(wslots)]
        if len(sub) < 200: continue
        agg = sub.groupby("road_id").agg(
            typh=("relative_speed","mean"),
            bl=("bl","mean"),
            n_slot=("slot","nunique")
        ).reset_index()
        # 至少观测一半 slot
        agg = agg[agg["n_slot"] >= max(2, len(wslots)//2)]
        agg["dev"] = agg["typh"] - agg["bl"]
        agg = agg.merge(rr, on="road_id", how="left")
        agg = agg[agg["road_category"].isin(KEEP)]
        if len(agg) < 200: continue

        # baseline quintile
        agg["bl_q"] = pd.qcut(agg["bl"].rank(method="first"), 5, labels=range(1,6))
        # commercial score quintile
        m = agg.merge(lu[["road_id","retail_density","food_drink_density",
                          "recreation_density","finance_density",
                          "tourism_density","work_density"]],
                      on="road_id", how="left")
        m["comm"] = (m["retail_density"].fillna(0) +
                     m["food_drink_density"].fillna(0) +
                     m["recreation_density"].fillna(0) +
                     m["finance_density"].fillna(0))
        m["work_score"] = m["work_density"].fillna(0)
        m["tour_score"] = m["tourism_density"].fillna(0)
        m["comm_q"] = pd.qcut(m["comm"].rank(method="first"), 5, labels=range(1,6))

        d_q1 = agg.loc[agg["bl_q"]==1,"dev"].mean()
        d_q5 = agg.loc[agg["bl_q"]==5,"dev"].mean()
        d_clow = m.loc[m["comm_q"]==1,"dev"].mean()
        d_chigh= m.loc[m["comm_q"]==5,"dev"].mean()

        rows.append({
            "day": day, "day_label": day_label,
            "window": wname, "window_label": wlabel,
            "n_roads": len(agg),
            "mean_dev": agg["dev"].mean(),
            "median_dev": agg["dev"].median(),
            "pct_better_10": (agg["dev"]>0.10).mean()*100,
            "pct_worse_10":  (agg["dev"]<-0.10).mean()*100,
            "bl_q1_dev": d_q1, "bl_q5_dev": d_q5,
            "bl_q1q5_gap": d_q1 - d_q5,
            "comm_low_dev": d_clow, "comm_high_dev": d_chigh,
            "comm_hl_gap": d_chigh - d_clow,
        })

        for q in range(1,6):
            quint_rows.append({"day": day, "window": wname,
                               "type":"baseline",
                               "q":q, "dev": agg.loc[agg["bl_q"]==q,"dev"].mean()})
            quint_rows.append({"day": day, "window": wname,
                               "type":"commercial",
                               "q":q, "dev": m.loc[m["comm_q"]==q,"dev"].mean()})

panel = pd.DataFrame(rows)
qpanel = pd.DataFrame(quint_rows)
print("\n=== Heterogeneity panel ===")
print(panel.to_string(index=False, formatters={
    "mean_dev":"{:+.3f}".format,"median_dev":"{:+.3f}".format,
    "pct_better_10":"{:.1f}".format,"pct_worse_10":"{:.1f}".format,
    "bl_q1_dev":"{:+.3f}".format,"bl_q5_dev":"{:+.3f}".format,
    "bl_q1q5_gap":"{:+.3f}".format,
    "comm_low_dev":"{:+.3f}".format,"comm_high_dev":"{:+.3f}".format,
    "comm_hl_gap":"{:+.3f}".format,
}))
panel.to_csv(f"{OUT}/heterogeneity_panel.csv", index=False)
qpanel.to_csv(f"{OUT}/heterogeneity_quintiles.csv", index=False)

# ── 图: 两个 panel × 3 天 ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey=True)

w_order = [w[0] for w in WINDOWS]
w_label = {w[0]: w[1] for w in WINDOWS}

day_color = {"2025-09-23":"#FB8C00", "2025-09-24":"#C62828", "2025-09-25":"#1976D2"}

# 第一行: baseline quintile
for j,(day,day_label) in enumerate(DAYS):
    ax = axes[0, j]
    sub = qpanel[(qpanel["day"]==day)&(qpanel["type"]=="baseline")]
    if not len(sub): continue
    pv = sub.pivot(index="window", columns="q", values="dev").reindex(w_order)
    # 一条线 = 一个 quintile, 横轴 = window
    cmap = plt.cm.RdYlGn_r
    for q in range(1,6):
        if q not in pv.columns: continue
        col = cmap((q-1)/4)
        ax.plot(range(len(pv)), pv[q].values, marker="o",
                color=col, lw=2.0, ms=7,
                label=f"Q{q}{' slow' if q==1 else (' fast' if q==5 else '')}")
    ax.axhline(0, color="#666", lw=0.6)
    ax.set_xticks(range(len(pv)))
    ax.set_xticklabels([w_label[w].split("–")[0] for w in pv.index],
                       rotation=30, ha="right")
    ax.set_title(f"{day} — {day_label}\nBaseline-congestion quintile",
                 fontweight="bold", fontsize=11.5, loc="left")
    if j==0: ax.set_ylabel("Mean dev (typhoon − baseline)")
    ax.grid(alpha=0.25)
    if j==2: ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

# 第二行: commercial quintile
for j,(day,day_label) in enumerate(DAYS):
    ax = axes[1, j]
    sub = qpanel[(qpanel["day"]==day)&(qpanel["type"]=="commercial")]
    if not len(sub): continue
    pv = sub.pivot(index="window", columns="q", values="dev").reindex(w_order)
    cmap = plt.cm.YlOrRd
    for q in range(1,6):
        if q not in pv.columns: continue
        col = cmap(0.2 + (q-1)/4*0.7)
        ax.plot(range(len(pv)), pv[q].values, marker="s",
                color=col, lw=2.0, ms=7,
                label=f"Q{q}{' low' if q==1 else (' high' if q==5 else '')}")
    ax.axhline(0, color="#666", lw=0.6)
    ax.set_xticks(range(len(pv)))
    ax.set_xticklabels([w_label[w].split("–")[0] for w in pv.index],
                       rotation=30, ha="right")
    ax.set_title(f"{day} — {day_label}\nCommercial-density quintile",
                 fontweight="bold", fontsize=11.5, loc="left")
    if j==0: ax.set_ylabel("Mean dev (typhoon − baseline)")
    ax.grid(alpha=0.25)
    if j==2: ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

fig.suptitle("Heterogeneity across time windows × typhoon stages "
             "(Ragasa, workdays 09-23 to 09-25)",
             fontweight="bold", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.96])
out_fig = f"{OUT}/图25h_heterogeneity_panel.png"
fig.savefig(out_fig, dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved: {out_fig}")

# ── 图2: Q1−Q5 gap & comm-high−low gap 跨时段对比 ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
ax = axes[0]
for day, day_label in DAYS:
    sub = panel[panel["day"]==day]
    sub = sub.set_index("window").reindex(w_order).reset_index()
    ax.plot(range(len(sub)), sub["bl_q1q5_gap"], marker="o", lw=2.4, ms=7,
            color=day_color[day],
            label=f"{day} ({day_label})")
ax.axhline(0, color="#666", lw=0.6)
ax.set_xticks(range(len(WINDOWS)))
ax.set_xticklabels([w[1].split("–")[0] for w in WINDOWS], rotation=30, ha="right")
ax.set_ylabel("Q1(slow) dev  −  Q5(fast) dev")
ax.set_title("Baseline-congestion gap (Q1 − Q5)\n"
             "= how much more congested roads improve",
             fontweight="bold", loc="left", fontsize=12)
ax.grid(alpha=0.25)
ax.legend(loc="upper left", fontsize=9.5)

ax = axes[1]
for day, day_label in DAYS:
    sub = panel[panel["day"]==day]
    sub = sub.set_index("window").reindex(w_order).reset_index()
    ax.plot(range(len(sub)), sub["comm_hl_gap"], marker="s", lw=2.4, ms=7,
            color=day_color[day],
            label=f"{day} ({day_label})")
ax.axhline(0, color="#666", lw=0.6)
ax.set_xticks(range(len(WINDOWS)))
ax.set_xticklabels([w[1].split("–")[0] for w in WINDOWS], rotation=30, ha="right")
ax.set_ylabel("Comm-high dev  −  Comm-low dev")
ax.set_title("Commercial-density gap (high − low)\n"
             "= how much more dense-commercial roads improve",
             fontweight="bold", loc="left", fontsize=12)
ax.grid(alpha=0.25)
ax.legend(loc="upper left", fontsize=9.5)

fig.suptitle("Magnitude of heterogeneity across windows × stages",
             fontweight="bold", fontsize=13)
fig.tight_layout(rect=[0,0,1,0.95])
out_fig2 = f"{OUT}/图25i_heterogeneity_gaps.png"
fig.savefig(out_fig2, dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out_fig2}")
