"""
Three figures for morning clearance paper section:
  Fig 1: Map — 586 normally-S roads, Sep 23 state (green=became F, red=remained S, gray=N)
  Fig 2: Employee ratio Q1 vs Q4 bar chart
  Fig 3: Student ratio Q1 vs Q4 bar chart
"""
import os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely import wkb as shapely_wkb
import geopandas as gpd
import contextily as cx
import warnings
warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"
OUT  = "/Users/helloling/workspace/thesis"
DEV_HI, DEV_LO = 0.03, -0.03

CTRL_DATES = ["2025-09-16", "2025-09-26", "2025-09-29", "2025-09-30",
              "2025-10-02", "2025-10-06", "2025-10-08", "2025-10-09"]

HK_BBOX = (113.82, 22.15, 114.45, 22.60)

def get_ep_key(wb):
    try:
        g = shapely_wkb.loads(bytes(wb))
        coords = list(g.coords) if g.geom_type == "LineString" else \
                 [c for line in g.geoms for c in line.coords]
        s = (round(coords[0][0], 4), round(coords[0][1], 4))
        e = (round(coords[-1][0], 4), round(coords[-1][1], 4))
        return str((min(s, e), max(s, e)))
    except:
        return None

# ─── Load geometry cache ────────────────────────────────────────────────
print("Building geometry cache...", flush=True)
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
ep_to_rid = ep.set_index("ep_key")["road_id"].to_dict()

geom_per_rid = {}
folder = f"{FLOW}/2025-09-23"
for s in [10, 14, 17, 20, 22, 26, 30, 36]:
    files = [f for f in os.listdir(folder) if f"_slot{s:02d}_" in f]
    if not files: continue
    df = pd.read_parquet(f"{folder}/{files[0]}", columns=["geometry"])
    for g in df["geometry"]:
        if g is None: continue
        epk = get_ep_key(g)
        if epk and epk in ep_to_rid:
            rid = ep_to_rid[epk]
            if rid not in geom_per_rid:
                try:
                    geom_per_rid[rid] = shapely_wkb.loads(bytes(g))
                except: pass
print(f"  {len(geom_per_rid):,} geometries cached")

# ─── Load data ───────────────────────────────────────────────────────────
print("Loading...", flush=True)
ts = pd.read_parquet(f"{DATA}/yagiasha_road_timeseries.parquet")
ts["ds"] = pd.to_datetime(ts["dt"]).dt.strftime("%Y-%m-%d")

# Control workdays slot 17 — per-road S rate
ctrl = ts[(ts["ds"].isin(CTRL_DATES)) & (ts["slot"] == 17)].copy()
ctrl["is_S"] = (ctrl["dev"] < DEV_LO).astype(int)
ctrl_S = ctrl.groupby("road_id")["is_S"].mean().reset_index()
ctrl_S.columns = ["road_id", "ctrl_S_rate"]
ctrl_dev = ctrl.groupby("road_id")["dev"].mean().reset_index()
ctrl_dev.columns = ["road_id", "ctrl_dev_mean"]

# Sep 23 slot 17
sep23 = ts[(ts["ds"] == "2025-09-23") & (ts["slot"] == 17)].copy()
sep23["state"] = sep23["dev"].apply(
    lambda d: "F" if d > DEV_HI else ("S" if d < DEV_LO else "N"))

# Merge
df = sep23[["road_id", "dev", "state"]].rename(columns={"dev": "sep23_dev"})
df = df.merge(ctrl_S, on="road_id", how="inner")
df = df.merge(ctrl_dev, on="road_id", how="inner")
df["improvement"] = df["sep23_dev"] - df["ctrl_dev_mean"]
df["normally_S"] = (df["ctrl_S_rate"] >= 0.5).astype(int)
df["became_F"] = ((df["normally_S"] == 1) & (df["state"] == "F")).astype(int)

# Merge demographics
rt = pd.read_parquet(f"{DATA}/regression_table.parquet")
demo_cols = ["road_id", "ratio_雇员_500m", "ratio_学生_500m",
             "population_density_500m", "median_income_500m"]
rt_demo = rt[demo_cols].drop_duplicates("road_id")
df = df.merge(rt_demo, on="road_id", how="inner")
df = df.dropna(subset=["ratio_雇员_500m", "ratio_学生_500m"])

print(f"  Total: {len(df)} roads with demographics")
ns = df[df["normally_S"] == 1]
print(f"  Normally-S: {len(ns)} roads, became_F: {ns['became_F'].sum()} ({ns['became_F'].mean():.1%})")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Map of 586 normally-S roads on Sep 23
# ══════════════════════════════════════════════════════════════════════════
print("Plotting Figure 1...", flush=True)

rows = []
for _, row in ns.iterrows():
    g = geom_per_rid.get(row["road_id"])
    if g is None: continue
    rows.append({"road_id": row["road_id"], "geometry": g, "state": row["state"]})
gdf_ns = gpd.GeoDataFrame(rows, crs="EPSG:4326")

state_colors = {"F": "#2ca02c", "S": "#d62728", "N": "#7f7f7f"}
state_labels = {"F": "Became Faster", "S": "Remained Slow", "N": "Near baseline"}
state_lw =    {"F": 0.9, "S": 0.9, "N": 0.4}
state_zorder = {"F": 5, "S": 4, "N": 2}

fig, ax = plt.subplots(figsize=(11, 8.5))

for st in ["N", "S", "F"]:
    sub = gdf_ns[gdf_ns["state"] == st]
    if len(sub) == 0: continue
    cnt = len(sub)
    sub_m = sub.to_crs("EPSG:3857")
    sub_m.plot(ax=ax, color=state_colors[st], linewidth=state_lw[st],
               alpha=0.88, zorder=state_zorder[st],
               label=f"{state_labels[st]} — {cnt} roads")

bbox_4326 = gpd.GeoSeries([
    gpd.points_from_xy([HK_BBOX[0], HK_BBOX[2]], [HK_BBOX[1], HK_BBOX[3]])[0],
    gpd.points_from_xy([HK_BBOX[0], HK_BBOX[2]], [HK_BBOX[1], HK_BBOX[3]])[1]
], crs="EPSG:4326").to_crs("EPSG:3857").total_bounds
ax.set_xlim(bbox_4326[0], bbox_4326[2])
ax.set_ylim(bbox_4326[1], bbox_4326[3])

try:
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                   attribution_size=7, zorder=0)
except: pass

ax.set_title("Normally-Congested Roads on Sep 23 Morning (08:30)\n"
             f"{len(ns)} roads (ctrl S rate >= 50%),  "
             f"{ns['became_F'].mean()*100:.0f}% became Faster,  "
             f"mean improvement = +{ns['improvement'].mean():.3f}",
             fontsize=14, fontweight="bold", pad=10)
ax.set_xticks([]); ax.set_yticks([])
ax.legend(loc="lower right", fontsize=10, framealpha=0.92, edgecolor="#999")
ax.set_aspect("equal")

plt.tight_layout()
out1 = f"{OUT}/图59f_normallyS_becameF_map.png"
fig.savefig(out1, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  saved -> {out1}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Employee ratio Q1 vs Q4
# ══════════════════════════════════════════════════════════════════════════
print("Plotting Figure 2...", flush=True)

ns["emp_q"] = pd.qcut(ns["ratio_雇员_500m"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
emp_q14 = ns[ns["emp_q"].astype(str).isin(["Q1", "Q4"])].groupby("emp_q", observed=True).agg(
    became_F=("became_F", "mean"),
    improvement=("improvement", "mean"),
    n=("became_F", "count"),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))

# Left: became_F %
ax = axes[0]
vals = emp_q14["became_F"].values * 100
bars = ax.bar([0, 1], vals, color=["#7f7f7f", "#2ca02c"], width=0.5, edgecolor="white")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Q1 (low)", "Q4 (high)"], fontsize=11)
ax.set_ylabel("% Became Faster on Sep 23", fontsize=10)
ax.set_title("Employee Ratio", fontsize=12, fontweight="bold")
ax.set_ylim(0, 75)
for i, (v, n) in enumerate(zip(vals, emp_q14["n"].values)):
    ax.text(i, v + 1.5, f"{v:.1f}%\n(n={n})", ha="center", fontsize=10, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Right: improvement
ax = axes[1]
vals2 = emp_q14["improvement"].values
bars = ax.bar([0, 1], vals2, color=["#7f7f7f", "#2ca02c"], width=0.5, edgecolor="white")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Q1 (low)", "Q4 (high)"], fontsize=11)
ax.set_ylabel("Mean Improvement (sep23 dev − ctrl dev)", fontsize=10)
ax.set_title("Employee Ratio", fontsize=12, fontweight="bold")
for i, v in enumerate(vals2):
    ax.text(i, v + 0.006, f"+{v:.3f}", ha="center", fontsize=10, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle("Normally-Congested Roads: Morning Clearance by Employee Ratio Quartile\n"
             "Ragasa Sep 23 S3, 08:30",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out2 = f"{OUT}/图59f_employee_clearance.png"
fig.savefig(out2, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  saved -> {out2}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Student ratio Q1 vs Q4
# ══════════════════════════════════════════════════════════════════════════
print("Plotting Figure 3...", flush=True)

ns["stu_q"] = pd.qcut(ns["ratio_学生_500m"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
stu_q14 = ns[ns["stu_q"].astype(str).isin(["Q1", "Q4"])].groupby("stu_q", observed=True).agg(
    became_F=("became_F", "mean"),
    improvement=("improvement", "mean"),
    n=("became_F", "count"),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))

# Left: became_F %
ax = axes[0]
vals = stu_q14["became_F"].values * 100
bars = ax.bar([0, 1], vals, color=["#7f7f7f", "#2ca02c"], width=0.5, edgecolor="white")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Q1 (low)", "Q4 (high)"], fontsize=11)
ax.set_ylabel("% Became Faster on Sep 23", fontsize=10)
ax.set_title("Student Ratio", fontsize=12, fontweight="bold")
ax.set_ylim(0, 75)
for i, (v, n) in enumerate(zip(vals, stu_q14["n"].values)):
    ax.text(i, v + 1.5, f"{v:.1f}%\n(n={n})", ha="center", fontsize=10, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Right: improvement
ax = axes[1]
vals2 = stu_q14["improvement"].values
bars = ax.bar([0, 1], vals2, color=["#7f7f7f", "#2ca02c"], width=0.5, edgecolor="white")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Q1 (low)", "Q4 (high)"], fontsize=11)
ax.set_ylabel("Mean Improvement (sep23 dev − ctrl dev)", fontsize=10)
ax.set_title("Student Ratio", fontsize=12, fontweight="bold")
for i, v in enumerate(vals2):
    ax.text(i, v + 0.006, f"+{v:.3f}", ha="center", fontsize=10, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle("Normally-Congested Roads: Morning Clearance by Student Ratio Quartile\n"
             "Ragasa Sep 23 S3, 08:30",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out3 = f"{OUT}/图59f_student_clearance.png"
fig.savefig(out3, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  saved -> {out3}")

print("Done.")
