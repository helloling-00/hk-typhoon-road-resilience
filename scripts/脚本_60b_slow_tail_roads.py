"""
Slow-tail truncation analysis: which roads remained slow during S8/S10?
Compare typhoon-slow roads vs typhoon-fast roads on POI, demographics, road type.
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
OUT  = "/Users/helloling/workspace/thesis"
DEV_HI, DEV_LO = 0.03, -0.03

# ─── Load ─────────────────────────────────────────────────────────────────
print("Loading...", flush=True)
ts = pd.read_parquet(f"{DATA}/yagiasha_road_timeseries.parquet")
ts["dt"] = pd.to_datetime(ts["dt"])
ts["ds"] = ts["dt"].dt.strftime("%Y-%m-%d")

# S8-S10-S8 period: Sep 23 14:20 to Sep 24 20:20
typhoon = ts[(ts["dt"] >= "2025-09-23 14:20") & (ts["dt"] <= "2025-09-24 20:20")].copy()
typhoon["state"] = typhoon["dev"].apply(
    lambda d: "F" if d > DEV_HI else ("S" if d < DEV_LO else "N"))

# Per-road summary during typhoon
road_ty = typhoon.groupby("road_id").agg(
    n_slots=("state", "count"),
    mean_dev=("dev", "mean"),
    F_rate=("state", lambda x: (x == "F").mean()),
    S_rate=("state", lambda x: (x == "S").mean()),
    N_rate=("state", lambda x: (x == "N").mean()),
).reset_index()

# Control workdays
CTRL = ["2025-09-16", "2025-09-26", "2025-09-29", "2025-09-30",
        "2025-10-02", "2025-10-06", "2025-10-08", "2025-10-09"]
ctrl = ts[ts["ds"].isin(CTRL)].copy()
ctrl["state"] = ctrl["dev"].apply(
    lambda d: "F" if d > DEV_HI else ("S" if d < DEV_LO else "N"))

road_ctrl = ctrl.groupby("road_id").agg(
    mean_dev=("dev", "mean"),
    F_rate=("state", lambda x: (x == "F").mean()),
    S_rate=("state", lambda x: (x == "S").mean()),
).rename(columns={"mean_dev": "ctrl_dev", "F_rate": "ctrl_F", "S_rate": "ctrl_S"}).reset_index()

# Merge
road = road_ty.merge(road_ctrl, on="road_id", how="inner")
# Filter to roads with ≥3 typhoon slots
road = road[road["n_slots"] >= 3].copy()
print(f"  Roads with ≥3 typhoon slots: {len(road)}")

# ─── Classify roads ─────────────────────────────────────────────────────
# Typhoon-persistent-slow: S_rate > 0 during typhoon and at least 20% of slots
road["ty_slow"] = (road["S_rate"] >= 0.2).astype(int)
# Typhoon-cleared: F_rate > 0 during typhoon and at least 40% of slots
road["ty_fast"] = (road["F_rate"] >= 0.4).astype(int)
# Mixed: both
road["ty_mixed"] = ((road["ty_slow"] == 1) & (road["ty_fast"] == 1)).astype(int)

print(f"  Typhoon-slow (S≥20%): {road['ty_slow'].sum()}")
print(f"  Typhoon-fast (F≥40%): {road['ty_fast'].sum()}")
print(f"  Both slow & fast:     {road['ty_mixed'].sum()}")

# Pure types (exclude mixed)
road["group"] = "Other"
road.loc[(road["ty_slow"] == 1) & (road["ty_fast"] == 0), "group"] = "Typhoon-Slow"
road.loc[(road["ty_fast"] == 1) & (road["ty_slow"] == 0), "group"] = "Typhoon-Fast"
print(f"  Pure Typhoon-Slow: {sum(road['group']=='Typhoon-Slow')}")
print(f"  Pure Typhoon-Fast: {sum(road['group']=='Typhoon-Fast')}")

# ─── Merge with POI + demographics ──────────────────────────────────────
rt = pd.read_parquet(f"{DATA}/regression_table.parquet")
feat_cols = [
    "road_id", "road_category", "road_length_m",
    "intersection_degree", "dist_to_coast_m",
    "work_density", "education_density", "retail_density", "food_drink_density",
    "recreation_density", "medical_density", "transport_density",
    "tourism_density", "finance_density", "civic_density",
    "incident_count_500m",
    "population_density_500m", "median_income_500m",
    "ratio_学生_500m", "ratio_雇员_500m", "ratio_退休人士_500m",
    "ratio_age_65plus_500m",
]
rt_sub = rt[feat_cols].drop_duplicates("road_id")
road = road.merge(rt_sub, on="road_id", how="inner")

# Collapse road category
cat_map = {
    "motorway": "Highway", "motorway_link": "Highway",
    "trunk": "Highway", "trunk_link": "Highway",
    "primary": "Arterial", "primary_link": "Arterial",
    "secondary": "Arterial", "secondary_link": "Arterial",
    "tertiary": "Local", "tertiary_link": "Local",
    "street": "Local", "service": "Local",
}
road["road_class"] = road["road_category"].map(cat_map)

print(f"\n  With features: {len(road)}")

# ─── Comparison table ────────────────────────────────────────────────────
compare = road[road["group"].isin(["Typhoon-Slow", "Typhoon-Fast"])]

poi_vars = ["work_density", "education_density", "retail_density", "food_drink_density",
            "recreation_density", "medical_density", "transport_density",
            "tourism_density", "finance_density", "civic_density"]
demo_vars = ["population_density_500m", "median_income_500m",
             "ratio_学生_500m", "ratio_雇员_500m", "ratio_退休人士_500m", "ratio_age_65plus_500m"]
struct_vars = ["intersection_degree", "road_length_m"]

print(f"\n{'='*90}")
print("  COMPARISON: Typhoon-Slow vs Typhoon-Fast roads")
print(f"{'='*90}")
print(f"  {'Variable':30s} {'Typhoon-Slow':>14s} {'Typhoon-Fast':>14s} {'Ratio':>8s}")
print(f"  {'─'*30} {'─'*14} {'─'*14} {'─'*8}")

all_vars = struct_vars + poi_vars + demo_vars + ["ctrl_S", "ctrl_F", "n_slots", "mean_dev"]
for var in all_vars:
    if var not in compare.columns:
        continue
    slow_val = compare[compare["group"] == "Typhoon-Slow"][var].mean()
    fast_val = compare[compare["group"] == "Typhoon-Fast"][var].mean()
    ratio = slow_val / fast_val if fast_val != 0 else np.inf
    marker = " ***" if abs(ratio - 1) > 0.5 else (" **" if abs(ratio - 1) > 0.3 else "")
    print(f"  {var:30s} {slow_val:>14.4f} {fast_val:>14.4f} {ratio:>7.2f}{marker}")

# Road class distribution
print(f"\n  === Road class distribution ===")
xtab = compare.groupby("group")["road_class"].value_counts(normalize=True).unstack()
print(xtab.round(3).to_string())

# ─── Bar plot: key differences ──────────────────────────────────────────
key_vars = [
    ("intersection_degree", "Intersection density"),
    ("work_density", "Office density"),
    ("retail_density", "Retail density"),
    ("recreation_density", "Recreation density"),
    ("population_density_500m", "Population density"),
    ("ratio_退休人士_500m", "Retired ratio"),
    ("ctrl_S", "Ctrl Slow rate"),
]

fig, axes = plt.subplots(2, 4, figsize=(16, 9))
axes = axes.flatten()

for ax, (var, label) in zip(axes, key_vars):
    slow_data = compare[compare["group"] == "Typhoon-Slow"][var].dropna()
    fast_data = compare[compare["group"] == "Typhoon-Fast"][var].dropna()
    ax.boxplot([slow_data, fast_data], labels=["Typhoon\nSlow", "Typhoon\nFast"],
               patch_artist=True, widths=0.5, showfliers=False,
               medianprops={"color": "black", "lw": 1.5})
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)

# Extra: scatter plot: ctrl_S rate vs typhoon S rate
ax = axes[-1]
ax.scatter(road["ctrl_S"], road["S_rate"], alpha=0.15, s=4, color="#333")
ax.plot([0, 1], [0, 1], "r--", lw=0.8)
ax.set_xlabel("Control S rate", fontsize=9)
ax.set_ylabel("Typhoon S rate", fontsize=9)
ax.set_title("Ctrl vs Typhoon: S rate", fontsize=10, fontweight="bold")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

fig.suptitle("Typhoon-Slow vs Typhoon-Fast Roads  —  S8/S10 Period",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out1 = f"{OUT}/图60b_slow_tail_comparison.png"
fig.savefig(out1, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n  saved -> {out1}")

# ─── Top roads that were still slow ──────────────────────────────────────
print(f"\n  === Profile of Typhoon-Slow roads ===")
slow_roads = road[road["group"] == "Typhoon-Slow"]
print(f"  n={len(slow_roads)}")
print(f"  Mean dev during typhoon: {slow_roads['mean_dev'].mean():.4f}")
print(f"  Mean S rate: {slow_roads['S_rate'].mean():.3f}")
print(f"  Mean ctrl S rate: {slow_roads['ctrl_S'].mean():.3f}")
print(f"  Mean ctrl dev: {slow_roads['ctrl_dev'].mean():.4f}")

# Save
road[["road_id", "group", "n_slots", "mean_dev", "F_rate", "S_rate",
      "ctrl_dev", "ctrl_F", "ctrl_S"] + [v for v in all_vars if v in road.columns]] \
    .to_csv(f"{OUT}/preS8_typhoon_slow_fast.csv", index=False)
print("  CSV saved")

print("\nDone.")
