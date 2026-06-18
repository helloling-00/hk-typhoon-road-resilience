"""
Descriptive statistics: what drives F→S reversal vs F→F persistence?
Merge transition classes with regression_table features (POI, population, road class, etc.).
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
STATES = ["F","N","S"]

def classify(d):
    if d > DEV_HI: return "F"
    if d < DEV_LO: return "S"
    return "N"

# ─── Load transition data ──────────────────────────────────────────────────
print("Loading transition data...", flush=True)
ts = pd.read_parquet(f"{DATA}/yagiasha_road_timeseries.parquet")
ts["ds"] = pd.to_datetime(ts["dt"]).dt.strftime("%Y-%m-%d")
sep23 = ts[ts["ds"]=="2025-09-23"].copy()

rt = pd.read_parquet(f"{DATA}/regression_table.parquet")
length_per_rid = rt.drop_duplicates("road_id").set_index("road_id")["road_length_m"]

s17 = sep23[sep23["slot"]==17][["road_id","dev"]].rename(columns={"dev":"dev_morn"})
s26 = sep23[sep23["slot"]==26][["road_id","dev"]].rename(columns={"dev":"dev_mid"})
both = s17.merge(s26, on="road_id", how="inner")
both["length_m"] = both["road_id"].map(length_per_rid)
both = both.dropna(subset=["length_m"])
both["state_morn"] = both["dev_morn"].apply(classify)
both["state_mid"]  = both["dev_mid"].apply(classify)
both["trans"] = both["state_morn"] + "→" + both["state_mid"]

# Keep only the 4 main transitions for clarity
TRANSITIONS = ["F→F", "F→S", "N→S", "S→S"]
both = both[both["trans"].isin(TRANSITIONS)].copy()

# ─── Merge with regression_table ───────────────────────────────────────────
print("Merging with regression_table...", flush=True)
feat_cols = [
    "road_id", "road_category", "baseline_avg_speed",
    "intersection_degree", "dist_to_coast_m",
    "population_total_500m", "population_density_500m", "median_income_500m",
    "working_pop_ratio_500m", "ratio_学生_500m", "ratio_雇员_500m",
    "ratio_age_0_14_500m", "ratio_age_25_44_500m", "ratio_age_45_64_500m", "ratio_age_65plus_500m",
    "work_density", "education_density", "retail_density", "food_drink_density",
    "recreation_density", "medical_density", "transport_density",
    "tourism_density", "finance_density", "civic_density",
    "incident_count_500m", "severe_incident_500m", "closure_nearby_500m",
]
rt_sub = rt[feat_cols].drop_duplicates("road_id")
merged = both.merge(rt_sub, on="road_id", how="inner")
merged["length_km"] = merged["length_m"] / 1000

print(f"  Merged: {len(merged):,} roads ({merged['length_km'].sum():.0f} km)")
for t in TRANSITIONS:
    sub = merged[merged["trans"]==t]
    print(f"    {t}: {len(sub):,} roads, {sub['length_km'].sum():.0f} km")

# ─── Road category cross-tab ───────────────────────────────────────────────
print("\n=== Road category distribution by transition (% of road-km) ===")
cat_tab = pd.DataFrame({
    t: merged[merged["trans"]==t].groupby("road_category")["length_km"].sum()
    for t in TRANSITIONS
}).fillna(0)
cat_pct = cat_tab / cat_tab.sum() * 100
print(cat_pct.round(1).to_string())

# ─── POI density comparison ────────────────────────────────────────────────
poi_cols = ["work_density","education_density","retail_density","food_drink_density",
            "recreation_density","medical_density","transport_density",
            "tourism_density","finance_density","civic_density"]
poi_labels = ["Work","Education","Retail","Food/Drink","Recreation","Medical",
              "Transport","Tourism","Finance","Civic"]

print("\n=== Mean POI density by transition (per km² in 500m buffer) ===")
poi_means = pd.DataFrame({
    t: merged[merged["trans"]==t][poi_cols].mean() for t in TRANSITIONS
})
poi_means.index = poi_labels
print(poi_means.round(1).to_string())

# ─── Demographic comparison ────────────────────────────────────────────────
demo_cols = ["population_density_500m","median_income_500m","working_pop_ratio_500m",
             "ratio_学生_500m","ratio_age_25_44_500m","ratio_age_65plus_500m"]
demo_labels = ["Pop density","Median income","Working pop ratio",
               "Student ratio","Age 25-44 ratio","Age 65+ ratio"]

print("\n=== Mean demographics by transition ===")
demo_means = pd.DataFrame({
    t: merged[merged["trans"]==t][demo_cols].mean() for t in TRANSITIONS
})
demo_means.index = demo_labels
print(demo_means.round(3).to_string())

# ─── Structural comparison ─────────────────────────────────────────────────
struct_cols = ["baseline_avg_speed","intersection_degree","dist_to_coast_m"]
struct_labels = ["Baseline speed (km/h)","Intersection degree","Dist to coast (m)"]

print("\n=== Mean structural features by transition ===")
struct_means = pd.DataFrame({
    t: merged[merged["trans"]==t][struct_cols].mean() for t in TRANSITIONS
})
struct_means.index = struct_labels
print(struct_means.round(1).to_string())

# ─── Incident comparison ───────────────────────────────────────────────────
inc_cols = ["incident_count_500m","severe_incident_500m","closure_nearby_500m"]
inc_labels = ["Incidents nearby","Severe incidents","Closures nearby"]

print("\n=== Mean incident features by transition ===")
inc_means = pd.DataFrame({
    t: merged[merged["trans"]==t][inc_cols].mean() for t in TRANSITIONS
})
inc_means.index = inc_labels
print(inc_means.round(2).to_string())

# ─── Visualization 1: POI density radar / grouped bar ──────────────────────
print("\nPlotting...", flush=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

colors = {"F→F": "#2ca02c", "F→S": "#d62728", "N→S": "#ff7f0e", "S→S": "#7b1fa2"}

# 1) POI density bars
ax = axes[0]
x = np.arange(len(poi_labels))
w = 0.2
for i, t in enumerate(TRANSITIONS):
    ax.bar(x + i*w, poi_means[t].values, w, color=colors[t], alpha=0.85, label=t)
ax.set_xticks(x + 1.5*w)
ax.set_xticklabels(poi_labels, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("POI density (count/km² in 500m buffer)", fontsize=9)
ax.set_title("POI density by transition class", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")

# 2) F→S / F→F ratio for POIs
ax = axes[1]
ratios = poi_means["F→S"] / poi_means["F→F"]
bars = ax.barh(range(len(poi_labels)), ratios.values, color=["#d62728" if v>1 else "#2ca02c" for v in ratios])
ax.axvline(1, color="black", lw=0.8, ls="--")
ax.set_yticks(range(len(poi_labels)))
ax.set_yticklabels(poi_labels, fontsize=8)
ax.set_xlabel("F→S / F→F ratio (>1 = enriched in reversal roads)", fontsize=9)
ax.set_title("F→S vs F→F: POI enrichment ratio", fontsize=11, fontweight="bold")
for i, v in enumerate(ratios):
    ax.text(v+0.02, i, f"{v:.2f}", va="center", fontsize=8, fontweight="bold")

# 3) Demographics grouped bar
ax = axes[2]
x = np.arange(len(demo_labels))
for i, t in enumerate(TRANSITIONS):
    vals = demo_means[t].values
    ax.bar(x + i*w, vals, w, color=colors[t], alpha=0.85, label=t)
ax.set_xticks(x + 1.5*w)
ax.set_xticklabels(demo_labels, rotation=30, ha="right", fontsize=8)
ax.set_title("Demographics by transition class", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)

# 4) F→S / F→F ratio for demographics
ax = axes[3]
demo_ratio = demo_means["F→S"] / demo_means["F→F"]
bars = ax.barh(range(len(demo_labels)), demo_ratio.values, color=["#d62728" if v>1 else "#2ca02c" for v in demo_ratio])
ax.axvline(1, color="black", lw=0.8, ls="--")
ax.set_yticks(range(len(demo_labels)))
ax.set_yticklabels(demo_labels, fontsize=8)
ax.set_xlabel("F→S / F→F ratio", fontsize=9)
ax.set_title("F→S vs F→F: Demographic enrichment", fontsize=11, fontweight="bold")
for i, v in enumerate(demo_ratio):
    ax.text(v+0.01, i, f"{v:.2f}", va="center", fontsize=8, fontweight="bold")

# 5) Structural features grouped bar
ax = axes[4]
x = np.arange(len(struct_labels))
for i, t in enumerate(TRANSITIONS):
    vals = struct_means[t].values
    # Normalize for display (intersection_degree is small, dist_to_coast is large)
    ax.bar(x + i*w, vals, w, color=colors[t], alpha=0.85, label=t)
ax.set_xticks(x + 1.5*w)
ax.set_xticklabels(struct_labels, rotation=15, ha="right", fontsize=8)
ax.set_title("Structural features by transition class", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)

# 6) Road category distribution as stacked %
ax = axes[5]
cat_order = ["motorway","trunk","primary","secondary","tertiary","street","service"]
cat_names = {"motorway":"Motorway","trunk":"Trunk","primary":"Primary",
             "secondary":"Secondary","tertiary":"Tertiary","street":"Street","service":"Service"}
cat_colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2"]
cat_plot = cat_pct.reindex([c for c in cat_order if c in cat_pct.index])
bottom = np.zeros(len(TRANSITIONS))
for j, cat in enumerate(cat_plot.index):
    vals = cat_plot.loc[cat].values
    ax.bar(TRANSITIONS, vals, 0.55, bottom=bottom, color=cat_colors[j],
           alpha=0.85, label=cat_names.get(cat, cat))
    bottom += vals
ax.set_ylabel("% of road-km", fontsize=9)
ax.set_title("Road category mix by transition", fontsize=11, fontweight="bold")
ax.legend(fontsize=7, loc="upper right", ncol=2)

fig.suptitle("Transition descriptive statistics  —  Ragasa Sep 23, 08:30 → 13:00",
             fontsize=14, fontweight="bold")
plt.tight_layout()
out1 = f"{OUT}/图57_transition_descriptive.png"
fig.savefig(out1, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  saved -> {out1}")

# ─── Key summary table ─────────────────────────────────────────────────────
print("\n=== Key summary: F→S vs F→F ===")
for cols, labels in [(poi_cols, poi_labels), (demo_cols, demo_labels), (struct_cols, struct_labels)]:
    for c, l in zip(cols, labels):
        ff = merged[merged["trans"]=="F→F"][c].mean()
        fs = merged[merged["trans"]=="F→S"][c].mean()
        print(f"  {l:25s}: F→F={ff:.3f}  F→S={fs:.3f}  ratio={fs/ff:.3f}")

print("\nDone.")
