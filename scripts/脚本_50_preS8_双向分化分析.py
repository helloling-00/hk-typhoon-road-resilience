"""
Pre-S8 bidirectional flow analysis (Yagiasha Sep 23):
  Fig 50a — Stacked share-of-roads (faster/normal/slower) over time,
            with incident overlay to separate demand vs supply.
  Fig 50b — Maps: morning-faster cohort (slot 15) vs midday-slower cohort (slot 26).
  Fig 50c — POI / structural feature comparison between the two cohorts.

Key idea: mean relative-speed hides the ~30%/30% bidirectional churn at midday.
"""
import os, gc
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from shapely import wkb as shapely_wkb
import warnings
warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"
OUT  = "/Users/helloling/workspace/thesis"

DEV_HI =  0.03   # faster threshold
DEV_LO = -0.03   # slower threshold
S8_TIME_HHMM = 14 + 20/60  # 14:20

# ─── Load timeseries ─────────────────────────────────────────────────────────
print("Loading yagiasha timeseries...", flush=True)
ts = pd.read_parquet(f"{DATA}/yagiasha_road_timeseries.parquet")
ts["dt"] = pd.to_datetime(ts["dt"])
ts["ds"] = ts["dt"].dt.strftime("%Y-%m-%d")

sep23 = ts[ts["ds"] == "2025-09-23"].copy()
print(f"  Sep 23 rows: {len(sep23):,}, unique roads: {sep23['road_id'].nunique():,}")

# ─── 1) Share-of-roads per slot ──────────────────────────────────────────────
def classify(d):
    if d > DEV_HI: return "faster"
    if d < DEV_LO: return "slower"
    return "normal"

sep23["cls"] = sep23["dev"].apply(classify)
agg = sep23.groupby(["slot","cls"]).size().unstack(fill_value=0)
for c in ["faster","normal","slower"]:
    if c not in agg.columns: agg[c] = 0
agg = agg[["faster","normal","slower"]]
totals = agg.sum(axis=1)
share = agg.div(totals, axis=0)
share["mean_dev"] = sep23.groupby("slot")["dev"].mean()
share["hour"] = share.index * 0.5
share = share.reset_index()

# ─── Incident counts per slot (supply-side proxy) ────────────────────────────
print("Loading incidents Sep 23...", flush=True)
inc_paths = sorted([f"{DATA}/incident_parquet/date=2025-09-23/{p}"
                    for p in os.listdir(f"{DATA}/incident_parquet/date=2025-09-23")])
inc_list = []
for p in inc_paths:
    try:
        d = pd.read_parquet(p, columns=["ts","magnitude_of_delay","closed"])
        inc_list.append(d)
    except: pass
inc = pd.concat(inc_list, ignore_index=True) if inc_list else pd.DataFrame()
inc["ts"] = pd.to_datetime(inc["ts"])
inc["slot"] = inc["ts"].dt.hour*2 + (inc["ts"].dt.minute >= 30).astype(int)
inc_per_slot = inc.groupby("slot").agg(
    n_incidents=("ts","count"),
    n_closed=("closed", lambda x: int((x==True).sum())),
    n_severe=("magnitude_of_delay", lambda x: int((x>=3).sum())),
).reset_index()
share = share.merge(inc_per_slot, on="slot", how="left").fillna({"n_incidents":0,"n_closed":0,"n_severe":0})

# Print key stats
print("\n  Pre-S8 key slots:")
for s in [12, 14, 15, 16, 17, 22, 24, 25, 26, 27, 28]:
    r = share[share["slot"]==s].iloc[0]
    print(f"    slot{s:02d} ({r['hour']:5.1f}h)  faster={r['faster']:.1%}  "
          f"slower={r['slower']:.1%}  mean={r['mean_dev']:+.4f}  "
          f"inc={int(r['n_incidents'])} closed={int(r['n_closed'])}")

# ─── Figure 50a — diverging bars (faster up, slower down) + incidents ───────
# Easier-to-read alternative to a stacked share plot:
#   y > 0  = % of roads going FASTER than baseline by >0.03
#   y < 0  = % of roads going SLOWER than baseline by >0.03
#   gap    = % of "normal" roads (implicit)
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                              gridspec_kw={"height_ratios":[3, 1]})

x = share["hour"].values
w = 0.42
ax.bar(x, share["faster"]*100, width=w, color="#2ca02c", alpha=0.85,
       label=f"% roads FASTER than baseline (dev > +{DEV_HI})")
ax.bar(x, -share["slower"]*100, width=w, color="#d62728", alpha=0.85,
       label=f"% roads SLOWER than baseline (dev < {DEV_LO})")
ax.axhline(0, color="black", lw=0.8)

# Phase windows + S8 vertical
ax.axvspan(6.5, 9.5, color="gold", alpha=0.12, zorder=0)
ax.axvspan(11.5, 14.0, color="lightcoral", alpha=0.12, zorder=0)
ax.axvline(S8_TIME_HHMM, color="orange", lw=2.5, ls="--", alpha=0.9)
ax.annotate("S8 raised\n14:20", xy=(S8_TIME_HHMM, 38), xytext=(15.2, 38),
            color="orange", fontweight="bold", fontsize=10,
            arrowprops=dict(arrowstyle="->", color="orange", lw=1.2))

# Annotate the two morals at the top of each phase
ax.annotate("Morning rush 07:30\n34% faster vs 17% slower\n→ NET FASTER",
            xy=(7.5, 34), xytext=(7.5, 49),
            ha="center", fontsize=9.5, fontweight="bold", color="#1a6b1a",
            arrowprops=dict(arrowstyle="->", color="#1a6b1a", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", fc="#eaf6ea", ec="#1a6b1a", alpha=0.9))
ax.annotate("Midday 13:00\n28% faster AND 28% slower\n→ BIDIRECTIONAL CHURN",
            xy=(13, -28), xytext=(13, -50),
            ha="center", fontsize=9.5, fontweight="bold", color="#8a1f1f",
            arrowprops=dict(arrowstyle="->", color="#8a1f1f", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", fc="#fbeaea", ec="#8a1f1f", alpha=0.9))

ax.set_ylim(-55, 55)
ax.set_ylabel("Share of roads (%)\n← slower      faster →", fontsize=10)
ax.set_title("Pre-S8 (Yagiasha, Sep 23) — % of roads faster vs slower than baseline, every 30 min",
             fontweight="bold", pad=10)
ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
ax.grid(axis="y", alpha=0.25)
yticks = [-40,-20,0,20,40]
ax.set_yticks(yticks); ax.set_yticklabels([f"{abs(v)}%" for v in yticks])

# Bottom: incident count
ax2.bar(x, share["n_incidents"], width=w, color="#666", alpha=0.8, label="All incidents")
ax2.bar(x, share["n_closed"],    width=w, color="#000", label="Formal closures")
ax2.axvline(S8_TIME_HHMM, color="orange", lw=2.5, ls="--", alpha=0.9)
ax2.axvspan(11.5, 14.0, color="lightcoral", alpha=0.12, zorder=0)
ax2.axvspan(6.5, 9.5, color="gold", alpha=0.12, zorder=0)
ax2.set_xlabel("Hour of day (HKT)", fontsize=10)
ax2.set_ylabel("# new incidents\nreported", fontsize=10)
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(axis="y", alpha=0.25)
ax2.set_xticks(range(0, 25, 2))
ax2.set_xlim(-0.5, 24.5)
# Annotate the midday incident spike
ax2.annotate("3.5× normal\n(49 incidents)", xy=(13, 49), xytext=(10, 60),
             fontsize=9, color="#8a1f1f", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#8a1f1f", lw=1.0))

plt.tight_layout()
fig.savefig(f"{OUT}/图50a_preS8_路占比与事故.png", dpi=160, bbox_inches="tight")
print(f"\n  saved -> 图50a_preS8_路占比与事故.png")
plt.close()

# ─── 2) Build geometry cache for Sep 23 ──────────────────────────────────────
print("\nBuilding geometry cache for Sep 23...", flush=True)
folder = f"{FLOW}/2025-09-23"
files = sorted([f for f in os.listdir(folder) if "_slot15_" in f or "_slot26_" in f])

ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
ep_to_rid = ep.set_index("ep_key")["road_id"].to_dict()

def get_ep_key(wb):
    try:
        g = shapely_wkb.loads(bytes(wb))
        coords = list(g.coords) if g.geom_type=="LineString" else \
                 [c for line in g.geoms for c in line.coords]
        s = (round(coords[0][0],4), round(coords[0][1],4))
        e = (round(coords[-1][0],4), round(coords[-1][1],4))
        return str((min(s,e), max(s,e)))
    except: return None

# Pull geometry per road_id (use slot 15 + slot 26 files; either is fine)
geom_per_rid = {}
for f in files:
    df = pd.read_parquet(f"{folder}/{f}", columns=["geometry"])
    for g in df["geometry"]:
        if g is None: continue
        epk = get_ep_key(g)
        if epk and epk in ep_to_rid:
            rid = ep_to_rid[epk]
            if rid not in geom_per_rid:
                try:
                    geom_per_rid[rid] = shapely_wkb.loads(bytes(g))
                except: pass
print(f"  geometries cached for {len(geom_per_rid)} roads")

# ─── 3) Identify cohorts ─────────────────────────────────────────────────────
slot_morn  = sep23[sep23["slot"]==15].copy()  # 07:30 — strongest faster signal
slot_mid   = sep23[sep23["slot"]==26].copy()  # 13:00 — peak dip

morn_fast = slot_morn[slot_morn["dev"] >  DEV_HI]["road_id"].tolist()
morn_slow = slot_morn[slot_morn["dev"] < DEV_LO]["road_id"].tolist()
mid_fast  = slot_mid [slot_mid ["dev"] >  DEV_HI]["road_id"].tolist()
mid_slow  = slot_mid [slot_mid ["dev"] < DEV_LO]["road_id"].tolist()

print(f"\n  Morning (07:30) fast: {len(morn_fast)} roads | slow: {len(morn_slow)} roads")
print(f"  Midday (13:00)  fast: {len(mid_fast)}  roads | slow: {len(mid_slow)}  roads")

# Persistence: same road slow at midday AND not slow in morning?
mid_slow_set  = set(mid_slow)
morn_fast_set = set(morn_fast)
flip_roads = mid_slow_set - set(morn_slow) - morn_fast_set
persistent_slow = mid_slow_set & set(morn_slow)
print(f"  Midday-slow that flipped (not slow in morning): {len(flip_roads)}")
print(f"  Midday-slow that were already slow in morning: {len(persistent_slow)}")

# ─── Figure 50b — Maps of two cohorts ────────────────────────────────────────
print("\nDrawing maps...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

def plot_cohort(ax, fast_ids, slow_ids, title):
    n_f = n_s = 0
    for rid in fast_ids:
        g = geom_per_rid.get(rid)
        if g is None: continue
        if g.geom_type == "LineString":
            xs, ys = zip(*g.coords); ax.plot(xs, ys, color="#2ca02c", lw=0.6, alpha=0.7)
            n_f += 1
        else:
            for line in g.geoms:
                xs, ys = zip(*line.coords); ax.plot(xs, ys, color="#2ca02c", lw=0.6, alpha=0.7)
                n_f += 1
    for rid in slow_ids:
        g = geom_per_rid.get(rid)
        if g is None: continue
        if g.geom_type == "LineString":
            xs, ys = zip(*g.coords); ax.plot(xs, ys, color="#d62728", lw=0.7, alpha=0.8)
            n_s += 1
        else:
            for line in g.geoms:
                xs, ys = zip(*line.coords); ax.plot(xs, ys, color="#d62728", lw=0.7, alpha=0.8)
                n_s += 1
    # Restrict to HK proper bounds
    ax.set_xlim(113.85, 114.45)
    ax.set_ylim(22.18, 22.58)
    ax.set_aspect("equal")
    ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.2)
    ax.plot([], [], color="#2ca02c", label=f"Faster ({n_f} roads drawn)")
    ax.plot([], [], color="#d62728", label=f"Slower ({n_s} roads drawn)")
    ax.legend(loc="lower left", fontsize=9)

plot_cohort(axes[0], morn_fast, morn_slow,
            f"Morning rush 07:30  —  faster {len(morn_fast)} | slower {len(morn_slow)}")
plot_cohort(axes[1], mid_fast, mid_slow,
            f"Midday dip 13:00  —  faster {len(mid_fast)} | slower {len(mid_slow)}")

fig.suptitle("Pre-S8 Bidirectional Flow — Where the Roads Are (Yagiasha Sep 23)",
             fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(f"{OUT}/图50b_preS8_两时刻地图.png", dpi=160, bbox_inches="tight")
print(f"  saved -> 图50b_preS8_两时刻地图.png")
plt.close()

# ─── 4) POI / structural feature comparison ──────────────────────────────────
print("\nBuilding feature comparison...", flush=True)
rt = pd.read_parquet(f"{DATA}/regression_table.parquet")
yagi = rt[rt["typhoon"]=="Ragasa"].drop_duplicates("road_id").set_index("road_id")

feat_cols = [
    "road_length_m", "intersection_degree", "dist_to_coast_m",
    "population_density_500m", "median_income_500m", "working_pop_ratio_500m",
    "ratio_退休人士_500m", "ratio_学生_500m",
    "work_density", "education_density", "retail_density",
    "food_drink_density", "recreation_density", "medical_density",
    "transport_density", "tourism_density", "finance_density", "civic_density",
    "incident_count_500m", "severe_incident_500m", "closure_nearby_500m",
]
feat_cols = [c for c in feat_cols if c in yagi.columns]
LABEL_MAP = {
    "ratio_退休人士_500m": "retired_ratio_500m",
    "ratio_学生_500m":    "student_ratio_500m",
    "ratio_其他职业_500m": "other_job_ratio_500m",
    "ratio_料理家务者_500m": "homemaker_ratio_500m",
    "ratio_无酬家庭从业员_500m": "unpaid_family_ratio_500m",
    "ratio_无酬照顾者_500m": "unpaid_caregiver_ratio_500m",
    "ratio_自营作业者_500m": "self_employed_ratio_500m",
    "ratio_雇主_500m": "employer_ratio_500m",
    "ratio_雇员_500m": "employee_ratio_500m",
}

cohorts = {
    "Morning faster":  morn_fast,
    "Midday slower":   mid_slow,
    "Network avg":     list(set(sep23["road_id"]))
}

stats = {}
for name, rids in cohorts.items():
    sub = yagi.loc[yagi.index.intersection(rids), feat_cols]
    stats[name] = sub.mean()

cmp = pd.DataFrame(stats)
cmp["MornFast/Net"] = cmp["Morning faster"] / cmp["Network avg"].replace(0, np.nan)
cmp["MidSlow/Net"]  = cmp["Midday slower"]  / cmp["Network avg"].replace(0, np.nan)
cmp["MidSlow/MornFast"] = cmp["Midday slower"] / cmp["Morning faster"].replace(0, np.nan)

print("\n  Feature comparison (mean per road):")
print(cmp.round(3).to_string())

cmp.to_csv(f"{OUT}/preS8_cohort_features.csv")

# ─── Figure 50c — feature comparison bar chart ──────────────────────────────
focus_feats = [
    "retail_density","food_drink_density","education_density","transport_density",
    "recreation_density","tourism_density","work_density","finance_density",
    "population_density_500m","ratio_学生_500m","ratio_退休人士_500m",
    "incident_count_500m","closure_nearby_500m",
]
focus_feats = [f for f in focus_feats if f in cmp.index]

ratio_df = cmp.loc[focus_feats, ["MornFast/Net","MidSlow/Net"]].copy()
ratio_df = ratio_df.sort_values("MidSlow/Net", ascending=True)
ratio_df.index = [LABEL_MAP.get(i, i) for i in ratio_df.index]

fig, ax = plt.subplots(figsize=(10, 6))
y = np.arange(len(ratio_df))
ax.barh(y-0.2, ratio_df["MornFast/Net"], 0.4, color="#2ca02c", alpha=0.8,
        label="Morning faster cohort vs network")
ax.barh(y+0.2, ratio_df["MidSlow/Net"], 0.4, color="#d62728", alpha=0.8,
        label="Midday slower cohort vs network")
ax.axvline(1.0, color="black", lw=1, ls="--", alpha=0.6)
ax.set_yticks(y); ax.set_yticklabels(ratio_df.index, fontsize=9)
ax.set_xlabel("Ratio of cohort mean to network mean (1.0 = equal)")
ax.set_title("What's around the two cohorts?\n(POI/demographic features near each road, 500m)",
             fontweight="bold")
ax.legend(loc="lower right")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT}/图50c_preS8_POI对比.png", dpi=160, bbox_inches="tight")
print(f"\n  saved -> 图50c_preS8_POI对比.png")
plt.close()

print("\nDone.")
