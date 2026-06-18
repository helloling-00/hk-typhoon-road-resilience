"""
图47 — Speed shape comparison: Mina (no S8) vs Yagiasha (S8+)
Shows that midday congestion dip only occurs pre-S8 (Yagiasha Sep 23),
not during weaker typhoon Mina (max S3, no S8).
Conclusion: S8 is the behavioural trigger, not S1/S3.
"""
import os, gc, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely import wkb as shapely_wkb
import warnings
warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"
OUT  = "/Users/helloling/workspace/thesis"

# ── Load lookups ──────────────────────────────────────────────────────────────
print("Loading lookups...", flush=True)
bl = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")

def get_ep_key(wb):
    try:
        g = shapely_wkb.loads(bytes(wb))
        coords = list(g.coords) if g.geom_type == "LineString" else \
                 [c for line in g.geoms for c in line.coords]
        s = (round(coords[0][0],4), round(coords[0][1],4))
        e = (round(coords[-1][0],4), round(coords[-1][1],4))
        return str((min(s,e), max(s,e)))
    except: return None

# ── Pre-build WKB cache from a sample day ─────────────────────────────────────
def build_wkb_cache(day):
    folder = f"{FLOW}/{day}"
    if not os.path.exists(folder): return {}
    uniq = {}
    for s in range(0, 48):
        files = [f for f in os.listdir(folder) if f"_slot{s:02d}_" in f]
        if not files: continue
        df = pd.read_parquet(f"{folder}/{files[0]}", columns=["geometry"])
        for g in df["geometry"]:
            if g is not None:
                k = id(bytes(g)[:8])
                if k not in uniq: uniq[k] = g
    wkb_ep = {}
    for g in uniq.values():
        epk = get_ep_key(g)
        if epk: wkb_ep[bytes(g)] = epk
    print(f"  {day}: {len(wkb_ep)} unique geometries cached", flush=True)
    return wkb_ep

# ── Compute per-slot deviations for a single day ──────────────────────────────
def compute_day_deviations(day, wkb_cache):
    folder = f"{FLOW}/{day}"
    if not os.path.exists(folder): return None

    def lookup_epk(g):
        if g is None: return None
        b = bytes(g)
        if b in wkb_cache: return wkb_cache[b]
        epk = get_ep_key(g)
        if epk: wkb_cache[b] = epk
        return epk

    all_slots = sorted([int(f.split("_slot")[1][:2])
                        for f in os.listdir(folder)
                        if "_slot" in f and f.endswith(".parquet")])

    records = []
    for s in all_slots:
        files = [f for f in os.listdir(folder) if f"_slot{s:02d}_" in f]
        if not files: continue
        try:
            df = pd.read_parquet(f"{folder}/{files[0]}",
                                 columns=["relative_speed","geometry","road_closure"])
            df = df[df["road_closure"] != 1].copy()
            if len(df) < 50: continue

            df["ep_key"] = df["geometry"].apply(lookup_epk)
            df = df.merge(ep[["ep_key","road_id"]], on="ep_key", how="inner")
            if len(df) < 50: continue

            agg = df.groupby("road_id")["relative_speed"].mean().rename("obs")
            agg = agg.reset_index().set_index("road_id")

            # Baseline lookup
            bl_idx = bl.set_index(["day_type","slot","road_id"])["mean_speed"]
            idx = pd.MultiIndex.from_arrays(
                [["WORKDAY"]*len(agg), [s]*len(agg), agg.index],
                names=["day_type","slot","road_id"])
            agg["baseline"] = bl_idx.reindex(idx).values
            agg = agg.dropna(subset=["baseline"])
            if len(agg) < 100: continue

            agg["dev"] = agg["obs"] - agg["baseline"]
            records.append({
                "day": day, "slot": s, "hour": s * 0.5,
                "n_roads": len(agg),
                "mean_dev": float(agg["dev"].mean()),
                "median_dev": float(agg["dev"].median()),
                "std_dev": float(agg["dev"].std()),
                "pct_faster": float((agg["dev"] > 0).mean()),
                "pct_slower_003": float((agg["dev"] < -0.03).mean()),
                "pct_faster_003": float((agg["dev"] > 0.03).mean()),
            })
        except Exception as e:
            pass
        gc.collect()

    return pd.DataFrame(records) if records else None

# ── Process all days ──────────────────────────────────────────────────────────
MINA_DAYS = ["2025-09-17", "2025-09-18", "2025-09-19"]
YAGI_DAYS = ["2025-09-22", "2025-09-23", "2025-09-24", "2025-09-25"]

# Build WKB cache from first available day
all_days = MINA_DAYS + YAGI_DAYS
first_available = None
for d in all_days:
    if os.path.exists(f"{FLOW}/{d}"):
        first_available = d
        break
print(f"Building WKB cache from {first_available}...", flush=True)
wkb_cache = build_wkb_cache(first_available)

# Compute all days
all_data = []
for d in all_days:
    print(f"Processing {d}...", flush=True)
    df = compute_day_deviations(d, wkb_cache)
    if df is not None and len(df) > 0:
        all_data.append(df)
        print(f"  {d}: {len(df)} slots, mean_dev={df['mean_dev'].mean():+.4f}", flush=True)
    gc.collect()

combined = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
print(f"\nTotal: {len(combined)} slots across {combined['day'].nunique()} days")

# ── Plot ────────────────────────────────────────────────────────────────────
print("Plotting...", flush=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.3),
                                facecolor="#1a1a2e")
for ax in [ax1, ax2]:
    ax.set_facecolor("#16213e")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.axhline(0, color="#aaaaaa", linewidth=0.5, linestyle="--")
    ax.set_ylim(-0.035, 0.055)
    ax.set_xlabel("Hour of Day", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("Mean Speed Deviation", color="#aaaaaa", fontsize=9)

# Color palette for Mina days
mina_colors = {"2025-09-17": "#4fc3f7", "2025-09-18": "#81d4fa", "2025-09-19": "#b3e5fc"}
# Color palette for Yagiasha days
yagi_colors = {"2025-09-22": "#aaaaaa", "2025-09-23": "#ff5252", "2025-09-24": "#81c784", "2025-09-25": "#64b5f6"}

# --- LEFT: Mina ---
ax1.set_title("Mina (max Signal 3, no S8)", color="white", fontsize=12, fontweight="bold", pad=10)
for d in MINA_DAYS:
    sub = combined[combined["day"] == d]
    if len(sub) == 0: continue
    label = {"2025-09-17":"Sep 17 (S1↑→S3↓)","2025-09-18":"Sep 18 (S3)","2025-09-19":"Sep 19 (S3)"}.get(d, d)
    lw = 2.0 if d == "2025-09-18" else 1.2
    alpha = 1.0 if d == "2025-09-18" else 0.7
    sub = sub.sort_values("hour")
    ax1.plot(sub["hour"], sub["mean_dev"], color=mina_colors[d], linewidth=lw,
             alpha=alpha, label=label)
# Highlight midday zone (11:00–14:00)
ax1.axvspan(11, 14, alpha=0.10, color="#ffc800")
ax1.text(12.5, 0.050, "midday", color="#ffc800", fontsize=7, ha="center", alpha=0.8)
ax1.legend(loc="lower left", fontsize=7, facecolor="#16213e", edgecolor="#444444",
           labelcolor="#cccccc")

# --- RIGHT: Yagiasha ---
ax2.set_title("Yagiasha (S8+ at 14:20 Sep 23)", color="white", fontsize=12, fontweight="bold", pad=10)
for d in YAGI_DAYS:
    sub = combined[combined["day"] == d]
    if len(sub) == 0: continue
    if d == "2025-09-22":
        label = "Sep 22 (S1, data gap)"
        ls = ":"
        lw = 0.8; alpha = 0.4
    else:
        label = {"2025-09-23":"Sep 23 (pre-S8 → S8↑)","2025-09-24":"Sep 24 (S10↓→S3↓)","2025-09-25":"Sep 25 (S1↓→Clear)"}.get(d, d)
        ls = "-"
        lw = 3.0 if d == "2025-09-23" else 1.5
        alpha = 1.0 if d == "2025-09-23" else 0.7
    sub = sub.sort_values("hour")
    ax2.plot(sub["hour"], sub["mean_dev"], color=yagi_colors[d], linewidth=lw,
             linestyle=ls, alpha=alpha, label=label)
# Highlight midday zone (11:00–14:00)
ax2.axvspan(11, 14, alpha=0.15, color="#ff5252")
ax2.text(12.5, 0.050, "midday dip\npre-S8", color="#ff5252", fontsize=7, ha="center", alpha=0.9)
# S8 line
ax2.axvline(14.33, color="#ffc800", linewidth=0.8, linestyle="--", alpha=0.7)
ax2.text(14.4, -0.030, "S8\n14:20", color="#ffc800", fontsize=6.5, alpha=0.9)
ax2.legend(loc="lower left", fontsize=7, facecolor="#16213e", edgecolor="#444444",
           labelcolor="#cccccc")

fig.suptitle("Speed Shape Comparison: Without S8 (Mina) vs With S8 (Yagiasha)",
             color="white", fontsize=14, fontweight="bold", y=1.01)

plt.tight_layout()
out_path = f"{OUT}/图47_速度形状对比_米娜vs叶加沙.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#1a1a2e")
print(f"Saved: {out_path}")
plt.close()
