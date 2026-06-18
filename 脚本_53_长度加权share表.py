"""
Length-weighted share of clearly-faster / near-baseline / clearly-slower roads
at two key Yagiasha pre-S8 timestamps:
   slot 17 (08:30) — morning peak
   slot 26 (13:00) — midday dip
For Sep 23 (Yagiasha) AND all clean control workdays (averaged + per-day rows).

Thresholds: ±0.03 and ±0.05 (both reported).
Length weighting: % of road kilometers (not road count).
"""
import os, gc
import pandas as pd, numpy as np
from shapely import wkb as shapely_wkb
import warnings
warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"
OUT  = "/Users/helloling/workspace/thesis"

CONTROL_DAYS = ["2025-09-16","2025-09-26","2025-09-29","2025-09-30",
                "2025-10-02","2025-10-06","2025-10-08","2025-10-09"]
KEY_SLOTS = {17:"08:30 (morning peak)", 26:"13:00 (midday dip)"}

print("Loading lookups...", flush=True)
bl = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
bl_idx = bl[bl["day_type"]=="WORKDAY"].set_index(["slot","road_id"])["mean_speed"]
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
ep_to_rid = ep.set_index("ep_key")["road_id"].to_dict()

# Road length (per road_id)
rt = pd.read_parquet(f"{DATA}/regression_table.parquet")
length_per_rid = rt.drop_duplicates("road_id").set_index("road_id")["road_length_m"]
print(f"  length lookup: {len(length_per_rid):,} roads, "
      f"median={length_per_rid.median():.1f}m, total={length_per_rid.sum()/1000:.1f}km")

def get_ep_key(wb):
    try:
        g = shapely_wkb.loads(bytes(wb))
        coords = list(g.coords) if g.geom_type=="LineString" else \
                 [c for line in g.geoms for c in line.coords]
        s = (round(coords[0][0],4), round(coords[0][1],4))
        e = (round(coords[-1][0],4), round(coords[-1][1],4))
        return str((min(s,e), max(s,e)))
    except: return None

def slot_devs(day, slot, wkb_cache):
    pat = [f for f in os.listdir(f"{FLOW}/{day}") if f"_slot{slot:02d}_" in f]
    if not pat: return None
    df = pd.read_parquet(f"{FLOW}/{day}/{pat[0]}",
                         columns=["relative_speed","geometry","road_closure"])
    df = df[df["road_closure"]!=1].dropna(subset=["relative_speed"])
    if len(df) < 50: return None
    def lookup(g):
        if g is None: return None
        b = bytes(g)
        if b in wkb_cache: return wkb_cache[b]
        k = get_ep_key(g)
        if k: wkb_cache[b] = k
        return k
    df["ep_key"] = df["geometry"].apply(lookup)
    df["road_id"] = df["ep_key"].map(ep_to_rid)
    df = df.dropna(subset=["road_id"])
    df["road_id"] = df["road_id"].astype(int)
    agg = df.groupby("road_id")["relative_speed"].mean().rename("obs").reset_index()
    agg["baseline"] = bl_idx.reindex(
        pd.MultiIndex.from_arrays([[slot]*len(agg), agg["road_id"].values],
                                  names=["slot","road_id"])).values
    agg = agg.dropna(subset=["baseline"])
    if len(agg) < 100: return None
    agg["dev"] = agg["obs"] - agg["baseline"]
    agg["length_m"] = agg["road_id"].map(length_per_rid)
    agg = agg.dropna(subset=["length_m"])
    return agg

def compute_shares(agg):
    """Return length-weighted shares at ±0.03 and ±0.05."""
    L = agg["length_m"].sum()
    if L == 0: return None
    out = {"n_roads": len(agg), "total_km": L/1000}
    for thr in [0.03, 0.05]:
        f = agg[agg["dev"] >  thr]["length_m"].sum() / L
        s = agg[agg["dev"] < -thr]["length_m"].sum() / L
        n = 1 - f - s
        out[f"faster_{thr:.2f}"] = f
        out[f"near_{thr:.2f}"]   = n
        out[f"slower_{thr:.2f}"] = s
    return out

# ── Compute Sep 23 (Yagiasha pre-S8) ─────────────────────────────────────────
print("\nComputing Sep 23 ...", flush=True)
wkb23 = {}
yagi_results = {}
for slot in KEY_SLOTS:
    agg = slot_devs("2025-09-23", slot, wkb23)
    yagi_results[slot] = compute_shares(agg) if agg is not None else None
del wkb23; gc.collect()

# ── Compute each control day ─────────────────────────────────────────────────
ctrl_results = {slot: [] for slot in KEY_SLOTS}
for day in CONTROL_DAYS:
    print(f"\nComputing {day} ...", flush=True)
    cache = {}
    for slot in KEY_SLOTS:
        agg = slot_devs(day, slot, cache)
        if agg is not None:
            r = compute_shares(agg); r["day"] = day
            ctrl_results[slot].append(r)
    del cache; gc.collect()

# ── Assemble tables ──────────────────────────────────────────────────────────
def fmt_pct(x): return f"{x*100:5.1f}%"

for slot, label in KEY_SLOTS.items():
    print("\n" + "="*88)
    print(f"  SLOT {slot}  ({label})  —  length-weighted shares")
    print("="*88)
    rows = []
    # Per-day controls
    for r in ctrl_results[slot]:
        rows.append({"day": r["day"], "n_roads": r["n_roads"],
                     "km": r["total_km"],
                     "faster_03": r["faster_0.03"], "near_03": r["near_0.03"], "slower_03": r["slower_0.03"],
                     "faster_05": r["faster_0.05"], "near_05": r["near_0.05"], "slower_05": r["slower_0.05"]})
    # Control mean
    if ctrl_results[slot]:
        cm = {k: np.mean([r[k] for r in ctrl_results[slot]])
              for k in ["faster_0.03","near_0.03","slower_0.03","faster_0.05","near_0.05","slower_0.05"]}
        rows.append({"day": "CONTROL MEAN", "n_roads": int(np.mean([r["n_roads"] for r in ctrl_results[slot]])),
                     "km": float(np.mean([r["total_km"] for r in ctrl_results[slot]])),
                     "faster_03": cm["faster_0.03"], "near_03": cm["near_0.03"], "slower_03": cm["slower_0.03"],
                     "faster_05": cm["faster_0.05"], "near_05": cm["near_0.05"], "slower_05": cm["slower_0.05"]})
    # Yagiasha
    if yagi_results[slot]:
        r = yagi_results[slot]
        rows.append({"day": "Sep 23 (Yagiasha)", "n_roads": r["n_roads"],
                     "km": r["total_km"],
                     "faster_03": r["faster_0.03"], "near_03": r["near_0.03"], "slower_03": r["slower_0.03"],
                     "faster_05": r["faster_0.05"], "near_05": r["near_0.05"], "slower_05": r["slower_0.05"]})

    # Print
    print(f"{'day':<22s} {'n':>6s} {'km':>8s}  | "
          f"{'fast≥0.03':>9s} {'near':>6s} {'slow≥0.03':>9s}  | "
          f"{'fast≥0.05':>9s} {'near':>6s} {'slow≥0.05':>9s}")
    print("-"*88)
    for r in rows:
        sep = "*" if r["day"] in ("CONTROL MEAN","Sep 23 (Yagiasha)") else " "
        print(f"{sep}{r['day']:<21s} {r['n_roads']:>6d} {r['km']:>8.1f}  | "
              f"{fmt_pct(r['faster_03']):>9s} {fmt_pct(r['near_03']):>6s} "
              f"{fmt_pct(r['slower_03']):>9s}  | "
              f"{fmt_pct(r['faster_05']):>9s} {fmt_pct(r['near_05']):>6s} "
              f"{fmt_pct(r['slower_05']):>9s}")

    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/preS8_lengthshare_slot{slot:02d}.csv", index=False)
    print(f"\n  saved -> preS8_lengthshare_slot{slot:02d}.csv")

print("\nDone.")
