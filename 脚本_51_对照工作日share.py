"""
Falsification check: does the 28%/28% bidirectional split at Sep 23 13:00
also appear on a normal non-typhoon workday?

Compute per-slot share-of-roads (faster/slower vs WORKDAY baseline) for:
  - Sep 16 (clean Tuesday, before any typhoon)
  - Sep 30 (clean Tuesday between Ragasa and Matmo)
And compare against Sep 23 (Yagiasha pre-S8).
"""
import os, gc
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from shapely import wkb as shapely_wkb
import warnings
warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"
OUT  = "/Users/helloling/workspace/thesis"
DEV_HI, DEV_LO = 0.03, -0.03

print("Loading baseline & ep_to_road...", flush=True)
bl = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
bl_idx = bl[bl["day_type"]=="WORKDAY"].set_index(["slot","road_id"])["mean_speed"]
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

def share_for_day(day):
    folder = f"{FLOW}/{day}"
    files = sorted(os.listdir(folder))
    rows = []
    wkb_ep = {}
    for f in files:
        if not f.endswith(".parquet") or "_slot" not in f: continue
        s = int(f.split("_slot")[1][:2])
        try:
            df = pd.read_parquet(f"{folder}/{f}",
                                 columns=["relative_speed","geometry","road_closure"])
            df = df[df["road_closure"] != 1].copy()
            if len(df) < 50: continue
            def lookup(g):
                if g is None: return None
                b = bytes(g)
                if b in wkb_ep: return wkb_ep[b]
                k = get_ep_key(g)
                if k: wkb_ep[b] = k
                return k
            df["ep_key"] = df["geometry"].apply(lookup)
            df["road_id"] = df["ep_key"].map(ep_to_rid)
            df = df.dropna(subset=["road_id"])
            df["road_id"] = df["road_id"].astype(int)
            agg = df.groupby("road_id")["relative_speed"].mean().rename("obs").reset_index()
            agg["slot"] = s
            agg["baseline"] = bl_idx.reindex(
                pd.MultiIndex.from_arrays([[s]*len(agg), agg["road_id"].values],
                                          names=["slot","road_id"])).values
            agg = agg.dropna(subset=["baseline"])
            if len(agg) < 100: continue
            agg["dev"] = agg["obs"] - agg["baseline"]
            n = len(agg)
            rows.append({
                "slot": s, "hour": s*0.5, "n": n,
                "mean_dev": float(agg["dev"].mean()),
                "pct_faster": float((agg["dev"]> DEV_HI).mean()),
                "pct_slower": float((agg["dev"]< DEV_LO).mean()),
            })
        except Exception as e:
            print(f"  skipped {f}: {e}")
        gc.collect()
    out = pd.DataFrame(rows).sort_values("slot").reset_index(drop=True)
    return out

print("\nComputing Sep 16 (control Tue, pre-Mitag)...", flush=True)
s16 = share_for_day("2025-09-16")
print(f"  {len(s16)} slots")

print("\nComputing Sep 30 (control Tue, post-Ragasa)...", flush=True)
s30 = share_for_day("2025-09-30")
print(f"  {len(s30)} slots")

print("\nComputing Oct 08 (control Wed, between Ragasa and Matmo)...", flush=True)
s08 = share_for_day("2025-10-08")
print(f"  {len(s08)} slots")

# Reuse Sep 23 from existing timeseries (matches earlier Fig 50a)
print("\nLoading Sep 23 from timeseries...", flush=True)
ts = pd.read_parquet(f"{DATA}/yagiasha_road_timeseries.parquet")
ts["ds"] = pd.to_datetime(ts["dt"]).dt.strftime("%Y-%m-%d")
sep23 = ts[ts["ds"]=="2025-09-23"]
s23 = sep23.groupby("slot").agg(
    n=("road_id","count"),
    mean_dev=("dev","mean"),
    pct_faster=("dev", lambda x: (x>DEV_HI).mean()),
    pct_slower=("dev", lambda x: (x<DEV_LO).mean()),
).reset_index()
s23["hour"] = s23["slot"]*0.5

# Pretty summary
print("\n  Slot-by-slot (key hours):")
print(f"{'slot':>4} {'hr':>5}  | "
      f"{'Sep16 f%':>9} {'s%':>5}  | "
      f"{'Sep30 f%':>9} {'s%':>5}  | "
      f"{'Oct08 f%':>9} {'s%':>5}  | "
      f"{'Sep23 f%':>9} {'s%':>5}")
for s in [12, 14, 15, 16, 17, 22, 24, 25, 26, 27, 28]:
    def fmt(d, s):
        r = d[d.slot==s]
        if len(r)==0: return "  --       --"
        r = r.iloc[0]
        return f"{r['pct_faster']*100:7.1f}% {r['pct_slower']*100:5.1f}%"
    print(f"{s:>4} {s*0.5:>5.1f}  | "
          f"{fmt(s16,s)}  | {fmt(s30,s)}  | {fmt(s08,s)}  | {fmt(s23,s)}")

# ─── Figure 51: 4-panel comparison ───────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=True)

def plot_div(ax, df, title, span_storm=False):
    x = df["hour"].values
    ax.bar(x, df["pct_faster"]*100, width=0.42, color="#2ca02c", alpha=0.85,
           label="% Faster (dev > +0.03)")
    ax.bar(x, -df["pct_slower"]*100, width=0.42, color="#d62728", alpha=0.85,
           label="% Slower (dev < -0.03)")
    ax.axhline(0, color="black", lw=0.8)
    if span_storm:
        ax.axvspan(11.5, 14.0, color="lightcoral", alpha=0.15)
        ax.axvline(14+20/60, color="orange", lw=2.0, ls="--", alpha=0.9)
        ax.text(14.3, -38, "S8↑", color="orange", fontweight="bold", fontsize=10)
    ax.set_ylim(-50, 50)
    yt = [-40,-20,0,20,40]
    ax.set_yticks(yt); ax.set_yticklabels([f"{abs(v)}%" for v in yt])
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Hour of day (HKT)")
    ax.set_xticks(range(0, 25, 4))
    ax.grid(axis="y", alpha=0.25)

axes[0].set_ylabel("Share of roads (%)\n← slower      faster →")
plot_div(axes[0], s16, "Sep 16 (Tue) — control workday\nbefore any typhoon")
plot_div(axes[1], s30, "Sep 30 (Tue) — control workday\npost-Ragasa")
plot_div(axes[2], s08, "Oct 08 (Wed) — control workday\nbetween Ragasa & Matmo")
plot_div(axes[3], s23, "Sep 23 (Tue) — Yagiasha pre-S8", span_storm=True)
axes[0].legend(loc="upper right", fontsize=8)

fig.suptitle("Is the 28%/28% bidirectional split at midday unusual?  "
             "Compare Yagiasha (Sep 23) to three normal Tuesdays.",
             fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(f"{OUT}/图51_对照工作日share对比.png", dpi=160, bbox_inches="tight")
print(f"\n  saved -> 图51_对照工作日share对比.png")

# ─── One-line numerical summary ──────────────────────────────────────────────
print("\n" + "="*70)
print("ANSWER: midday (slot 26 = 13:00) bidirectional rate")
print("="*70)
for label, d in [("Sep 16 (control)", s16), ("Sep 30 (control)", s30),
                 ("Oct 08 (control)", s08),
                 ("Sep 23 (Yagiasha pre-S8)", s23)]:
    r = d[d.slot==26]
    if len(r):
        r = r.iloc[0]
        print(f"  {label:30s}  faster {r['pct_faster']*100:5.1f}%  "
              f"slower {r['pct_slower']*100:5.1f}%  mean {r['mean_dev']:+.4f}")

print("\nDone.")
