"""
Replicates ecdf.jpg and speed_shape figures:
  - ECDF of Δ: Bottom-5%-baseline roads vs Others, by signal level
  - Speed shape (p5): event vs baseline p5, by signal level and typhoon
"""
import os, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely import wkb as shapely_wkb
from datetime import datetime
import warnings; warnings.filterwarnings("ignore")

plt.rcParams.update({"figure.dpi":140,"savefig.dpi":260,"font.size":13,
                     "axes.titlesize":14,"axes.labelsize":13,
                     "xtick.labelsize":11,"ytick.labelsize":11,
                     "lines.linewidth":2.0,"legend.fontsize":11})

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"
OUT  = "/Users/helloling/workspace/thesis"

ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
bl = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
bl_idx = bl.set_index(["day_type","slot","road_id"])["mean_speed"]

# ── signal timeline ───────────────────────────────────────────────────────────
signal_schedule = [
    ("Yagiasha", "WORKDAY", [
        (datetime(2025,9,22,12,20), datetime(2025,9,22,21,40), 1),
        (datetime(2025,9,22,21,40), datetime(2025,9,23,14,20), 3),
        (datetime(2025,9,23,14,20), datetime(2025,9,24, 1,40), 8),
        (datetime(2025,9,24, 1,40), datetime(2025,9,24, 2,40), 9),
        (datetime(2025,9,24, 2,40), datetime(2025,9,24,13,20),10),
        (datetime(2025,9,24,13,20), datetime(2025,9,24,20,20), 8),
        (datetime(2025,9,24,20,20), datetime(2025,9,25, 8,20), 3),
        (datetime(2025,9,25, 8,20), datetime(2025,9,25,11,20), 1),
    ]),
    ("Mina", "WORKDAY", [
        (datetime(2025,9,17,21,20), datetime(2025,9,19, 9,20), 1),
        (datetime(2025,9,19, 9,20), datetime(2025,9,20, 9,20), 3),
        (datetime(2025,9,20, 9,20), datetime(2025,9,20,10,40), 1),
    ]),
    ("Madum", "WORKDAY", [
        (datetime(2025,10,3,19,40), datetime(2025,10,4,12,20), 1),
        (datetime(2025,10,4,12,20), datetime(2025,10,5,15,40), 3),
        (datetime(2025,10,5,15,40), datetime(2025,10,5,22,20), 1),
    ]),
]
def slot_to_dt(day_str, slot):
    base = datetime.strptime(day_str, "%Y-%m-%d")
    return base.replace(hour=slot//2, minute=30*(slot%2))

def get_signal(ts):
    for _, _, phases in signal_schedule:
        for start, end, sig in phases:
            if start <= ts < end:
                return sig
    return 0

def get_ep_key(wb):
    try:
        g = shapely_wkb.loads(bytes(wb))
        coords = list(g.coords) if g.geom_type=="LineString" else \
                 [c for line in g.geoms for c in line.coords]
        s=(round(coords[0][0],4),round(coords[0][1],4))
        e=(round(coords[-1][0],4),round(coords[-1][1],4))
        return str((min(s,e),max(s,e)))
    except: return None

ep_lkp = ep.set_index("ep_key")["road_id"]

# ── road baseline rank ────────────────────────────────────────────────────────
road_mean_bl = bl.groupby("road_id")["mean_speed"].mean()
thresh = road_mean_bl.quantile(0.05)
bottom5 = set(road_mean_bl[road_mean_bl <= thresh].index)
print(f"Bottom-5% threshold: {thresh:.4f}  ({len(bottom5):,} roads)")

# ── collect typhoon period observations ──────────────────────────────────────
typhoon_days = {
    "Yagiasha": ["2025-09-22","2025-09-23","2025-09-24","2025-09-25"],
    "Mina":     ["2025-09-17","2025-09-18","2025-09-19","2025-09-20"],
    "Madum":    ["2025-10-03","2025-10-04","2025-10-05"],
}

records = []   # (typhoon, signal, road_id, obs, baseline, delta)
p5_records = []  # (typhoon, signal, slot_ts, p5_event, p5_baseline)

for typhoon, days in typhoon_days.items():
    day_type = "WORKDAY"
    print(f"\n{typhoon}:")
    for day in days:
        folder = f"{FLOW}/{day}"
        if not os.path.exists(folder): continue
        for slot in range(48):
            ts = slot_to_dt(day, slot)
            sig = get_signal(ts)
            if sig == 0: continue
            files = [f for f in os.listdir(folder) if f"_slot{slot:02d}_" in f]
            if not files: continue
            try:
                df = pd.read_parquet(f"{folder}/{files[0]}",
                                     columns=["relative_speed","geometry","road_closure"])
            except: continue
            df = df[df["road_closure"]!=1].copy()
            if len(df)<50: continue

            df["ep_key"] = df["geometry"].apply(get_ep_key)
            df = df.merge(ep[["ep_key","road_id"]], on="ep_key", how="inner")
            if len(df)==0: continue
            agg = df.groupby("road_id")["relative_speed"].mean().reset_index()
            agg.columns = ["road_id","obs"]

            # baseline join
            idx = pd.MultiIndex.from_arrays(
                [[day_type]*len(agg),[slot]*len(agg),agg["road_id"]],
                names=["day_type","slot","road_id"])
            agg["baseline"] = bl_idx.reindex(idx).values
            agg = agg.dropna(subset=["baseline"])
            agg["delta"] = agg["obs"] - agg["baseline"]

            # p5 for speed_shape
            p5_ev = agg["obs"].quantile(0.05)
            p5_bl = agg["baseline"].quantile(0.05)
            p5_records.append({"typhoon":typhoon,"signal":sig,"ts":ts,
                                "p5_event":p5_ev,"p5_baseline":p5_bl})

            for _, row in agg.iterrows():
                records.append({"typhoon":typhoon,"signal":sig,
                                 "road_id":int(row["road_id"]),
                                 "obs":row["obs"],"baseline":row["baseline"],
                                 "delta":row["delta"]})
        print(f"  {day}: {len(records)} obs so far")

df_all = pd.DataFrame(records)
df_p5  = pd.DataFrame(p5_records)
print(f"\nTotal typhoon observations: {len(df_all):,}")
df_all["group"] = df_all["road_id"].apply(lambda r: "Bottom 5%" if r in bottom5 else "Others")
print(df_all.groupby(["group","signal"])["delta"].describe()[["mean","50%","count"]].to_string())

# ── Figure 1: ECDF ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1,2, figsize=(13,5), sharey=True)
fig.suptitle("ECDF of Δ during typhoons: slow roads (Bottom 5%) vs others",
             fontsize=14, fontweight="bold")

for ax, (label, mask) in zip(axes, [
    ("3 ≤ Signal < 8", (df_all["signal"]>=3)&(df_all["signal"]<8)),
    ("Signal ≥ 8",     df_all["signal"]>=8),
]):
    sub = df_all[mask]
    for grp, color in [("Bottom 5%","#1565C0"),("Others","#E65100")]:
        vals = np.sort(sub[sub["group"]==grp]["delta"].values)
        ecdf = np.arange(1, len(vals)+1)/len(vals)
        ax.plot(vals, ecdf, color=color, lw=2, label=grp)
    ax.axvline(0, color="steelblue", lw=1.2, ls="--", alpha=0.7)
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_xlabel("Δ = agg_speed − baseline_speed")
    ax.set_ylabel("ECDF")
    ax.set_xlim(-1,1); ax.set_ylim(0,1)
    ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(f"{OUT}/图14_ECDF慢路vs其他.png", dpi=260, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved 图14")

# ── Figure 2: speed_shape (p5 event vs baseline, by signal) ─────────────────
fig, axes = plt.subplots(3,1, figsize=(14,13), sharex=False)
fig.suptitle("Speed Shape (p5): Event vs Baseline by Signal Level", fontsize=14, fontweight="bold")
colors_sig = {1:"#4CAF50", 3:"#2196F3", 8:"#FF9800", 9:"#FF5722", 10:"#D32F2F"}
markers_sig = {1:"o", 3:"s", 8:"^", 9:"D", 10:"*"}

for ax, typhoon in zip(axes, ["Yagiasha","Mina","Madum"]):
    sub = df_p5[df_p5["typhoon"]==typhoon].sort_values("ts")
    for sig in sorted(sub["signal"].unique()):
        g = sub[sub["signal"]==sig]
        c = colors_sig.get(sig,"gray")
        m = markers_sig.get(sig,"o")
        ax.plot(g["ts"], g["p5_event"],    color=c, marker=m, ms=5, lw=1.8,
                label=f"Signal {sig} - Event p5")
        ax.plot(g["ts"], g["p5_baseline"], color=c, marker=m, ms=5, lw=1.8,
                ls="--", alpha=0.55, label=f"Signal {sig} - Baseline p5")
    ax.set_title(f"Speed Shape (p5) by Signal Level — {typhoon}", fontsize=12)
    ax.set_ylabel("relative_speed (p5)")
    ax.legend(fontsize=9, ncol=3)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(
        matplotlib.dates.DateFormatter("%m-%d %H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")

plt.tight_layout()
fig.savefig(f"{OUT}/图15_速度形状p5对比.png", dpi=260, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved 图15")

# ── Summary statistics: by quintile ──────────────────────────────────────────
df_all["quintile"] = pd.qcut(
    df_all["road_id"].map(road_mean_bl), q=5,
    labels=["Q1 (slowest)","Q2","Q3","Q4","Q5 (fastest)"])
qsum = df_all.groupby(["quintile","signal"])["delta"].agg(
    mean_delta="mean", pct_pos=lambda x:(x>0.02).mean(),
    pct_neg=lambda x:(x<-0.02).mean(), n="count").reset_index()
print("\n=== Quintile × Signal breakdown ===")
for sig in [3,8,10]:
    print(f"\n  Signal {sig}:")
    print(qsum[qsum["signal"]==sig][["quintile","mean_delta","pct_pos","pct_neg","n"]].to_string(index=False))
