"""
Three morning peaks side-by-side numerical analysis.
"""
import os, glob
import pandas as pd, numpy as np
from shapely import wkb as shapely_wkb
import warnings; warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"
MORN_SLOTS = list(range(12, 23))  # 06:00 -> 11:00

bl = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
bl_idx = bl.set_index(["day_type","slot","road_id"])["mean_speed"]

def get_ep_key(wb):
    try:
        g = shapely_wkb.loads(bytes(wb))
        coords = list(g.coords) if g.geom_type=="LineString" else \
                 [c for line in g.geoms for c in line.coords]
        s = (round(coords[0][0],4), round(coords[0][1],4))
        e = (round(coords[-1][0],4), round(coords[-1][1],4))
        return str((min(s,e), max(s,e)))
    except: return None

def load_slot(day, slot, day_type):
    pat = f"{FLOW}/{day}/traffic_flow_zoom15_{day}_slot{slot:02d}_*.parquet"
    fs = glob.glob(pat)
    if not fs: return None
    df = pd.read_parquet(fs[0],
                         columns=["relative_speed","geometry","road_closure"])
    df = df[df["road_closure"]!=1].dropna(subset=["relative_speed"])
    if len(df) < 50: return None
    df["ep_key"] = df["geometry"].apply(get_ep_key)
    df = df.merge(ep[["ep_key","road_id"]], on="ep_key", how="inner")
    if len(df) < 50: return None
    obs = df.groupby("road_id")["relative_speed"].mean()
    idx = pd.MultiIndex.from_arrays(
        [[day_type]*len(obs), [slot]*len(obs), obs.index],
        names=["day_type","slot","road_id"])
    bl_vals = bl_idx.reindex(idx).values
    valid = ~np.isnan(bl_vals)
    if valid.sum() < 50: return None
    return {"slot": slot, "n_roads": int(valid.sum()),
            "mean_speed": obs[valid].mean(),
            "mean_baseline": bl_vals[valid].mean()}

def load_day(day, day_type):
    out=[]
    for s in MORN_SLOTS:
        r=load_slot(day,s,day_type)
        if r is not None: out.append(r)
    return pd.DataFrame(out)

s3  = load_day("2025-09-23", "WORKDAY")
s10 = load_day("2025-09-24", "WORKDAY")
bl_curve = (bl_idx.loc["WORKDAY"].groupby("slot").mean()
            .reindex(MORN_SLOTS).reset_index()
            .rename(columns={"mean_speed":"bl"}))

def stt(s):
    h,m=divmod(s*30,60); return f"{h:02d}:{m:02d}"

merged = bl_curve.merge(s3[["slot","mean_speed","n_roads"]].rename(columns={"mean_speed":"S3","n_roads":"n_S3"}),on="slot",how="left")
merged = merged.merge(s10[["slot","mean_speed","n_roads"]].rename(columns={"mean_speed":"S10","n_roads":"n_S10"}),on="slot",how="left")
merged["time"]=merged["slot"].map(stt)
merged["d_S3"]  = merged["S3"]  - merged["bl"]
merged["d_S10"] = merged["S10"] - merged["bl"]
merged["d_S10_S3"] = merged["S10"] - merged["S3"]

cols=["time","bl","S3","S10","d_S3","d_S10","d_S10_S3","n_S3","n_S10"]
print("\n=== Morning peak slot-by-slot ===")
print(merged[cols].to_string(index=False,
      formatters={"bl":"{:.3f}".format,"S3":"{:.3f}".format,"S10":"{:.3f}".format,
                  "d_S3":"{:+.3f}".format,"d_S10":"{:+.3f}".format,
                  "d_S10_S3":"{:+.3f}".format,
                  "n_S3":"{:.0f}".format,"n_S10":"{:.0f}".format}))

print("\n=== Aggregate (06:00-11:00) ===")
print(f"  Baseline mean : {merged['bl'].mean():.3f}   min={merged['bl'].min():.3f} @ {merged.loc[merged['bl'].idxmin(),'time']}")
print(f"  S3 mean       : {merged['S3'].mean():.3f}   min={merged['S3'].min():.3f} @ {merged.loc[merged['S3'].idxmin(),'time']}")
print(f"  S10 mean      : {merged['S10'].mean():.3f}   min={merged['S10'].min():.3f} @ {merged.loc[merged['S10'].idxmin(),'time']}")
print(f"  d(S3-bl) mean : {merged['d_S3'].mean():+.3f}   max={merged['d_S3'].max():+.3f} @ {merged.loc[merged['d_S3'].idxmax(),'time']}")
print(f"  d(S10-bl) mean: {merged['d_S10'].mean():+.3f}   max={merged['d_S10'].max():+.3f} @ {merged.loc[merged['d_S10'].idxmax(),'time']}")

# 07:30 (slot 15) baseline trough  vs  S3 / S10 at the same slot
trough_slot = merged.loc[merged["bl"].idxmin(),"slot"]
ts = merged[merged["slot"]==trough_slot].iloc[0]
print(f"\n=== Baseline trough @ {ts['time']} ===")
print(f"  baseline = {ts['bl']:.3f}")
print(f"  S3       = {ts['S3']:.3f}  ({ts['d_S3']:+.3f},  +{(ts['S3']-ts['bl'])/ts['bl']*100:.1f}%)")
print(f"  S10      = {ts['S10']:.3f}  ({ts['d_S10']:+.3f}, +{(ts['S10']-ts['bl'])/ts['bl']*100:.1f}%)")

# baseline drop magnitude vs S3 / S10 drop magnitude
def drop(series):
    return series.iloc[0] - series.min()
print(f"\n=== Morning-peak drop (06:00 → trough) ===")
print(f"  baseline drop = {drop(merged['bl']):.3f}")
print(f"  S3 drop       = {drop(merged['S3']):.3f}")
print(f"  S10 drop      = {drop(merged['S10']):.3f}")

merged.to_csv(f"{DATA}/../morning_compare_S3_S10_baseline.csv", index=False)
print(f"\nCSV saved.")
