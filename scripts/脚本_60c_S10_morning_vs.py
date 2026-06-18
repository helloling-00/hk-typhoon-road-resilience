"""
S10 morning peak vs S3 morning peak vs control mornings.
Plus S8 PM peak vs control PM peak for comparison.
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

CTRL = ["2025-09-16", "2025-09-26", "2025-09-29", "2025-09-30",
        "2025-10-02", "2025-10-06", "2025-10-08", "2025-10-09"]

# ─── Load ─────────────────────────────────────────────────────────────────
print("Loading...", flush=True)
ts = pd.read_parquet(f"{DATA}/yagiasha_road_timeseries.parquet")
ts["dt"] = pd.to_datetime(ts["dt"])
ts["ds"] = ts["dt"].dt.strftime("%Y-%m-%d")
ts["hour"] = ts["dt"].dt.hour

# Define periods
# S10 morning: Sep 24, 06:00-10:00 (slots 12-19)
s10_morn = ts[(ts["ds"] == "2025-09-24") & (ts["hour"] >= 6) & (ts["hour"] < 10)].copy()
# S3 morning: Sep 23, 06:00-10:00
s3_morn = ts[(ts["ds"] == "2025-09-23") & (ts["hour"] >= 6) & (ts["hour"] < 10)].copy()
# Control mornings
ctrl_morn = ts[(ts["ds"].isin(CTRL)) & (ts["hour"] >= 6) & (ts["hour"] < 10)].copy()
# S8 PM: Sep 23, 16:00-20:00 (slots 32-39)
s8_pm = ts[(ts["ds"] == "2025-09-23") & (ts["hour"] >= 16) & (ts["hour"] < 20)].copy()
# Control PM
ctrl_pm = ts[(ts["ds"].isin(CTRL)) & (ts["hour"] >= 16) & (ts["hour"] < 20)].copy()

def stats_str(series, label):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    f = (series > DEV_HI).mean() * 100
    s = (series < DEV_LO).mean() * 100
    n = ((series >= DEV_LO) & (series <= DEV_HI)).mean() * 100
    print(f"  {label:30s} n={len(series):>8,}  mean={series.mean():>8.4f}  median={series.median():>8.4f}  "
          f"IQR={q3-q1:>6.4f}  F={f:>5.1f}%  S={s:>5.1f}%  N={n:>5.1f}%")
    return {"n": len(series), "mean": series.mean(), "median": series.median(),
            "IQR": q3 - q1, "F%": f, "S%": s, "N%": n}

# ─── Per-road aggregation for paired comparison ───────────────────────────
print("\n=== Pooled stats ===")
s_all = {}
for label, data in [("S10 Morning", s10_morn), ("S3 Morning", s3_morn),
                     ("Ctrl Morning", ctrl_morn),
                     ("S8 PM", s8_pm), ("Ctrl PM", ctrl_pm)]:
    s_all[label] = stats_str(data["dev"], label)

# ─── Per-road: S10 morning vs S3 morning (paired) ────────────────────────
print("\n=== Per-road paired: S10 vs S3 morning ===")

s10_rd = s10_morn.groupby("road_id")["dev"].mean().reset_index()
s10_rd.columns = ["road_id", "s10_dev"]
s3_rd = s3_morn.groupby("road_id")["dev"].mean().reset_index()
s3_rd.columns = ["road_id", "s3_dev"]
ctrl_rd = ctrl_morn.groupby("road_id")["dev"].mean().reset_index()
ctrl_rd.columns = ["road_id", "ctrl_dev"]

pair = s10_rd.merge(s3_rd, on="road_id", how="inner").merge(ctrl_rd, on="road_id", how="inner")
pair["delta_s10_s3"] = pair["s10_dev"] - pair["s3_dev"]
pair["state_s10"] = pair["s10_dev"].apply(
    lambda d: "F" if d > DEV_HI else ("S" if d < DEV_LO else "N"))
pair["state_s3"] = pair["s3_dev"].apply(
    lambda d: "F" if d > DEV_HI else ("S" if d < DEV_LO else "N"))

print(f"  Paired roads: {len(pair)}")
print(f"  S10 mean dev: {pair['s10_dev'].mean():.4f}, S3 mean dev: {pair['s3_dev'].mean():.4f}")
print(f"  Delta (S10-S3): mean={pair['delta_s10_s3'].mean():.4f}, "
      f"median={pair['delta_s10_s3'].median():.4f}")

# Transition matrix S3 → S10
print(f"\n  === S3 morning state → S10 morning state ===")
xtab = pd.crosstab(pair["state_s3"], pair["state_s10"], normalize="index") * 100
print(xtab.round(1).to_string())
print(f"\n  Counts:")
print(pd.crosstab(pair["state_s3"], pair["state_s10"]).to_string())

# ─── Per-road: S8 PM vs Ctrl PM (paired) ─────────────────────────────────
print("\n=== Per-road paired: S8 PM vs Ctrl PM ===")
s8pm_rd = s8_pm.groupby("road_id")["dev"].mean().reset_index()
s8pm_rd.columns = ["road_id", "s8pm_dev"]
cpm_rd = ctrl_pm.groupby("road_id")["dev"].mean().reset_index()
cpm_rd.columns = ["road_id", "ctrlpm_dev"]

pair_pm = s8pm_rd.merge(cpm_rd, on="road_id", how="inner")
pair_pm["delta"] = pair_pm["s8pm_dev"] - pair_pm["ctrlpm_dev"]
pair_pm["state_s8"] = pair_pm["s8pm_dev"].apply(
    lambda d: "F" if d > DEV_HI else ("S" if d < DEV_LO else "N"))
pair_pm["state_ctrl"] = pair_pm["ctrlpm_dev"].apply(
    lambda d: "F" if d > DEV_HI else ("S" if d < DEV_LO else "N"))

print(f"  Paired roads: {len(pair_pm)}")
print(f"  S8 PM mean dev: {pair_pm['s8pm_dev'].mean():.4f}, Ctrl PM mean dev: {pair_pm['ctrlpm_dev'].mean():.4f}")
print(f"  Delta: mean={pair_pm['delta'].mean():.4f}")

print(f"\n  === Ctrl PM state → S8 PM state ===")
xtab_pm = pd.crosstab(pair_pm["state_ctrl"], pair_pm["state_s8"], normalize="index") * 100
print(xtab_pm.round(1).to_string())
print(f"\n  Counts:")
print(pd.crosstab(pair_pm["state_ctrl"], pair_pm["state_s8"]).to_string())

# ─── Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 1. Distribution comparison: S10 vs S3 vs Ctrl morning
ax = axes[0]
for label, data, color, ls in [
    ("Ctrl Morning", ctrl_morn["dev"], "#7f7f7f", "--"),
    ("S3 Morning", s3_morn["dev"], "#ffd54f", "-"),
    ("S10 Morning", s10_morn["dev"], "#c62828", "-"),
]:
    vals = data.dropna()
    hist, bins = np.histogram(vals, bins=60, range=(-0.2, 0.3), density=True)
    ax.plot((bins[:-1] + bins[1:]) / 2, hist, color=color, ls=ls, lw=2, label=label, alpha=0.9)
ax.axvline(DEV_HI, color="#2ca02c", lw=0.7, ls=":")
ax.axvline(DEV_LO, color="#d62728", lw=0.7, ls=":")
ax.axvline(0, color="black", lw=0.7, ls="-")
ax.set_xlabel("Deviation from Baseline")
ax.set_ylabel("Density")
ax.set_title("Morning Peak Speed Distribution\nS3 vs S10 vs Control")
ax.legend(fontsize=9)

# 2. S3→S10 transition scatter
ax = axes[1]
ax.scatter(pair["s3_dev"], pair["s10_dev"], alpha=0.15, s=3, color="#333")
ax.plot([-0.3, 0.4], [-0.3, 0.4], "k--", lw=0.8)
ax.axhline(0, color="black", lw=0.5)
ax.axvline(0, color="black", lw=0.5)
ax.axhline(DEV_HI, color="#2ca02c", lw=0.5, ls=":")
ax.axvline(DEV_HI, color="#2ca02c", lw=0.5, ls=":")
ax.axhline(DEV_LO, color="#d62728", lw=0.5, ls=":")
ax.axvline(DEV_LO, color="#d62728", lw=0.5, ls=":")
ax.set_xlabel("S3 Morning Deviation (Sep 23)")
ax.set_ylabel("S10 Morning Deviation (Sep 24)")
ax.set_title(f"S3 → S10 Morning: Per-Road Change\n"
             f"n={len(pair):,}, Δmean={pair['delta_s10_s3'].mean():.4f}")
ax.set_xlim(-0.30, 0.45)
ax.set_ylim(-0.30, 0.45)

# Annotate quadrants
f_s3 = (pair["s3_dev"] > DEV_HI).mean() * 100
f_s10 = (pair["s10_dev"] > DEV_HI).mean() * 100
s_s3 = (pair["s3_dev"] < DEV_LO).mean() * 100
s_s10 = (pair["s10_dev"] < DEV_LO).mean() * 100
ax.text(0.35, 0.35, f"F→F\n{f_s3:.0f}%→{f_s10:.0f}%", fontsize=8, ha="center", color="#2ca02c")
ax.text(-0.25, 0.35, f"S→F\n{s_s3:.0f}%→?", fontsize=8, ha="center", color="#1f77b4")

# 3. S8 PM vs Ctrl PM scatter
ax = axes[2]
ax.scatter(pair_pm["ctrlpm_dev"], pair_pm["s8pm_dev"], alpha=0.15, s=3, color="#333")
ax.plot([-0.3, 0.4], [-0.3, 0.4], "k--", lw=0.8)
ax.axhline(0, color="black", lw=0.5)
ax.axvline(0, color="black", lw=0.5)
ax.axhline(DEV_HI, color="#2ca02c", lw=0.5, ls=":")
ax.axvline(DEV_HI, color="#2ca02c", lw=0.5, ls=":")
ax.axhline(DEV_LO, color="#d62728", lw=0.5, ls=":")
ax.axvline(DEV_LO, color="#d62728", lw=0.5, ls=":")
ax.set_xlabel("Control PM Deviation")
ax.set_ylabel("S8 PM Deviation (Sep 23)")
ax.set_title(f"S8 PM vs Control PM: Per-Road Change\n"
             f"n={len(pair_pm):,}, Δmean={pair_pm['delta'].mean():.4f}")
ax.set_xlim(-0.30, 0.45)
ax.set_ylim(-0.30, 0.45)

fig.suptitle("Typhoon Ragasa: S10 Morning vs S3 Morning vs S8 PM Peak",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out1 = f"{OUT}/图60c_S10_morning_comparison.png"
fig.savefig(out1, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n  saved -> {out1}")

# ─── Key numbers for the paper ────────────────────────────────────────────
print(f"\n{'='*60}")
print("  KEY NUMBERS FOR PAPER")
print(f"{'='*60}")
# S3 morning clearance
s3m_f = (pair["s3_dev"] > DEV_HI).mean() * 100
s3m_s = (pair["s3_dev"] < DEV_LO).mean() * 100
s10m_f = (pair["s10_dev"] > DEV_HI).mean() * 100
s10m_s = (pair["s10_dev"] < DEV_LO).mean() * 100
print(f"  S3 morning:  F={s3m_f:.1f}%, S={s3m_s:.1f}%, mean dev={pair['s3_dev'].mean():.4f}")
print(f"  S10 morning: F={s10m_f:.1f}%, S={s10m_s:.1f}%, mean dev={pair['s10_dev'].mean():.4f}")
print(f"  Change: F +{s10m_f-s3m_f:.1f}pp, S {s10m_s-s3m_s:+.1f}pp, mean dev {pair['delta_s10_s3'].mean():+.4f}")

s8pm_f = (pair_pm["s8pm_dev"] > DEV_HI).mean() * 100
s8pm_s = (pair_pm["s8pm_dev"] < DEV_LO).mean() * 100
cpm_f = (pair_pm["ctrlpm_dev"] > DEV_HI).mean() * 100
cpm_s = (pair_pm["ctrlpm_dev"] < DEV_LO).mean() * 100
print(f"\n  Ctrl PM: F={cpm_f:.1f}%, S={cpm_s:.1f}%, mean dev={pair_pm['ctrlpm_dev'].mean():.4f}")
print(f"  S8 PM:   F={s8pm_f:.1f}%, S={s8pm_s:.1f}%, mean dev={pair_pm['s8pm_dev'].mean():.4f}")
print(f"  Change:  F +{s8pm_f-cpm_f:.1f}pp, S {s8pm_s-cpm_s:+.1f}pp, mean dev {pair_pm['delta'].mean():+.4f}")

print("\nDone.")
