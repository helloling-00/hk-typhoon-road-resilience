"""
S8-S10-S8 intensity summary: deviation stats by signal level × time period.
Table + boxplot showing speed polarization during typhoon.
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

def tag_signal(dt):
    if dt < pd.Timestamp("2025-09-22 21:40"):
        return "Pre-S3"
    elif dt < pd.Timestamp("2025-09-23 14:20"):
        return "S3 (pre)"
    elif dt < pd.Timestamp("2025-09-24 01:40"):
        return "S8"
    elif dt < pd.Timestamp("2025-09-24 13:20"):
        return "S10"
    elif dt < pd.Timestamp("2025-09-24 20:20"):
        return "S8 (2nd)"
    elif dt < pd.Timestamp("2025-09-25 08:20"):
        return "S3 (after)"
    else:
        return "Post"

def tag_period(dt):
    h = dt.hour + dt.minute / 60
    if 0 <= h < 6:
        return "Night (00-06)"
    elif 6 <= h < 10:
        return "Morning pk (06-10)"
    elif 10 <= h < 16:
        return "Midday (10-16)"
    elif 16 <= h < 20:
        return "PM pk (16-20)"
    else:
        return "Evening (20-24)"

ragasa = ts[(ts["dt"] >= "2025-09-22 21:40") & (ts["dt"] <= "2025-09-25 08:20")].copy()
ragasa["signal"] = ragasa["dt"].apply(tag_signal)
ragasa["period"] = ragasa["dt"].apply(tag_period)

# Control workdays for baseline comparison
CTRL = ["2025-09-16", "2025-09-26", "2025-09-29", "2025-09-30",
        "2025-10-02", "2025-10-06", "2025-10-08", "2025-10-09"]
ts["ds"] = ts["dt"].dt.strftime("%Y-%m-%d")
ctrl = ts[ts["ds"].isin(CTRL)].copy()
ctrl["period"] = ctrl["dt"].apply(tag_period)

# ─── Table 1: By signal level ─────────────────────────────────────────────
print("\n" + "=" * 100)
print("  TABLE 1: Deviation by Signal Level")
print("=" * 100)

signal_order = ["S3 (pre)", "S8", "S10", "S8 (2nd)", "S3 (after)"]

def stats_str(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    faster = (series > DEV_HI).mean() * 100
    slower = (series < DEV_LO).mean() * 100
    neutral = ((series >= DEV_LO) & (series <= DEV_HI)).mean() * 100
    return {
        "n_roads": len(series),
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "IQR": q3 - q1,
        "Faster%": faster,
        "Slower%": slower,
        "Neutral%": neutral,
        "Q1": q1, "Q3": q3,
    }

print(f"  {'Signal':14s} {'n_roads':>8s} {'Mean dev':>9s} {'Median':>8s} {'IQR':>7s} {'Faster%':>8s} {'Slower%':>8s} {'Neutral%':>9s}")
print(f"  {'─'*14} {'─'*8} {'─'*9} {'─'*8} {'─'*7} {'─'*8} {'─'*8} {'─'*9}")

signal_stats = {}
for sig in signal_order:
    sub = ragasa[ragasa["signal"] == sig]
    s = stats_str(sub["dev"])
    signal_stats[sig] = s
    print(f"  {sig:14s} {s['n_roads']:>8,} {s['mean']:>9.4f} {s['median']:>8.4f} {s['IQR']:>7.4f} {s['Faster%']:>7.1f}% {s['Slower%']:>7.1f}% {s['Neutral%']:>8.1f}%")

# Control baseline
ctrl_s = stats_str(ctrl["dev"])
signal_stats["Control"] = ctrl_s
print(f"  {'Control':14s} {ctrl_s['n_roads']:>8,} {ctrl_s['mean']:>9.4f} {ctrl_s['median']:>8.4f} {ctrl_s['IQR']:>7.4f} {ctrl_s['Faster%']:>7.1f}% {ctrl_s['Slower%']:>7.1f}% {ctrl_s['Neutral%']:>8.1f}%")

# ─── Table 2: Signal × Period ─────────────────────────────────────────────
print("\n" + "=" * 100)
print("  TABLE 2: Deviation by Signal × Time Period")
print("=" * 100)

period_order = ["Night (00-06)", "Morning pk (06-10)", "Midday (10-16)", "PM pk (16-20)", "Evening (20-24)"]

print(f"  {'Signal × Period':28s} {'n_roads':>8s} {'Mean dev':>9s} {'Median':>8s} {'IQR':>7s} {'Faster%':>8s} {'Slower%':>8s}")
print(f"  {'─'*28} {'─'*8} {'─'*9} {'─'*8} {'─'*7} {'─'*8} {'─'*8}")

cross_stats = {}
for sig in signal_order:
    for per in period_order:
        sub = ragasa[(ragasa["signal"] == sig) & (ragasa["period"] == per)]
        if len(sub) < 50:
            continue
        s = stats_str(sub["dev"])
        key = f"{sig} × {per}"
        cross_stats[key] = s
        print(f"  {key:28s} {s['n_roads']:>8,} {s['mean']:>9.4f} {s['median']:>8.4f} {s['IQR']:>7.4f} {s['Faster%']:>7.1f}% {s['Slower%']:>7.1f}%")

# Control by period
for per in period_order:
    sub = ctrl[ctrl["period"] == per]
    s = stats_str(sub["dev"])
    print(f"  {'Control × ' + per:28s} {s['n_roads']:>8,} {s['mean']:>9.4f} {s['median']:>8.4f} {s['IQR']:>7.4f} {s['Faster%']:>7.1f}% {s['Slower%']:>7.1f}%")

# ─── Boxplot: deviation distribution by signal ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: by signal level
ax = axes[0]
sig_data = []
sig_labels = []
for sig in signal_order:
    devs = ragasa[ragasa["signal"] == sig]["dev"].dropna()
    if len(devs) > 0:
        sig_data.append(devs.values)
        sig_labels.append(sig)

bp = ax.boxplot(sig_data, labels=sig_labels, patch_artist=True, widths=0.55,
                showfliers=False, medianprops={"color": "black", "lw": 1.5})
colors_sig = ["#ffd54f", "#ef9a9a", "#c62828", "#ef9a9a", "#ffd54f"]
for patch, c in zip(bp["boxes"], colors_sig):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.axhline(0, color="black", lw=0.7, ls="--")
ax.axhline(DEV_HI, color="#2ca02c", lw=0.7, ls=":")
ax.axhline(DEV_LO, color="#d62728", lw=0.7, ls=":")
ax.set_ylabel("Deviation from Baseline", fontsize=11)
ax.set_title("Speed Deviation by Signal Level", fontsize=12, fontweight="bold")
ax.tick_params(axis="x", rotation=20)

# Right: by period (pool Typhoon S8+S10)
ax = axes[1]
typhoon = ragasa[ragasa["signal"].isin(["S8", "S10", "S8 (2nd)"])]
per_data, per_labels = [], []
for per in period_order:
    devs = typhoon[typhoon["period"] == per]["dev"].dropna()
    if len(devs) > 0:
        per_data.append(devs.values)
        per_labels.append(per)

bp2 = ax.boxplot(per_data, labels=per_labels, patch_artist=True, widths=0.55,
                 showfliers=False, medianprops={"color": "black", "lw": 1.5})
for patch in bp2["boxes"]:
    patch.set_facecolor("#ef5350")
    patch.set_alpha(0.6)
ax.axhline(0, color="black", lw=0.7, ls="--")
ax.axhline(DEV_HI, color="#2ca02c", lw=0.7, ls=":")
ax.axhline(DEV_LO, color="#d62728", lw=0.7, ls=":")
ax.set_ylabel("Deviation from Baseline", fontsize=11)
ax.set_title("S8/S10 Speed Deviation by Time of Day", fontsize=12, fontweight="bold")
ax.tick_params(axis="x", rotation=20)

fig.suptitle("Typhoon Ragasa: During-Event Speed Distribution  —  S8 → S10 → S8",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out1 = f"{OUT}/图60a_intensity_boxplot.png"
fig.savefig(out1, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n  saved -> {out1}")

# ─── Save CSV ─────────────────────────────────────────────────────────────
rows = []
for sig in signal_order:
    s = signal_stats[sig]
    rows.append({"signal": sig, **s})
s = signal_stats["Control"]
rows.append({"signal": "Control", **s})
pd.DataFrame(rows).to_csv(f"{OUT}/preS8_intensity_by_signal.csv", index=False)

rows2 = []
for key, s in cross_stats.items():
    rows2.append({"group": key, **s})
pd.DataFrame(rows2).to_csv(f"{OUT}/preS8_intensity_cross.csv", index=False)
print("  CSVs saved")

# ─── Key insight ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  KEY NUMBERS")
print(f"{'='*60}")
s8s10 = ragasa[ragasa["signal"].isin(["S8", "S10", "S8 (2nd)"])]
s3pre = ragasa[ragasa["signal"] == "S3 (pre)"]
print(f"  S8/S10 pooled: mean dev = {s8s10['dev'].mean():.4f}, "
      f"Faster = {(s8s10['dev'] > DEV_HI).mean()*100:.1f}%, "
      f"Slower = {(s8s10['dev'] < DEV_LO).mean()*100:.1f}%")
print(f"  S3 (pre):      mean dev = {s3pre['dev'].mean():.4f}, "
      f"Faster = {(s3pre['dev'] > DEV_HI).mean()*100:.1f}%, "
      f"Slower = {(s3pre['dev'] < DEV_LO).mean()*100:.1f}%")
print(f"  Control:       mean dev = {ctrl['dev'].mean():.4f}, "
      f"Faster = {(ctrl['dev'] > DEV_HI).mean()*100:.1f}%, "
      f"Slower = {(ctrl['dev'] < DEV_LO).mean()*100:.1f}%")

# Polarization check: IQR
print(f"\n  IQR: S3={signal_stats['S3 (pre)']['IQR']:.4f}, "
      f"S8={signal_stats['S8']['IQR']:.4f}, "
      f"S10={signal_stats['S10']['IQR']:.4f}, "
      f"Ctrl={ctrl_s['IQR']:.4f}")

print("\nDone.")
