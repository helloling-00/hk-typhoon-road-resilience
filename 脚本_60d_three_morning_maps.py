"""
Morning peak relative speed comparison: Ctrl vs S3 vs S10.
Line chart: X = time slots (06:00-10:00), Y = mean relative speed.
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
OUT  = "/Users/helloling/workspace/thesis"

CTRL = ["2025-09-16", "2025-09-26", "2025-09-29", "2025-09-30",
        "2025-10-02", "2025-10-06", "2025-10-08", "2025-10-09"]

# ─── Load ─────────────────────────────────────────────────────────────────
print("Loading...", flush=True)
ts = pd.read_parquet(f"{DATA}/yagiasha_road_timeseries.parquet")
ts["ds"] = pd.to_datetime(ts["dt"]).dt.strftime("%Y-%m-%d")
ts["hour"] = ts["dt"].dt.hour + ts["dt"].dt.minute / 60
ts["rel_speed"] = ts["obs"] / ts["bl"]

# Morning peak: 06:00-10:00 (slots 12-19), half-hour bins
def half_hour_slot(hour_float):
    return int(hour_float * 2)  # 6.0→12, 6.5→13, ...

ts["slot_bin"] = ts["hour"].apply(half_hour_slot)

# Control: aggregate per slot across all control days
ctrl_morn = ts[(ts["ds"].isin(CTRL)) & (ts["slot_bin"] >= 12) & (ts["slot_bin"] <= 19)].copy()
ctrl_line = ctrl_morn.groupby("slot_bin")["rel_speed"].mean()
ctrl_ste = ctrl_morn.groupby("slot_bin")["rel_speed"].sem()

# S3: Sep 23 morning
s3_morn = ts[(ts["ds"] == "2025-09-23") & (ts["slot_bin"] >= 12) & (ts["slot_bin"] <= 19)].copy()
s3_line = s3_morn.groupby("slot_bin")["rel_speed"].mean()
s3_ste = s3_morn.groupby("slot_bin")["rel_speed"].sem()

# S10: Sep 24 morning
s10_morn = ts[(ts["ds"] == "2025-09-24") & (ts["slot_bin"] >= 12) & (ts["slot_bin"] <= 19)].copy()
s10_line = s10_morn.groupby("slot_bin")["rel_speed"].mean()
s10_ste = s10_morn.groupby("slot_bin")["rel_speed"].sem()

# ─── Plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
fig.subplots_adjust(left=0.11, right=0.97, top=0.90, bottom=0.13)

slots = np.arange(12, 20)  # 12-19
time_labels = ["06:00", "06:30", "07:00", "07:30", "08:00", "08:30", "09:00", "09:30"]

# Control
ax.plot(slots, ctrl_line.values, color="#7f7f7f", lw=2.2, marker="o", markersize=7,
        label="Normal Workday (8 days mean)", zorder=2)
ax.fill_between(slots, ctrl_line.values - ctrl_ste.values,
                ctrl_line.values + ctrl_ste.values,
                color="#7f7f7f", alpha=0.15)

# S3
ax.plot(slots, s3_line.values, color="#ffa000", lw=2.5, marker="s", markersize=8,
        label="S3 Morning (Sep 23) — Pre-S8", zorder=3)
ax.fill_between(slots, s3_line.values - s3_ste.values,
                s3_line.values + s3_ste.values,
                color="#ffa000", alpha=0.12)

# S10
ax.plot(slots, s10_line.values, color="#c62828", lw=2.8, marker="D", markersize=8,
        label="S10 Morning (Sep 24) — Hurricane", zorder=4)
ax.fill_between(slots, s10_line.values - s10_ste.values,
                s10_line.values + s10_ste.values,
                color="#c62828", alpha=0.12)

ax.axhline(1.0, color="black", lw=1.2, ls="--", alpha=0.6, zorder=0)
ax.set_xticks(slots)
ax.set_xticklabels(time_labels, fontsize=10)
ax.set_xlabel("Time (HKT)", fontsize=11)
ax.set_ylabel("Relative Speed (observed / baseline)", fontsize=11)
ax.set_ylim(0.92, 1.18)
ax.grid(alpha=0.2, lw=0.5)
ax.legend(fontsize=10, loc="upper left", framealpha=0.9, edgecolor="#ccc")
ax.set_title("Morning Peak Relative Speed: Normal vs S3 vs S10\n"
             "Ragasa, 06:00–10:00 HKT",
             fontsize=13, fontweight="bold")

plt.tight_layout()
out1 = f"{OUT}/图60d_morning_speed_lines.png"
fig.savefig(out1, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  saved -> {out1}")

# ─── Print values ─────────────────────────────────────────────────────────
print("\n=== Relative speed by slot ===")
print(f"  Time    Ctrl     S3       S10")
for slot in slots:
    c = ctrl_line.get(slot, np.nan)
    s3 = s3_line.get(slot, np.nan)
    s10 = s10_line.get(slot, np.nan)
    print(f"  {time_labels[slot-12]:5s}  {c:.4f}   {s3:.4f}   {s10:.4f}")

print(f"\n  Ctrl mean: {ctrl_line.mean():.4f}")
print(f"  S3 mean:   {s3_line.mean():.4f}")
print(f"  S10 mean:  {s10_line.mean():.4f}")
print("Done.")
