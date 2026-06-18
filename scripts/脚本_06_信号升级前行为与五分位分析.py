"""
两项专项分析：
1. 信号3→信号8升级前的拥堵行为（09-23 12:00-14:20）
2. 各信号等级下"原本快/慢的路"五分位偏差对比
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")
# ── Readable thesis-figure style ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 260,
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
    "lines.linewidth": 2.4,
})


DATA = "/Users/helloling/workspace/thesis/data"
OUT  = "/Users/helloling/workspace/thesis"

print("Loading Yagiasha timeseries...")
yagi = pd.read_parquet(f"{DATA}/yagiasha_road_timeseries.parquet")
bl   = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
road_bl = bl.groupby("road_id")["mean_speed"].mean()

# ── 信号段标注 ────────────────────────────────────────────────────────────────
SIGNALS = [
    (pd.Timestamp("2025-09-22 12:20"), pd.Timestamp("2025-09-22 21:40"), 1),
    (pd.Timestamp("2025-09-22 21:40"), pd.Timestamp("2025-09-23 14:20"), 3),
    (pd.Timestamp("2025-09-23 14:20"), pd.Timestamp("2025-09-24 01:40"), 8),
    (pd.Timestamp("2025-09-24 01:40"), pd.Timestamp("2025-09-24 02:40"), 9),
    (pd.Timestamp("2025-09-24 02:40"), pd.Timestamp("2025-09-24 13:20"), 10),
    (pd.Timestamp("2025-09-24 13:20"), pd.Timestamp("2025-09-24 20:20"), 8),
    (pd.Timestamp("2025-09-24 20:20"), pd.Timestamp("2025-09-25 08:20"), 3),
    (pd.Timestamp("2025-09-25 08:20"), pd.Timestamp("2025-09-25 11:20"), 1),
]
SIG_COLORS = {1: "#FFF9C4", 3: "#FFE0B2", 8: "#FFCDD2", 9: "#F48FB1", 10: "#EF9A9A"}

def assign_signal(dt):
    for start, end, sig in SIGNALS:
        if start <= dt < end:
            return sig
    return 0

yagi["signal"] = yagi["dt"].apply(assign_signal)

# ── ANALYSIS 1: 09-23 的分时偏差（信号3期间 + 信号3→8升级前后）────────────────
print("Analysis 1: 09-23 pre-Signal-8 behavior...")

yagi_23 = yagi[yagi["ds"] == "2025-09-23"].copy()
slot_stats_23 = yagi_23.groupby("slot").agg(
    mean_dev=("dev", "mean"),
    median_dev=("dev", "median"),
    pct_neg=("dev", lambda x: (x < -0.02).mean()),
    pct_pos=("dev", lambda x: (x > 0.02).mean()),
    n=("dev", "count"),
).reset_index()
slot_stats_23["hour"] = slot_stats_23["slot"] * 0.5
slot_stats_23["signal"] = slot_stats_23["slot"].apply(
    lambda s: 8 if s >= 28 else 3)  # S8 at slot 28 (14:00)

# Also get a control WORKDAY comparison for the same slots
# Use 09-16 (Tuesday, workday)
import os
from shapely import wkb as shapely_wkb
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
bl_idx = bl.set_index(["day_type","slot","road_id"])["mean_speed"]

def get_ep_key(wb):
    try:
        g = shapely_wkb.loads(bytes(wb))
        coords = list(g.coords) if g.geom_type == "LineString" else \
                 [c for line in g.geoms for c in line.coords]
        s = (round(coords[0][0], 4), round(coords[0][1], 4))
        e = (round(coords[-1][0], 4), round(coords[-1][1], 4))
        return str((min(s, e), max(s, e)))
    except:
        return None

def load_slot(day, slot_num, day_type):
    FLOW = f"{DATA}/flow_parquet2"
    folder = f"{FLOW}/{day}"
    files = [f for f in os.listdir(folder) if f"_slot{slot_num:02d}_" in f]
    if not files:
        return None
    df = pd.read_parquet(f"{folder}/{files[0]}",
                         columns=["relative_speed","geometry","road_closure"])
    df = df[df["road_closure"] != 1].copy()
    if len(df) < 50:
        return None
    df["ep_key"] = df["geometry"].apply(get_ep_key)
    df = df.merge(ep[["ep_key","road_id"]], on="ep_key", how="inner")
    if len(df) == 0:
        return None
    agg = df.groupby("road_id")["relative_speed"].mean().reset_index()
    agg = agg.set_index("road_id")
    idx = pd.MultiIndex.from_arrays([[day_type]*len(agg), [slot_num]*len(agg), agg.index],
                                    names=["day_type","slot","road_id"])
    agg["bl"] = bl_idx.reindex(idx).values
    agg = agg.dropna(subset=["bl"])
    if len(agg) < 50:
        return None
    agg["dev"] = agg["relative_speed"] - agg["bl"]
    return agg

# Load control day (09-16 Tue) for slots 20-30 (10:00-15:00)
print("  Loading control day (09-16)...")
ctrl_rows = []
for s in range(20, 32):
    df = load_slot("2025-09-16", s, "WORKDAY")
    if df is not None:
        ctrl_rows.append({"slot": s, "hour": s*0.5,
                          "mean_dev": df["dev"].mean(),
                          "pct_neg": (df["dev"] < -0.02).mean()})
ctrl_23 = pd.DataFrame(ctrl_rows)

# ── FIGURE 1: 09-23 全天分析 ─────────────────────────────────────────────────
print("Building Figure 1: 09-23 pre-Signal-8 analysis...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Shade periods
for start, end, sig in SIGNALS:
    if start.date() == pd.Timestamp("2025-09-23").date() or \
       (start < pd.Timestamp("2025-09-23") and end > pd.Timestamp("2025-09-23")):
        plot_start = max(start, pd.Timestamp("2025-09-23 00:00"))
        plot_end = min(end, pd.Timestamp("2025-09-24 00:00"))
        if plot_start < plot_end:
            for ax in [ax1, ax2]:
                ax.axvspan(plot_start.hour + plot_start.minute/60,
                           min(plot_end.hour + plot_end.minute/60, 24),
                           alpha=0.15, color=SIG_COLORS.get(sig, "#ddd"), zorder=0)

# Highlight pre-Signal-8 window (12:00-14:20)
for ax in [ax1, ax2]:
    ax.axvspan(12.0, 14.33, alpha=0.25, color="#FF9800", zorder=1,
               label="Pre-Signal-8 window (12:00–14:20)")
    ax.axvline(14.33, color="red", lw=2, ls="--", alpha=0.9, label="Signal 8 raised (14:20)")

# Panel 1: Mean deviation
ax1.plot(slot_stats_23["hour"], slot_stats_23["mean_dev"],
         color="#7B1FA2", lw=2.5, zorder=3, label="09-23 (Yagiasha Signal 3→8)")
ax1.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
if not ctrl_23.empty:
    ax1.plot(ctrl_23["hour"], ctrl_23["mean_dev"],
             color="#BDBDBD", lw=1.8, ls="--", label="Control workday (09-16)")
# Annotate minimum
min_slot = slot_stats_23.loc[slot_stats_23["mean_dev"].idxmin()]
ax1.annotate(f"Min: {min_slot['mean_dev']:+.3f}\n@ {min_slot['hour']:.1f}h\n(pre-Signal-8\nrush?)",
             xy=(min_slot["hour"], min_slot["mean_dev"]),
             xytext=(11.5, -0.025),
             fontsize=18, color="darkred",
             arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5))
ax1.set_ylabel("Network Mean Speed Deviation")
ax1.set_title("September 23 (Yagiasha): Network Speed Deviation — Pre-Signal-8 Behavioral Anomaly",
              fontsize=18, fontweight="bold")
ax1.legend(fontsize=18, loc="upper right")
ax1.grid(alpha=0.3)
ax1.set_ylim(-0.04, 0.12)

# Panel 2: % roads slower
ax2.plot(slot_stats_23["hour"], slot_stats_23["pct_neg"],
         color="#F44336", lw=2.5, zorder=3, label="% Roads slower than baseline (09-23)")
ax2.plot(slot_stats_23["hour"], slot_stats_23["pct_pos"],
         color="#2196F3", lw=2.5, zorder=3, label="% Roads faster than baseline (09-23)")
if not ctrl_23.empty:
    ax2.plot(ctrl_23["hour"], ctrl_23["pct_neg"],
             color="#FFCDD2", lw=1.8, ls="--", label="Control: % slower (09-16)")
ax2.axhline(0.5, color="gray", lw=0.5, ls=":", alpha=0.5)
# Annotate peak %slower
peak_neg = slot_stats_23.loc[slot_stats_23["pct_neg"].idxmax()]
ax2.annotate(f"{peak_neg['pct_neg']:.0%} roads slower\n@ {peak_neg['hour']:.1f}h",
             xy=(peak_neg["hour"], peak_neg["pct_neg"]),
             xytext=(10.5, 0.38),
             fontsize=18, color="darkred",
             arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5))
ax2.set_xlabel("Hour of Day (HKT)")
ax2.set_ylabel("Fraction of Roads")
ax2.set_title("Fraction of Roads Faster/Slower Than Workday Baseline",
              fontsize=13, fontweight="bold")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax2.legend(fontsize=18, loc="upper right")
ax2.grid(alpha=0.3)
ax2.set_xlim(0, 24)
ax2.set_xticks(range(0, 25, 2))
ax2.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], fontsize=18)

# Signal annotations on top panel
for sig_h, sig_v in [(14.33, "S8\nraised")]:
    ax1.text(sig_h + 0.3, 0.09, sig_v, fontsize=18, color="red", fontweight="bold")
ax1.text(0.5, 0.085, "S3", fontsize=15, color="darkorange", fontweight="bold", transform=ax1.get_xaxis_transform())
ax1.text(15.5/24, 0.085, "S8", fontsize=15, color="darkred", fontweight="bold", transform=ax1.get_xaxis_transform())

fig.suptitle("Pre-Escalation Behavioral Signature: Roads Slower Than Normal Before Signal 8 Is Raised\n"
             "Yagiasha, 23 September 2025 — Signal 3 Active, Signal 8 Raised at 14:20 HKT",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/图11_信号3升级前拥堵行为.png", dpi=260, bbox_inches="tight")
plt.close()
print(f"Saved: 图11_信号3升级前拥堵行为.png")


# ── ANALYSIS 2: 各信号等级五分位偏差 ─────────────────────────────────────────
print("\nAnalysis 2: Quintile deviation by signal level...")

def quintile_stats(df_signal, signal_label):
    road_dev = df_signal.groupby("road_id")["dev"].mean()
    merged = pd.concat([road_dev, road_bl], axis=1).dropna()
    merged.columns = ["dev", "bl"]
    merged["q"] = pd.qcut(merged["bl"], q=5,
                           labels=["Q1\nslowest","Q2","Q3","Q4","Q5\nfastest"])
    stats = merged.groupby("q", observed=True).agg(
        mean_dev=("dev","mean"),
        median_dev=("dev","median"),
        pct_pos=("dev", lambda x: (x>0.02).mean()),
        pct_neg=("dev", lambda x: (x<-0.02).mean()),
        n=("dev","count"),
    ).reset_index()
    stats["signal"] = signal_label
    return stats, merged

# Signals 1, 3, 8, 10
quintile_all = []
sig_colors2 = {1: "#FFF9C4", 3: "#FFE0B2", 8: "#FF8A65", 10: "#D32F2F"}
sig_labels2 = {1: "Signal 1", 3: "Signal 3", 8: "Signal 8", 10: "Signal 10"}

for sig in [1, 3, 8, 10]:
    sub = yagi[yagi["signal"] == sig]
    if len(sub) < 1000:
        continue
    stats, merged = quintile_stats(sub, sig)
    quintile_all.append((sig, stats, merged))
    print(f"  Signal {sig}: {len(merged):,} roads")
    print(stats[["q","mean_dev","pct_pos","pct_neg"]].to_string(index=False))
    print()

# ── FIGURE 2: 四幅子图 — 各信号等级五分位 ────────────────────────────────────
print("Building Figure 2: quintile by signal level...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, (sig, stats, merged) in zip(axes, quintile_all):
    q_labels = stats["q"].tolist()
    x = np.arange(len(q_labels))
    w = 0.35

    bars_f = ax.bar(x - w/2, stats["pct_pos"], w, color="#2196F3", alpha=0.8,
                    label="Faster (dev>0.02)")
    bars_s = ax.bar(x + w/2, stats["pct_neg"], w, color="#F44336", alpha=0.8,
                    label="Slower (dev<-0.02)")

    # Mean deviation as line overlay
    ax2_twin = ax.twinx()
    ax2_twin.plot(x, stats["mean_dev"], color="#7B1FA2", lw=2.5, marker="o",
                  markersize=6, label="Mean deviation")
    ax2_twin.axhline(0, color="purple", lw=0.8, ls="--", alpha=0.4)
    ax2_twin.set_ylabel("Mean Deviation", color="#7B1FA2", fontsize=18)
    ax2_twin.tick_params(axis="y", colors="#7B1FA2", labelsize=8)
    # Set symmetric y range for twin axis
    ymax = max(abs(stats["mean_dev"]).max() * 1.5, 0.05)
    ax2_twin.set_ylim(-ymax, ymax)

    # Value labels on bars
    for bar in bars_f:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=13)
    for bar in bars_s:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=13)

    ax.set_xticks(x)
    ax.set_xticklabels(q_labels, fontsize=18)
    ax.set_xlabel("Baseline Speed Quintile")
    ax.set_ylabel("Fraction of Roads")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title(f"{sig_labels2[sig]} (n={len(merged):,} roads)",
                 fontsize=13, fontweight="bold",
                 color={"1":"#F9A825","3":"#E65100","8":"#B71C1C","10":"#880E4F"}[str(sig)])
    ax.legend(fontsize=13, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # Annotate the trend direction
    trend = "Ceiling effect: Q5 motorways\nalready at free-flow, no room to accelerate" if sig >= 8 else ""
    if trend:
        ax.text(0.5, -0.12, trend, transform=ax.transAxes, fontsize=13,
                ha="center", color="gray", style="italic")

fig.suptitle(
    "Road Speed Quintile Analysis by Signal Level — Yagiasha Typhoon\n"
    "Q1 = normally slowest roads, Q5 = motorways at free-flow speed\n"
    "Blue bars: % faster than baseline | Red bars: % slower | Purple line: mean deviation",
    fontsize=18, fontweight="bold",
)
plt.tight_layout()
plt.savefig(f"{OUT}/图12_各信号等级五分位偏差.png", dpi=260, bbox_inches="tight")
plt.close()
print(f"Saved: 图12_各信号等级五分位偏差.png")


# ── FIGURE 3: 合并对比 — 五分位 mean deviation across signals ───────────────
print("Building Figure 3: quintile gradient across all signals...")

fig, ax = plt.subplots(figsize=(12, 6))

colors_by_sig = {1: "#FBC02D", 3: "#F57C00", 8: "#E53935", 10: "#880E4F"}
q_labels = ["Q1\n(slowest)", "Q2", "Q3", "Q4", "Q5\n(fastest)"]
x = np.arange(5)

for sig, stats, _ in quintile_all:
    ax.plot(x, stats["mean_dev"], color=colors_by_sig[sig], lw=2.5, marker="o",
            markersize=8, label=f"Signal {sig}")
    # Fill between zero line and values to show direction
    ax.fill_between(x, 0, stats["mean_dev"], color=colors_by_sig[sig], alpha=0.08)

ax.axhline(0, color="black", lw=1, ls="--", alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(q_labels, fontsize=13)
ax.set_xlabel("Baseline Speed Quintile (Q1=Slowest, Q5=Fastest=Motorways)")
ax.set_ylabel("Mean Speed Deviation from Baseline")
ax.set_title("Speed Deviation by Road-Quintile Across All Signal Levels\n"
             "Demand suppression most visible on normally-congested roads; "
             "motorways show ceiling effect",
             fontsize=18, fontweight="bold")
ax.legend(fontsize=13)
ax.grid(alpha=0.3)

# Annotation
ax.annotate("Demand suppression\n(roads empty out)",
            xy=(0, 0.095), xytext=(0.8, 0.085),
            fontsize=18.5, color="darkblue",
            arrowprops=dict(arrowstyle="->", color="darkblue"))
ax.annotate("Ceiling effect:\nmotorways already\nat free-flow speed",
            xy=(4, -0.003), xytext=(3.2, 0.03),
            fontsize=18.5, color="darkred",
            arrowprops=dict(arrowstyle="->", color="darkred"))

plt.tight_layout()
plt.savefig(f"{OUT}/图13_五分位偏差信号等级梯度.png", dpi=260, bbox_inches="tight")
plt.close()
print(f"Saved: 图13_五分位偏差信号等级梯度.png")

# ── Print numerical summary ──────────────────────────────────────────────────
print("\n=== Quintile Mean Deviation Summary ===")
print(f"{'Signal':<8} {'Q1(slow)':<12} {'Q2':<10} {'Q3':<10} {'Q4':<10} {'Q5(fast)':<12}")
for sig, stats, _ in quintile_all:
    devs = stats["mean_dev"].values
    print(f"S{sig:<7} {devs[0]:+.4f}     {devs[1]:+.4f}    {devs[2]:+.4f}    {devs[3]:+.4f}    {devs[4]:+.4f}")

# Pre-Signal-8 stats
print("\n=== 09-23 Pre-Signal-8 Window (12:00-14:20) ===")
pre_s8 = slot_stats_23[(slot_stats_23["slot"] >= 24) & (slot_stats_23["slot"] < 28)]
print(pre_s8[["slot","hour","mean_dev","pct_neg","pct_pos"]].to_string(index=False))
