"""
Redraw pre-event figures for PPT: (A) Pre-S1 anticipatory mobility, (B) Pre-S8 congestion surge
Clean single-panel dark-theme figures suitable for slides.
"""
import os, gc, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from datetime import datetime
from shapely import wkb as shapely_wkb
import warnings
warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"
OUT  = "/Users/helloling/workspace/thesis"

BG    = "#1a1a2e"; PANEL = "#16213e"; TEAL  = "#0f8b8d"
YELLOW= "#ffc800"; LGRAY = "#aaaaaa"; WHITE = "#f0f0f0"

# ── Load lookups ─────────────────────────────────────────────────────────────
print("Loading lookups...", flush=True)
bl = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
bl_idx = bl.set_index(["day_type","slot","road_id"])["mean_speed"]

def get_ep_key(wb):
    try:
        g = shapely_wkb.loads(bytes(wb))
        coords = list(g.coords) if g.geom_type == "LineString" else \
                 [c for line in g.geoms for c in line.coords]
        s = (round(coords[0][0],4), round(coords[0][1],4))
        e = (round(coords[-1][0],4), round(coords[-1][1],4))
        return str((min(s,e), max(s,e)))
    except: return None

_wkb_ep_cache = {}
def build_wkb_cache(day):
    folder = f"{FLOW}/{day}"
    if not os.path.exists(folder): return {}
    uniq = {}
    for s in [0, 12, 24, 36]:
        files = [f for f in os.listdir(folder) if f"_slot{s:02d}_" in f]
        if not files: continue
        df = pd.read_parquet(f"{folder}/{files[0]}", columns=["geometry"])
        for g in df["geometry"]:
            if g is not None:
                key = id(bytes(g)[:8])
                if key not in uniq: uniq[key] = g
    ep_map = {}
    for g in uniq.values():
        epk = get_ep_key(g)
        if epk: ep_map[bytes(g)] = epk
    _wkb_ep_cache[day] = ep_map
    return ep_map

def compute_day_timeseries(day, day_type, slot_range=None):
    folder = f"{FLOW}/{day}"
    if not os.path.exists(folder): return None
    all_slots = sorted([int(f.split("_slot")[1][:2])
                        for f in os.listdir(folder)
                        if "_slot" in f and f.endswith(".parquet")])
    if slot_range:
        all_slots = [s for s in all_slots if s in slot_range]
    wkb_ep = build_wkb_cache(day)
    rows = []
    for s in all_slots:
        files = [f for f in os.listdir(folder) if f"_slot{s:02d}_" in f]
        if not files: continue
        try:
            df = pd.read_parquet(f"{folder}/{files[0]}",
                                 columns=["relative_speed","geometry","road_closure"])
            df = df[df["road_closure"] != 1].copy()
            if len(df) < 50: continue

            def lookup_epk(g):
                if g is None: return None
                b = bytes(g)
                if b in wkb_ep: return wkb_ep[b]
                epk = get_ep_key(g)
                if epk: wkb_ep[b] = epk
                return epk

            df["ep_key"] = df["geometry"].apply(lookup_epk)
            df = df.merge(ep[["ep_key","road_id"]], on="ep_key", how="inner")
            if len(df) < 50: continue
            agg = df.groupby("road_id")["relative_speed"].mean().rename("obs")
            agg = agg.reset_index().set_index("road_id")
            idx = pd.MultiIndex.from_arrays(
                [[day_type]*len(agg), [s]*len(agg), agg.index],
                names=["day_type","slot","road_id"])
            agg["baseline"] = bl_idx.reindex(idx).values
            agg = agg.dropna(subset=["baseline"])
            if len(agg) < 100: continue
            agg["dev"] = agg["obs"] - agg["baseline"]
            base_dt = datetime.strptime(day, "%Y-%m-%d")
            rows.append({
                "datetime": base_dt + pd.Timedelta(minutes=s*30),
                "slot": s, "hour": s*0.5,
                "mean_dev": float(agg["dev"].mean()),
                "p25": float(agg["dev"].quantile(0.25)),
                "p75": float(agg["dev"].quantile(0.75)),
                "pct_faster": float((agg["dev"] > 0).mean()),
                "n_roads": len(agg),
            })
        except: pass
        gc.collect()
    return pd.DataFrame(rows) if rows else None

# ── Compute pre-S1 data ─────────────────────────────────────────────────────
print("\nComputing pre-S1 time series...")
# Mina: Sep 17 (Wed), S1@21:20 → analyze up to slot 42 (21:00)
mina = compute_day_timeseries("2025-09-17", "WORKDAY", range(0, 43))
# Madum: Oct 3 (Fri), S1@19:40 → analyze up to slot 39 (19:30)
madum = compute_day_timeseries("2025-10-03", "WORKDAY", range(0, 40))
# Control: Sep 16 (Tue)
ctrl = compute_day_timeseries("2025-09-16", "WORKDAY", range(0, 43))

# Yagiasha: from existing CSV (Sep 22, only slots 0-9 before gap)
yagi = pd.read_csv(f"{OUT}/ragasa_timeseries.csv", parse_dates=["datetime"])
yagi["hour"] = yagi["slot"] * 0.5
yagi_pre = yagi[(yagi.day == "2025-09-22") & (yagi.slot <= 9)].copy()

print(f"  Mina: {len(mina) if mina is not None else 0} slots")
print(f"  Madum: {len(madum) if madum is not None else 0} slots")
print(f"  Control: {len(ctrl) if ctrl is not None else 0} slots")
print(f"  Yagiasha pre-S1: {len(yagi_pre)} slots")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE A: Pre-Signal 1 Anticipatory Mobility
# ═══════════════════════════════════════════════════════════════════════════════
print("\nDrawing Figure A: Pre-S1...")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=BG)
ax.set_facecolor(PANEL)
for sp in ax.spines.values():
    sp.set_color(LGRAY); sp.set_linewidth(0.5)
ax.tick_params(colors=WHITE, length=3, width=0.5)

# Control band (grey)
if ctrl is not None and len(ctrl) > 0:
    ax.fill_between(ctrl["hour"], ctrl["p25"], ctrl["p75"],
                    color="#666666", alpha=0.25, lw=0, label="Control day IQR (Sep 16)")
    ax.plot(ctrl["hour"], ctrl["mean_dev"], color="#888888",
            lw=1.5, ls="--", label="Control day (Sep 16)")

# Typhoon lines
colors = {"Mina": "#4fc3f7", "Madum": "#81c784", "Yagiasha": "#ef5350"}
for name, df, s1_h, s1_label in [
    ("Mina",    mina,      21.33, "S1 21:20"),
    ("Madum",   madum,     19.67, "S1 19:40"),
    ("Yagiasha", yagi_pre, 12.33, "S1 12:20"),
]:
    if df is None or len(df) == 0: continue
    c = colors[name]
    ax.plot(df["hour"], df["mean_dev"], color=c, lw=2.2, label=f"{name}")
    ax.fill_between(df["hour"], df["p25"], df["p75"], color=c, alpha=0.10, lw=0)
    ax.axvline(s1_h, color=c, lw=1.2, ls=(0,(4,3)), alpha=0.7)
    ax.text(s1_h+0.15, 0.94, s1_label, color=c, fontsize=7.5,
            fontweight="bold", transform=ax.get_xaxis_transform(), va="top")

# Annotations
ax.annotate("Anticipatory peak\n(+0.012 at 17:00)",
            xy=(17, 0.012), xytext=(17, 0.022),
            fontsize=8.5, color="#4fc3f7", fontweight="bold",
            ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-|>", color="#4fc3f7", lw=0.9),
            bbox=dict(boxstyle="round,pad=0.2", fc=PANEL, ec="#4fc3f7", alpha=0.9))

ax.axhline(0, color=LGRAY, lw=0.6, ls="--")
ax.set_xlim(5, 22)
ax.set_xticks(range(6, 23, 2))
ax.set_xticklabels([f"{h:02d}:00" for h in range(6, 23, 2)])
plt.setp(ax.get_xticklabels(), color=WHITE, fontsize=9)

ax.set_ylabel("Mean speed deviation\n(typhoon − baseline)", color=WHITE, fontsize=9)
ax.set_title("Pre-Signal 1: Anticipatory Mobility Before Formal Warning",
             color=WHITE, fontsize=11, fontweight="bold", pad=6)

leg = ax.legend(loc="lower left", fontsize=7.5, framealpha=0.5,
                facecolor=PANEL, edgecolor=LGRAY, labelcolor=WHITE, ncol=2)
plt.tight_layout(pad=0.8)
fig_a = f"{OUT}/图43a_preS1_anticipatory.png"
plt.savefig(fig_a, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  Saved: {fig_a}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE B: Pre-Signal 8 Congestion Surge (Yagiasha Sep 23)
# ═══════════════════════════════════════════════════════════════════════════════
print("Drawing Figure B: Pre-S8 surge...")
# Use existing Ragasa/Yagiasha data for Sep 23
yagi23 = yagi[yagi.day == "2025-09-23"].copy()
# S8 at 14:20 = hour 14.33
s8_hour = 14.33

fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=BG)
ax.set_facecolor(PANEL)
for sp in ax.spines.values():
    sp.set_color(LGRAY); sp.set_linewidth(0.5)
ax.tick_params(colors=WHITE, length=3, width=0.5)

# Phase shading: pre-S8 (grey) vs post-S8 (red tint)
ax.axvspan(5, s8_hour, color="#333344", alpha=0.4, lw=0)
ax.axvspan(s8_hour, 23, color="#442222", alpha=0.4, lw=0)

# IQR band
ax.fill_between(yagi23["hour"], yagi23["p25"], yagi23["p75"],
                color=TEAL, alpha=0.10, lw=0)

# Mean deviation line
ax.plot(yagi23["hour"], yagi23["mean_dev"], color=TEAL, lw=2.2, zorder=5)

# S8 line
ax.axvline(s8_hour, color=YELLOW, lw=2.0, ls=(0,(4,3)), alpha=0.9, zorder=6)
ax.text(s8_hour+0.15, 0.94, "S8↑ 14:20", color=YELLOW, fontsize=8.5,
        fontweight="bold", transform=ax.get_xaxis_transform(), va="top")

# Before/after labels
ax.text(10, 0.88, "Pre-S8\nrush", color="#aaaaaa", fontsize=8,
        ha="center", transform=ax.get_xaxis_transform(), style="italic")
ax.text(18, 0.88, "Network\nclearance", color="#ef9a9a", fontsize=8,
        ha="center", transform=ax.get_xaxis_transform(), style="italic")

# Key annotations
# The dip
dip_slot = yagi23.loc[yagi23["mean_dev"].idxmin()]
ax.annotate(f"−0.010 at {dip_slot['hour']:.1f}h\n(31.3% roads slower)",
            xy=(dip_slot["hour"], dip_slot["mean_dev"]),
            xytext=(dip_slot["hour"]-1.5, dip_slot["mean_dev"]-0.018),
            fontsize=8, color="#e08080", fontweight="bold",
            ha="center", va="top",
            arrowprops=dict(arrowstyle="-|>", color="#e08080", lw=0.9),
            bbox=dict(boxstyle="round,pad=0.2", fc=PANEL, ec="#e08080", alpha=0.9))

# The surge
peak_s8 = yagi23[yagi23["hour"] > s8_hour].loc[yagi23["mean_dev"].idxmax()]
ax.annotate(f"+0.045 at {peak_s8['hour']:.1f}h\n(network cleared)",
            xy=(peak_s8["hour"], peak_s8["mean_dev"]),
            xytext=(peak_s8["hour"]+1.0, peak_s8["mean_dev"]+0.008),
            fontsize=8, color=TEAL, fontweight="bold",
            ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=0.9),
            bbox=dict(boxstyle="round,pad=0.2", fc=PANEL, ec=TEAL, alpha=0.9))

ax.axhline(0, color=LGRAY, lw=0.6, ls="--")
ax.set_xlim(5, 23)
ax.set_xticks(range(6, 23, 2))
ax.set_xticklabels([f"{h:02d}:00" for h in range(6, 23, 2)])
plt.setp(ax.get_xticklabels(), color=WHITE, fontsize=9)

ax.set_ylabel("Mean speed deviation\n(typhoon − baseline)", color=WHITE, fontsize=9)
ax.set_title("Pre-Signal 8: Congestion Surge → Network Clearance (Yagiasha, Sep 23)",
             color=WHITE, fontsize=11, fontweight="bold", pad=6)

plt.tight_layout(pad=0.8)
fig_b = f"{OUT}/图43b_preS8_surge.png"
plt.savefig(fig_b, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  Saved: {fig_b}")
print("\nDone.")
