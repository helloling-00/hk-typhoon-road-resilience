"""
图66 横版：1 行 × 3 列  —  Sep 23 Signal-3 morning clearance @ 08:30 (slot 17)
  Panel 1: Normal workday baseline
  Panel 2: Sep 23 actual speeds
  Panel 3: Deviation (Sep 23 − baseline) with road-km share annotation
"""
import ast, glob, pickle, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from shapely import wkb as shapely_wkb
from shapely.geometry import LineString
import geopandas as gpd
import contextily as ctx
import warnings; warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"
OUT  = "/Users/helloling/workspace/thesis"

BG = "white"
TXT = "#202020"
SUB = "#5a6a7a"
SLOT_AM = 17           # 08:30
EVENT_DATE = "2025-09-23"
F_THR = 0.03           # faster threshold
S_THR = -0.03          # slower threshold

KEEP_CATS = {"motorway","motorway_link","trunk","trunk_link",
             "primary","primary_link","secondary","secondary_link",
             "tertiary","tertiary_link"}

print("Loading baseline...", flush=True)
bl = pd.read_parquet(f"{DATA}/baseline_speed.parquet",
                     columns=["day_type","slot","road_id","mean_speed"])
bl_wk_17 = bl[(bl["day_type"]=="WORKDAY") & (bl["slot"]==SLOT_AM)][
    ["road_id","mean_speed"]].set_index("road_id")["mean_speed"]
del bl

ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
ep_lkp = ep.set_index("ep_key")["road_id"]

def get_ep_key(wkb_bytes):
    try:
        g = shapely_wkb.loads(wkb_bytes)
        c = list(g.coords)
        s4 = (round(c[0][0],4), round(c[0][1],4))
        e4 = (round(c[-1][0],4), round(c[-1][1],4))
        return str((min(s4,e4), max(s4,e4)))
    except: return None

def read_speed(date_str, slot):
    pat = f"{FLOW}/{date_str}/traffic_flow_zoom15_{date_str}_slot{slot:02d}_*.parquet"
    fs = glob.glob(pat)
    if not fs: return pd.Series(dtype=float)
    df = pd.read_parquet(fs[0],
                         columns=["geometry","road_closure","relative_speed"])
    df = df[df["road_closure"]!=1].dropna(subset=["relative_speed"])
    rows = []
    for _, row in df.iterrows():
        epk = get_ep_key(row["geometry"])
        if epk and epk in ep_lkp.index:
            rows.append((int(ep_lkp[epk]), row["relative_speed"]))
    if not rows: return pd.Series(dtype=float)
    s = pd.DataFrame(rows, columns=["road_id","speed"]).groupby("road_id")["speed"].mean()
    return s

print(f"Reading Sep 23 slot {SLOT_AM} (08:30)...", flush=True)
spd_event = read_speed(EVENT_DATE, SLOT_AM)
print(f"  {len(spd_event):,} road observations on {EVENT_DATE}")

# Third panel uses pre-computed deviation from yagiasha_road_timeseries (脚本_55/50d data source)
ts = pd.read_parquet(f"{DATA}/yagiasha_road_timeseries.parquet",
                     columns=["road_id","dt","slot","dev"])
ts["ds"] = pd.to_datetime(ts["dt"]).dt.strftime("%Y-%m-%d")
dev = ts[(ts["ds"]==EVENT_DATE) & (ts["slot"]==SLOT_AM)].set_index("road_id")["dev"].dropna()
print(f"  deviation (50d source): n={len(dev):,}  mean={dev.mean():.3f}")

print("Loading geometries...", flush=True)
with open(f"{DATA}/osm_cache/road_wkb_store.pkl","rb") as f:
    wkb_store = pickle.load(f)
ep_df = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
rr = pd.read_parquet(f"{DATA}/road_registry.parquet")[
        ["road_id","road_category"]].drop_duplicates("road_id")

def build_geom(row):
    rid = row["road_id"]
    if rid in wkb_store:
        try: return shapely_wkb.loads(wkb_store[rid])
        except: pass
    try:
        pts = ast.literal_eval(row["ep_key"])
        return LineString([pts[0], pts[1]])
    except: return None

ep_df["geometry"] = ep_df.apply(build_geom, axis=1)
ep_df = ep_df.dropna(subset=["geometry"])
ep_df = ep_df.merge(rr, on="road_id", how="left")
gdf = gpd.GeoDataFrame(ep_df, geometry="geometry",
                       crs="EPSG:4326").to_crs("EPSG:3857")
gdf_main = gdf[gdf["road_category"].isin(KEEP_CATS)].copy()
gdf_main["length_m"] = gdf_main.geometry.length
print(f"  {len(gdf_main):,} road segments")

# ── road-km share of faster / slower (using thresholds) ─────────────────────
g_dev = gdf_main.merge(dev.rename("dev"), on="road_id", how="inner")
total_km = g_dev["length_m"].sum() / 1000
faster_km = g_dev.loc[g_dev["dev"] >  F_THR, "length_m"].sum() / 1000
slower_km = g_dev.loc[g_dev["dev"] <  S_THR, "length_m"].sum() / 1000
pct_fast = faster_km / total_km * 100
pct_slow = slower_km / total_km * 100
print(f"\nRoad-km totals (Sep 23 08:30):")
print(f"  total observed: {total_km:,.1f} km")
print(f"  faster (dev > {F_THR}): {faster_km:,.1f} km  ({pct_fast:.1f}%)")
print(f"  slower (dev < {S_THR}): {slower_km:,.1f} km  ({pct_slow:.1f}%)")

# ── plotting setup ─────────────────────────────────────────────────────────
SPEED_CMAP = matplotlib.colormaps["RdYlGn"]
SPEED_NORM = mcolors.Normalize(vmin=0.3, vmax=1.0)
DEV_CMAP   = matplotlib.colormaps["RdBu"]
DEV_NORM   = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)

# Fixed HK extent so all three panels render at identical size/aspect
HK_BBOX_4326 = (113.82, 22.15, 114.45, 22.60)
_bbox_pts = gpd.GeoSeries(
    gpd.points_from_xy([HK_BBOX_4326[0], HK_BBOX_4326[2]],
                       [HK_BBOX_4326[1], HK_BBOX_4326[3]]),
    crs="EPSG:4326").to_crs("EPSG:3857")
HK_XLIM = (_bbox_pts.iloc[0].x, _bbox_pts.iloc[1].x)
HK_YLIM = (_bbox_pts.iloc[0].y, _bbox_pts.iloc[1].y)

def _set_extent(ax):
    ax.set_xlim(HK_XLIM); ax.set_ylim(HK_YLIM); ax.set_aspect("equal")

def _add_basemap(ax):
    try:
        ctx.add_basemap(ax, crs="EPSG:3857",
                        source=ctx.providers.CartoDB.Positron,
                        zoom=12, alpha=0.85, attribution_size=6)
    except Exception as e:
        print(f"basemap failed: {e}", flush=True)

def plot_speed(ax, spd_series, title, subtitle):
    ax.set_facecolor(BG); ax.set_axis_off()
    g = gdf_main[gdf_main["road_id"].isin(spd_series.index)].copy()
    g["spd"] = g["road_id"].map(spd_series)
    for lo, hi, lw, zo in [(0.8,1.01,0.30,1),(0.5,0.8,0.45,2),(0.0,0.5,0.65,3)]:
        grp = g[(g["spd"]>=lo) & (g["spd"]<hi)]
        if not len(grp): continue
        colors = [SPEED_CMAP(SPEED_NORM(v)) for v in grp["spd"]]
        grp.plot(ax=ax, color=colors, linewidth=lw, alpha=0.88, zorder=zo)
    _set_extent(ax)
    _add_basemap(ax)
    ax.set_title(title, color=TXT, fontsize=13, fontweight="bold", pad=8)
    ax.text(0.5, 0.01, subtitle, transform=ax.transAxes,
            ha="center", fontsize=10, color=SUB)

def plot_deviation(ax, dev_series, title, subtitle):
    """Binary faster/slower (50d style) on dark basemap."""
    ax.set_facecolor(BG); ax.set_axis_off()
    g = gdf_main[gdf_main["road_id"].isin(dev_series.index)].copy()
    g["dev"] = g["road_id"].map(dev_series)
    slower = g[g["dev"] < S_THR]
    faster = g[g["dev"] > F_THR]
    if len(slower):
        slower.plot(ax=ax, color="#d62728", linewidth=1.0, alpha=0.90,
                    zorder=4, label=f"Slower (dev < −0.03)  —  25.2% by length")
    if len(faster):
        faster.plot(ax=ax, color="#2ca02c", linewidth=1.0, alpha=0.90,
                    zorder=5, label=f"Faster (dev > +0.03)  —  45.6% by length")
    _set_extent(ax)
    _add_basemap(ax)
    ax.set_title(title, color=TXT, fontsize=13, fontweight="bold", pad=8)
    ax.text(0.5, 0.01, subtitle, transform=ax.transAxes,
            ha="center", fontsize=10, color=SUB)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92,
              facecolor="white", edgecolor="#999", labelcolor=TXT)

# ── Figure: 1 row × 3 cols ─────────────────────────────────────────────────
print("\nPlotting...", flush=True)
fig = plt.figure(figsize=(30, 12))
fig.patch.set_facecolor(BG)
gs = GridSpec(2, 3, height_ratios=[10, 0.35],
              hspace=0.18, wspace=0.02,
              top=0.92, bottom=0.04, left=0.01, right=0.99)
axes = [fig.add_subplot(gs[0, c]) for c in range(3)]
ax_cb_spd = fig.add_subplot(gs[1, :2])

plot_speed(axes[0], bl_wk_17,
           "Normal Workday Baseline  —  Morning Peak (08:30)",
           "Mean of all non-typhoon workdays at slot 17")
plot_speed(axes[1], spd_event,
           "Ragasa  Sep 23  Signal 3  —  08:30",
           "Pre-escalation morning, day before Signal 8/10")
plot_deviation(axes[2], dev,
               "Pre-Event Morning Clearance under Signal 3",
               "Ragasa, Sep 23, 08:30   ·   "
               "45.6% of road-km cleared (faster)   ·   25.2% slower")

# colourbars
sm_spd = ScalarMappable(cmap=SPEED_CMAP, norm=SPEED_NORM); sm_spd.set_array([])
cb1 = fig.colorbar(sm_spd, cax=ax_cb_spd, orientation="horizontal")
cb1.set_label("Relative Speed  (0.3 = heavy congestion  →  1.0 = free flow)",
              color=TXT, fontsize=11)
cb1.set_ticks([0.3, 0.5, 0.7, 0.9, 1.0])
cb1.ax.xaxis.set_tick_params(color=TXT)
plt.setp(cb1.ax.xaxis.get_ticklabels(), color=TXT, fontsize=10)

fig.suptitle(
    "Pre-Event Morning Clearance under Signal 3  (Ragasa, Sep 23, 08:30)",
    color=TXT, fontsize=16, fontweight="bold")

out = f"{OUT}/图66_preS3_morning_clearance_3panel.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"\nSaved: {out}")
