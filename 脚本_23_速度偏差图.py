"""
图23：台风 Ragasa 道路速度对比与偏差图
布局：3行 × 2列
  行0 正常基线均值（baseline_speed）| 早高峰 / 晚高峰
  行1 Ragasa 台风实测速度          | 早高峰 / 晚高峰
  行2 速度偏差（台风 − 基线）       | 早高峰 / 晚高峰
黑色背景，主干道及以上等级路段
"""
import ast, glob, pickle, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from shapely import wkb as shapely_wkb
from shapely.geometry import LineString
import geopandas as gpd
import contextily as ctx
import warnings; warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"
OUT  = "/Users/helloling/workspace/thesis"

BG = "#080c14"

AM_SLOTS = list(range(14, 19))   # 07:00–09:30  (5 slots)
PM_SLOTS = list(range(34, 40))   # 17:00–20:00  (6 slots)
TYPHOON_DATE = "2025-09-24"      # Ragasa S10 峰值，工作日

KEEP_CATS = {"motorway","motorway_link","trunk","trunk_link",
             "primary","primary_link","secondary","secondary_link",
             "tertiary","tertiary_link"}

# ── 加载数据 ─────────────────────────────────────────────────────────────────
print("Loading...", flush=True)
bl = pd.read_parquet(f"{DATA}/baseline_speed.parquet",
                     columns=["day_type","slot","road_id","mean_speed"])
bl_wk = bl[bl["day_type"] == "WORKDAY"][["road_id","slot","mean_speed"]]
bl_lkp = bl_wk.set_index(["road_id","slot"])["mean_speed"]
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

# ── 基线均值（每路段在给定 slots 的平均 mean_speed） ─────────────────────────
def baseline_avg(road_ids, slots):
    rows = []
    for sl in slots:
        sub = bl_wk[bl_wk["slot"] == sl]
        rows.append(sub)
    df = pd.concat(rows)
    df = df[df["road_id"].isin(road_ids)]
    return df.groupby("road_id")["mean_speed"].mean()

# ── 读台风期流量文件 ──────────────────────────────────────────────────────────
def read_speed_df(date_str, slots):
    rows = []
    for sl in slots:
        pat = f"{FLOW}/{date_str}/traffic_flow_zoom15_{date_str}_slot{sl:02d}_*.parquet"
        fs = glob.glob(pat)
        if not fs: continue
        df = pd.read_parquet(fs[0],
                             columns=["geometry","road_closure","relative_speed"])
        df = df[df["road_closure"] != 1].dropna(subset=["relative_speed"])
        for _, row in df.iterrows():
            epk = get_ep_key(row["geometry"])
            if epk and epk in ep_lkp.index:
                rows.append({"road_id": int(ep_lkp[epk]),
                             "slot": sl, "speed": row["relative_speed"]})
    if not rows:
        return pd.DataFrame(columns=["road_id","slot","speed"])
    out = pd.DataFrame(rows)
    return out.groupby(["road_id","slot"])["speed"].mean().reset_index()

def speed_avg(speed_df):
    return speed_df.groupby("road_id")["speed"].mean()

def deviation(speed_df):
    def d(row):
        key = (row["road_id"], row["slot"])
        return row["speed"] - bl_lkp[key] if key in bl_lkp.index else np.nan
    speed_df = speed_df.copy()
    speed_df["dev"] = speed_df.apply(d, axis=1)
    return speed_df.groupby("road_id")["dev"].mean().dropna()

print("Reading typhoon flow...", flush=True)
df_t_am = read_speed_df(TYPHOON_DATE, AM_SLOTS)
df_t_pm = read_speed_df(TYPHOON_DATE, PM_SLOTS)
spd_t_am = speed_avg(df_t_am)
spd_t_pm = speed_avg(df_t_pm)
dev_am   = deviation(df_t_am)
dev_pm   = deviation(df_t_pm)

for label, dev in [("AM", dev_am), ("PM", dev_pm)]:
    print(f"  {label}: mean={dev.mean():.3f} | "
          f"worse(<-0.10): {(dev<-0.10).sum():,} | "
          f"better(>+0.10): {(dev>0.10).sum():,}")

# ── 加载几何 ──────────────────────────────────────────────────────────────────
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
print(f"  {len(gdf_main):,} road segments", flush=True)

# 基线均值（只需 gdf_main 中的 road_id）
all_road_ids = set(gdf_main["road_id"].values)
bl_am = baseline_avg(all_road_ids, AM_SLOTS)
bl_pm = baseline_avg(all_road_ids, PM_SLOTS)

# ── 色彩 ─────────────────────────────────────────────────────────────────────
SPEED_CMAP = matplotlib.colormaps["RdYlGn"]
SPEED_NORM = mcolors.Normalize(vmin=0.3, vmax=1.0)

DEV_CMAP = matplotlib.colormaps["RdBu"]
DEV_NORM  = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)

def lw_for_cat(cat):
    if "motorway" in cat: return 0.80
    if "trunk"    in cat: return 0.65
    if "primary"  in cat: return 0.50
    if "secondary"in cat: return 0.40
    return 0.30  # tertiary

def _add_basemap(ax):
    try:
        ctx.add_basemap(ax, crs="EPSG:3857",
                        source=ctx.providers.CartoDB.DarkMatter,
                        zoom=12, alpha=0.38)
    except Exception as e:
        print(f"basemap failed: {e}", flush=True)

# ── 绘制速度图（一格） ────────────────────────────────────────────────────────
def plot_speed(ax, spd_series, title, subtitle):
    ax.set_facecolor(BG); ax.set_axis_off()
    g = gdf_main[gdf_main["road_id"].isin(spd_series.index)].copy()
    g["spd"] = g["road_id"].map(spd_series)
    # 快路先画，慢路后画（慢路顶层可见）
    for lo, hi, lw, zo in [(0.8,1.01,0.30,1),(0.5,0.8,0.45,2),(0.0,0.5,0.65,3)]:
        grp = g[(g["spd"]>=lo) & (g["spd"]<hi)]
        if not len(grp): continue
        colors = [SPEED_CMAP(SPEED_NORM(v)) for v in grp["spd"]]
        grp.plot(ax=ax, color=colors, linewidth=lw, alpha=0.88, zorder=zo)
    _add_basemap(ax)   # 路线画完后叠底图，海面/陆地可见
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=8)
    ax.text(0.5, 0.01, subtitle, transform=ax.transAxes,
            ha="center", fontsize=9.5, color="#aabbcc")

# ── 绘制偏差图（一格） ────────────────────────────────────────────────────────
def plot_deviation(ax, dev_series, title, subtitle):
    ax.set_facecolor(BG); ax.set_axis_off()
    g = gdf_main[gdf_main["road_id"].isin(dev_series.index)].copy()
    g["dev"] = g["road_id"].map(dev_series)
    g["absdev"] = g["dev"].abs()
    neutral = g[g["absdev"] < 0.05]
    if len(neutral):
        neutral.plot(ax=ax, color="#1e3040", linewidth=0.15, alpha=0.50, zorder=1)
    for lo, hi, lw, zo in [(0.05,0.15,0.40,2),(0.15,0.30,0.65,3),(0.30,1.0,1.00,4)]:
        pos = g[(g["dev"] >= lo) & (g["dev"] < hi)]
        neg = g[(g["dev"] <= -lo) & (g["dev"] > -hi)]
        for grp in [pos, neg]:
            if not len(grp): continue
            colors = [DEV_CMAP(DEV_NORM(v)) for v in grp["dev"]]
            grp.plot(ax=ax, color=colors, linewidth=lw, alpha=0.95, zorder=zo)
    _add_basemap(ax)
    n_worse  = (dev_series < -0.10).sum()
    n_better = (dev_series >  0.10).sum()
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=8)
    ax.text(0.5, 0.01,
            f"{subtitle}   ·   slower: {n_worse:,} roads  |  faster: {n_better:,} roads",
            transform=ax.transAxes, ha="center", fontsize=9.5, color="#aabbcc")

# ── 画布：GridSpec 精确控制 ───────────────────────────────────────────────────
print("Plotting...", flush=True)
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(22, 28))
fig.patch.set_facecolor(BG)
# 3个地图行 + 2个色条行（色条行很矮）
gs = GridSpec(5, 2,
              height_ratios=[8, 8, 8, 0.35, 0.35],
              hspace=0.10, wspace=0.02,
              top=0.95, bottom=0.01, left=0.01, right=0.99)
axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)]
ax_cb_spd = fig.add_subplot(gs[3, :])   # 速度色条
ax_cb_dev = fig.add_subplot(gs[4, :])   # 偏差色条

# 行0：正常基线速度
plot_speed(axes[0][0], bl_am,
           "Normal Workday Baseline  —  Morning Peak",
           "07:00–09:30  (mean of all non-typhoon workdays)")
plot_speed(axes[0][1], bl_pm,
           "Normal Workday Baseline  —  Evening Peak",
           "17:00–20:00  (mean of all non-typhoon workdays)")

# 行1：台风实测速度
plot_speed(axes[1][0], spd_t_am,
           "Ragasa Signal 10  —  Morning Peak",
           "Sep 24  07:00–09:30  (typhoon peak)")
plot_speed(axes[1][1], spd_t_pm,
           "Ragasa Signal 8↓  —  Evening Peak",
           "Sep 24  17:00–20:00  (signal declining)")

# 行2：速度偏差
plot_deviation(axes[2][0], dev_am,
               "Speed Deviation  —  Morning Peak",
               "Typhoon − Baseline  (red = slower, blue = faster)")
plot_deviation(axes[2][1], dev_pm,
               "Speed Deviation  —  Evening Peak",
               "Typhoon − Baseline  (red = slower, blue = faster)")

# ── 色条（独立行） ────────────────────────────────────────────────────────────
sm_spd = ScalarMappable(cmap=SPEED_CMAP, norm=SPEED_NORM)
sm_spd.set_array([])
cb1 = fig.colorbar(sm_spd, cax=ax_cb_spd, orientation="horizontal")
cb1.set_label("Relative Speed  (0.3 = heavy congestion  →  1.0 = free flow)",
              color="white", fontsize=10.5)
cb1.set_ticks([0.3, 0.5, 0.7, 0.9, 1.0])
cb1.ax.xaxis.set_tick_params(color="white")
plt.setp(cb1.ax.xaxis.get_ticklabels(), color="white", fontsize=9.5)

sm_dev = ScalarMappable(cmap=DEV_CMAP, norm=DEV_NORM)
sm_dev.set_array([])
cb2 = fig.colorbar(sm_dev, cax=ax_cb_dev, orientation="horizontal")
cb2.set_label("Speed Deviation (typhoon − baseline)  ·  "
              "Red = slower than normal   Blue = faster (emptier)",
              color="white", fontsize=10.5)
cb2.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
cb2.ax.xaxis.set_tick_params(color="white")
plt.setp(cb2.ax.xaxis.get_ticklabels(), color="white", fontsize=9.5)

fig.suptitle(
    "Typhoon Ragasa: Road Speed During Typhoon vs Normal Workday Baseline\n"
    "Morning Peak (07:00–09:30)  ·  Evening Peak (17:00–20:00)",
    color="white", fontsize=14, fontweight="bold")

out = f"{OUT}/图23_速度偏差图.png"
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out}", flush=True)
