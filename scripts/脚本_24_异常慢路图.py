"""
图24：台风 Ragasa 期间异常变慢的路段
定义：速度偏差 < -0.10（台风期比正常基线慢 10 个百分点以上）
左：早高峰 07:00–09:30（S10 峰值）
右：晚高峰 17:00–20:00（S8↓ 降级）
底图：CartoDB DarkMatter alpha=0.38（同图19）
全量路网作暗灰参考底层，异常路段用红-橙色突出显示
"""
import ast, glob, pickle, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from shapely import wkb as shapely_wkb
from shapely.geometry import LineString
import geopandas as gpd
import contextily as ctx
import warnings; warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"
OUT  = "/Users/helloling/workspace/thesis"
BG   = "#080c14"

AM_SLOTS     = list(range(14, 19))   # 07:00–09:30
PM_SLOTS     = list(range(34, 40))   # 17:00–20:00
TYPHOON_DATE = "2025-09-24"

# ── 加载 ─────────────────────────────────────────────────────────────────────
print("Loading...", flush=True)
bl = pd.read_parquet(f"{DATA}/baseline_speed.parquet",
                     columns=["day_type","slot","road_id","mean_speed"])
bl_lkp = (bl[bl["day_type"]=="WORKDAY"]
          .set_index(["road_id","slot"])["mean_speed"])
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

def read_speed_df(date_str, slots):
    rows = []
    for sl in slots:
        pat = f"{FLOW}/{date_str}/traffic_flow_zoom15_{date_str}_slot{sl:02d}_*.parquet"
        fs = glob.glob(pat)
        if not fs: continue
        df = pd.read_parquet(fs[0],
                             columns=["geometry","road_closure","relative_speed"])
        df = df[df["road_closure"]!=1].dropna(subset=["relative_speed"])
        for _, row in df.iterrows():
            epk = get_ep_key(row["geometry"])
            if epk and epk in ep_lkp.index:
                rows.append({"road_id": int(ep_lkp[epk]),
                             "slot": sl, "speed": row["relative_speed"]})
    if not rows: return pd.DataFrame(columns=["road_id","slot","speed"])
    out = pd.DataFrame(rows)
    return out.groupby(["road_id","slot"])["speed"].mean().reset_index()

def deviation(speed_df):
    speed_df = speed_df.copy()
    speed_df["dev"] = speed_df.apply(
        lambda r: r["speed"] - bl_lkp[(r["road_id"], r["slot"])]
                  if (r["road_id"], r["slot"]) in bl_lkp.index else np.nan,
        axis=1)
    avg = speed_df.groupby("road_id")["dev"].mean().dropna()
    # 同时保存台风期实测均速，便于标注
    spd = speed_df.groupby("road_id")["speed"].mean()
    return pd.DataFrame({"dev": avg, "typhoon_spd": spd}).dropna()

print("Reading flow data...", flush=True)
df_am = read_speed_df(TYPHOON_DATE, AM_SLOTS)
df_pm = read_speed_df(TYPHOON_DATE, PM_SLOTS)
res_am = deviation(df_am)
res_pm = deviation(df_pm)

THRESH = -0.10
slow_am = res_am[res_am["dev"] < THRESH]
slow_pm = res_pm[res_pm["dev"] < THRESH]
print(f"  AM anomalously slow: {len(slow_am):,} roads  (dev<{THRESH})")
print(f"  PM anomalously slow: {len(slow_pm):,} roads  (dev<{THRESH})")

# ── 几何 ─────────────────────────────────────────────────────────────────────
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

# 主干道以上作底层参考
MAIN_CATS = {"motorway","motorway_link","trunk","trunk_link",
             "primary","primary_link","secondary","secondary_link",
             "tertiary","tertiary_link"}
gdf_ref = gdf[gdf["road_category"].isin(MAIN_CATS)]

# ── 色彩：按偏差深浅着色（越红越慢） ─────────────────────────────────────────
SLOW_CMAP = matplotlib.colormaps["YlOrRd_r"]           # 反转：深红=最堵，浅黄=轻微慢
SLOW_NORM = mcolors.Normalize(vmin=-0.50, vmax=-0.10)  # -0.50→index 0→深红，-0.10→index 1→浅黄

def lw_for_dev(dev):
    # 偏差越大线越粗
    return 0.5 + abs(dev) * 2.0

# ── 绘图 ─────────────────────────────────────────────────────────────────────
print("Plotting...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(24, 13))
fig.patch.set_facecolor(BG)
plt.subplots_adjust(wspace=0.03, left=0.01, right=0.99, top=0.91, bottom=0.10)

panels = [
    (axes[0], slow_am, res_am,
     "Ragasa Signal 10  —  Morning Peak  (07:00–09:30)",
     "Sep 24  typhoon peak"),
    (axes[1], slow_pm, res_pm,
     "Ragasa Signal 8↓  —  Evening Peak  (17:00–20:00)",
     "Sep 24  signal declining"),
]

for ax, slow_df, all_df, title, subtitle in panels:
    ax.set_facecolor(BG)
    ax.set_axis_off()

    # 1. 全量主干道参考底层（极暗）
    gdf_ref.plot(ax=ax, color="#1c2a38", linewidth=0.18, alpha=0.70, zorder=1)

    # 2. 台风期有数据但正常的路（速度 ≥ baseline 或偏差小）深蓝灰
    normal_ids = set(all_df[all_df["dev"] >= THRESH].index)
    gdf_normal = gdf[gdf["road_id"].isin(normal_ids)]
    if len(gdf_normal):
        gdf_normal.plot(ax=ax, color="#2a4a6a", linewidth=0.22, alpha=0.55, zorder=2)

    # 3. 异常慢的路：按偏差深浅着色，偏差越大越粗
    gdf_slow = gdf[gdf["road_id"].isin(slow_df.index)].copy()
    gdf_slow["dev"] = gdf_slow["road_id"].map(slow_df["dev"])
    gdf_slow["spd"] = gdf_slow["road_id"].map(slow_df["typhoon_spd"])
    # 按偏差分层，最严重的画在最上层
    for lo, hi, lw, zo in [(-0.20,-0.10, 0.60, 3),
                             (-0.35,-0.20, 0.90, 4),
                             (-0.50,-0.35, 1.30, 5)]:
        grp = gdf_slow[(gdf_slow["dev"]>=lo) & (gdf_slow["dev"]<hi)]
        if not len(grp): continue
        colors = [SLOW_CMAP(SLOW_NORM(v)) for v in grp["dev"]]
        grp.plot(ax=ax, color=colors, linewidth=lw, alpha=0.95, zorder=zo)

    # 底图
    try:
        ctx.add_basemap(ax, crs="EPSG:3857",
                        source=ctx.providers.CartoDB.DarkMatter,
                        zoom=12, alpha=0.38)
    except Exception:
        pass

    n_slow = len(slow_df)
    pct = n_slow / len(all_df) * 100
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
    ax.text(0.50, 0.02,
            f"{subtitle}   ·   {n_slow:,} roads slower than normal ({pct:.1f}% of observed)",
            transform=ax.transAxes, ha="center", fontsize=10.5, color="#aabbcc")

# 图例
legend_handles = [
    Line2D([0],[0], color=SLOW_CMAP(SLOW_NORM(-0.15)), lw=2.0,
           label="Slightly slower  (deviation −0.10 to −0.20)"),
    Line2D([0],[0], color=SLOW_CMAP(SLOW_NORM(-0.28)), lw=2.5,
           label="Moderately slower  (−0.20 to −0.35)"),
    Line2D([0],[0], color=SLOW_CMAP(SLOW_NORM(-0.43)), lw=3.5,
           label="Severely slower  (< −0.35)  ← darkest red"),
    Line2D([0],[0], color="#2a4a6a", lw=1.5,
           label="Observed but normal / faster than baseline"),
    Line2D([0],[0], color="#1c2a38", lw=1.0,
           label="Reference road network (not observed this period)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           framealpha=0.65, facecolor="#0d1420", edgecolor="#445566",
           labelcolor="white", fontsize=10, bbox_to_anchor=(0.5, 0.01))

fig.suptitle(
    "Typhoon Ragasa: Roads Slower Than Normal Workday Baseline\n"
    "Colour intensity = degree of slowdown relative to baseline speed",
    color="white", fontsize=14, fontweight="bold")

out = f"{OUT}/图24_异常慢路图.png"
fig.savefig(out, dpi=250, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out}", flush=True)

# ── 打印最慢路段 top20 ────────────────────────────────────────────────────────
print("\n=== Top 20 slowest roads (AM peak, deviation < -0.10) ===")
top = slow_am.nsmallest(20, "dev").copy()
top["road_category"] = top.index.map(
    rr.set_index("road_id")["road_category"])
print(top[["dev","typhoon_spd","road_category"]].to_string())
