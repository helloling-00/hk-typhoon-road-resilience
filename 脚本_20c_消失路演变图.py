"""
图20：随信号增强累计消失的路段
每格只画"消失路"（亮色），存活路不显示，背景纯黑
信号越强 → 亮线越多 → 视觉上地图越满
"""
import ast, pickle, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely import wkb as shapely_wkb
from shapely.geometry import LineString
import geopandas as gpd
import warnings; warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
OUT  = "/Users/helloling/workspace/thesis"

print("Loading...", flush=True)
with open(f"{DATA}/osm_cache/typhoon_road_sets.pkl","rb") as f:
    tsets = pickle.load(f)

r_s3  = tsets["r_s3"]
r_s8  = tsets["r_s8"]
r_s10 = tsets["r_s10"]

ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
rr = pd.read_parquet(f"{DATA}/road_registry.parquet")[
        ["road_id","road_category"]].drop_duplicates("road_id")
with open(f"{DATA}/osm_cache/road_wkb_store.pkl","rb") as f:
    wkb_store = pickle.load(f)

def build_geom(row):
    rid = row["road_id"]
    if rid in wkb_store:
        try: return shapely_wkb.loads(wkb_store[rid])
        except: pass
    try:
        pts = ast.literal_eval(row["ep_key"])
        return LineString([pts[0], pts[1]])
    except: return None

ep["geometry"] = ep.apply(build_geom, axis=1)
ep = ep.dropna(subset=["geometry"])
ep = ep.merge(rr, on="road_id", how="left")
gdf = gpd.GeoDataFrame(ep, geometry="geometry", crs="EPSG:4326").to_crs("EPSG:3857")
n_total = len(gdf)

# 各信号期消失集合（该期未观测到的路）
lost_s3  = gdf[~gdf["road_id"].isin(r_s3)]   # 125,590
lost_s8  = gdf[~gdf["road_id"].isin(r_s8)]   # 135,573
lost_s10 = gdf[~gdf["road_id"].isin(r_s10)]  # 144,021

BG = "#000000"
COLOR = "#FF6600"   # 单一亮橙色，简洁有力

print("Plotting...", flush=True)
fig, axes = plt.subplots(1, 3, figsize=(27, 12))
fig.patch.set_facecolor(BG)
plt.subplots_adjust(wspace=0.03, left=0.01, right=0.99, top=0.88, bottom=0.10)

panels = [
    (axes[0], lost_s3,  "Signal 3",  "Sep 22 21:40 – Sep 25 08:20"),
    (axes[1], lost_s8,  "Signal 8",  "Sep 23 14:20 – Sep 24 20:20"),
    (axes[2], lost_s10, "Signal 10  ▲ PEAK", "Sep 24  02:40–13:20"),
]

for ax, lost_gdf, sig_label, time_label in panels:
    ax.set_facecolor(BG)
    ax.set_axis_off()
    lost_gdf.plot(ax=ax, color=COLOR, linewidth=0.20, alpha=0.85)
    n_lost = len(lost_gdf)
    pct = n_lost / n_total * 100
    ax.set_title(f"Ragasa  —  {sig_label}\n{time_label}",
                 color="white", fontsize=12.5, fontweight="bold", pad=8)
    ax.text(0.50, 0.03, f"{n_lost:,} roads with no probe data  ({pct:.0f}% of network)",
            transform=ax.transAxes, ha="center", color="#ffaa66", fontsize=11)

fig.suptitle(
    "Roads Without Floating-Car Data During Typhoon Ragasa  —  By Signal Level\n"
    "Each line = a road segment that fell below TomTom's minimum probe-vehicle threshold",
    color="white", fontsize=14, fontweight="bold", y=0.97)

# 共用图例
handles = [
    Line2D([0],[0], color=COLOR, lw=2.5,
           label="Road segment with no observed speed data in this signal period"),
    Line2D([0],[0], color=BG,    lw=0,
           label="(Surviving roads not shown)"),
]
fig.legend(handles=handles, loc="lower center", framealpha=0,
           labelcolor="white", fontsize=11, bbox_to_anchor=(0.5, 0.01))

out = f"{OUT}/图20_消失路演变图.png"
fig.savefig(out, dpi=280, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out}", flush=True)

# GIF版本
print("Making GIF...", flush=True)
try:
    from PIL import Image
    frame_files = []
    for i, (lost_gdf, sig_label, time_label) in enumerate([
        (lost_s3,  "Signal 3",        "Sep 22 21:40 – Sep 25 08:20  |  59 slots"),
        (lost_s8,  "Signal 8",        "Sep 23 14:20 – Sep 24 20:20  |  39 slots"),
        (lost_s10, "Signal 10  PEAK", "Sep 24  02:40–13:20  |  22 slots"),
        (lost_s8,  "Signal 8",        "Sep 24  13:20–20:20  (descending)"),
        (lost_s3,  "Signal 3",        "Sep 24 20:20 – Sep 25 08:20  (descending)"),
    ]):
        fig_g, ax_g = plt.subplots(figsize=(13, 11))
        fig_g.patch.set_facecolor(BG)
        ax_g.set_facecolor(BG)
        ax_g.set_axis_off()
        lost_gdf.plot(ax=ax_g, color=COLOR, linewidth=0.20, alpha=0.85)
        n_lost = len(lost_gdf)
        pct = n_lost / n_total * 100
        ax_g.set_title(
            f"Ragasa  —  {sig_label}\n{time_label}\n"
            f"{n_lost:,} roads with no data  ({pct:.0f}% of network)",
            color="white", fontsize=12, fontweight="bold", pad=10)
        fname = f"/tmp/dis_frame_{i:02d}.png"
        fig_g.savefig(fname, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig_g)
        frame_files.append(fname)
        print(f"  Frame {i+1}: {n_lost:,} disappeared ({pct:.0f}%)", flush=True)

    imgs = [Image.open(f) for f in frame_files]
    durations = [1400, 1400, 2200, 1400, 1400]
    out_gif = f"{OUT}/图20_消失路动画.gif"
    imgs[0].save(out_gif, save_all=True, append_images=imgs[1:],
                 loop=0, duration=durations)
    print(f"Saved GIF: {out_gif}")
except ImportError:
    print("PIL not found")
