"""
图20：Ragasa台风不同信号阶段路段存活演变
设计：每格以全量基线路网作为暗色参考底层，存活路段用亮色叠加
消失的路 = 可见的灰色暗线；存活的路 = 鲜艳彩色
四格：Ragasa S1 / S3 / S8 / S10
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

DATA  = "/Users/helloling/workspace/thesis/data"
OUT   = "/Users/helloling/workspace/thesis"
WKB_CACHE = f"{DATA}/osm_cache/road_wkb_store.pkl"

# ── 加载路段集合 ──────────────────────────────────────────────────────────────
print("Loading road sets...", flush=True)
with open(f"{DATA}/osm_cache/typhoon_road_sets.pkl","rb") as f:
    tsets = pickle.load(f)

bl_roads  = tsets["bl_road_ids"]
r_s1      = tsets["r_s1"]
r_s3      = tsets["r_s3"]
r_s8      = tsets["r_s8"]
r_s10     = tsets["r_s10"]

# ── 加载 TomTom 路段几何 ──────────────────────────────────────────────────────
print("Loading geometries...", flush=True)
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
rr = pd.read_parquet(f"{DATA}/road_registry.parquet")[
        ["road_id","road_category"]].drop_duplicates("road_id")
with open(WKB_CACHE,"rb") as f:
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
gdf_all = gpd.GeoDataFrame(ep, geometry="geometry", crs="EPSG:4326").to_crs("EPSG:3857")
print(f"  {len(gdf_all):,} roads", flush=True)

cat_color = {
    "motorway":      "#FF4444", "motorway_link": "#FF7777",
    "trunk":         "#FF9900", "trunk_link":    "#FFCC66",
    "primary":       "#FFE033", "primary_link":  "#FFF080",
    "secondary":     "#44BB44", "secondary_link":"#88DD88",
    "tertiary":      "#44AAFF", "tertiary_link": "#88CCFF",
    "street":        "#CC66FF",
    "service":       "#C0B0A0",
}
def get_color(cat):
    if pd.isna(cat): return "#C0B0A0"
    return cat_color.get(str(cat).lower(), "#C0B0A0")

def lw_for_cat(cat, scale=1.0):
    if pd.isna(cat): return 0.17 * scale
    c = str(cat).lower()
    if "motorway" in c: return 0.30 * scale
    if "trunk" in c:    return 0.26 * scale
    if "primary" in c:  return 0.22 * scale
    return 0.18 * scale

cat_order = ["service","street","tertiary_link","tertiary",
             "secondary_link","secondary","primary_link","primary",
             "trunk_link","trunk","motorway_link","motorway"]

BG = "#080c14"

# ── 绘制单格 ──────────────────────────────────────────────────────────────────
def draw_panel(ax, survive_set, title, subtitle, note_s1=False):
    ax.set_facecolor(BG)
    ax.set_axis_off()

    n_total = len(gdf_all)
    n_surv  = sum(1 for rid in gdf_all["road_id"] if rid in survive_set)
    n_lost  = n_total - n_surv
    pct_lost = n_lost / n_total * 100

    # 1) 消失路：暗橙灰，明显可见但不抢眼
    gdf_lost = gdf_all[~gdf_all["road_id"].isin(survive_set)]
    gdf_lost.plot(ax=ax, color="#3a2e20", linewidth=0.18, alpha=0.85)

    # 2) 存活路：鲜艳彩色，按类别从细到粗叠加
    gdf_surv = gdf_all[gdf_all["road_id"].isin(survive_set)]
    for cat in cat_order:
        grp = gdf_surv[gdf_surv["road_category"] == cat]
        if len(grp) == 0: continue
        grp.plot(ax=ax, color=get_color(cat), linewidth=lw_for_cat(cat), alpha=0.95)

    ax.set_title(title, color="white", fontsize=12, pad=6, fontweight="bold")
    ax.text(0.50, -0.01,
            f"{subtitle}   ·   Survived {n_surv:,} / {n_total:,}  ({100-pct_lost:.0f}%)   "
            f"Lost {n_lost:,} ({pct_lost:.0f}%)",
            transform=ax.transAxes, color="#99bbcc", fontsize=9,
            ha="center", va="top")
    if note_s1:
        ax.text(0.50, 0.04,
                "⚠ Data limited: 09-22 had 32 missing slots (05:00–20:30)",
                transform=ax.transAxes, color="#ffaa44", fontsize=8.5,
                ha="center", va="bottom")

# ── 静态4格图 ─────────────────────────────────────────────────────────────────
print("Drawing 4-panel figure...", flush=True)
fig, axes = plt.subplots(2, 2, figsize=(24, 19))
fig.patch.set_facecolor(BG)
plt.subplots_adjust(hspace=0.10, wspace=0.04, top=0.92, bottom=0.08, left=0.01, right=0.99)

draw_panel(axes[0,0], r_s1,
           "Ragasa  —  Signal 1",
           "Sep 22  12:20–21:40  (ascending)  |  27 slots",
           note_s1=True)
draw_panel(axes[0,1], r_s3,
           "Ragasa  —  Signal 3",
           "Sep 22 21:40 – Sep 25 08:20  |  59 slots")
draw_panel(axes[1,0], r_s8,
           "Ragasa  —  Signal 8",
           "Sep 23 14:20 – Sep 24 20:20  |  39 slots")
draw_panel(axes[1,1], r_s10,
           "Ragasa  —  Signal 10  ▲ PEAK",
           "Sep 24  02:40–13:20  |  22 slots")

# 图例
legend_items = [
    ("Motorway",                           "#FF4444"),
    ("Trunk",                              "#FF9900"),
    ("Primary",                            "#FFE033"),
    ("Secondary",                          "#44BB44"),
    ("Tertiary",                           "#44AAFF"),
    ("Street / residential / unclassified","#CC66FF"),
    ("Service / other",                    "#C0B0A0"),
    ("Disappeared — below probe threshold","#3a2e20"),
]
handles = [Line2D([0],[0], color=c, lw=2.2, label=l) for l, c in legend_items]
fig.legend(handles=handles, loc="lower center", ncol=4,
           framealpha=0.60, facecolor="#0d1420", edgecolor="#445566",
           labelcolor="white", fontsize=10, bbox_to_anchor=(0.5, 0.01))

fig.suptitle(
    "Typhoon Ragasa: Floating-Car Data Survival by Signal Level\n"
    "Bright = roads observed in this period  ·  Dark amber = disappeared (below TomTom probe-density threshold)",
    color="white", fontsize=13.5, fontweight="bold", y=0.965)

out_static = f"{OUT}/图20_路段消失演变图.png"
fig.savefig(out_static, dpi=250, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out_static}", flush=True)

# ── GIF：Ragasa 信号全程升降 ──────────────────────────────────────────────────
print("Drawing GIF...", flush=True)
frames_def = [
    (r_s1,  "Ragasa  Signal 1  (ascending)\nSep 22  12:20–21:40  [data limited: 27 slots]", True),
    (r_s3,  "Ragasa  Signal 3  (ascending)\nSep 22 21:40 – Sep 23 14:20  [59 slots]", False),
    (r_s8,  "Ragasa  Signal 8  (ascending)\nSep 23 14:20 – Sep 24 01:40  [39 slots]", False),
    (r_s10, "Ragasa  Signal 10  ▲ PEAK\nSep 24  02:40–13:20  [22 slots]", False),
    (r_s8,  "Ragasa  Signal 8  (descending)\nSep 24  13:20–20:20", False),
    (r_s3,  "Ragasa  Signal 3  (descending)\nSep 24 20:20 – Sep 25 08:20", False),
    (r_s1,  "Ragasa  Signal 1  (clearing)\nSep 25  08:20–11:20", False),
]

fig_g, ax_g = plt.subplots(figsize=(13, 11))
fig_g.patch.set_facecolor(BG)
plt.subplots_adjust(top=0.88, bottom=0.04, left=0.01, right=0.99)

frame_files = []
for i, (surv_set, label, warn) in enumerate(frames_def):
    ax_g.cla()
    ax_g.set_facecolor(BG)
    ax_g.set_axis_off()

    gdf_lost = gdf_all[~gdf_all["road_id"].isin(surv_set)]
    gdf_surv = gdf_all[gdf_all["road_id"].isin(surv_set)]

    gdf_lost.plot(ax=ax_g, color="#3a2e20", linewidth=0.18, alpha=0.85)
    for cat in cat_order:
        grp = gdf_surv[gdf_surv["road_category"] == cat]
        if len(grp): grp.plot(ax=ax_g, color=get_color(cat), linewidth=lw_for_cat(cat), alpha=0.95)

    n_surv = len(gdf_surv)
    n_lost = len(gdf_lost)
    pct_lost = n_lost / len(gdf_all) * 100

    title_str = f"{label}\nSurvived: {n_surv:,}  |  Lost: {n_lost:,} ({pct_lost:.0f}%)"
    if warn:
        title_str += "\n⚠ Limited data: 32 slots missing on Sep 22"
    ax_g.set_title(title_str, color="white", fontsize=11, pad=8, fontweight="bold")

    fname = f"/tmp/gif_frame_{i:02d}.png"
    fig_g.savefig(fname, dpi=150, bbox_inches="tight", facecolor=BG)
    frame_files.append(fname)
    print(f"  Frame {i+1}/{len(frames_def)}: survived {n_surv:,}, lost {n_lost:,} ({pct_lost:.0f}%)", flush=True)

plt.close(fig_g)

try:
    from PIL import Image
    imgs = [Image.open(f) for f in frame_files]
    # 在 S10 帧停留更长（800ms vs 其他1200ms）
    durations = [1200]*len(imgs)
    durations[3] = 2000  # S10 peak stays longer
    out_gif = f"{OUT}/图20_路段消失动画.gif"
    imgs[0].save(out_gif, save_all=True, append_images=imgs[1:],
                 loop=0, duration=durations)
    print(f"Saved GIF: {out_gif}")
except ImportError:
    print("PIL not found — frames at /tmp/gif_frame_*.png")
    print("Run: convert -delay 120 -loop 0 /tmp/gif_frame_*.png 图20_路段消失动画.gif")
