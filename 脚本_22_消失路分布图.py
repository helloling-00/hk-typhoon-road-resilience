"""
图22：Ragasa台风期间消失路段的空间分布
两个时段对比：
  左：S10 白天（Sep 24 06:00–13:20，工作日上午，台风峰值）
  右：S8 夜间（Sep 23 22:00 – Sep 24 01:40，工作日深夜）
参考基准：baseline_speed n_obs≥2 的工作日同时段路段
底图：CartoDB Positron（浅色）
"""
import ast, glob, pickle, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from shapely import wkb as shapely_wkb
from shapely.geometry import LineString
import geopandas as gpd
import contextily as ctx
import warnings; warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"
OUT  = "/Users/helloling/workspace/thesis"

# ── 定义两个时段的 (date, slot) 列表 ─────────────────────────────────────────
# S10 白天：Sep 24 06:00–13:20  slot12–26（工作日）
S10_DAY = [("2025-09-24", s) for s in range(12, 27)]   # 15 slots

# S8 夜间：Sep 23 22:00–23:30 + Sep 24 00:00–01:30  slot44-47 + slot0-3（工作日）
S8_NIGHT = ([("2025-09-23", s) for s in range(44, 48)] +
            [("2025-09-24", s) for s in range(0, 4)])   # 8 slots

# ── 加载 baseline，构建 (day_type, slot) → set[road_id] ──────────────────────
print("Loading baseline...", flush=True)
bl = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
bl2 = bl[bl["n_obs"] >= 2]
ref_lookup = {}
for (dt, sl), grp in bl2.groupby(["day_type", "slot"]):
    ref_lookup[(dt, sl)] = frozenset(grp["road_id"].values)
del bl, bl2

# ── 加载 ep_to_road ───────────────────────────────────────────────────────────
print("Loading ep_to_road...", flush=True)
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
ep_lkp = ep.set_index("ep_key")["road_id"]

def get_ep_key(wkb_bytes):
    try:
        g = shapely_wkb.loads(wkb_bytes)
        coords = list(g.coords)
        s, e = coords[0], coords[-1]
        s4 = (round(s[0],4), round(s[1],4))
        e4 = (round(e[0],4), round(e[1],4))
        return str((min(s4,e4), max(s4,e4)))
    except: return None

def observed_in_slots(slot_list):
    seen = set()
    for date_str, slot_i in slot_list:
        for f in glob.glob(f"{FLOW}/{date_str}/traffic_flow_zoom15_{date_str}_slot{slot_i:02d}_*.parquet"):
            df = pd.read_parquet(f, columns=["geometry","road_closure"])
            df = df[df["road_closure"] != 1]
            for wb in df["geometry"]:
                epk = get_ep_key(wb)
                if epk and epk in ep_lkp.index:
                    seen.add(int(ep_lkp[epk]))
    return seen

def ref_for_slots(slot_list, day_type="WORKDAY"):
    ref = set()
    for _, slot_i in slot_list:
        ref.update(ref_lookup.get((day_type, slot_i), frozenset()))
    return ref

# ── 读取各时段数据 ────────────────────────────────────────────────────────────
print("Reading S10 daytime flow...", flush=True)
obs_s10_day  = observed_in_slots(S10_DAY)
ref_s10_day  = ref_for_slots(S10_DAY)
dis_s10_day  = ref_s10_day - obs_s10_day
surv_s10_day = ref_s10_day & obs_s10_day
print(f"  ref={len(ref_s10_day):,}  obs={len(obs_s10_day):,}  disappeared={len(dis_s10_day):,}  ({len(dis_s10_day)/len(ref_s10_day)*100:.1f}%)", flush=True)

print("Reading S8 nighttime flow...", flush=True)
obs_s8_night  = observed_in_slots(S8_NIGHT)
ref_s8_night  = ref_for_slots(S8_NIGHT)
dis_s8_night  = ref_s8_night - obs_s8_night
surv_s8_night = ref_s8_night & obs_s8_night
print(f"  ref={len(ref_s8_night):,}  obs={len(obs_s8_night):,}  disappeared={len(dis_s8_night):,}  ({len(dis_s8_night)/len(ref_s8_night)*100:.1f}%)", flush=True)

# ── 加载路段几何 ──────────────────────────────────────────────────────────────
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
gdf_all = gpd.GeoDataFrame(ep_df, geometry="geometry", crs="EPSG:4326").to_crs("EPSG:3857")
print(f"  {len(gdf_all):,} roads with geometry", flush=True)

# ── 分类函数 ──────────────────────────────────────────────────────────────────
def classify(gdf, ref_set, dis_set, surv_set):
    def cat(rid):
        if rid in dis_set:  return "disappeared"
        if rid in surv_set: return "survived"
        return "not_ref"
    gdf = gdf.copy()
    gdf["status"] = gdf["road_id"].apply(cat)
    return gdf

gdf_s10 = classify(gdf_all, ref_s10_day,  dis_s10_day,  surv_s10_day)
gdf_s8  = classify(gdf_all, ref_s8_night, dis_s8_night, surv_s8_night)

# ── 绘图 ──────────────────────────────────────────────────────────────────────
print("Plotting...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(26, 14))
fig.patch.set_facecolor("white")

STYLE = {
    #  status       color       lw     alpha  zorder
    "not_ref":    ("#cccccc",  0.12,  0.35,  1),
    "disappeared":("#DC2626",  0.22,  0.80,  2),
    "survived":   ("#1D4ED8",  0.55,  0.95,  5),  # 最上层，更粗
}

panels = [
    (axes[0], gdf_s10,
     "Ragasa  Signal 10  —  Daytime\nSep 24  06:00–13:20  (typhoon peak, 15 slots)",
     dis_s10_day, ref_s10_day),
    (axes[1], gdf_s8,
     "Ragasa  Signal 8  —  Nighttime\nSep 23 22:00 – Sep 24 01:40  (8 slots)",
     dis_s8_night, ref_s8_night),
]

for ax, gdf, title, dis_set, ref_set in panels:
    ax.set_axis_off()

    # 绘制顺序：灰底 → 红（消失）→ 蓝（存活，最上层最粗）
    for status in ["not_ref", "disappeared", "survived"]:
        color, lw, alpha, zo = STYLE[status]
        sub = gdf[gdf["status"] == status]
        if len(sub): sub.plot(ax=ax, color=color, linewidth=lw, alpha=alpha, zorder=zo)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=13, alpha=0.55)

    n_dis = len(dis_set)
    n_ref = len(ref_set)
    pct   = n_dis / n_ref * 100
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.text(0.50, 0.015,
            f"Disappeared: {n_dis:,} / {n_ref:,} reference roads  ({pct:.1f}%)",
            transform=ax.transAxes, ha="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# 图例
legend_handles = [
    Line2D([0],[0], color="#DC2626", lw=2.5, label="Disappeared (in reference, not observed during typhoon)"),
    Line2D([0],[0], color="#2563EB", lw=2.5, label="Survived (observed despite typhoon signal)"),
    Line2D([0],[0], color="#cccccc", lw=1.5, label="Not in reference (below n_obs≥2 threshold normally)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           fontsize=10.5, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

fig.suptitle(
    "Spatial Distribution of Disappeared Roads  —  Typhoon Ragasa\n"
    "Reference: workday roads with n_obs≥2 in baseline  ·  Red = missing during typhoon  ·  Blue = still observed",
    fontsize=13.5, fontweight="bold", y=0.98)

plt.subplots_adjust(wspace=0.03, left=0.01, right=0.99, top=0.91, bottom=0.07)

out = f"{OUT}/图22_消失路分布图.png"
fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}", flush=True)
