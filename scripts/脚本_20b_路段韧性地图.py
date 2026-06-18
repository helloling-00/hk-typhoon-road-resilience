"""
图20：台风Ragasa期间路段数据韧性地图（单图）
每条路按其"消失阈值"着色：
  蓝色  = 存活至 Signal 10（最强韧）
  绿色  = Signal 8 时消失（S10时不见）
  黄色  = Signal 3 时消失（S8时不见）
  红色  = Signal 1 时消失（S3时不见）
  深灰  = 台风全程未见（never observed during Ragasa）
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

r_s1  = tsets["r_s1"]
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

# 分配韧性等级
def resilience_tier(rid):
    if rid in r_s10: return "survived_s10"
    if rid in r_s8:  return "lost_at_s10"
    if rid in r_s3:  return "lost_at_s8"
    if rid in r_s1:  return "lost_at_s3"
    return "never_ragasa"

print("Assigning resilience tiers...", flush=True)
gdf["tier"] = gdf["road_id"].apply(resilience_tier)
print(gdf["tier"].value_counts().to_string())

# 韧性等级配色与线宽（从底层到顶层绘制）
tier_cfg = [
    # tier,            color,     lw,    alpha,  zorder
    ("never_ragasa",  "#252525",  0.14,  0.80,   1),
    ("lost_at_s3",    "#CC2200",  0.18,  0.90,   2),  # 红：S1即消失
    ("lost_at_s8",    "#FF8800",  0.20,  0.92,   3),  # 橙：S3存活S8消失
    ("lost_at_s10",   "#FFE033",  0.22,  0.93,   4),  # 黄：S8存活S10消失
    ("survived_s10",  "#00AAFF",  0.26,  0.95,   5),  # 蓝：S10仍存活
]

BG = "#080c14"
print("Plotting...", flush=True)
fig, ax = plt.subplots(figsize=(16, 14))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_axis_off()

for tier, color, lw, alpha, zo in tier_cfg:
    grp = gdf[gdf["tier"] == tier]
    if len(grp) == 0: continue
    grp.plot(ax=ax, color=color, linewidth=lw, alpha=alpha, zorder=zo)

# 图例
counts = gdf["tier"].value_counts()
legend_items = [
    ("survived_s10",  "#00AAFF", f"Survived Signal 10  ({counts.get('survived_s10',0):,} roads, {counts.get('survived_s10',0)/len(gdf)*100:.0f}%)"),
    ("lost_at_s10",   "#FFE033", f"Lost at Signal 10  ({counts.get('lost_at_s10',0):,}, {counts.get('lost_at_s10',0)/len(gdf)*100:.0f}%)"),
    ("lost_at_s8",    "#FF8800", f"Lost at Signal 8  ({counts.get('lost_at_s8',0):,}, {counts.get('lost_at_s8',0)/len(gdf)*100:.0f}%)"),
    ("lost_at_s3",    "#CC2200", f"Lost at Signal 3  ({counts.get('lost_at_s3',0):,}, {counts.get('lost_at_s3',0)/len(gdf)*100:.0f}%)"),
    ("never_ragasa",  "#252525", f"Never observed during Ragasa  ({counts.get('never_ragasa',0):,}, {counts.get('never_ragasa',0)/len(gdf)*100:.0f}%)"),
]
handles = [Line2D([0],[0], color=c, lw=2.5, label=l) for _, c, l in legend_items]
leg = ax.legend(handles=handles, loc="lower left", framealpha=0.60,
                facecolor="#0d1420", edgecolor="#445566",
                labelcolor="white", fontsize=11,
                title="Data Survival under Typhoon Ragasa (Signal 1→10)",
                title_fontsize=11.5)
leg.get_title().set_color("white")

ax.set_title(
    "Road Segment Data Resilience  —  Typhoon Ragasa (max Signal 10)\n"
    "Colour shows the last signal level at which floating-car data was observed",
    color="white", fontsize=13.5, pad=14, fontweight="bold")

fig.text(0.5, 0.012,
    f"TomTom floating-car data · {len(gdf):,} road segments · "
    f"Signal periods: S1 (Sep 22–25), S3, S8, S10 · Hong Kong SAR",
    ha="center", color="#7799aa", fontsize=9.5)

plt.tight_layout(rect=[0, 0.03, 1, 1])
out = f"{OUT}/图20_路段韧性地图.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"\nSaved: {out}")
