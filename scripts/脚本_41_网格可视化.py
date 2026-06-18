"""
脚本_41_网格可视化.py
单个网格：道路颜色 = mean_deviation，叠加POI，展示Y和X变量
选格：22.304054_114.171826（旺角高Y，pct_better=0.83）
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from shapely.geometry import LineString, box, Point
from shapely import wkb
import osmium, ast, os, warnings
warnings.filterwarnings("ignore")
import contextily as ctx

DATA    = "/Users/helloling/workspace/thesis/data"
PBF     = "/Users/helloling/workspace/thesis/hong-kong-260502.osm.pbf"
GRID_ID = "22.304054_114.171826"
GRID_LAT, GRID_LON = 22.304054, 114.171826
CELL_LAT = 500 / 111000
CELL_LON = 500 / (111000 * np.cos(np.radians(22.3)))
DELTA    = 0.05

# ── 1. 格子边界 ───────────────────────────────────────────────────────────────
lat0 = GRID_LAT - CELL_LAT / 2;  lat1 = GRID_LAT + CELL_LAT / 2
lon0 = GRID_LON - CELL_LON / 2;  lon1 = GRID_LON + CELL_LON / 2
grid_box = box(lon0, lat0, lon1, lat1)
print(f"Grid bounds: lon {lon0:.4f}–{lon1:.4f}, lat {lat0:.4f}–{lat1:.4f}")

# ── 2. 该格内路段及其deviation ────────────────────────────────────────────────
rt = pd.read_parquet(f"{DATA}/regression_table.parquet")
rr = pd.read_parquet(f"{DATA}/road_registry.parquet").drop_duplicates("road_id")

def parse_center(s):
    try:
        (lon1_,lat1_),(lon2_,lat2_) = ast.literal_eval(s)
        return (lon1_+lon2_)/2,(lat1_+lat2_)/2
    except: return None,None

centers = rr["ep_key"].apply(parse_center)
rr = rr.copy()
rr["cx"] = centers.apply(lambda x: x[0])
rr["cy"] = centers.apply(lambda x: x[1])
rr["grid_lat"] = (rr["cy"]//CELL_LAT)*CELL_LAT+CELL_LAT/2
rr["grid_lon"] = (rr["cx"]//CELL_LON)*CELL_LON+CELL_LON/2
rr["grid_id"]  = rr["grid_lat"].round(6).astype(str)+"_"+rr["grid_lon"].round(6).astype(str)

grid_road_ids = set(rr[rr.grid_id==GRID_ID]["road_id"].tolist())

# Ragasa S3+ MIDDAY，按road_id取均值
rag = rt[(rt.typhoon=="Ragasa") & (rt.signal_level>=3) &
         (rt.time_group=="MIDDAY") & (rt.road_id.isin(grid_road_ids))].copy()
road_dev = (rag.groupby("road_id")
              .agg(mean_dev=("mean_deviation","mean"),
                   road_length_m=("road_length_m","first"),
                   road_broad=("road_broad","first"))
              .reset_index())
road_dev["clearly_better"] = road_dev["mean_dev"] > DELTA
road_dev["clearly_worse"]  = road_dev["mean_dev"] < -DELTA

print(f"Roads in grid with data: {len(road_dev)}")
print(f"  clearly better: {road_dev.clearly_better.sum()}")
print(f"  clearly worse:  {road_dev.clearly_worse.sum()}")
print(f"  in between:     {(~road_dev.clearly_better & ~road_dev.clearly_worse).sum()}")

# ── 3. 提取道路几何 ───────────────────────────────────────────────────────────
print("Extracting road geometries...", flush=True)
target_eps = {}
for rid in road_dev["road_id"]:
    rows = rr[rr.road_id==rid]
    if len(rows)>0:
        target_eps[rows["ep_key"].iloc[0]] = rid

flow_dir = f"{DATA}/flow_parquet2/2025-09-19"
slots = sorted([f for f in os.listdir(flow_dir)
                if any(x in f for x in ["slot16","slot18","slot20","slot22","slot24","slot26","slot28"])])

ep_to_geom = {}
for fname in slots:
    fp = os.path.join(flow_dir, fname)
    flow = pd.read_parquet(fp)
    for _, row in flow.iterrows():
        try:
            g = wkb.loads(row["geometry"])
            coords = list(g.geoms[0].coords) if g.geom_type=="MultiLineString" else list(g.coords)
            if len(coords)<2: continue
            lo1,la1 = round(coords[0][0],4), round(coords[0][1],4)
            lo2,la2 = round(coords[-1][0],4), round(coords[-1][1],4)
            ep = str(((min(lo1,lo2),min(la1,la2)),(max(lo1,lo2),max(la1,la2))))
            if ep in target_eps and ep not in ep_to_geom:
                ep_to_geom[ep] = g
        except: pass
    if len(ep_to_geom) >= len(target_eps)*0.85: break

print(f"  Found {len(ep_to_geom)}/{len(target_eps)} geometries")

# 合并几何到road_dev
ep_rid = {v:k for k,v in target_eps.items()}   # rid→ep
road_dev["ep_key"] = road_dev["road_id"].map(lambda r: ep_rid.get(r))
road_dev["geometry"] = road_dev["ep_key"].map(lambda ep: ep_to_geom.get(ep))
road_dev = road_dev[road_dev["geometry"].notna()].copy()
print(f"  Roads with geometry: {len(road_dev)}")

gdf_roads = gpd.GeoDataFrame(road_dev, geometry="geometry", crs="EPSG:4326")

# ── 4. POI ────────────────────────────────────────────────────────────────────
print("Extracting POI...", flush=True)
PAD = 0.003
bbox_poi = (lon0-PAD, lat0-PAD, lon1+PAD, lat1+PAD)

POI_COLORS = {
    "work":("#4e79a7","Work"), "education":("#f28e2b","Education"),
    "retail":("#e15759","Retail"), "food_drink":("#76b7b2","Food & Drink"),
    "recreation":("#59a14f","Recreation"), "medical":("#edc948","Medical"),
    "transport":("#b07aa1","Transport"), "tourism":("#ff9da7","Tourism"),
    "finance":("#9c755f","Finance"), "civic":("#bab0ac","Civic"),
}

def classify(tags):
    am=tags.get("amenity",""); sh=tags.get("shop",""); to=tags.get("tourism","")
    of=tags.get("office",""); le=tags.get("leisure","")
    rw=tags.get("railway",""); pt=tags.get("public_transport","")
    lu=tags.get("landuse","")
    if rw in ("station","subway_entrance","tram_stop","halt") or \
       pt in ("station","stop_position","platform") or \
       am in ("bus_station","ferry_terminal","taxi","parking"): return "transport"
    if am in ("hospital","clinic","pharmacy","doctors","dentist"): return "medical"
    if am in ("school","university","college","kindergarten","library") or lu=="education": return "education"
    if am in ("restaurant","cafe","fast_food","bar","pub","food_court"): return "food_drink"
    if sh in ("supermarket","convenience","mall","department_store","clothes",
              "electronics","hardware","books","bakery") or am=="marketplace": return "retail"
    if to in ("hotel","hostel","attraction","museum","viewpoint","gallery") or \
       am in ("arts_centre","cinema","theatre"): return "tourism"
    if am in ("bank","atm") or of=="financial": return "finance"
    if le in ("park","sports_centre","fitness_centre","swimming_pool","playground","pitch") or \
       am in ("gym",): return "recreation"
    if of in ("company","commercial","it","consulting","architect","engineer",
              "insurance","lawyer","ngo") or lu in ("commercial","office","industrial"): return "work"
    if am in ("police","post_office","fire_station","courthouse",
              "townhall","community_centre") or of=="government": return "civic"
    return None

class POIHandler(osmium.SimpleHandler):
    def __init__(self, bbox):
        super().__init__(); self.bbox=bbox; self.features=[]
    def node(self, n):
        if not n.location.valid(): return
        lo,la = n.location.lon, n.location.lat
        bx = self.bbox
        if not (bx[0]<=lo<=bx[2] and bx[1]<=la<=bx[3]): return
        tags = {t.k:t.v for t in n.tags}
        cat = classify(tags)
        if cat: self.features.append({"cat":cat,"lon":lo,"lat":la})

h = POIHandler(bbox_poi)
h.apply_file(PBF, locations=True)
poi_df = pd.DataFrame(h.features)
if len(poi_df)>0:
    gdf_poi = gpd.GeoDataFrame(poi_df,
        geometry=[Point(r.lon,r.lat) for r in poi_df.itertuples()], crs="EPSG:4326")
    gdf_poi = gdf_poi[gdf_poi.geometry.within(grid_box)].copy()
else:
    gdf_poi = gpd.GeoDataFrame()
print(f"  POI in grid: {len(gdf_poi)}")
if len(gdf_poi)>0: print(gdf_poi["cat"].value_counts().to_string())

# ── 5. 网格变量值（从grid_pct_better取） ─────────────────────────────────────
grd = pd.read_parquet(f"{DATA}/grid_pct_better.parquet")
grd_row = grd[(grd.grid_id==GRID_ID) & (grd.time_group=="MIDDAY")]
if len(grd_row)>0:
    gv = grd_row.iloc[0]
else:
    gv = None

# ── 6. 作图 ───────────────────────────────────────────────────────────────────
print("Plotting...", flush=True)
fig = plt.figure(figsize=(16, 9))
gs  = fig.add_gridspec(1, 2, width_ratios=[1.6, 1])
ax_map = fig.add_subplot(gs[0])
ax_tab = fig.add_subplot(gs[1]); ax_tab.set_axis_off()

# 转投影
gdf_roads_p = gdf_roads.to_crs(epsg=3857)
gdf_poi_p   = gdf_poi.to_crs(epsg=3857) if len(gdf_poi)>0 else None
gdf_box_p   = gpd.GeoDataFrame(geometry=[grid_box], crs="EPSG:4326").to_crs(epsg=3857)

# 道路颜色：deviation → RdYlGn
dev_vals = gdf_roads["mean_dev"].values
dev_max  = max(abs(dev_vals.min()), abs(dev_vals.max()), 0.15)
norm_dev = mcolors.TwoSlopeNorm(vmin=-dev_max, vcenter=0, vmax=dev_max)
cmap_dev = cm.RdYlGn

lw_map = {"motorway":4,"trunk":3.5,"primary":3,"secondary":2.5,"tertiary":2,"other":1.5}

# 画道路
for _, row in gdf_roads_p.iterrows():
    color = cmap_dev(norm_dev(row["mean_dev"]))
    lw    = lw_map.get(str(row.get("road_broad","other")), 1.5)
    geom  = row.geometry
    lines = list(geom.geoms) if geom.geom_type=="MultiLineString" else [geom]
    for line in lines:
        x, y = line.xy
        ax_map.plot(x, y, color=color, linewidth=lw,
                    solid_capstyle="round", zorder=4)

# 网格边界
gdf_box_p.boundary.plot(ax=ax_map, color="black", linewidth=2.5,
                         linestyle="--", zorder=6)

# POI
if gdf_poi_p is not None and len(gdf_poi_p)>0:
    for cat, (color, label) in POI_COLORS.items():
        sub = gdf_poi_p[gdf_poi_p["cat"]==cat]
        if len(sub)>0:
            sub.plot(ax=ax_map, color=color, markersize=22,
                     alpha=0.8, zorder=5, marker="o")

ctx.add_basemap(ax_map, crs="EPSG:3857",
                source=ctx.providers.CartoDB.Positron, zoom=16)

# colorbar
sm = cm.ScalarMappable(norm=norm_dev, cmap=cmap_dev); sm.set_array([])
cb = plt.colorbar(sm, ax=ax_map, shrink=0.55, pad=0.02)
cb.set_label("Speed Deviation\n(typhoon − baseline)", fontsize=9)

ax_map.set_axis_off()
pct_b = gv["pct_better"] if gv is not None else float("nan")
ax_map.set_title(
    f"Grid {GRID_ID}  (Mong Kok / 旺角)\n"
    f"Ragasa S3+, Midday  |  Y = pct_better = {pct_b:.2f}  |  {len(gdf_roads)} roads shown",
    fontsize=10, fontweight="bold", pad=6)

# 图例（道路宽度 + POI颜色）
road_handles = [
    mlines.Line2D([],[],color="gray",lw=lw_map.get(t,1.5),linestyle="-",label=t.capitalize())
    for t in ["primary","secondary","tertiary","other"]
]
poi_handles = [
    mlines.Line2D([],[],color=c,marker="o",markersize=7,
                  linestyle="None",label=label)
    for cat,(c,label) in POI_COLORS.items()
    if gdf_poi_p is not None and (gdf_poi_p["cat"]==cat).sum()>0
]
grid_h = mpatches.Patch(facecolor="none",edgecolor="black",
                         linestyle="--",linewidth=2,label="500×500m grid")
ax_map.legend(handles=[grid_h]+road_handles+poi_handles,
              loc="lower left", fontsize=7, title="Legend",
              title_fontsize=8, framealpha=0.9)

# ── 右侧：变量值表格 + 路段列表 ───────────────────────────────────────────────
y0 = 0.98; dy = 0.038

def draw_row(ax, y, cells, widths, bg, fc="black", bold=False):
    x = 0.0
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, y-dy*0.85), 1.0, dy*0.88, boxstyle="square,pad=0",
        linewidth=0, facecolor=bg, transform=ax.transAxes, clip_on=False))
    for txt, w in zip(cells, widths):
        ax.text(x+0.01, y-dy*0.35, str(txt), transform=ax.transAxes,
                fontsize=7.5, va="center", color=fc,
                fontweight="bold" if bold else "normal")
        x += w

# 标题
draw_row(ax_tab, y0, ["Variable", "Value"], [0.65, 0.35], "#222222", "white", bold=True)
y0 -= dy

# Y变量
rows_tab = []
if gv is not None:
    rows_tab += [
        ("── OUTCOME ──────────────────", "", "#e8e8e8"),
        ("pct_better (Y)", f"{gv['pct_better']:.3f}", "#fce8e8"),
        ("pct_worse",      f"{gv['pct_worse']:.3f}", "#fce8e8"),
        ("mean_deviation", f"{gv['mean_dev']:+.3f}", "#fce8e8"),
        ("n_roads in grid", f"{int(gv['n_roads'])}", "#fce8e8"),
        ("total_length (km)", f"{gv['total_length_m']/1000:.2f}", "#fce8e8"),
        ("── POI DENSITY ─────────────", "", "#e8e8e8"),
    ]
    for cat in ["work","education","retail","food_drink","recreation",
                "medical","transport","tourism","finance","civic"]:
        col = f"log_{cat}"
        val = gv.get(col, float("nan"))
        cnt = int((gdf_poi["cat"]==cat).sum()) if len(gdf_poi)>0 else 0
        rows_tab.append((f"log_{cat}", f"{val:.2f}  ({cnt} pts)", "#e8fce8"))
    rows_tab += [
        ("── DEMOGRAPHICS ────────────", "", "#e8e8e8"),
        ("log_pop_density", f"{gv.get('log_pop_density',float('nan')):.2f}", "#fff8e1"),
        ("log_income",      f"{gv.get('log_income',float('nan')):.2f}", "#fff8e1"),
        ("working_pop_ratio", f"{gv.get('working_pop_ratio_500m',float('nan')):.3f}", "#fff8e1"),
        ("── ROAD STRUCTURE ──────────", "", "#e8e8e8"),
        ("log_intersection", f"{gv.get('log_intersection',float('nan')):.2f}", "#e8f0fe"),
        ("log_road_density", f"{gv.get('log_road_density',float('nan')):.2f}", "#e8f0fe"),
        ("log_dist_coast",   f"{gv.get('log_dist_coast2',float('nan')):.2f}", "#e8f0fe"),
    ]

for label, val, bg in rows_tab:
    draw_row(ax_tab, y0, [label, val], [0.65, 0.35], bg)
    y0 -= dy
    if y0 < 0.02: break

# 各路段deviation列表
y0 -= dy*0.3
ax_tab.text(0.01, y0, "Road-level deviations (Ragasa S3+ MIDDAY avg):",
            transform=ax_tab.transAxes, fontsize=7.5, fontweight="bold")
y0 -= dy*0.8
for _, r in road_dev.sort_values("mean_dev").iterrows():
    col = "#d73027" if r.mean_dev < -DELTA else ("#1a9850" if r.mean_dev > DELTA else "#999999")
    tag = "▼worse" if r.mean_dev < -DELTA else ("▲better" if r.mean_dev > DELTA else "  ~")
    ax_tab.text(0.01, y0,
        f"id={r.road_id}  {r.road_broad:<10}  {r.mean_dev:+.3f}  {tag}",
        transform=ax_tab.transAxes, fontsize=6.8, color=col, va="center")
    y0 -= dy*0.75
    if y0 < 0.01: break

ax_tab.set_title(f"Grid Summary — Midday, Ragasa S3+\n(δ = {DELTA} for 'clearly better')",
                  fontsize=9.5, fontweight="bold", pad=6)

plt.tight_layout()
out = "/Users/helloling/workspace/thesis/图41_网格可视化.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()
