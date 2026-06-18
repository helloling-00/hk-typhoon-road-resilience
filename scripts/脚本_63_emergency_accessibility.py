"""
应急可达性分析（Buffer 法）
对每个 estate, 算 3 km buffer 内主干路 baseline / 09-24 中段平均速度
+ 最近 fire / hospital / police 的距离
+ 与人口结构 (老人比、收入、人口规模) 的关联
"""
import os, ast, pickle
import pandas as pd, numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely import wkb as shapely_wkb
from shapely.geometry import LineString, Point
import contextily as ctx
import warnings; warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 220,
    "font.size": 12, "axes.titlesize": 13.5, "axes.labelsize": 12,
})

DATA = "/Users/helloling/workspace/thesis/data"
OUT  = "/Users/helloling/workspace/thesis"

BUFFER_M = 3000   # 3 km
KEEP = {"motorway","motorway_link","trunk","trunk_link",
        "primary","primary_link","secondary","secondary_link",
        "tertiary","tertiary_link"}

print("Loading...")
estates = pd.read_parquet(f"{DATA}/estate_features.parquet")
emerg   = gpd.read_file(f"{DATA}/osm_cache/hk_emergency.gpkg")
md      = pd.read_csv(f"{OUT}/midday_road_dev.csv")
rr      = pd.read_parquet(f"{DATA}/road_registry.parquet")[
            ["road_id","road_category"]].drop_duplicates("road_id")
ep_df   = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
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

ep_df["geometry"] = ep_df.apply(build_geom, axis=1)
ep_df = ep_df.dropna(subset=["geometry"])
roads = gpd.GeoDataFrame(ep_df[["road_id","geometry"]],
                         crs="EPSG:4326").to_crs(epsg=2326)
roads = roads.merge(rr, on="road_id").merge(md[["road_id","bl_speed","typh_speed","dev"]], on="road_id")
roads = roads[roads["road_category"].isin(KEEP)].copy()
roads["length_m"] = roads.geometry.length
print(f"  {len(roads):,} main roads with mid-day dev")

# Estates → GeoDataFrame
est = gpd.GeoDataFrame(
    estates,
    geometry=gpd.points_from_xy(estates["lon"], estates["lat"]),
    crs="EPSG:4326").to_crs(epsg=2326)

# Emergency 拆分
emerg = emerg.to_crs(epsg=2326)
emerg["centroid"] = emerg.geometry.centroid
fire = emerg[emerg["amenity"]=="fire_station"].copy()
hosp = emerg[emerg["amenity"]=="hospital"].copy()
pol  = emerg[emerg["amenity"]=="police"].copy()
fire_pts = gpd.GeoDataFrame(geometry=fire["centroid"].values, crs=emerg.crs)
hosp_pts = gpd.GeoDataFrame(geometry=hosp["centroid"].values, crs=emerg.crs)
pol_pts  = gpd.GeoDataFrame(geometry=pol["centroid"].values,  crs=emerg.crs)
print(f"  fire={len(fire_pts)} hosp={len(hosp_pts)} police={len(pol_pts)}")

# 每个 estate: 3 km buffer 内主干路 length-weighted dev
print(f"Computing accessibility (buffer {BUFFER_M/1000:.1f} km)...")
est["buf"] = est.geometry.buffer(BUFFER_M)
buf_gdf = est[["estate","buf"]].set_geometry("buf")

joined = gpd.sjoin(roads, buf_gdf, predicate="intersects", how="inner")
agg = joined.groupby("estate").apply(lambda g: pd.Series({
    "n_roads": len(g),
    "tot_length_km": g["length_m"].sum() / 1000,
    "bl_acc":  np.average(g["bl_speed"],   weights=g["length_m"]),
    "typh_acc":np.average(g["typh_speed"], weights=g["length_m"]),
    "dev_acc": np.average(g["dev"],        weights=g["length_m"]),
    "n_worse_10": (g["dev"] < -0.10).sum(),
    "n_better_10":(g["dev"] >  0.10).sum(),
})).reset_index()
print(f"  {len(agg):,} estates with road coverage in buffer")

# 最近 fire / hospital / police 距离 (米)
def nearest_dist(pt, pts_gdf):
    if len(pts_gdf)==0: return np.nan
    return pts_gdf.distance(pt).min()
agg = agg.merge(est[["estate","geometry","total_pop","ratio_age_65plus",
                     "ratio_age_0_14","working_pop","median_income","lat","lon"]],
                on="estate")
agg = gpd.GeoDataFrame(agg, geometry="geometry", crs=est.crs)
agg["d_fire_m"]  = agg.geometry.apply(lambda p: nearest_dist(p, fire_pts))
agg["d_hosp_m"]  = agg.geometry.apply(lambda p: nearest_dist(p, hosp_pts))
agg["d_pol_m"]   = agg.geometry.apply(lambda p: nearest_dist(p, pol_pts))

# Travel-time proxy:
# baseline t = d_nearest / (free_flow_kmh * bl_acc) 单位换 min
# typhoon  t = d_nearest / (free_flow_kmh * typh_acc)
FREE_FLOW_KMH = 50  # 主干道 free-flow 估算
def t_min(d_m, acc):
    if pd.isna(d_m) or pd.isna(acc) or acc<=0.05: return np.nan
    return (d_m/1000) / (FREE_FLOW_KMH*acc) * 60

agg["t_fire_bl"]   = [t_min(d, a) for d,a in zip(agg["d_fire_m"], agg["bl_acc"])]
agg["t_fire_typh"] = [t_min(d, a) for d,a in zip(agg["d_fire_m"], agg["typh_acc"])]
agg["d_t_fire"]    = agg["t_fire_typh"] - agg["t_fire_bl"]
agg["t_hosp_bl"]   = [t_min(d, a) for d,a in zip(agg["d_hosp_m"], agg["bl_acc"])]
agg["t_hosp_typh"] = [t_min(d, a) for d,a in zip(agg["d_hosp_m"], agg["typh_acc"])]
agg["d_t_hosp"]    = agg["t_hosp_typh"] - agg["t_hosp_bl"]

# ── 1. 总体描述 ───────────────────────────────────────────────────────────────
print("\n=== Estate-level accessibility (mid-day 09-24, S10/S8) ===")
print(f"  estates with coverage : {len(agg):,} / {len(est):,}")
print(f"  bl  accessibility mean: {agg['bl_acc'].mean():.3f}")
print(f"  typh accessibility mean: {agg['typh_acc'].mean():.3f}")
print(f"  dev mean              : {agg['dev_acc'].mean():+.3f}")
print(f"  estates with dev > 0  : {(agg['dev_acc']>0).sum()} ({(agg['dev_acc']>0).mean()*100:.1f}%)")
print(f"  estates with dev < 0  : {(agg['dev_acc']<0).sum()} ({(agg['dev_acc']<0).mean()*100:.1f}%)")
print(f"  estates dev < -0.02   : {(agg['dev_acc']<-0.02).sum()}")
print(f"  estates dev < -0.05   : {(agg['dev_acc']<-0.05).sum()}")

print(f"\n  d_t_fire mean change  : {agg['d_t_fire'].mean():+.2f} min  "
      f"(estate 平均提速节省时间)")
print(f"  estates d_t_fire > 0  : {(agg['d_t_fire']>0).sum()} (worse)")
print(f"  estates d_t_fire <-1m : {(agg['d_t_fire']<-1).sum()} (>1 min faster)")
print(f"  d_t_hosp mean change  : {agg['d_t_hosp'].mean():+.2f} min")

# ── 2. 最差 / 最好 estate ────────────────────────────────────────────────────
worst = agg.nsmallest(20, "dev_acc")[
    ["estate","bl_acc","typh_acc","dev_acc","d_fire_m","d_hosp_m",
     "t_fire_bl","t_fire_typh","d_t_fire","total_pop","ratio_age_65plus"]]
best  = agg.nlargest(20, "dev_acc")[
    ["estate","bl_acc","typh_acc","dev_acc","d_fire_m","d_hosp_m",
     "t_fire_bl","t_fire_typh","d_t_fire","total_pop","ratio_age_65plus"]]
print("\n=== Worst-20 estates (accessibility deteriorated) ===")
print(worst.to_string(index=False, formatters={
    "bl_acc":"{:.3f}".format,"typh_acc":"{:.3f}".format,"dev_acc":"{:+.3f}".format,
    "d_fire_m":"{:.0f}".format,"d_hosp_m":"{:.0f}".format,
    "t_fire_bl":"{:.2f}".format,"t_fire_typh":"{:.2f}".format,"d_t_fire":"{:+.2f}".format,
    "total_pop":"{:.0f}".format,"ratio_age_65plus":"{:.2%}".format,
}))
print("\n=== Best-20 estates (most improved) ===")
print(best.to_string(index=False, formatters={
    "bl_acc":"{:.3f}".format,"typh_acc":"{:.3f}".format,"dev_acc":"{:+.3f}".format,
    "d_fire_m":"{:.0f}".format,"d_hosp_m":"{:.0f}".format,
    "t_fire_bl":"{:.2f}".format,"t_fire_typh":"{:.2f}".format,"d_t_fire":"{:+.2f}".format,
    "total_pop":"{:.0f}".format,"ratio_age_65plus":"{:.2%}".format,
}))

# ── 3. 与人口结构相关性 ──────────────────────────────────────────────────────
print("\n=== Pearson correlation of dev_acc with demographics ===")
for c in ["total_pop","working_pop","median_income",
          "ratio_age_65plus","ratio_age_0_14",
          "d_fire_m","d_hosp_m","bl_acc"]:
    if c in agg.columns:
        x = agg[c].dropna()
        y = agg.loc[x.index, "dev_acc"]
        r = np.corrcoef(x,y)[0,1] if len(x)>5 else np.nan
        print(f"  {c:<22}: r = {r:+.3f}  (n={len(x)})")

# 老人比例五分位
agg["age65_q"] = pd.qcut(agg["ratio_age_65plus"].rank(method="first"), 5,
                         labels=["Q1 young","Q2","Q3","Q4","Q5 old"])
g_age = agg.groupby("age65_q").agg(
    n=("estate","count"),
    bl_acc=("bl_acc","mean"),
    typh_acc=("typh_acc","mean"),
    dev_acc=("dev_acc","mean"),
    d_t_fire=("d_t_fire","mean"),
    pct_worse=("dev_acc", lambda x: (x<-0.02).mean()*100),
).reset_index()
print("\n=== Accessibility by elderly-ratio quintile ===")
print(g_age.to_string(index=False, formatters={
    "bl_acc":"{:.3f}".format,"typh_acc":"{:.3f}".format,"dev_acc":"{:+.3f}".format,
    "d_t_fire":"{:+.2f}".format,"pct_worse":"{:.1f}".format,
}))

# 基线可达性五分位 (平时难到达的小区是否变得更难?)
agg["bl_q"] = pd.qcut(agg["bl_acc"].rank(method="first"), 5,
                      labels=["Q1 hard","Q2","Q3","Q4","Q5 easy"])
g_bl = agg.groupby("bl_q").agg(
    n=("estate","count"),
    bl_acc=("bl_acc","mean"),
    dev_acc=("dev_acc","mean"),
    pct_worse=("dev_acc", lambda x: (x<-0.02).mean()*100),
).reset_index()
print("\n=== dev by baseline-accessibility quintile ===")
print(g_bl.to_string(index=False, formatters={
    "bl_acc":"{:.3f}".format,"dev_acc":"{:+.3f}".format,"pct_worse":"{:.1f}".format,
}))

# ── 4. 图: 4 panel ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 11))

# (a) baseline vs typhoon scatter
ax = plt.subplot(2,2,1)
sc = ax.scatter(agg["bl_acc"], agg["typh_acc"],
                c=agg["ratio_age_65plus"]*100, cmap="viridis",
                s=20, alpha=0.7, edgecolor="white", lw=0.3)
mn, mx = 0.4, 1.02
ax.plot([mn,mx],[mn,mx], color="#888", lw=1, ls="--")
ax.set_xlim(mn,mx); ax.set_ylim(mn,mx)
ax.set_xlabel("Baseline accessibility (buffer mean speed)")
ax.set_ylabel("Typhoon (09-24 mid-day) accessibility")
plt.colorbar(sc, ax=ax, label="% age 65+")
ax.set_title("(a) Estate accessibility: baseline vs typhoon\n(above 1:1 = improved)",
             fontweight="bold", loc="left")
ax.grid(alpha=0.25)

# (b) elderly-ratio quintile bar
ax = plt.subplot(2,2,2)
xs = np.arange(len(g_age))
ax.bar(xs, g_age["dev_acc"], color="#7b1fa2", alpha=0.8, edgecolor="white")
for x,v,p in zip(xs, g_age["dev_acc"], g_age["pct_worse"]):
    ax.text(x, v+0.001, f"{v:+.3f}\n({p:.0f}% worse)",
            ha="center", va="bottom", fontsize=9.5)
ax.set_xticks(xs); ax.set_xticklabels(g_age["age65_q"])
ax.set_ylabel("Mean accessibility deviation")
ax.axhline(0, color="#666", lw=0.6)
ax.set_title("(b) Accessibility change by elderly ratio quintile",
             fontweight="bold", loc="left")
ax.grid(axis="y", alpha=0.25)

# (c) Estate map colored by dev_acc
ax = plt.subplot(2,2,3)
agg_geo = agg.copy()
vmin, vmax = -0.05, 0.10
sc = ax.scatter(agg_geo.geometry.x, agg_geo.geometry.y,
                c=agg_geo["dev_acc"].clip(vmin,vmax),
                cmap="RdYlGn", vmin=vmin, vmax=vmax,
                s=18, alpha=0.85, edgecolor="white", lw=0.3)
try:
    ctx.add_basemap(ax, crs=agg_geo.crs,
                    source=ctx.providers.CartoDB.PositronNoLabels,
                    zoom=11, alpha=0.55)
except Exception as e:
    print(f"basemap failed: {e}")
ax.set_axis_off()
plt.colorbar(sc, ax=ax, label="dev_acc (typhoon − baseline)", shrink=0.7)
ax.set_title("(c) Per-estate accessibility change\n(red = worse, green = better)",
             fontweight="bold", loc="left")

# (d) baseline accessibility quintile bar
ax = plt.subplot(2,2,4)
xs = np.arange(len(g_bl))
ax.bar(xs, g_bl["dev_acc"], color="#1565c0", alpha=0.8, edgecolor="white")
for x,v,b in zip(xs, g_bl["dev_acc"], g_bl["bl_acc"]):
    ax.text(x, v+0.002, f"{v:+.3f}", ha="center", va="bottom", fontsize=10)
ax.set_xticks(xs)
ax.set_xticklabels([f"{q}\nbl={b:.2f}" for q,b in zip(g_bl["bl_q"], g_bl["bl_acc"])])
ax.set_ylabel("Mean accessibility deviation")
ax.axhline(0, color="#666", lw=0.6)
ax.set_title("(d) Improvement by baseline accessibility quintile",
             fontweight="bold", loc="left")
ax.grid(axis="y", alpha=0.25)

fig.suptitle("Emergency accessibility to residential estates "
             "(09-24 mid-day, S10/S8, 3 km buffer)",
             fontweight="bold", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.96])
out_fig = f"{OUT}/图25j_emergency_accessibility.png"
fig.savefig(out_fig, dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved: {out_fig}")

# ── 5. CSV ───────────────────────────────────────────────────────────────────
out_cols = ["estate","total_pop","working_pop","median_income",
            "ratio_age_65plus","ratio_age_0_14",
            "n_roads","tot_length_km","bl_acc","typh_acc","dev_acc",
            "n_worse_10","n_better_10",
            "d_fire_m","t_fire_bl","t_fire_typh","d_t_fire",
            "d_hosp_m","t_hosp_bl","t_hosp_typh","d_t_hosp",
            "d_pol_m","lat","lon"]
agg[out_cols].to_csv(f"{OUT}/emergency_accessibility_estate.csv", index=False)
worst.to_csv(f"{OUT}/emergency_worst20_estates.csv", index=False)
best.to_csv(f"{OUT}/emergency_best20_estates.csv", index=False)
print("CSVs saved.")
