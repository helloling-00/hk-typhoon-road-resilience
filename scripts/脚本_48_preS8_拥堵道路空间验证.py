"""
脚本_48 — Verify: where are the pre-S8 congested roads at Sep 23 13:00?
Map congested vs fast roads, overlay supermarkets/convenience stores.
Buffer analysis: what POIs are near congested vs fast roads?
Tests the "shopping" vs "commuting" hypothesis.
"""
import os, gc, pandas as pd, numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from shapely import wkb as shapely_wkb
from shapely.geometry import Point, LineString, box
import contextily as ctx
import warnings
warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"
OUT  = "/Users/helloling/workspace/thesis"

# ── Load shops ────────────────────────────────────────────────────────────────
print("Loading shops...", flush=True)
shops = gpd.read_file(f"{DATA}/osm_cache/hk_shops.gpkg")
supermarkets = shops[shops["shop"] == "supermarket"].copy()
convenience = shops[shops["shop"] == "convenience"].copy()
print(f"  Supermarkets: {len(supermarkets)}")
print(f"  Convenience stores: {len(convenience)}")

# ── Load lookups ──────────────────────────────────────────────────────────────
print("Loading lookups...", flush=True)
EP = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
BL = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
bl_s26 = BL[(BL.slot == 26) & (BL.day_type == "WORKDAY")].set_index("road_id")["mean_speed"]

def get_ep_key(wb):
    try:
        g = shapely_wkb.loads(bytes(wb))
        coords = list(g.coords) if g.geom_type == "LineString" else \
                 [c for line in g.geoms for c in line.coords]
        s = (round(coords[0][0],4), round(coords[0][1],4))
        e = (round(coords[-1][0],4), round(coords[-1][1],4))
        return str((min(s,e), max(s,e)))
    except: return None

# ── Build WKB cache for Sep 23 ────────────────────────────────────────────────
print("Building WKB cache for Sep 23...", flush=True)
day = "2025-09-23"
folder = f"{FLOW}/{day}"
uniq = {}
for s in [0, 12, 24, 26, 36]:
    files = [f for f in os.listdir(folder) if f"_slot{s:02d}_" in f]
    if not files: continue
    df = pd.read_parquet(f"{folder}/{files[0]}", columns=["geometry"])
    for g in df["geometry"]:
        if g is not None:
            k = id(bytes(g)[:8])
            if k not in uniq: uniq[k] = g
wkb_ep = {}
for g in uniq.values():
    epk = get_ep_key(g)
    if epk: wkb_ep[bytes(g)] = epk
print(f"  {len(wkb_ep)} unique geometries cached")

def lookup_epk(g):
    if g is None: return None
    b = bytes(g)
    if b in wkb_ep: return wkb_ep[b]
    epk = get_ep_key(g)
    if epk: wkb_ep[b] = epk
    return epk

# ── Load slot 26 (13:00) ─────────────────────────────────────────────────────
print("Loading slot 26...", flush=True)
files_s26 = [f for f in os.listdir(folder) if "_slot26_" in f]
if not files_s26:
    print("ERROR: slot 26 file not found")
    exit(1)

df = pd.read_parquet(f"{folder}/{files_s26[0]}",
                     columns=["relative_speed","geometry","road_closure"])
df = df[df["road_closure"] != 1].copy()

df["ep_key"] = df["geometry"].apply(lookup_epk)
df = df.merge(EP[["ep_key","road_id"]], on="ep_key", how="inner")

# Aggregate to road_id; keep first geometry
geoms = {}
for _, row in df.iterrows():
    rid = row["road_id"]
    if rid not in geoms:
        try:
            g = shapely_wkb.loads(bytes(row["geometry"]))
            geoms[rid] = g
        except: pass

agg = df.groupby("road_id")["relative_speed"].mean().rename("speed_obs").reset_index()
agg = agg.set_index("road_id")
agg["geometry"] = agg.index.map(geoms)

# Add baseline
idx = pd.MultiIndex.from_arrays(
    [["WORKDAY"]*len(agg), [26]*len(agg), agg.index],
    names=["day_type","slot","road_id"])
agg["baseline"] = bl_s26.reindex(agg.index).values
agg = agg.dropna(subset=["baseline"])
agg["deviation"] = agg["speed_obs"] - agg["baseline"]
agg = agg.reset_index()

print(f"  {len(agg)} roads with deviation at slot 26")

# Classify
agg["congested"] = agg["deviation"] < -0.03
agg["clearly_fast"] = agg["deviation"] > 0.03
agg["normal"] = agg["deviation"].abs() <= 0.03

n_cong = agg["congested"].sum()
n_fast = agg["clearly_fast"].sum()
n_norm = agg["normal"].sum()
print(f"  Congested (<-0.03):  {n_cong} roads ({n_cong/len(agg):.1%})")
print(f"  Clearly fast (>0.03): {n_fast} roads ({n_fast/len(agg):.1%})")
print(f"  Normal (±0.03): {n_norm} roads ({n_norm/len(agg):.1%})")

# ── Create GeoDataFrame for roads ─────────────────────────────────────────────
road_gdf = gpd.GeoDataFrame(
    agg.dropna(subset=["geometry"]),
    geometry="geometry", crs="EPSG:4326"
)
print(f"  Road GDF: {len(road_gdf)} roads")

# ═══════════════════════════════════════════════════════════════════════════════
# BUFFER ANALYSIS: what POIs are near congested vs fast roads?
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("BUFFER ANALYSIS: POI counts near congested vs fast roads")
print("="*70)

# Convert to projected CRS for buffer (meters)
# HK uses EPSG:2326 (Hong Kong 1980 Grid)
road_proj = road_gdf.to_crs(epsg=2326)
sup_proj = supermarkets.to_crs(epsg=2326)
conv_proj = convenience.to_crs(epsg=2326)

# Buffer each road at 200m and 500m, count POIs
for radius_m in [200, 500]:
    print(f"\n--- Buffer radius: {radius_m}m ---")
    road_proj["buffer"] = road_proj.geometry.buffer(radius_m)

    # Count supermarkets and convenience stores in each buffer
    sup_idx = gpd.sjoin(sup_proj, road_proj.set_geometry("buffer"), how="inner", predicate="within")
    conv_idx = gpd.sjoin(conv_proj, road_proj.set_geometry("buffer"), how="inner", predicate="within")

    sup_counts = sup_idx.groupby("road_id").size()
    conv_counts = conv_idx.groupby("road_id").size()

    road_proj["n_supermarkets"] = road_proj["road_id"].map(sup_counts).fillna(0)
    road_proj["n_convenience"] = road_proj["road_id"].map(conv_counts).fillna(0)
    road_proj["n_all_shops"] = road_proj["n_supermarkets"] + road_proj["n_convenience"]

    cong = road_proj[road_proj["congested"]]
    fast = road_proj[road_proj["clearly_fast"]]
    norm = road_proj[road_proj["normal"]]

    for label, subset in [("Congested", cong), ("Normal", norm), ("Fast", fast)]:
        if len(subset) == 0: continue
        print(f"  {label:12s} (n={len(subset):5d}): "
              f"supermarkets={subset['n_supermarkets'].mean():.2f}  "
              f"convenience={subset['n_convenience'].mean():.2f}  "
              f"all_shops={subset['n_all_shops'].mean():.2f}  "
              f"any_shop={((subset['n_all_shops']) > 0).mean():.1%}")

    # Statistical test: congested vs fast
    from scipy import stats
    for poi_col in ["n_supermarkets", "n_convenience", "n_all_shops"]:
        c_vals = cong[poi_col]
        f_vals = fast[poi_col]
        t, p = stats.ttest_ind(c_vals, f_vals, equal_var=False)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        direction = "↑" if c_vals.mean() > f_vals.mean() else "↓"
        print(f"    {direction} {poi_col}: cong={c_vals.mean():.3f} vs fast={f_vals.mean():.3f}  "
              f"diff={c_vals.mean()-f_vals.mean():+.3f}  p={p:.4f} {sig}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAP 1: Congested vs Fast roads + Supermarkets
# ═══════════════════════════════════════════════════════════════════════════════
print("\nPlotting map...", flush=True)

# Convert to web mercator for basemap
road_web = road_gdf.to_crs(epsg=3857)
sup_web = supermarkets.to_crs(epsg=3857)
conv_web = convenience.to_crs(epsg=3857)

# Hong Kong focus bounds (web mercator)
fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), facecolor="white")

for ax_idx, (ax, focus, title) in enumerate([
    (axes[0], "full", "Hong Kong — All Roads"),
    (axes[1], "kowloon", "Kowloon + HK Island (Zoom)"),
]):
    # Set extent
    if focus == "full":
        ax.set_xlim(113.82e5, 114.40e5)  # rough mercator
        ax.set_ylim(22.15e5, 22.55e5)
    else:
        ax.set_xlim(113.87e5, 114.28e5)
        ax.set_ylim(22.26e5, 22.38e5)

    # Plot normal roads (grey, thin)
    normal_roads = road_web[road_web["normal"]]
    for _, row in normal_roads.head(500).iterrows():
        g = row.geometry
        try:
            lines = list(g.geoms) if g.geom_type == "MultiLineString" else [g]
        except: continue
        for line in lines:
            try:
                x, y = line.xy
                ax.plot(x, y, color="#cccccc", linewidth=0.3, alpha=0.4)
            except: pass

    # Plot fast roads (green)
    fast_roads = road_web[road_web["clearly_fast"]]
    for _, row in fast_roads.iterrows():
        g = row.geometry
        try:
            lines = list(g.geoms) if g.geom_type == "MultiLineString" else [g]
        except: continue
        for line in lines:
            try:
                x, y = line.xy
                ax.plot(x, y, color="#1a9850", linewidth=0.7, alpha=0.7)
            except: pass

    # Plot congested roads (red)
    cong_roads = road_web[road_web["congested"]]
    for _, row in cong_roads.iterrows():
        g = row.geometry
        try:
            lines = list(g.geoms) if g.geom_type == "MultiLineString" else [g]
        except: continue
        for line in lines:
            try:
                x, y = line.xy
                ax.plot(x, y, color="#d73027", linewidth=0.9, alpha=0.85, zorder=5)
            except: pass

    # Plot supermarkets (yellow dots)
    sup_pts = sup_web[sup_web.geometry.type == "Point"]
    ax.scatter(sup_pts.geometry.x, sup_pts.geometry.y,
              s=3, color="#ffc800", alpha=0.6, zorder=4, label="Supermarket")
    # Plot convenience stores (orange dots, smaller)
    conv_pts = conv_web[conv_web.geometry.type == "Point"]
    ax.scatter(conv_pts.geometry.x, conv_pts.geometry.y,
              s=1, color="#ff8c00", alpha=0.4, zorder=3, label="Convenience")

    ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron, zoom=11)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Legend
    legend_lines = [
        Line2D([0],[0], color="#d73027", lw=2, label=f"Congested (n={n_cong})"),
        Line2D([0],[0], color="#1a9850", lw=2, label=f"Faster (n={n_fast})"),
    ]
    ax.legend(handles=legend_lines, loc="lower right", fontsize=7.5,
              title="Sep 23 13:00 Speed Deviation", title_fontsize=8, framealpha=0.85)

fig.suptitle("Where Are the Pre-S8 Congested Roads?  ·  Yagiasha Sep 23, 13:00",
             fontsize=14, fontweight="bold", y=1.01)

plt.tight_layout()
out_path = f"{OUT}/图48_preS8_拥堵道路空间分布.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# MAP 2: Congested roads with shops — zoomed Kowloon side
# ═══════════════════════════════════════════════════════════════════════════════
print("Plotting zoomed map with shops...", flush=True)
fig, ax = plt.subplots(figsize=(12, 10), facecolor="white")

ax.set_xlim(113.90e5, 114.22e5)
ax.set_ylim(22.28e5, 22.37e5)

# Plot normal roads
for _, row in road_web[road_web["normal"]].head(800).iterrows():
    g = row.geometry
    try:
        lines = list(g.geoms) if g.geom_type == "MultiLineString" else [g]
    except: continue
    for line in lines:
        try:
            x, y = line.xy
            ax.plot(x, y, color="#cccccc", linewidth=0.3, alpha=0.35)
        except: pass

# Plot fast roads
for _, row in road_web[road_web["clearly_fast"]].iterrows():
    g = row.geometry
    try:
        lines = list(g.geoms) if g.geom_type == "MultiLineString" else [g]
    except: continue
    for line in lines:
        try:
            x, y = line.xy
            ax.plot(x, y, color="#1a9850", linewidth=0.8, alpha=0.7)
        except: pass

# Plot congested roads (thicker)
for _, row in road_web[road_web["congested"]].iterrows():
    g = row.geometry
    try:
        lines = list(g.geoms) if g.geom_type == "MultiLineString" else [g]
    except: continue
    for line in lines:
        try:
            x, y = line.xy
            ax.plot(x, y, color="#d73027", linewidth=1.2, alpha=0.9, zorder=5)
        except: pass

# Supermarkets with larger markers
sup_pts2 = sup_web[sup_web.geometry.type == "Point"]
ax.scatter(sup_pts2.geometry.x, sup_pts2.geometry.y,
          s=8, color="#ffc800", alpha=0.8, zorder=6, edgecolors="white", linewidths=0.3,
          label="Supermarket")
# Convenience stores
conv_pts2 = conv_web[conv_web.geometry.type == "Point"]
ax.scatter(conv_pts2.geometry.x, conv_pts2.geometry.y,
          s=3, color="#ff8c00", alpha=0.6, zorder=5, edgecolors="white", linewidths=0.2,
          label="Convenience store")

ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron, zoom=12)
ax.set_axis_off()
ax.set_title("Kowloon + HK Island  ·  Congested Roads & Food Shops  ·  Sep 23 13:00",
             fontsize=13, fontweight="bold")

legend_lines = [
    Line2D([0],[0], color="#d73027", lw=2.5, label=f"Congested roads (dev < −0.03, n={n_cong})"),
    Line2D([0],[0], color="#1a9850", lw=2, label=f"Faster roads (dev > +0.03, n={n_fast})"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#ffc800", markersize=7,
           label="Supermarket"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#ff8c00", markersize=5,
           label="Convenience store"),
]
ax.legend(handles=legend_lines, loc="lower right", fontsize=8,
          title="Roads & POIs", title_fontsize=9, framealpha=0.9)

plt.tight_layout()
out_path2 = f"{OUT}/图48b_preS8_拥堵道路_商铺.png"
plt.savefig(out_path2, dpi=250, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path2}")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"  Total roads: {len(agg)}")
print(f"  Congested:   {n_cong} ({n_cong/len(agg):.1%}) — mean dev = {agg[agg['congested']]['deviation'].mean():+.4f}")
print(f"  Fast:        {n_fast} ({n_fast/len(agg):.1%}) — mean dev = {agg[agg['clearly_fast']]['deviation'].mean():+.4f}")
print(f"  Normal:      {n_norm} ({n_norm/len(agg):.1%})")

print("\nDone.")
