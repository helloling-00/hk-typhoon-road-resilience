"""
Pre-S8 congestion spatial map: Sep 23 13:00 vs control Sep 16 13:00
Show WHERE roads are congested before S8, and how they differ from normal.
"""
import os, gc, pandas as pd, numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from shapely import wkb as shapely_wkb
from shapely.geometry import LineString
import contextily as ctx
import warnings
warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"
OUT  = "/Users/helloling/workspace/thesis"

EP = pd.read_parquet(f"{DATA}/ep_to_road.parquet")

# ── Load road geometries for a given slot ────────────────────────────────────
def get_ep_key(wb):
    try:
        g = shapely_wkb.loads(bytes(wb))
        coords = list(g.coords) if g.geom_type == "LineString" else \
                 [c for line in g.geoms for c in line.coords]
        s = (round(coords[0][0],4), round(coords[0][1],4))
        e = (round(coords[-1][0],4), round(coords[-1][1],4))
        return str((min(s,e), max(s,e)))
    except: return None

def load_slot_roads(day, slot_num, day_type="WORKDAY"):
    folder = f"{FLOW}/{day}"
    if not os.path.exists(folder): return None
    files = [f for f in os.listdir(folder) if f"_slot{slot_num:02d}_" in f]
    if not files: return None
    try:
        df = pd.read_parquet(f"{folder}/{files[0]}",
                             columns=["relative_speed","geometry","road_closure","road_category"])
        df = df[df["road_closure"] != 1].copy()
        if len(df) < 100: return None

        # Extract ep_key and geometry
        geoms = []; ep_keys = []; speeds = []
        for _, row in df.iterrows():
            g = row["geometry"]
            if g is None: continue
            epk = get_ep_key(g)
            if epk is None: continue
            try:
                gobj = shapely_wkb.loads(bytes(g))
            except: continue
            ep_keys.append(epk)
            geoms.append(gobj)
            speeds.append(row["relative_speed"])

        df2 = pd.DataFrame({"ep_key": ep_keys, "speed": speeds, "geometry": geoms})
        # Join road_id
        df2 = df2.merge(EP[["ep_key","road_id"]], on="ep_key", how="inner")
        if len(df2) < 100: return None
        # Aggregate to road_id (take first geometry)
        agg = df2.groupby("road_id").agg(
            speed=("speed", "mean"),
            geometry=("geometry", "first")
        ).reset_index()
        return agg
    except Exception as e:
        print(f"  Error: {e}")
        return None

# ── Load baseline for slot 26 (13:00) ─────────────────────────────────────────
print("Loading data...", flush=True)
bl = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
bl_slot26 = bl[(bl.slot == 26) & (bl.day_type == "WORKDAY")].set_index("road_id")["mean_speed"]

# Load typhoon slot 26 (Sep 23 13:00) and control (Sep 16 13:00)
typhoon = load_slot_roads("2025-09-23", 26)
control = load_slot_roads("2025-09-16", 26)

print(f"  Typhoon roads: {len(typhoon) if typhoon is not None else 0}")
print(f"  Control roads: {len(control) if control is not None else 0}")

# Add baseline and deviation
def add_deviation(df, bl_series):
    if df is None: return None
    df = df.copy()
    df["baseline"] = df["road_id"].map(bl_series)
    df = df.dropna(subset=["baseline"])
    df["deviation"] = df["speed"] - df["baseline"]
    return df

typhoon = add_deviation(typhoon, bl_slot26)
control = add_deviation(control, bl_slot26)

# ── Create GeoDataFrames ──────────────────────────────────────────────────────
typhoon_gdf = gpd.GeoDataFrame(
    typhoon.dropna(subset=["deviation"]),
    geometry="geometry", crs="EPSG:4326"
)
control_gdf = gpd.GeoDataFrame(
    control.dropna(subset=["deviation"]),
    geometry="geometry", crs="EPSG:4326"
)
print(f"  Typhoon GDF: {len(typhoon_gdf)} roads")
print(f"  Control GDF: {len(control_gdf)} roads")

# Stats
print(f"\n  Typhoon 13:00: mean dev = {typhoon_gdf['deviation'].mean():+.4f}, "
      f"pct_slower_003 = {(typhoon_gdf['deviation'] < -0.03).mean():.1%}")
print(f"  Control 13:00: mean dev = {control_gdf['deviation'].mean():+.4f}, "
      f"pct_slower_003 = {(control_gdf['deviation'] < -0.03).mean():.1%}")

# Convert to web mercator for plotting
typhoon_plot = typhoon_gdf.to_crs(epsg=3857)
control_plot = control_gdf.to_crs(epsg=3857)

# ── Set color scale ───────────────────────────────────────────────────────────
vmax = 0.10
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
cmap = plt.cm.RdYlGn  # red=slow, green=fast

# ── Figure: two side-by-side maps ─────────────────────────────────────────────
print("Plotting...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), facecolor="white")

# Common Hong Kong extent (web mercator)
# Kowloon + HK Island focus
bounds = {
    "minx": 113.82, "maxx": 114.30,
    "miny": 22.22, "maxy": 22.42,
}

for ax, gdf, title, subtitle in [
    (axes[0], control_plot, "Control Day (Sep 16, 13:00)",
     f"mean dev = {control_gdf['deviation'].mean():+.4f}"),
    (axes[1], typhoon_plot, "Pre-S8 (Sep 23, 13:00) — Yagiasha",
     f"mean dev = {typhoon_gdf['deviation'].mean():+.4f}  ·  "
     f"{(typhoon_gdf['deviation'] < -0.03).mean():.1%} roads clearly slower"),
]:
    for _, row in gdf.iterrows():
        val = row["deviation"]
        if pd.isna(val): continue
        color = cmap(norm(val))
        g = row.geometry
        try:
            lines = list(g.geoms) if g.geom_type == "MultiLineString" else [g]
        except:
            continue
        for line in lines:
            try:
                x, y = line.xy
                ax.plot(x, y, color=color, linewidth=0.6, solid_capstyle="round", alpha=0.85)
            except: pass

    ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron, zoom=11)
    ax.set_axis_off()
    ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold")

# Colorbar
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, shrink=0.55, pad=0.02, aspect=30)
cbar.set_label("Speed Deviation (typhoon − baseline)", fontsize=9)
cbar.ax.tick_params(labelsize=8)

fig.suptitle("Where Does the Pre-S8 Congestion Surge Occur?  ·  Yagiasha Sep 23 vs Control Sep 16, 13:00",
             fontsize=13, fontweight="bold", y=1.01)

# Legend for road types (line width not shown, just explain color)
legend_lines = [
    Line2D([0],[0], color="#d73027", lw=2, label="Slower (dev < 0)"),
    Line2D([0],[0], color="#1a9850", lw=2, label="Faster (dev > 0)"),
    Line2D([0],[0], color="#ffffbf", lw=2, label="Near baseline"),
]
axes[1].legend(handles=legend_lines, loc="lower right", fontsize=7.5,
               title="Speed deviation", title_fontsize=8, framealpha=0.85)

plt.tight_layout()
out_path = f"{OUT}/图45_preS8_空间对比.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
plt.close()
