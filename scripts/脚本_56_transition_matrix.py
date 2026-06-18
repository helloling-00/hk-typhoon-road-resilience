"""
Per-road state transition between 08:30 (slot 17, morning peak) and
13:00 (slot 26, midday dip), Ragasa Sep 23.

State at each time = {Faster, Near, Slower} based on dev threshold ±0.03.
Transition matrix is 3×3 (9 cells), length-weighted.
"""
import os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely import wkb as shapely_wkb
import geopandas as gpd
import contextily as cx
import warnings
warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
FLOW = f"{DATA}/flow_parquet2"
OUT  = "/Users/helloling/workspace/thesis"
DEV_HI, DEV_LO = 0.03, -0.03

STATES = ["F","N","S"]
STATE_NAMES = {"F":"Faster","N":"Near","S":"Slower"}
def classify(d):
    if d > DEV_HI: return "F"
    if d < DEV_LO: return "S"
    return "N"

# ─── Load deviations + lengths ───────────────────────────────────────────────
print("Loading deviations + lengths...", flush=True)
ts = pd.read_parquet(f"{DATA}/yagiasha_road_timeseries.parquet")
ts["ds"] = pd.to_datetime(ts["dt"]).dt.strftime("%Y-%m-%d")
sep23 = ts[ts["ds"]=="2025-09-23"].copy()

rt = pd.read_parquet(f"{DATA}/regression_table.parquet")
length_per_rid = rt.drop_duplicates("road_id").set_index("road_id")["road_length_m"]

s17 = sep23[sep23["slot"]==17][["road_id","dev"]].rename(columns={"dev":"dev_morn"})
s26 = sep23[sep23["slot"]==26][["road_id","dev"]].rename(columns={"dev":"dev_mid"})
both = s17.merge(s26, on="road_id", how="inner")
both["length_m"] = both["road_id"].map(length_per_rid)
both = both.dropna(subset=["length_m"])
both["state_morn"] = both["dev_morn"].apply(classify)
both["state_mid"]  = both["dev_mid"].apply(classify)

print(f"  Roads observed at both 08:30 and 13:00: {len(both):,}")
print(f"  Total km: {both['length_m'].sum()/1000:.1f}")

# ─── Build 3×3 length-weighted transition matrix ─────────────────────────────
mat = np.zeros((3,3))
for i, sm in enumerate(STATES):
    for j, sd in enumerate(STATES):
        mat[i,j] = both.loc[(both["state_morn"]==sm) & (both["state_mid"]==sd),
                            "length_m"].sum()
total_km = mat.sum() / 1000
mat_pct = mat / mat.sum() * 100

# Marginals
row_sum = mat.sum(axis=1) / mat.sum() * 100   # 08:30 distribution
col_sum = mat.sum(axis=0) / mat.sum() * 100   # 13:00 distribution

print("\n  Length-weighted transition matrix (% of total km observed at both):")
print(f"    {'13:00→':>8s}  {'F':>8s} {'N':>8s} {'S':>8s}   {'row %':>8s}")
for i, sm in enumerate(STATES):
    row = "  ".join([f"{mat_pct[i,j]:7.2f}%" for j in range(3)])
    print(f"  08:30 {sm}    {row}    {row_sum[i]:7.2f}%")
print(f"  col %       {col_sum[0]:7.2f}% {col_sum[1]:7.2f}% {col_sum[2]:7.2f}%")

# Persistent / improving / deteriorating shares
diag = mat_pct[0,0] + mat_pct[1,1] + mat_pct[2,2]
deteriorate = mat_pct[0,1] + mat_pct[0,2] + mat_pct[1,2]   # F→N, F→S, N→S
improve     = mat_pct[2,1] + mat_pct[2,0] + mat_pct[1,0]   # S→N, S→F, N→F
print(f"\n  Stayed in same state:   {diag:5.1f}% of road-km")
print(f"  Deteriorated (worsened):{deteriorate:5.1f}%  (F→N + F→S + N→S)")
print(f"  Improved:               {improve:5.1f}%  (S→N + S→F + N→F)")

# ─── Heatmap visualization ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 6.2))
im = ax.imshow(mat_pct, cmap="YlOrRd", vmin=0, vmax=mat_pct.max()*1.05)

# Cell labels
for i in range(3):
    for j in range(3):
        v = mat_pct[i,j]
        color = "white" if v > mat_pct.max()*0.55 else "black"
        ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                fontsize=15, fontweight="bold", color=color)

ax.set_xticks(range(3))
ax.set_xticklabels([f"{s}\n{STATE_NAMES[s]}" for s in STATES], fontsize=11)
ax.set_yticks(range(3))
ax.set_yticklabels([f"{s}\n{STATE_NAMES[s]}" for s in STATES], fontsize=11)
ax.set_xlabel("State at 13:00 (midday dip)", fontsize=12, fontweight="bold")
ax.set_ylabel("State at 08:30 (morning peak)", fontsize=12, fontweight="bold")
ax.set_title("Road-state transitions  08:30 → 13:00  (Ragasa Sep 23, S3)",
             fontsize=18, fontweight="bold", pad=12)

# Marginal annotations on the right and bottom
for i, p in enumerate(row_sum):
    ax.text(2.65, i, f"{p:.1f}%", va="center", ha="left",
            fontsize=10, color="#555", style="italic")
for j, p in enumerate(col_sum):
    ax.text(j, 2.65, f"{p:.1f}%", ha="center", va="center",
            fontsize=10, color="#555", style="italic")
ax.text(2.65, -0.55, "row %\n(08:30)", fontsize=8.5, color="#555",
        style="italic", ha="left")
ax.text(-0.55, 2.65, "col %\n(13:00)", fontsize=8.5, color="#555",
        style="italic", ha="left")

cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
cbar.set_label("% of road-km", fontsize=10)

plt.tight_layout()
out = f"{OUT}/图50f_transition_matrix.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n  saved -> {out}")

# ─── Save CSV for thesis table ───────────────────────────────────────────────
df_mat = pd.DataFrame(mat_pct,
                       index=[f"08:30 {s}" for s in STATES],
                       columns=[f"13:00 {s}" for s in STATES]).round(2)
df_mat["row_total"] = row_sum.round(2)
df_mat.loc["col_total"] = list(col_sum.round(2)) + [100.0]
df_mat.to_csv(f"{OUT}/preS8_transition_matrix.csv")
print(f"  saved -> preS8_transition_matrix.csv")

# ─── Optional: map showing 4 most interesting cells ──────────────────────────
print("\nBuilding optional transition map...", flush=True)
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
ep_to_rid = ep.set_index("ep_key")["road_id"].to_dict()

def get_ep_key(wb):
    try:
        g = shapely_wkb.loads(bytes(wb))
        coords = list(g.coords) if g.geom_type=="LineString" else \
                 [c for line in g.geoms for c in line.coords]
        s = (round(coords[0][0],4), round(coords[0][1],4))
        e = (round(coords[-1][0],4), round(coords[-1][1],4))
        return str((min(s,e), max(s,e)))
    except: return None

folder = f"{FLOW}/2025-09-23"
geom_per_rid = {}
for s in [10, 14, 17, 22, 26, 30]:
    files = [f for f in os.listdir(folder) if f"_slot{s:02d}_" in f]
    if not files: continue
    df = pd.read_parquet(f"{folder}/{files[0]}", columns=["geometry"])
    for g in df["geometry"]:
        if g is None: continue
        epk = get_ep_key(g)
        if epk and epk in ep_to_rid:
            rid = ep_to_rid[epk]
            if rid not in geom_per_rid:
                try: geom_per_rid[rid] = shapely_wkb.loads(bytes(g))
                except: pass
print(f"  {len(geom_per_rid):,} road geometries cached")

# Tag each row with transition class
both["trans"] = both["state_morn"] + "→" + both["state_mid"]

# ─── Single combined map: all 4 transitions, F→S thickest ──────────────────
HK_BBOX = (113.82, 22.15, 114.45, 22.60)

TRANSITIONS = [
    # (cls, color, lw, alpha, zorder, label)
    ("S→S", "#7b1fa2", 0.9, 0.82, 3, f"S → S  Persistent congestion  ({mat_pct[2,2]:.1f}%)"),
    ("N→S", "#ff7f0e", 1.0, 0.85, 4, f"N → S  New midday congestion  ({mat_pct[1,2]:.1f}%)"),
    ("F→F", "#2ca02c", 1.1, 0.85, 5, f"F → F  Persistent clearance  ({mat_pct[0,0]:.1f}%)"),
    ("F→S", "#d62728", 2.2, 0.92, 6, f"F → S  Reversal to congestion  ({mat_pct[0,2]:.1f}%)"),
]

fig, ax = plt.subplots(figsize=(15, 12))
bbox_3857 = gpd.GeoSeries([
    gpd.points_from_xy([HK_BBOX[0], HK_BBOX[2]],
                       [HK_BBOX[1], HK_BBOX[3]])[0],
    gpd.points_from_xy([HK_BBOX[0], HK_BBOX[2]],
                       [HK_BBOX[1], HK_BBOX[3]])[1]
], crs="EPSG:4326").to_crs("EPSG:3857").total_bounds

for cls, color, lw, alpha, zorder, label in TRANSITIONS:
    sub = both[both["trans"]==cls]
    rows = [{"geometry": geom_per_rid[rid]} for rid in sub["road_id"] if rid in geom_per_rid]
    if rows:
        gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs("EPSG:3857")
        gdf.plot(ax=ax, color=color, linewidth=lw, alpha=alpha, zorder=zorder)

ax.set_xlim(bbox_3857[0], bbox_3857[2])
ax.set_ylim(bbox_3857[1], bbox_3857[3])
try:
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                   attribution_size=7, zorder=0)
except Exception:
    pass

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], color=clr, lw=1.8, label=lbl)
    for _, clr, _, _, _, lbl in TRANSITIONS
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=18,
          framealpha=0.92, edgecolor="#999")

ax.set_title("Road-state transitions  08:30 → 13:00  (Ragasa Sep 23, S3)\n"
             f"Length-weighted, {len(both):,} roads ({total_km:.0f} km)",
             fontsize=18, fontweight="bold", pad=12)
ax.set_xticks([]); ax.set_yticks([])
ax.set_aspect("equal")

plt.tight_layout()
out2 = f"{OUT}/图50g_transition_map.png"
fig.savefig(out2, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  saved -> {out2}")

print("\nDone.")
