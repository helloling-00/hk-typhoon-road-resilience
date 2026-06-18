"""
Build per-road school / higher-ed / elderly-facility counts at 500m AND 1km.
Also recompute demographic ratios at 1km buffer.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import re

DATA = "/Users/helloling/workspace/thesis/data"

# ── 1. Road midpoints from ep_key ──────────────────────────────────────────
print("Extracting road midpoints from ep_to_road...", flush=True)
ep = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
def parse_ep(s):
    nums = list(map(float, re.findall(r"[-+]?\d*\.?\d+", s)))
    if len(nums) >= 4:
        return ((nums[0]+nums[2])/2, (nums[1]+nums[3])/2)
    return (np.nan, np.nan)
mids = ep["ep_key"].apply(parse_ep)
ep["lon"] = mids.apply(lambda t: t[0])
ep["lat"] = mids.apply(lambda t: t[1])
ep = ep.dropna(subset=["lon","lat"])
print(f"  {len(ep):,} roads with valid midpoints")

# Project to meters (HK ~ EPSG:2326)
roads = gpd.GeoDataFrame(
    ep[["road_id","lon","lat"]],
    geometry=[Point(xy) for xy in zip(ep["lon"], ep["lat"])],
    crs="EPSG:4326",
).to_crs("EPSG:2326")
roads["x"] = roads.geometry.x; roads["y"] = roads.geometry.y
print(f"  projected to EPSG:2326")

# ── 2. Schools / elderly POIs ─────────────────────────────────────────────
print("\nLoading school POIs...", flush=True)
schools = gpd.read_file(f"{DATA}/osm_cache/hk_schools.gpkg").to_crs("EPSG:2326")
schools["geometry"] = schools.geometry.centroid
schools["x"] = schools.geometry.x; schools["y"] = schools.geometry.y

# Tag elderly facilities by name
elderly_pat = re.compile(r"老|elder|aged|senior|護老|安老|長者|耆", re.IGNORECASE)
schools["is_elderly"] = (
    (schools["amenity"]=="social_facility") &
    schools["name"].fillna("").str.contains(elderly_pat)
)
schools["cat"] = "other"
schools.loc[schools["amenity"].isin(["school","kindergarten","childcare"]), "cat"] = "school"
schools.loc[schools["amenity"].isin(["university","college"]), "cat"] = "higher_ed"
schools.loc[schools["is_elderly"], "cat"] = "elderly_facility"
print(schools["cat"].value_counts())

# ── 3. Spatial join via cKDTree ────────────────────────────────────────────
def count_within(road_xy, poi_xy, radius_m):
    if len(poi_xy) == 0:
        return np.zeros(len(road_xy), dtype=int)
    tree = cKDTree(poi_xy)
    counts = tree.query_ball_point(road_xy, r=radius_m, return_length=True)
    return np.array(counts)

road_xy = roads[["x","y"]].values
out = roads[["road_id"]].copy()
for cat in ["school","higher_ed","elderly_facility"]:
    pts = schools[schools["cat"]==cat][["x","y"]].values
    for r in [500, 1000]:
        out[f"{cat}_count_{r}m"] = count_within(road_xy, pts, r)
print("\nRoad-level counts (500m / 1km):")
print(out.describe()[["school_count_500m","school_count_1000m",
                       "higher_ed_count_500m","higher_ed_count_1000m",
                       "elderly_facility_count_500m","elderly_facility_count_1000m"]])

# ── 4. 1km demographic ratios (population-weighted) ────────────────────────
print("\nComputing 1km demographic ratios...", flush=True)
est = pd.read_parquet(f"{DATA}/estate_features.parquet")
# project estates
est_g = gpd.GeoDataFrame(est, geometry=[Point(xy) for xy in zip(est["lon"], est["lat"])],
                          crs="EPSG:4326").to_crs("EPSG:2326")
est_g["x"] = est_g.geometry.x; est_g["y"] = est_g.geometry.y

ratio_cols = [c for c in est.columns if c.startswith("ratio_")]
demo_cols = ["working_pop_ratio","median_income","total_pop"] + ratio_cols
est_xy = est_g[["x","y"]].values
tree = cKDTree(est_xy)

def weighted_ratios(road_xy, radius):
    nbr_idx = tree.query_ball_point(road_xy, r=radius)
    rows = []
    weights = est_g["total_pop"].fillna(0).values
    vals = {c: est_g[c].fillna(np.nan).values for c in demo_cols}
    for nbrs in nbr_idx:
        if not nbrs:
            rows.append({c: np.nan for c in demo_cols} | {"estate_count": 0, "pop_total": 0})
            continue
        w = weights[nbrs]
        row = {"estate_count": len(nbrs), "pop_total": w.sum()}
        for c in demo_cols:
            v = vals[c][nbrs]
            mask = ~np.isnan(v) & (w > 0)
            if mask.sum() == 0:
                row[c] = np.nan
            else:
                row[c] = (v[mask] * w[mask]).sum() / w[mask].sum()
        rows.append(row)
    return pd.DataFrame(rows)

demo_1km = weighted_ratios(road_xy, 1000)
demo_1km.columns = [f"{c}_1000m" for c in demo_1km.columns]
demo_1km["road_id"] = roads["road_id"].values

# Merge
out = out.merge(demo_1km, on="road_id", how="left")
out["log_school_count_500m"]  = np.log1p(out["school_count_500m"])
out["log_school_count_1000m"] = np.log1p(out["school_count_1000m"])
out["log_higher_ed_count_500m"]  = np.log1p(out["higher_ed_count_500m"])
out["log_higher_ed_count_1000m"] = np.log1p(out["higher_ed_count_1000m"])
out["log_elderly_facility_count_500m"]  = np.log1p(out["elderly_facility_count_500m"])
out["log_elderly_facility_count_1000m"] = np.log1p(out["elderly_facility_count_1000m"])

# population density 1km (people per km²)
out["population_density_1000m"] = out["pop_total_1000m"] / (np.pi * 1.0**2)
out["log_population_density_1000m"] = np.log1p(out["population_density_1000m"])

out.to_parquet(f"{DATA}/road_school_elderly_features.parquet", index=False)
print(f"\nsaved -> road_school_elderly_features.parquet  ({len(out):,} roads, {out.shape[1]} cols)")
print("\nNon-null sample:")
print(out.dropna().describe()[["school_count_500m","school_count_1000m",
       "elderly_facility_count_500m","elderly_facility_count_1000m",
       "ratio_学生_1000m","ratio_退休人士_1000m","ratio_age_65plus_1000m"]].T)
