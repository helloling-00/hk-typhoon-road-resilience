"""
Spatial regression: predict road deviation during typhoon from road attributes + spatial features.
Outcome: mean deviation during Yagiasha Signal ≥ 3 period (strongest typhoon)
Predictors:
  - Road functional category (Motorway/Trunk/Primary/Secondary/Tertiary/Residential/Service)
  - Baseline speed (proxy for road capacity / typical usage)
  - Geographic location (lat/lon centroid)
  - Distance to nearest coastline
  - Distance to nearest supermarket/wet market (panic buying)
  - Proximity to residential areas
  - Road orientation (N-S vs E-W, proxy for wind exposure)
"""

import os, ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point, MultiPolygon, Polygon, LineString
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")
# ── Readable thesis-figure style ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 260,
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
    "lines.linewidth": 2.4,
})


DATA = "/Users/helloling/workspace/thesis/data"
OUT  = "/Users/helloling/workspace/thesis"

# ── plotting helpers added by ChatGPT ────────────────────────────────────────
def normalize_road_category(x):
    """Normalize OSM-style road_category values so color dictionaries match."""
    if pd.isna(x):
        return "Other"
    s = str(x).strip().lower()
    mapping = {
        "motorway": "Motorway", "motorway_link": "Motorway",
        "trunk": "Trunk", "trunk_link": "Trunk",
        "primary": "Primary", "primary_link": "Primary",
        "secondary": "Secondary", "secondary_link": "Secondary",
        "tertiary": "Tertiary", "tertiary_link": "Tertiary",
        "residential": "Residential", "living_street": "Residential", "unclassified": "Residential",
        "service": "Service", "services": "Service",
    }
    return mapping.get(s, "Other")


def savefig_with_alias(fig, filename, *aliases, dpi=260):
    """Save the same figure under English and thesis Chinese filenames."""
    main_path = f"{OUT}/{filename}"
    fig.savefig(main_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    for alias in aliases:
        fig.savefig(f"{OUT}/{alias}", dpi=dpi, bbox_inches="tight", facecolor="white")
    return main_path


print("="*60)
print("Step 1: Build road-level spatial dataset")
print("="*60)

# ── load road deviation data from polarization analysis ──────────────────────
bl  = pd.read_parquet(f"{DATA}/baseline_speed.parquet")
ep  = pd.read_parquet(f"{DATA}/ep_to_road.parquet")
rr  = pd.read_parquet(f"{DATA}/road_registry.parquet")[["ep_key","road_id","road_category","road_subcategory"]]
yagi = pd.read_parquet(f"{DATA}/yagiasha_road_timeseries.parquet")

# Road metadata
rr["road_category"] = rr["road_category"].apply(normalize_road_category)
road_meta = rr.drop_duplicates("road_id").set_index("road_id")[["road_category","road_subcategory"]]
road_meta["mean_bl"] = bl.groupby("road_id")["mean_speed"].mean()

# Road centroid from ep_key
def ep_to_centroid(epk):
    try:
        pts = ast.literal_eval(epk)
        return ((pts[0][0]+pts[1][0])/2, (pts[0][1]+pts[1][1])/2)
    except:
        return (np.nan, np.nan)

def ep_to_linestring(epk):
    """Convert endpoint key to a road-centerline segment for the grey road basemap."""
    try:
        pts = ast.literal_eval(epk)
        return LineString([(pts[0][0], pts[0][1]), (pts[1][0], pts[1][1])])
    except Exception:
        return None

def ep_to_orientation(epk):
    """Angle of road in degrees (0=N-S, 90=E-W)."""
    try:
        pts = ast.literal_eval(epk)
        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        if dx == 0 and dy == 0:
            return np.nan
        angle = np.degrees(np.arctan2(abs(dx), abs(dy))) % 90
        return angle  # 0=N-S, 90=E-W
    except:
        return np.nan

print("Extracting road centroids from ep_key...")
ep_info = ep.copy()
ep_info[["lon","lat"]] = ep_info["ep_key"].apply(ep_to_centroid).apply(pd.Series)
ep_info["orientation"] = ep_info["ep_key"].apply(ep_to_orientation)
ep_road = ep_info.set_index("road_id")[["lon","lat","orientation"]]

road_meta = road_meta.join(ep_road, how="left")

# Deviation during Yagiasha active typhoon (Signal ≥ 3)
# Only use slots where signal ≥ 3
YAGIASHA_ACTIVE = [
    (pd.Timestamp("2025-09-22 21:40"), pd.Timestamp("2025-09-24 13:20")),  # S3→S10
    (pd.Timestamp("2025-09-24 13:20"), pd.Timestamp("2025-09-24 20:20")),  # S8
    (pd.Timestamp("2025-09-24 20:20"), pd.Timestamp("2025-09-25 08:20")),  # S3
]
def in_active_period(dt):
    for start, end in YAGIASHA_ACTIVE:
        if start <= dt < end:
            return True
    return False

yagi_active = yagi[yagi["dt"].apply(in_active_period)]
print(f"Yagiasha active (S≥3) observations: {len(yagi_active):,}")

road_dev = yagi_active.groupby("road_id").agg(
    mean_dev=("dev", "mean"),
    std_dev=("dev", "std"),
    n_slots=("dev", "count"),
    mean_obs=("obs", "mean"),
).rename(columns={"mean_dev":"dev_active", "std_dev":"dev_std"})

# Signal 10 specifically
yagi_s10 = yagi[
    (yagi["dt"] >= pd.Timestamp("2025-09-24 02:40")) &
    (yagi["dt"] < pd.Timestamp("2025-09-24 13:20"))
]
road_dev_s10 = yagi_s10.groupby("road_id").agg(
    dev_s10=("dev","mean"), n_s10=("dev","count")
)

# Merge all
road_df = road_meta.join(road_dev, how="inner").join(road_dev_s10, how="left")
road_df = road_df.dropna(subset=["lon","lat","dev_active","mean_bl"])
print(f"Roads with full spatial + deviation data: {len(road_df):,}")


# ── download HK spatial features from OSM ────────────────────────────────────
print("\n" + "="*60)
print("Step 2: Download OSM spatial features")
print("="*60)

CACHE_DIR = f"{DATA}/osm_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Coastline distance ────────────────────────────────────────────────────────
coast_cache = f"{CACHE_DIR}/hk_water.gpkg"
if os.path.exists(coast_cache):
    print("Loading cached HK water bodies...")
    water_gdf = gpd.read_file(coast_cache)
else:
    print("Downloading HK water/sea boundaries from OSM...")
    try:
        # Get natural=water and natural=bay areas
        water_gdf = ox.features_from_place(
            "Hong Kong",
            tags={"natural": ["water","bay","coastline"]}
        )
        water_gdf = water_gdf[water_gdf.geometry.notna()].to_crs("EPSG:4326")
        water_gdf.to_file(coast_cache, driver="GPKG")
        print(f"  Downloaded {len(water_gdf)} water features")
    except Exception as e:
        print(f"  Water download failed: {e}")
        water_gdf = gpd.GeoDataFrame()

# ── Supermarkets / wet markets ────────────────────────────────────────────────
shops_cache = f"{CACHE_DIR}/hk_shops.gpkg"
if os.path.exists(shops_cache):
    print("Loading cached HK shops...")
    shops_gdf = gpd.read_file(shops_cache)
else:
    print("Downloading HK shops (supermarket, convenience, wet market)...")
    try:
        shops_gdf = ox.features_from_place(
            "Hong Kong",
            tags={"shop": ["supermarket","convenience","greengrocer","butcher","seafood"],
                  "amenity": ["marketplace"]}
        )
        shops_gdf = shops_gdf[shops_gdf.geometry.notna()].to_crs("EPSG:4326")
        shops_gdf.to_file(shops_cache, driver="GPKG")
        print(f"  Downloaded {len(shops_gdf)} shop features")
    except Exception as e:
        print(f"  Shops download failed: {e}")
        shops_gdf = gpd.GeoDataFrame()

# ── Emergency services (hospitals, fire stations) ────────────────────────────
emerg_cache = f"{CACHE_DIR}/hk_emergency.gpkg"
if os.path.exists(emerg_cache):
    print("Loading cached HK emergency services...")
    emerg_gdf = gpd.read_file(emerg_cache)
else:
    print("Downloading HK emergency services...")
    try:
        emerg_gdf = ox.features_from_place(
            "Hong Kong",
            tags={"amenity": ["hospital","fire_station","police"]}
        )
        emerg_gdf = emerg_gdf[emerg_gdf.geometry.notna()].to_crs("EPSG:4326")
        emerg_gdf.to_file(emerg_cache, driver="GPKG")
        print(f"  Downloaded {len(emerg_gdf)} emergency features")
    except Exception as e:
        print(f"  Emergency download failed: {e}")
        emerg_gdf = gpd.GeoDataFrame()

# ── Land use (residential vs commercial) ─────────────────────────────────────
landuse_cache = f"{CACHE_DIR}/hk_landuse.gpkg"
if os.path.exists(landuse_cache):
    print("Loading cached HK land use...")
    lu_gdf = gpd.read_file(landuse_cache)
else:
    print("Downloading HK land use...")
    try:
        lu_gdf = ox.features_from_place(
            "Hong Kong",
            tags={"landuse": ["residential","commercial","industrial","retail","mixed"]}
        )
        lu_gdf = lu_gdf[lu_gdf.geometry.notna()].to_crs("EPSG:4326")
        lu_gdf.to_file(landuse_cache, driver="GPKG")
        print(f"  Downloaded {len(lu_gdf)} land use features")
    except Exception as e:
        print(f"  Land use download failed: {e}")
        lu_gdf = gpd.GeoDataFrame()


# ── Compute spatial distances ─────────────────────────────────────────────────
print("\n" + "="*60)
print("Step 3: Compute spatial features per road")
print("="*60)

road_coords = road_df[["lon","lat"]].values  # (n, 2)

def extract_points_from_gdf(gdf, geom_types=("Point","Polygon","MultiPolygon")):
    """Extract representative (lon, lat) for each feature."""
    pts = []
    for geom in gdf.geometry:
        if geom is None:
            continue
        if geom.geom_type == "Point":
            pts.append((geom.x, geom.y))
        elif geom.geom_type in ("Polygon","MultiPolygon"):
            c = geom.centroid
            pts.append((c.x, c.y))
        elif geom.geom_type == "LineString":
            c = geom.interpolate(0.5, normalized=True)
            pts.append((c.x, c.y))
        elif geom.geom_type == "MultiLineString":
            c = geom.centroid
            pts.append((c.x, c.y))
    return np.array(pts) if pts else np.zeros((0, 2))


def nearest_distance_deg(road_xy, feature_xy):
    """Return distance in degrees to nearest feature for each road."""
    if len(feature_xy) == 0:
        return np.full(len(road_xy), np.nan)
    tree = cKDTree(feature_xy)
    dists, _ = tree.query(road_xy, k=1)
    # Convert from degrees to km (approx): 1 degree ≈ 111 km
    return dists * 111.0


# Coastline distance — use HK boundary outline as proxy
print("Computing coastline distance (using water body boundaries)...")
if len(water_gdf) > 0:
    coast_pts = extract_points_from_gdf(water_gdf)
    print(f"  {len(coast_pts)} coast reference points")
    road_df["dist_coast_km"] = nearest_distance_deg(road_coords, coast_pts)
else:
    # Fallback: use distance from HK island center (22.26, 114.18)
    print("  Using HK Island center as fallback coastal reference")
    hk_center = np.array([[114.18, 22.26]])
    road_df["dist_coast_km"] = nearest_distance_deg(road_coords, hk_center)

print(f"  dist_coast_km stats: mean={road_df['dist_coast_km'].mean():.2f}, "
      f"max={road_df['dist_coast_km'].max():.2f} km")

# Supermarket distance
print("Computing distance to nearest supermarket/market...")
if len(shops_gdf) > 0:
    shop_pts = extract_points_from_gdf(shops_gdf)
    road_df["dist_shop_km"] = nearest_distance_deg(road_coords, shop_pts)
    print(f"  dist_shop_km stats: mean={road_df['dist_shop_km'].mean():.3f} km")
else:
    road_df["dist_shop_km"] = np.nan

# Emergency services distance
print("Computing distance to nearest emergency service...")
if len(emerg_gdf) > 0:
    emerg_pts = extract_points_from_gdf(emerg_gdf)
    road_df["dist_emergency_km"] = nearest_distance_deg(road_coords, emerg_pts)
else:
    road_df["dist_emergency_km"] = np.nan

# Land use: fraction of residential within 500m (≈0.0045 deg)
print("Computing residential proximity...")
if len(lu_gdf) > 0:
    res_gdf = lu_gdf[lu_gdf.get("landuse","").isin(["residential","mixed"])
                     if "landuse" in lu_gdf.columns else lu_gdf.index.isin([])]
    if len(res_gdf) > 0:
        res_pts = extract_points_from_gdf(res_gdf)
        road_df["dist_residential_km"] = nearest_distance_deg(road_coords, res_pts)
    else:
        road_df["dist_residential_km"] = np.nan
else:
    road_df["dist_residential_km"] = np.nan

# ── Regression analysis ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("Step 4: Regression analysis")
print("="*60)

# Build regression dataset
reg_df = road_df.copy()
# Outcome: deviation during active typhoon period (S≥3)
reg_df["outcome"] = reg_df["dev_active"]
# Also create direction outcome: 1=faster, -1=slower, 0=neutral
reg_df["direction"] = np.sign(reg_df["dev_active"])
reg_df.loc[reg_df["dev_active"].abs() <= 0.02, "direction"] = 0

# Road category dummies (Residential as reference)
cat_order = ["Motorway","Trunk","Primary","Secondary","Tertiary","Service","Other","Residential"]
reg_df["road_category"] = pd.Categorical(
    reg_df["road_category"].fillna("Other"),
    categories=cat_order,
    ordered=False
)
cat_dummies = pd.get_dummies(reg_df["road_category"], prefix="cat", drop_first=True)

# Feature matrix
feature_cols = []
predictor_data = {}

# 1. Road category dummies
for col in cat_dummies.columns:
    reg_df[col] = cat_dummies[col].astype(float)
    feature_cols.append(col)

# 2. Baseline speed (normalized)
reg_df["bl_norm"]  = (reg_df["mean_bl"] - reg_df["mean_bl"].mean()) / reg_df["mean_bl"].std()
feature_cols.append("bl_norm")

# 3. Geographic position
reg_df["lat_norm"] = (reg_df["lat"] - reg_df["lat"].mean()) / reg_df["lat"].std()
reg_df["lon_norm"] = (reg_df["lon"] - reg_df["lon"].mean()) / reg_df["lon"].std()
feature_cols.extend(["lat_norm","lon_norm"])

# 4. Distance features (where available)
if reg_df["dist_coast_km"].notna().sum() > 1000:
    reg_df["dist_coast_norm"] = (reg_df["dist_coast_km"] - reg_df["dist_coast_km"].mean()) / reg_df["dist_coast_km"].std()
    feature_cols.append("dist_coast_norm")

if reg_df["dist_shop_km"].notna().sum() > 1000:
    reg_df["dist_shop_norm"] = (reg_df["dist_shop_km"] - reg_df["dist_shop_km"].mean()) / reg_df["dist_shop_km"].std()
    feature_cols.append("dist_shop_norm")

if reg_df["dist_emergency_km"].notna().sum() > 1000:
    reg_df["dist_emerg_norm"] = (reg_df["dist_emergency_km"] - reg_df["dist_emergency_km"].mean()) / reg_df["dist_emergency_km"].std()
    feature_cols.append("dist_emerg_norm")

# 5. Road orientation
if reg_df["orientation"].notna().sum() > 1000:
    reg_df["orient_norm"] = (reg_df["orientation"] - 45) / 45  # -1=N-S, +1=E-W
    feature_cols.append("orient_norm")

print(f"Regression features: {feature_cols}")

# Filter to complete cases
reg_clean = reg_df[["outcome"] + feature_cols].dropna()
print(f"Complete observations: {len(reg_clean):,}")

X = reg_clean[feature_cols].values
y = reg_clean["outcome"].values

# OLS with statsmodels for interpretable coefficients
print("\n--- OLS Regression Results ---")
X_df = sm.add_constant(reg_clean[feature_cols], has_constant="raise")
ols_model = sm.OLS(y, X_df).fit()
print(ols_model.summary())

# Extract coefficients table (use model's own param names)
coef_df = pd.DataFrame({
    "feature": ols_model.params.index,
    "coef": ols_model.params.values,
    "std_err": ols_model.bse.values,
    "t_stat": ols_model.tvalues.values,
    "p_value": ols_model.pvalues.values,
    "ci_low": ols_model.conf_int()[0].values,
    "ci_high": ols_model.conf_int()[1].values,
})

print("\n--- Coefficients (sorted by |t|) ---")
print(coef_df.sort_values("t_stat", key=abs, ascending=False).to_string(index=False))
print(f"\nOLS R² = {ols_model.rsquared:.4f}, adj.R² = {ols_model.rsquared_adj:.4f}")

# Random Forest for non-linear importance
print("\n--- Random Forest Variable Importance ---")
rf = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=20,
                           n_jobs=-1, random_state=42)
rf.fit(X, y)
rf_r2 = r2_score(y, rf.predict(X))
# Cross-validated R²
cv_r2 = cross_val_score(rf, X, y, cv=KFold(5, shuffle=True, random_state=42),
                         scoring="r2").mean()
print(f"RF train R² = {rf_r2:.4f}, CV R² = {cv_r2:.4f}")

feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(feat_imp)

# ── FIGURES ──────────────────────────────────────────────────────────────────
print("\nBuilding figures...")

# ── Figure 1: Coefficient plot (OLS) ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# OLS coefficient plot (skip intercept)
coef_plot = coef_df[~coef_df["feature"].isin(["intercept", "const"])].sort_values("coef")
colors = ["#F44336" if p < 0.05 else "#BDBDBD" for p in coef_plot["p_value"]]
y_pos = range(len(coef_plot))
ax1.barh(y_pos, coef_plot["coef"], color=colors, alpha=0.8)
ax1.errorbar(coef_plot["coef"], y_pos,
             xerr=[coef_plot["coef"]-coef_plot["ci_low"],
                   coef_plot["ci_high"]-coef_plot["coef"]],
             fmt="none", color="black", capsize=3, lw=1.5)
ax1.axvline(0, color="black", lw=1, ls="--", alpha=0.6)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(coef_plot["feature"], fontsize=18)
ax1.set_xlabel("OLS Coefficient (change in deviation)")
ax1.set_title(f"OLS Regression Coefficients\n(R²={ols_model.rsquared:.3f}; red = p<0.05)",
              fontsize=13, fontweight="bold")
ax1.grid(axis="x", alpha=0.3)

# RF importance
feat_imp_plot = feat_imp.sort_values()
colors_rf = plt.cm.Blues(np.linspace(0.3, 0.9, len(feat_imp_plot)))
ax2.barh(range(len(feat_imp_plot)), feat_imp_plot.values,
         color=colors_rf[::-1], alpha=0.85)
ax2.set_yticks(range(len(feat_imp_plot)))
ax2.set_yticklabels(feat_imp_plot.index, fontsize=18)
ax2.set_xlabel("Random Forest Feature Importance")
ax2.set_title(f"Random Forest Variable Importance\n(CV R²={cv_r2:.3f})", fontsize=13, fontweight="bold")
ax2.grid(axis="x", alpha=0.3)

fig.suptitle("Predictors of Road-Level Speed Deviation During Yagiasha (Signal ≥ 3)",
             fontsize=15, fontweight="bold")
plt.tight_layout()
savefig_with_alias(fig, "spatial_regression.png", "图10_空间回归分析结果.png")
plt.close()
print(f"Saved: {OUT}/spatial_regression.png")


# ── Figure 2: Category-level deviation boxplot ────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

cat_order2 = ["Motorway","Trunk","Primary","Secondary","Tertiary","Residential","Service","Other"]
cat_colors2 = {
    "Motorway": "#D32F2F", "Trunk": "#F57C00", "Primary": "#FBC02D",
    "Secondary": "#388E3C", "Tertiary": "#1976D2", "Residential": "#7B1FA2",
    "Service": "#5D4037", "Other": "#78909C",
}

# Make the plot robust: if a category has too few observations, skip it;
# if all categories are skipped, still write a visible diagnostic panel.
data_by_cat, labels_used, colors_used = [], [], []
for cat in cat_order2:
    sub = road_df.loc[road_df["road_category"] == cat, "dev_active"].dropna()
    if len(sub) >= 20:
        data_by_cat.append(sub.values)
        labels_used.append(f"{cat}\n(n={len(sub):,})")
        colors_used.append(cat_colors2.get(cat, "#666"))

if data_by_cat:
    bp = ax1.boxplot(
        data_by_cat, patch_artist=True, notch=False,
        medianprops=dict(color="black", lw=2),
        whiskerprops=dict(lw=1.5), capprops=dict(lw=2),
    )
    for patch, col in zip(bp["boxes"], colors_used):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax1.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
    ax1.set_xticklabels(labels_used, fontsize=18, rotation=0)
else:
    ax1.text(0.5, 0.5, "No category has enough observations\nfor S≥3 boxplot", ha="center", va="center", transform=ax1.transAxes)
    ax1.set_xticks([])
ax1.set_title("Deviation During Yagiasha (S≥3)\nby Road Category", fontsize=13, fontweight="bold")
ax1.set_ylabel("Mean Speed Deviation from Baseline")
ax1.grid(axis="y", alpha=0.3)

# Signal 10 specifically
data_s10, labels_s10, colors_s10 = [], [], []
for cat in cat_order2:
    sub = road_df.loc[road_df["road_category"] == cat, "dev_s10"].dropna()
    if len(sub) >= 10:
        data_s10.append(sub.values)
        labels_s10.append(f"{cat}\n(n={len(sub):,})")
        colors_s10.append(cat_colors2.get(cat, "#666"))

if data_s10:
    bp2 = ax2.boxplot(data_s10, patch_artist=True, notch=False,
                      medianprops=dict(color="black", lw=2))
    for patch, col in zip(bp2["boxes"], colors_s10):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax2.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
    ax2.set_xticklabels(labels_s10, fontsize=18, rotation=0)
else:
    ax2.text(0.5, 0.5, "No category has enough observations\nfor Signal 10 boxplot", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xticks([])
ax2.set_title("Deviation During Signal 10 Only\nby Road Category", fontsize=13, fontweight="bold")
ax2.set_ylabel("Mean Speed Deviation from Baseline")
ax2.grid(axis="y", alpha=0.3)

fig.suptitle("Road Category as Predictor of Typhoon-Period Speed Deviation",
             fontsize=15, fontweight="bold")
plt.tight_layout()
savefig_with_alias(fig, "spatial_category_boxplot.png", "图05_道路类别偏差箱线图.png")
plt.close()
print(f"Saved: {OUT}/spatial_category_boxplot.png")


# ── Figure 3: Geographic scatter map of deviation ────────────────────────────
print("Building geographic deviation map...")

# Use a projected CRS for Hong Kong (EPSG:2326) to avoid longitude-latitude distortion.
# Optional boundary overlay: put hksar_18_district_boundary.json in either OUT or DATA.
boundary_gdf = None
for boundary_path in [f"{OUT}/hksar_18_district_boundary.json", f"{DATA}/hksar_18_district_boundary.json"]:
    if os.path.exists(boundary_path):
        try:
            boundary_gdf = gpd.read_file(boundary_path).to_crs("EPSG:2326")
            print(f"  Loaded HK boundary: {boundary_path}")
            break
        except Exception as e:
            print(f"  Failed to load boundary {boundary_path}: {e}")

map_base = road_df.dropna(subset=["lon", "lat"]).copy()
gdf_roads = gpd.GeoDataFrame(
    map_base,
    geometry=gpd.points_from_xy(map_base["lon"], map_base["lat"]),
    crs="EPSG:4326",
).to_crs("EPSG:2326")
gdf_roads["x"] = gdf_roads.geometry.x
gdf_roads["y"] = gdf_roads.geometry.y

# Build a light-grey road skeleton from the same road endpoints.
# This acts as a Hong Kong road basemap without requiring internet access.
line_src = ep.drop_duplicates("road_id")[["road_id", "ep_key"]].copy()
line_src["geometry"] = line_src["ep_key"].apply(ep_to_linestring)
line_src = line_src.dropna(subset=["geometry"])
road_lines = gpd.GeoDataFrame(line_src, geometry="geometry", crs="EPSG:4326").to_crs("EPSG:2326")
# Keep only roads inside the data crop; sample only if the file is extremely dense.
if len(road_lines) > 80000:
    road_lines = road_lines.sample(80000, random_state=42)

# Consistent crop around observed road points, with padding.
xmin, ymin, xmax, ymax = gdf_roads.total_bounds
xpad = (xmax - xmin) * 0.05
typad = (ymax - ymin) * 0.05

fig, axes = plt.subplots(1, 3, figsize=(24, 9))

for ax, (col, title, vmin, vmax) in zip(axes, [
    ("dev_active", "Mean Deviation\nYagiasha S≥3", -0.15, 0.15),
    ("dev_s10",    "Mean Deviation\nSignal 10 Only", -0.25, 0.25),
    ("mean_bl",    "Baseline Speed\n(reference)", 0.3, 1.0),
]):
    plot_gdf = gdf_roads.dropna(subset=[col])
    if len(plot_gdf) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        ax.set_axis_off()
        continue
    if len(plot_gdf) > 30000:
        plot_gdf = plot_gdf.sample(30000, random_state=42)

    if boundary_gdf is not None and len(boundary_gdf) > 0:
        boundary_gdf.boundary.plot(ax=ax, color="#9E9E9E", linewidth=0.35, alpha=0.8, zorder=1)
        try:
            boundary_gdf.plot(ax=ax, color="#F5F5F5", edgecolor="#BDBDBD", linewidth=0.25, alpha=0.25, zorder=0)
        except Exception:
            pass

    # Hong Kong road skeleton basemap
    if len(road_lines) > 0:
        road_lines.plot(ax=ax, color="#CFCFCF", linewidth=0.18, alpha=0.45, zorder=1.4)

    cmap = "RdBu" if "dev" in col else "YlOrRd"
    sc = ax.scatter(
        plot_gdf["x"], plot_gdf["y"],
        c=plot_gdf[col].clip(vmin, vmax),
        cmap=cmap, vmin=vmin, vmax=vmax,
        s=4.8, alpha=0.78, rasterized=True, zorder=2.5,
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.72)
    cbar.ax.tick_params(labelsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - typad, ymax + typad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("HK Grid Easting (m)", fontsize=14)
    ax.set_ylabel("HK Grid Northing (m)", fontsize=14)
    ax.grid(alpha=0.15, linewidth=0.5)

fig.suptitle("Geographic Distribution of Road Speed Deviation — Yagiasha Typhoon\n"
             "Blue = faster than baseline (demand suppression), Red = slower (supply disruption)",
             fontsize=15, fontweight="bold")
plt.tight_layout()
savefig_with_alias(fig, "spatial_deviation_map.png", "图09_空间偏差地理分布图.png")
plt.close()
print(f"Saved: {OUT}/spatial_deviation_map.png")

print("\n=== Spatial Analysis Complete ===")
