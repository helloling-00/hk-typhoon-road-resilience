"""
POI multicollinearity diagnostics + improved regressions.
  1. Compute total_poi_density and POI shares
  2. Correlation matrix among POI densities
  3. VIF
  4. Re-run 3 binary logit models:
     A) POI densities + total_poi_density control
     B) POI shares + total_poi_density
  5. Compare fits
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
OUT  = "/Users/helloling/workspace/thesis"
DEV_HI, DEV_LO = 0.03, -0.03

def classify(d):
    if d > DEV_HI: return "F"
    if d < DEV_LO: return "S"
    return "N"

# ─── Load & merge ─────────────────────────────────────────────────────────
print("Loading...", flush=True)
ts = pd.read_parquet(f"{DATA}/yagiasha_road_timeseries.parquet")
ts["ds"] = pd.to_datetime(ts["dt"]).dt.strftime("%Y-%m-%d")
sep23 = ts[ts["ds"]=="2025-09-23"].copy()

rt = pd.read_parquet(f"{DATA}/regression_table.parquet")
feat_cols = [
    "road_id", "road_category", "road_length_m",
    "intersection_degree", "dist_to_coast_m",
    "work_density", "education_density", "retail_density", "food_drink_density",
    "recreation_density", "medical_density", "transport_density",
    "tourism_density", "finance_density", "civic_density",
    "incident_count_500m", "severe_incident_500m",
]
rt_sub = rt[feat_cols].drop_duplicates("road_id")

s17 = sep23[sep23["slot"]==17][["road_id","dev"]].rename(columns={"dev":"dev_morn"})
s26 = sep23[sep23["slot"]==26][["road_id","dev"]].rename(columns={"dev":"dev_mid"})
both = s17.merge(s26, on="road_id", how="inner")
both["state_morn"] = both["dev_morn"].apply(classify)
both["state_mid"]  = both["dev_mid"].apply(classify)
both["trans"] = both["state_morn"] + "→" + both["state_mid"]

merged = both.merge(rt_sub, on="road_id", how="inner")

for c in ["incident_count_500m", "severe_incident_500m"]:
    merged[f"log_{c}"] = np.log1p(merged[c])

# Collapse road categories
cat_map = {
    "motorway": "highway", "motorway_link": "highway",
    "trunk": "highway", "trunk_link": "highway",
    "primary": "arterial", "primary_link": "arterial",
    "secondary": "arterial", "secondary_link": "arterial",
    "tertiary": "local", "tertiary_link": "local",
    "street": "local", "service": "local",
}
merged["road_broad"] = merged["road_category"].map(cat_map)

# ─── 1. Compute total POI density & shares ────────────────────────────────
POI_CATS = ["work","education","retail","food_drink","recreation",
            "medical","transport","tourism","finance","civic"]

merged["total_poi_density"] = sum(merged[f"{c}_density"] for c in POI_CATS)

for c in POI_CATS:
    merged[f"{c}_share"] = np.where(
        merged["total_poi_density"] > 0,
        merged[f"{c}_density"] / merged["total_poi_density"],
        0.0
    )

# ─── 2. Correlation matrix among POI density variables ────────────────────
print("\n" + "="*80)
print("  POI DENSITY CORRELATION MATRIX  (|r| > 0.70 flagged)")
print("="*80)

poi_dens_cols = [f"{c}_density" for c in POI_CATS]
corr = merged[poi_dens_cols].corr()

# Print matrix
print(f"\n{'':>16s}", end="")
for c in POI_CATS:
    print(f"{c:>10s}", end="")
print()
for ci in POI_CATS:
    print(f"  {ci:14s}", end="")
    for cj in POI_CATS:
        v = corr.loc[f"{ci}_density", f"{cj}_density"]
        flag = "***" if abs(v) > 0.7 else ""
        print(f" {v:8.3f}{'*' if abs(v)>0.7 else ' '}", end="")
    print()

# Flagged pairs
print(f"\n  Highly correlated pairs (|r| > 0.70):")
flagged = []
for i in range(len(POI_CATS)):
    for j in range(i+1, len(POI_CATS)):
        v = corr.iloc[i, j]
        if abs(v) > 0.70:
            flagged.append(f"  {POI_CATS[i]}_density — {POI_CATS[j]}_density: r = {v:.3f}")
if flagged:
    for s in flagged: print(s)
else:
    print("  (none)")

# ─── Heatmap ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(POI_CATS)))
ax.set_xticklabels(POI_CATS, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(POI_CATS)))
ax.set_yticklabels(POI_CATS, fontsize=9)
for i in range(len(POI_CATS)):
    for j in range(len(POI_CATS)):
        v = corr.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.6 else "black", fontweight="bold")
ax.set_title("POI Density Correlation Matrix", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.82)
plt.tight_layout()
fig.savefig(f"{OUT}/图59_poi_correlation.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  saved -> 图59_poi_correlation.png")

# ─── 3. VIF ───────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("  VIF (Variance Inflation Factor)")
print("="*80)

# Build X for VIF computation (same spec as Model A)
X_vars_base = [
    "intersection_degree", "dist_to_coast_m",
    "log_incident_count_500m", "log_severe_incident_500m",
]
X_vfs = merged[X_vars_base + poi_dens_cols].dropna()
X_broad_vif = pd.get_dummies(merged.loc[X_vfs.index, "road_broad"], prefix="road", drop_first=True)
X_vif = pd.concat([X_vfs, X_broad_vif.astype(float)], axis=1)
X_vif = X_vif.astype(float)
X_vif = X_vif.dropna(axis=1, how="all")  # drop any all-zero columns
X_vif_const = sm.add_constant(X_vif)

try:
    vif_df = pd.DataFrame({
        "variable": X_vif_const.columns,
        "VIF": [variance_inflation_factor(X_vif_const.values, i) for i in range(X_vif_const.shape[1])]
    }).sort_values("VIF", ascending=False)
    print(vif_df.to_string(index=False))
    vif_high = vif_df[vif_df["VIF"] > 10]
    if len(vif_high):
        print(f"\n  ⚠ VIF > 10: {', '.join(vif_high['variable'].tolist())}")
except Exception as e:
    print(f"  VIF computation failed: {e}")
    print("  Using correlation matrix as alternative multicollinearity check.")

# ─── 4. Run models ────────────────────────────────────────────────────────
OUTCOME_LABELS = {
    "y_morn_F": "Model 1: Morning clearance (F vs not-F at 08:30)",
    "y_mid_S":  "Model 2: Midday congestion (S vs not-S at 13:00)",
    "y_F_to_S": "Model 3: Reversal (F→S vs F→F, conditional on morning-F)",
}

STRUCT_VARS = ["intersection_degree", "dist_to_coast_m"]
INCIDENT_VARS = ["log_incident_count_500m", "log_severe_incident_500m"]

def build_spec(df, spec_type):
    """Build X for spec A (densities) or B (shares, drop civic_share as ref), plus controls."""
    if spec_type == "density":
        poi_vars = [f"{c}_density" for c in POI_CATS]
        extra = []
    else:  # shares — drop one category (civic) as reference to avoid perfect collinearity
        poi_vars = [f"{c}_share" for c in POI_CATS if c != "civic"]
        extra = ["total_poi_density"]
    X = pd.concat([
        df[STRUCT_VARS + INCIDENT_VARS + poi_vars + extra].astype(float),
        pd.get_dummies(df["road_broad"], prefix="road", drop_first=True).astype(float)
    ], axis=1)
    return X

def run_model(df, y_col, spec_type, spec_label):
    """Fit, print, return results."""
    X = build_spec(df, spec_type)
    y = df[y_col].values.astype(int)

    valid = X.notna().all(axis=1)
    X = X[valid]; y = y[valid.values]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    X_sm = sm.add_constant(X_scaled)
    model = sm.Logit(y, X_sm.astype(float))
    result = model.fit(disp=False, maxiter=500)

    ll_null = result.llnull
    ll_model = result.llf
    mcfadden_r2 = 1 - ll_model / ll_null
    y_pred_prob = result.predict(X_sm)
    auc = roc_auc_score(y, y_pred_prob)
    n1 = y.sum()
    bal = n1 / len(y) * 100

    # Print
    print(f"\n{'─'*80}")
    print(f"  {OUTCOME_LABELS[y_col]}  |  {spec_label}")
    print(f"{'─'*80}")
    print(f"  N = {len(y):,}  |  y=1: {n1:,} ({bal:.1f}%)  |  McFadden R² = {mcfadden_r2:.4f}  |  AUC = {auc:.4f}")
    print(f"  All continuous predictors standardized (mean=0, std=1)")
    print()
    print(f"  {'Variable':30s} {'Coef':>8s} {'SE':>8s} {'OR':>8s} {'p':>8s} {'95% CI low':>10s} {'95% CI high':>10s}")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*10}")

    rows = []
    for var in result.params.index:
        coef = result.params[var]
        se = result.bse[var]
        pval = result.pvalues[var]
        or_val = np.exp(coef)
        ci_low = np.exp(coef - 1.96*se)
        ci_high = np.exp(coef + 1.96*se)
        sig = "***" if pval<0.01 else ("**" if pval<0.05 else ("*" if pval<0.1 else ""))

        print(f"  {var:30s} {coef:>8.4f} {se:>8.4f} {or_val:>8.4f} {pval:>8.4f} {ci_low:>10.4f} {ci_high:>10.4f}  {sig}")
        rows.append({"variable": var, "coef": coef, "se": se, "OR": or_val,
                     "p": pval, "ci_95_low": ci_low, "ci_95_high": ci_high})

    return {"result": result, "pseudo_r2": mcfadden_r2, "auc": auc,
            "n": len(y), "bal_pct": bal, "rows": rows}

# Define outcomes
merged["y_morn_F"] = (merged["state_morn"] == "F").astype(int)
merged["y_mid_S"]  = (merged["state_mid"] == "S").astype(int)
morn_F = merged[merged["state_morn"] == "F"].copy()
morn_F["y_F_to_S"] = (morn_F["trans"] == "F→S").astype(int)

all_results = {}
for outcome in ["y_morn_F", "y_mid_S", "y_F_to_S"]:
    df = morn_F if outcome == "y_F_to_S" else merged
    for spec_type, spec_label in [("density", "Spec A: POI densities + total"),
                                   ("share",   "Spec B: POI shares + total")]:
        key = f"{outcome}_{spec_type}"
        all_results[key] = run_model(df, outcome, spec_type, spec_label)

# ─── 5. Model comparison table ────────────────────────────────────────────
print(f"\n\n{'='*90}")
print("  MODEL COMPARISON SUMMARY")
print(f"{'='*90}")
print(f"  {'Model':35s} {'Spec':12s} {'Pseudo R²':>10s} {'AUC':>8s} {'N':>6s} {'y=1%':>7s}")
print(f"  {'─'*35} {'─'*12} {'─'*10} {'─'*8} {'─'*6} {'─'*7}")
for outcome, label in OUTCOME_LABELS.items():
    for spec_type, spec_label in [("density", "Spec A"), ("share", "Spec B")]:
        r = all_results[f"{outcome}_{spec_type}"]
        print(f"  {label:35s} {spec_label:12s} {r['pseudo_r2']:>10.4f} {r['auc']:>8.4f} {r['n']:>6d} {r['bal_pct']:>6.1f}%")

# ─── 6. Save CSVs ─────────────────────────────────────────────────────────
for outcome in ["y_morn_F", "y_mid_S", "y_F_to_S"]:
    for stype in ["density", "share"]:
        pd.DataFrame(all_results[f"{outcome}_{stype}"]["rows"]).to_csv(
            f"{OUT}/preS8_logit_{outcome}_{stype}.csv", index=False)
print(f"\n  All CSVs saved -> preS8_logit_*_density.csv / preS8_logit_*_share.csv")

# ─── 7. Coefficient plot: significant POI variables only ──────────────────
# Compare Spec A vs Spec B for Model 3 (F→S reversal) — the key model
fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
for ax, (stype, slabel) in zip(axes, [("density", "Spec A: POI densities + total"),
                                        ("share", "Spec B: POI shares + total")]):
    r = all_results[f"y_F_to_S_{stype}"]
    res = r["result"]
    params = res.params.drop("const", errors="ignore")
    bse = res.bse.drop("const", errors="ignore")
    vals = params.values
    ses = bse.values
    cis = 1.96 * ses
    labels = params.index.tolist()
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in vals]
    ax.barh(range(len(labels)), vals, xerr=cis, color=colors, alpha=0.82, height=0.6)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_title(f"Model 3: F→S vs F→F  —  {slabel}\n"
                 f"Pseudo R²={r['pseudo_r2']:.4f}  AUC={r['auc']:.4f}  N={r['n']}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Coefficient (std. predictors)", fontsize=9)

fig.suptitle("Reversal Model (F→S vs F→F): Spec A vs Spec B  —  Ragasa Sep 23, S3",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT}/图59b_FS_spec_comparison.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  saved -> 图59b_FS_spec_comparison.png")

print("\n" + "="*80)
print("  INTERPRETATION NOTES")
print("="*80)
print("""
  All coefficients represent spatial associations, NOT causal effects.
  - POI data captured at static OSM snapshot; does not reflect hourly visitor flows.
  - 500m buffer blends land-use contexts; short roads (<200m) have more precise
    local attribution than long roads (>1km).
  - Incident counts are period-aggregated, not specific to Sep 23 midday.
  - Unobserved confounders (weather, real-time traffic management, firm-level
    work-from-home policies) are not in the model.
  - Significant coefficients indicate that roads with higher POI density/share
    in category X were more/less likely to experience the outcome, after
    controlling for road structure and total POI density.
  - "Share" coefficients have a compositional interpretation: given the same
    total POI count, roads with a higher share of category X ...
""")

print("Done.")
