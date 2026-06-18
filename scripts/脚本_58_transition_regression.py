"""
Multinomial logit: what predicts F→S reversal vs F→F persistence?
Tier 1: POI + structural + incidents (full sample ~2000 roads)
Tier 2: + demographics (census subset ~1100 roads)
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
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

# ─── Load and merge ────────────────────────────────────────────────────────
print("Loading data...", flush=True)
ts = pd.read_parquet(f"{DATA}/yagiasha_road_timeseries.parquet")
ts["ds"] = pd.to_datetime(ts["dt"]).dt.strftime("%Y-%m-%d")
sep23 = ts[ts["ds"]=="2025-09-23"].copy()

rt = pd.read_parquet(f"{DATA}/regression_table.parquet")

s17 = sep23[sep23["slot"]==17][["road_id","dev"]].rename(columns={"dev":"dev_morn"})
s26 = sep23[sep23["slot"]==26][["road_id","dev"]].rename(columns={"dev":"dev_mid"})
both = s17.merge(s26, on="road_id", how="inner")
both["state_morn"] = both["dev_morn"].apply(classify)
both["state_mid"]  = both["dev_mid"].apply(classify)
both["trans"] = both["state_morn"] + "→" + both["state_mid"]

TRANSITIONS = ["F→F", "F→S", "N→S", "S→S"]
both = both[both["trans"].isin(TRANSITIONS)].copy()
both["y"] = both["trans"].astype("category")
both["y"] = both["y"].cat.reorder_categories(TRANSITIONS)

# ─── Merge features ────────────────────────────────────────────────────────
feat_cols = [
    "road_id", "road_category", "road_length_m",
    "intersection_degree", "dist_to_coast_m",
    "work_density", "education_density", "retail_density", "food_drink_density",
    "recreation_density", "medical_density", "transport_density",
    "tourism_density", "finance_density", "civic_density",
    "incident_count_500m", "severe_incident_500m", "closure_nearby_500m",
]
rt_sub = rt[feat_cols].drop_duplicates("road_id")
merged = both.merge(rt_sub, on="road_id", how="inner")

# Log-transform skewed features
for c in ["incident_count_500m", "severe_incident_500m"]:
    merged[f"log_{c}"] = np.log1p(merged[c])

# Dummify road_category, collapse rare categories
cat_map = {
    "motorway": "highway", "motorway_link": "highway",
    "trunk": "highway", "trunk_link": "highway",
    "primary": "arterial", "primary_link": "arterial",
    "secondary": "arterial", "secondary_link": "arterial",
    "tertiary": "local", "tertiary_link": "local",
    "street": "local", "service": "local",
}
merged["road_broad"] = merged["road_category"].map(cat_map)

# ─── Tier 1: POI + structural + incidents ──────────────────────────────────
print(f"\nTier 1: n={len(merged)} roads")
X_vars_t1 = [
    "intersection_degree", "dist_to_coast_m",
    "work_density", "education_density", "retail_density", "food_drink_density",
    "recreation_density", "medical_density", "transport_density",
    "tourism_density", "finance_density", "civic_density",
    "log_incident_count_500m", "log_severe_incident_500m",
]
X_broad = pd.get_dummies(merged["road_broad"], prefix="road", drop_first=True)

X1 = pd.concat([merged[X_vars_t1], X_broad], axis=1)
X1 = X1.astype(float)
# Drop rows with any NaN
valid = X1.notna().all(axis=1)
X1 = X1[valid]
merged_t1 = merged.loc[valid].copy()
y_t1 = merged_t1["y"].values

# Standardize
scaler1 = StandardScaler()
X1_scaled = pd.DataFrame(scaler1.fit_transform(X1), columns=X1.columns, index=X1.index)

# Fit statsmodels MNLogit (F→F as reference)
y_cat = pd.Categorical(merged_t1["y"].values, categories=TRANSITIONS)
X1_sm = sm.add_constant(X1_scaled)
model1 = sm.MNLogit(y_cat, X1_sm.astype(float))
result1 = model1.fit(method="lbfgs", maxiter=500, disp=False)
print(result1.summary())

# ─── Coefficient plot ──────────────────────────────────────────────────────
coefs = result1.params.copy()  # rows=variables, cols=categories (0,1,2)
coefs = coefs.drop("const", axis=0, errors="ignore")
labels = coefs.index.tolist()
bse_df = result1.bse.copy().drop("const", axis=0, errors="ignore")

fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
for idx, (trans, ax) in enumerate(zip(["F→S", "N→S", "S→S"], axes)):
    vals = coefs.iloc[:, idx].values
    ses = bse_df.iloc[:, idx].values
    cis = 1.96 * ses

    colors = ["#d62728" if v > 0 else "#2ca02c" for v in vals]
    ax.barh(range(len(labels)), vals, xerr=cis, color=colors, alpha=0.8, height=0.6)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(range(len(labels)))
    if idx == 0:
        ax.set_yticklabels(labels, fontsize=9)
    else:
        ax.set_yticklabels([])
    ax.set_title(f"{trans} vs F→F (reference)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Coefficient (standardized)", fontsize=9)

fig.suptitle("Multinomial Logit: Predictors of Road-State Transitions  (Ragasa Sep 23, S3)\n"
             "Reference category: F→F (persistent clearance)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out1 = f"{OUT}/图58a_multinomial_coefs.png"
fig.savefig(out1, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n  saved -> {out1}")

# ─── Tier 2: add demographics (smaller sample) ─────────────────────────────
demo_cols = [
    "population_density_500m", "median_income_500m",
    "working_pop_ratio_500m", "ratio_学生_500m",
    "ratio_age_25_44_500m", "ratio_age_65plus_500m",
]
rt_demo = rt[["road_id"] + demo_cols].drop_duplicates("road_id")
merged2 = merged.merge(rt_demo, on="road_id", how="inner")
# Drop rows with NaN demographics
merged2 = merged2.dropna(subset=demo_cols)
print(f"\nTier 2 (with demographics): n={len(merged2)} roads")

X_vars_t2 = X_vars_t1 + demo_cols
X2 = pd.concat([merged2[X_vars_t2],
                pd.get_dummies(merged2["road_broad"], prefix="road", drop_first=True)], axis=1)
X2 = X2.astype(float)
y2 = merged2["y"].values

scaler2 = StandardScaler()
X2_scaled = pd.DataFrame(scaler2.fit_transform(X2), columns=X2.columns, index=X2.index)

y2_cat = pd.Categorical(merged2["y"].values, categories=TRANSITIONS)
X2_sm = sm.add_constant(X2_scaled)
model2 = sm.MNLogit(y2_cat, X2_sm.astype(float))
result2 = model2.fit(method="lbfgs", maxiter=500, disp=False)
print(result2.summary())

# ─── Coefficient plot with demographics ────────────────────────────────────
coefs2 = result2.params.copy()
coefs2 = coefs2.drop("const", axis=0, errors="ignore")
labels2 = coefs2.index.tolist()
bse_df2 = result2.bse.copy().drop("const", axis=0, errors="ignore")

fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
for idx, (trans, ax) in enumerate(zip(["F→S", "N→S", "S→S"], axes)):
    vals = coefs2.iloc[:, idx].values
    ses = bse_df2.iloc[:, idx].values
    cis = 1.96 * ses

    colors = ["#d62728" if v > 0 else "#2ca02c" for v in vals]
    ax.barh(range(len(labels2)), vals, xerr=cis, color=colors, alpha=0.8, height=0.6)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(range(len(labels2)))
    if idx == 0:
        ax.set_yticklabels(labels2, fontsize=8)
    else:
        ax.set_yticklabels([])
    ax.set_title(f"{trans} vs F→F (reference)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Coefficient (standardized)", fontsize=9)

fig.suptitle("Multinomial Logit with Demographics  —  Ragasa Sep 23, S3",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out2 = f"{OUT}/图58b_multinomial_coefs_demo.png"
fig.savefig(out2, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  saved -> {out2}")

# ─── Marginal effects (average) at mean ────────────────────────────────────
margins1 = result1.get_margeff(at="mean")
print("\n=== Marginal Effects (Tier 1, at mean) ===")
print(margins1.summary())

# ─── Key odds ratios for F→S ───────────────────────────────────────────────
print("\n=== Key odds ratios: F→S vs F→F (Tier 1) ===")
or_fs = np.exp(result1.params.iloc[:, 0])
for var, or_val in or_fs.drop("const").sort_values(ascending=False).items():
    stars = "***" if abs(or_val-1) > 0.3 else ("**" if abs(or_val-1) > 0.15 else "")
    print(f"  {var:30s}: {or_val:6.3f}  {stars}")
print("\n  (OR > 1 = higher odds of F→S relative to F→F)")

# ─── Summary table as CSV ──────────────────────────────────────────────────
summary_rows = []
cat_names = ["F→S", "N→S", "S→S"]
for ci, trans in enumerate(cat_names):
    for vi, var in enumerate(result1.params.index):
        coef = result1.params.iloc[vi, ci]
        se = result1.bse.iloc[vi, ci]
        pval = result1.pvalues.iloc[vi, ci]
        or_val = np.exp(coef)
        summary_rows.append({"transition": trans, "variable": var,
                             "coef": coef, "se": se, "pvalue": pval, "odds_ratio": or_val})
df_sum = pd.DataFrame(summary_rows)
df_sum.to_csv(f"{OUT}/preS8_regression_coefficients.csv", index=False)
print(f"\n  saved -> preS8_regression_coefficients.csv")

print("\nDone.")
