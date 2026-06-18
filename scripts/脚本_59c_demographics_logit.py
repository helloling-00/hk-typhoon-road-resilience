"""
Add demographic variables to the three binary logit models.
Key hypothesis: midday congestion = early return home → higher in:
  - high population density areas (more residents)
  - high working population ratio (more commuters returning)
  - high employee ratio (more office workers leaving early)
  - high student ratio areas (schools close → families at home?)
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

# ─── Load ─────────────────────────────────────────────────────────────────
print("Loading...", flush=True)
ts = pd.read_parquet(f"{DATA}/yagiasha_road_timeseries.parquet")
ts["ds"] = pd.to_datetime(ts["dt"]).dt.strftime("%Y-%m-%d")
sep23 = ts[ts["ds"]=="2025-09-23"].copy()

rt = pd.read_parquet(f"{DATA}/regression_table.parquet")

# Full feature set: POI + structural + incidents + demographics
feat_cols = [
    "road_id", "road_category", "road_length_m",
    "intersection_degree", "dist_to_coast_m",
    "work_density", "education_density", "retail_density", "food_drink_density",
    "recreation_density", "medical_density", "transport_density",
    "tourism_density", "finance_density", "civic_density",
    "incident_count_500m", "severe_incident_500m",
    # Demographics (from census, 500m buffer)
    "population_total_500m", "population_density_500m",
    "median_income_500m",
    "working_pop_ratio_500m",
    "ratio_学生_500m",
    "ratio_雇员_500m",
    "ratio_退休人士_500m",
    "ratio_age_0_14_500m",
    "ratio_age_25_44_500m",
    "ratio_age_65plus_500m",
]
rt_sub = rt[feat_cols].drop_duplicates("road_id")

s17 = sep23[sep23["slot"]==17][["road_id","dev"]].rename(columns={"dev":"dev_morn"})
s26 = sep23[sep23["slot"]==26][["road_id","dev"]].rename(columns={"dev":"dev_mid"})
both = s17.merge(s26, on="road_id", how="inner")
both["state_morn"] = both["dev_morn"].apply(classify)
both["state_mid"]  = both["dev_mid"].apply(classify)
both["trans"] = both["state_morn"] + "→" + both["state_mid"]

merged = both.merge(rt_sub, on="road_id", how="inner")

for c in ["incident_count_500m", "severe_incident_500m",
          "population_total_500m", "population_density_500m"]:
    merged[f"log_{c}"] = np.log1p(merged[c])

cat_map = {
    "motorway": "highway", "motorway_link": "highway",
    "trunk": "highway", "trunk_link": "highway",
    "primary": "arterial", "primary_link": "arterial",
    "secondary": "arterial", "secondary_link": "arterial",
    "tertiary": "local", "tertiary_link": "local",
    "street": "local", "service": "local",
}
merged["road_broad"] = merged["road_category"].map(cat_map)

# ─── Define variable blocks ──────────────────────────────────────────────
STRUCT_VARS = ["intersection_degree", "dist_to_coast_m"]
POI_VARS = ["work_density", "education_density", "retail_density", "food_drink_density",
            "recreation_density", "medical_density", "transport_density",
            "tourism_density", "finance_density", "civic_density"]
INCIDENT_VARS = ["log_incident_count_500m"]  # drop severe to avoid VIF > 100
DEMO_VARS = [
    "log_population_density_500m",
    "median_income_500m",
    "working_pop_ratio_500m",
    "ratio_学生_500m",
    "ratio_雇员_500m",
    "ratio_退休人士_500m",
    "ratio_age_25_44_500m",
    "ratio_age_65plus_500m",
]

# ─── Run models ──────────────────────────────────────────────────────────
OUTCOME_LABELS = {
    "y_morn_F": "Model 1: Morning clearance (F vs not-F at 08:30)",
    "y_mid_S":  "Model 2: Midday congestion (S vs not-S at 13:00)",
    "y_F_to_S": "Model 3: Reversal (F→S vs F→F, conditional on morning-F)",
}

merged["y_morn_F"] = (merged["state_morn"] == "F").astype(int)
merged["y_mid_S"]  = (merged["state_mid"] == "S").astype(int)
morn_F = merged[merged["state_morn"] == "F"].copy()
morn_F["y_F_to_S"] = (morn_F["trans"] == "F→S").astype(int)

def run_spec(df, y_col, var_blocks, spec_label):
    """Fit binary logit, return full diagnostics."""
    X_parts = []
    for block_name, block_vars in var_blocks:
        if block_name == "road_broad":
            X_parts.append(
                pd.get_dummies(df["road_broad"], prefix="road", drop_first=True).astype(float)
            )
        else:
            X_parts.append(df[block_vars].astype(float))
    X = pd.concat(X_parts, axis=1)
    y = df[y_col].values.astype(int)

    # Drop rows with any NaN in X or y
    valid = X.notna().all(axis=1)
    X = X[valid]
    y = y[valid.values]

    # Drop any perfectly collinear / constant columns
    X = X.loc[:, X.nunique() > 1]

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

    print(f"\n{'─'*85}")
    print(f"  {OUTCOME_LABELS[y_col]}")
    print(f"  {spec_label}")
    print(f"{'─'*85}")
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
            "n": len(y), "bal_pct": bal, "rows": rows, "X_scaled": X_scaled, "y": y}

# ─── Run 3 specs for each outcome ─────────────────────────────────────────
# Spec 1: POI only (baseline, full sample)
# Spec 2: Demographics only (census subset)
# Spec 3: POI + Demographics (census subset)

all_results = {}

for outcome in ["y_morn_F", "y_mid_S", "y_F_to_S"]:
    df = morn_F if outcome == "y_F_to_S" else merged

    # Spec 1: POI + structural + road class
    all_results[f"{outcome}_poi"] = run_spec(
        df, outcome,
        [("struct", STRUCT_VARS), ("poi", POI_VARS), ("incident", INCIDENT_VARS), ("road_broad", None)],
        "Spec 1: POI densities only"
    )

    # Spec 2: Demographics + structural + road class (drops POI)
    all_results[f"{outcome}_demo"] = run_spec(
        df, outcome,
        [("struct", STRUCT_VARS), ("demo", DEMO_VARS), ("incident", INCIDENT_VARS), ("road_broad", None)],
        "Spec 2: Demographics only"
    )

    # Spec 3: POI + Demographics + structural + road class
    all_results[f"{outcome}_full"] = run_spec(
        df, outcome,
        [("struct", STRUCT_VARS), ("poi", POI_VARS), ("demo", DEMO_VARS),
         ("incident", INCIDENT_VARS), ("road_broad", None)],
        "Spec 3: POI + Demographics (full model)"
    )

# ─── Model comparison ─────────────────────────────────────────────────────
print(f"\n\n{'='*90}")
print("  MODEL COMPARISON")
print(f"{'='*90}")
print(f"  {'Model':35s} {'Spec':25s} {'N':>6s} {'Pseudo R²':>10s} {'AUC':>8s} {'y=1%':>7s}")
print(f"  {'─'*35} {'─'*25} {'─'*6} {'─'*10} {'─'*8} {'─'*7}")
for outcome, label in OUTCOME_LABELS.items():
    for skey, sname in [("poi", "POI only"), ("demo", "Demographics only"), ("full", "POI + Demo")]:
        r = all_results[f"{outcome}_{skey}"]
        print(f"  {label:35s} {sname:25s} {r['n']:>6d} {r['pseudo_r2']:>10.4f} {r['auc']:>8.4f} {r['bal_pct']:>6.1f}%")

# ─── Coefficient plot: Model 3 (F→S) — full model ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 10), sharey=True)
for ax, (skey, sname) in zip(axes, [("poi", "Spec 1: POI only"),
                                      ("demo", "Spec 2: Demographics only"),
                                      ("full", "Spec 3: POI + Demographics")]):
    r = all_results[f"y_F_to_S_{skey}"]
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
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_title(f"Model 3: F→S vs F→F\n{sname}\n"
                 f"Pseudo R²={r['pseudo_r2']:.4f}  AUC={r['auc']:.4f}  N={r['n']}",
                 fontsize=9, fontweight="bold")

fig.suptitle("Reversal Model (F→S vs F→F): POI vs Demographics vs Full  —  Ragasa Sep 23, S3",
             fontsize=12, fontweight="bold")
plt.tight_layout()
out_plot = f"{OUT}/图59c_FS_demographics.png"
fig.savefig(out_plot, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n  saved -> {out_plot}")

# ─── Also plot morning and midday models ──────────────────────────────────
for outcome, plot_name, title_key in [
    ("y_morn_F", "图59c_morn_F.png", "Morning clearance (F vs not-F)"),
    ("y_mid_S", "图59c_mid_S.png", "Midday congestion (S vs not-S)")
]:
    fig, axes = plt.subplots(1, 3, figsize=(22, 10), sharey=True)
    for ax, (skey, sname) in zip(axes, [("poi", "POI only"),
                                          ("demo", "Demographics only"),
                                          ("full", "POI + Demo")]):
        r = all_results[f"{outcome}_{skey}"]
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
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_title(f"{title_key}\n{sname}\n"
                     f"Pseudo R²={r['pseudo_r2']:.4f}  AUC={r['auc']:.4f}  N={r['n']}",
                     fontsize=9, fontweight="bold")

    fig.suptitle(f"{title_key}  —  Ragasa Sep 23, S3", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUT}/{plot_name}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved -> {plot_name}")

# ─── Save CSVs ────────────────────────────────────────────────────────────
for outcome in ["y_morn_F", "y_mid_S", "y_F_to_S"]:
    for skey in ["poi", "demo", "full"]:
        pd.DataFrame(all_results[f"{outcome}_{skey}"]["rows"]).to_csv(
            f"{OUT}/preS8_logit_{outcome}_{skey}.csv", index=False)
print(f"\n  All CSVs saved")

# ─── Key insight: extract significant demographics for user ───────────────
print(f"\n{'='*90}")
print("  SIGNIFICANT DEMOGRAPHIC PREDICTORS")
print(f"{'='*90}")
for outcome, label in OUTCOME_LABELS.items():
    print(f"\n  {label}:")
    for skey, sname in [("demo", "Demographics only"), ("full", "POI + Demo")]:
        rows_df = pd.DataFrame(all_results[f"{outcome}_{skey}"]["rows"])
        sig_demo = rows_df[rows_df["variable"].isin(DEMO_VARS) & (rows_df["p"] < 0.1)]
        if len(sig_demo):
            print(f"    {sname}:")
            for _, row in sig_demo.iterrows():
                print(f"      {row['variable']:35s} OR={row['OR']:.3f}  p={row['p']:.4f}  "
                      f"95% CI=[{row['ci_95_low']:.3f}, {row['ci_95_high']:.3f}]")
        else:
            print(f"    {sname}: (none significant at p<0.1)")

print("\nDone.")
