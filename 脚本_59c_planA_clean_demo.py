"""
Plan A: collinearity-cleaned demographics for the three logit models.

Replaces the bloated DEMO block (8 highly correlated ratios) with 4–5 orthogonal
proxies. Reports VIF before fitting so the redesign is auditable.
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
    "population_total_500m", "population_density_500m", "median_income_500m",
    "working_pop_ratio_500m",
    "ratio_age_0_14_500m", "ratio_age_65plus_500m",
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
merged["log_median_income_500m"] = np.log(merged["median_income_500m"].clip(lower=1))

cat_map = {
    "motorway":"highway","motorway_link":"highway",
    "trunk":"highway","trunk_link":"highway",
    "primary":"arterial","primary_link":"arterial",
    "secondary":"arterial","secondary_link":"arterial",
    "tertiary":"local","tertiary_link":"local",
    "street":"local","service":"local",
}
merged["road_broad"] = merged["road_category"].map(cat_map)

STRUCT_VARS   = ["intersection_degree","dist_to_coast_m"]
POI_VARS      = ["work_density","education_density","retail_density","food_drink_density",
                 "recreation_density","medical_density","transport_density",
                 "tourism_density","finance_density","civic_density"]
INCIDENT_VARS = ["log_incident_count_500m"]

# ── Plan A: 5 demographic variables, no职业 ratios, age 用 0-14 + 65+ 两端 ─
DEMO_VARS_A = [
    "log_population_density_500m",
    "log_median_income_500m",
    "working_pop_ratio_500m",
    "ratio_age_0_14_500m",
    "ratio_age_65plus_500m",
]

# ── VIF check ──────────────────────────────────────────────────────────────
def report_vif(df, cols, label):
    sub = df[cols].dropna().copy()
    X = sm.add_constant(sub.astype(float))
    print(f"\nVIF — {label}  (n={len(sub):,})")
    for i, c in enumerate(X.columns):
        if c == "const": continue
        v = variance_inflation_factor(X.values, i)
        flag = "  ⚠" if v >= 5 else ""
        print(f"  {c:35s}  VIF = {v:7.2f}{flag}")

report_vif(merged, STRUCT_VARS + DEMO_VARS_A + INCIDENT_VARS, "Plan A demo + struct + incident")
report_vif(merged, STRUCT_VARS + POI_VARS + DEMO_VARS_A + INCIDENT_VARS, "Plan A full (POI + demo)")

OUTCOME_LABELS = {
    "y_morn_F": "Morning clearance (F vs not-F at 08:30)",
    "y_mid_S":  "Midday congestion (S vs not-S at 13:00)",
    "y_F_to_S": "Reversal (F→S vs F→F | morning-F)",
}

merged["y_morn_F"] = (merged["state_morn"] == "F").astype(int)
merged["y_mid_S"]  = (merged["state_mid"]  == "S").astype(int)
morn_F = merged[merged["state_morn"] == "F"].copy()
morn_F["y_F_to_S"] = (morn_F["trans"] == "F→S").astype(int)

def run_spec(df, y_col, var_blocks, spec_label):
    X_parts = []
    for block_name, block_vars in var_blocks:
        if block_name == "road_broad":
            X_parts.append(pd.get_dummies(df["road_broad"], prefix="road",
                                          drop_first=True).astype(float))
        else:
            X_parts.append(df[block_vars].astype(float))
    X = pd.concat(X_parts, axis=1)
    y = df[y_col].values.astype(int)
    valid = X.notna().all(axis=1)
    X = X[valid]; y = y[valid.values]
    X = X.loc[:, X.nunique() > 1]
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X),
                            columns=X.columns, index=X.index)
    X_sm = sm.add_constant(X_scaled)
    res = sm.Logit(y, X_sm.astype(float)).fit(disp=False, maxiter=500)
    mcfadden = 1 - res.llf / res.llnull
    auc = roc_auc_score(y, res.predict(X_sm))

    print(f"\n{'─'*92}")
    print(f"  {OUTCOME_LABELS[y_col]}   |   {spec_label}")
    print(f"{'─'*92}")
    print(f"  N={len(y):,}  y=1: {y.sum():,} ({y.mean()*100:.1f}%)  "
          f"McFadden R²={mcfadden:.4f}  AUC={auc:.4f}")
    print(f"  {'Variable':36s} {'Coef':>8s} {'SE':>8s} {'OR':>8s} {'p':>8s} {'sig':>5s}")
    rows = []
    for v in res.params.index:
        coef = res.params[v]; se = res.bse[v]; pval = res.pvalues[v]
        sig = "***" if pval<0.001 else ("**" if pval<0.01 else ("*" if pval<0.05 else ("." if pval<0.1 else "")))
        print(f"  {v:36s} {coef:>8.4f} {se:>8.4f} {np.exp(coef):>8.4f} {pval:>8.4f}  {sig:>4s}")
        rows.append({"variable":v,"coef":coef,"se":se,"OR":np.exp(coef),
                     "p":pval,"ci_95_low":np.exp(coef-1.96*se),
                     "ci_95_high":np.exp(coef+1.96*se),"sig":sig})
    return {"rows":rows,"pseudo_r2":mcfadden,"auc":auc,"n":len(y),
            "bal":y.mean()*100}

all_results = {}
for outcome in ["y_morn_F","y_mid_S","y_F_to_S"]:
    df = morn_F if outcome=="y_F_to_S" else merged
    all_results[f"{outcome}_demoA"] = run_spec(
        df, outcome,
        [("struct",STRUCT_VARS),("demo",DEMO_VARS_A),
         ("incident",INCIDENT_VARS),("road_broad",None)],
        "Plan A: Demographics only (5 vars)")
    all_results[f"{outcome}_fullA"] = run_spec(
        df, outcome,
        [("struct",STRUCT_VARS),("poi",POI_VARS),("demo",DEMO_VARS_A),
         ("incident",INCIDENT_VARS),("road_broad",None)],
        "Plan A: POI + Demographics (full)")

# ── Comparison vs original full spec ──────────────────────────────────────
print(f"\n\n{'='*92}")
print("  PLAN A SUMMARY")
print(f"{'='*92}")
print(f"  {'Outcome':40s} {'Spec':35s} {'N':>6s} {'PR²':>7s} {'AUC':>7s}")
for outcome,label in OUTCOME_LABELS.items():
    for skey,sname in [("demoA","demo only"),("fullA","POI + demo")]:
        r = all_results[f"{outcome}_{skey}"]
        print(f"  {label:40s} {sname:35s} {r['n']:>6d} {r['pseudo_r2']:>7.4f} {r['auc']:>7.4f}")

for outcome in ["y_morn_F","y_mid_S","y_F_to_S"]:
    for skey in ["demoA","fullA"]:
        pd.DataFrame(all_results[f"{outcome}_{skey}"]["rows"]).to_csv(
            f"{OUT}/preS8_logit_{outcome}_{skey}.csv", index=False)
print(f"\nCSVs saved -> preS8_logit_*_(demoA|fullA).csv")
