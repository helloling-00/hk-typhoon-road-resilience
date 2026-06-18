"""
Add school / higher_ed / elderly_facility counts and 1km demographics to the
three logit models. Keep the original (preferred) demo block intact.

Hypothesis being tested:
  - Morning F (8:30): roads near schools should be more likely to clear because
    students don't go to school during typhoon.
  - Midday S (13:00): roads near workplaces (workforce ratio) should be more
    likely to congest because office workers get released midday.
  - Elderly facilities: may show distinct pattern (less travel, slower clearance).
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

DATA = "/Users/helloling/workspace/thesis/data"
OUT  = "/Users/helloling/workspace/thesis"
DEV_HI, DEV_LO = 0.03, -0.03

def classify(d):
    if d > DEV_HI: return "F"
    if d < DEV_LO: return "S"
    return "N"

# ── Load ──────────────────────────────────────────────────────────────────
print("Loading...", flush=True)
ts = pd.read_parquet(f"{DATA}/yagiasha_road_timeseries.parquet")
ts["ds"] = pd.to_datetime(ts["dt"]).dt.strftime("%Y-%m-%d")
sep23 = ts[ts["ds"]=="2025-09-23"].copy()
rt = pd.read_parquet(f"{DATA}/regression_table.parquet")
se = pd.read_parquet(f"{DATA}/road_school_elderly_features.parquet")

base_cols = [
    "road_id","road_category","road_length_m",
    "intersection_degree","dist_to_coast_m",
    "work_density","education_density","retail_density","food_drink_density",
    "recreation_density","medical_density","transport_density",
    "tourism_density","finance_density","civic_density",
    "incident_count_500m","severe_incident_500m",
    "population_total_500m","population_density_500m","median_income_500m",
    "working_pop_ratio_500m","ratio_学生_500m","ratio_雇员_500m",
    "ratio_退休人士_500m","ratio_age_0_14_500m","ratio_age_25_44_500m",
    "ratio_age_65plus_500m",
]
rt_sub = rt[base_cols].drop_duplicates("road_id")

s17 = sep23[sep23["slot"]==17][["road_id","dev"]].rename(columns={"dev":"dev_morn"})
s26 = sep23[sep23["slot"]==26][["road_id","dev"]].rename(columns={"dev":"dev_mid"})
both = s17.merge(s26, on="road_id", how="inner")
both["state_morn"] = both["dev_morn"].apply(classify)
both["state_mid"]  = both["dev_mid"].apply(classify)
both["trans"] = both["state_morn"] + "→" + both["state_mid"]
merged = both.merge(rt_sub, on="road_id", how="inner").merge(se, on="road_id", how="left")

for c in ["incident_count_500m","severe_incident_500m",
          "population_total_500m","population_density_500m"]:
    merged[f"log_{c}"] = np.log1p(merged[c])

cat_map = {
    "motorway":"highway","motorway_link":"highway","trunk":"highway","trunk_link":"highway",
    "primary":"arterial","primary_link":"arterial","secondary":"arterial","secondary_link":"arterial",
    "tertiary":"local","tertiary_link":"local","street":"local","service":"local",
}
merged["road_broad"] = merged["road_category"].map(cat_map)

# ── Variable blocks ───────────────────────────────────────────────────────
STRUCT      = ["intersection_degree","dist_to_coast_m"]
POI         = ["work_density","education_density","retail_density","food_drink_density",
               "recreation_density","medical_density","transport_density",
               "tourism_density","finance_density","civic_density"]
INCIDENT    = ["log_incident_count_500m"]
DEMO_500    = ["log_population_density_500m","median_income_500m",
               "working_pop_ratio_500m","ratio_学生_500m","ratio_雇员_500m",
               "ratio_退休人士_500m","ratio_age_25_44_500m","ratio_age_65plus_500m"]
SCHOOL_500  = ["log_school_count_500m","log_higher_ed_count_500m",
               "log_elderly_facility_count_500m"]
SCHOOL_1KM  = ["log_school_count_1000m","log_higher_ed_count_1000m",
               "log_elderly_facility_count_1000m"]
DEMO_1KM    = ["log_population_density_1000m","median_income_1000m",
               "working_pop_ratio_1000m","ratio_学生_1000m","ratio_雇员_1000m",
               "ratio_退休人士_1000m","ratio_age_25_44_1000m","ratio_age_65plus_1000m"]

OUTCOME_LABELS = {
    "y_morn_F": "Morning clearance (F vs not-F at 08:30)",
    "y_mid_S":  "Midday congestion (S vs not-S at 13:00)",
    "y_F_to_S": "Reversal (F→S vs F→F | morning-F)",
}
merged["y_morn_F"] = (merged["state_morn"] == "F").astype(int)
merged["y_mid_S"]  = (merged["state_mid"]  == "S").astype(int)
morn_F = merged[merged["state_morn"] == "F"].copy()
morn_F["y_F_to_S"] = (morn_F["trans"] == "F→S").astype(int)

def run_spec(df, y_col, var_blocks, label):
    parts = []
    for name, vars_ in var_blocks:
        if name == "road_broad":
            parts.append(pd.get_dummies(df["road_broad"], prefix="road",
                                        drop_first=True).astype(float))
        else:
            parts.append(df[vars_].astype(float))
    X = pd.concat(parts, axis=1)
    y = df[y_col].values.astype(int)
    valid = X.notna().all(axis=1)
    X = X[valid]; y = y[valid.values]
    X = X.loc[:, X.nunique() > 1]
    X_sc = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    X_sm = sm.add_constant(X_sc)
    res = sm.Logit(y, X_sm.astype(float)).fit(disp=False, maxiter=500)
    pr2 = 1 - res.llf/res.llnull
    auc = roc_auc_score(y, res.predict(X_sm))

    print(f"\n{'─'*92}")
    print(f"  {OUTCOME_LABELS[y_col]}   |   {label}")
    print(f"{'─'*92}")
    print(f"  N={len(y):,}  y=1: {y.sum():,} ({y.mean()*100:.1f}%)  PR²={pr2:.4f}  AUC={auc:.4f}")
    print(f"  {'Variable':38s} {'OR':>7s} {'p':>8s} {'sig':>5s}")
    rows=[]
    for v in res.params.index:
        if v == "const": continue
        coef=res.params[v]; se=res.bse[v]; pv=res.pvalues[v]
        sig="***" if pv<0.001 else ("**" if pv<0.01 else ("*" if pv<0.05 else ("." if pv<0.1 else "")))
        flag = "★" if pv<0.05 and v not in STRUCT+POI+INCIDENT+["road_highway","road_local"] else ""
        print(f"  {v:38s} {np.exp(coef):>7.3f} {pv:>8.4f}  {sig:>4s} {flag}")
        rows.append({"variable":v,"coef":coef,"se":se,"OR":np.exp(coef),
                     "p":pv,"sig":sig,
                     "ci_95_low":np.exp(coef-1.96*se),"ci_95_high":np.exp(coef+1.96*se)})
    return {"rows":rows,"pr2":pr2,"auc":auc,"n":len(y)}

specs = [
    ("BASE = struct+POI+demo_500+incident+road",
     [("struct",STRUCT),("poi",POI),("demo",DEMO_500),
      ("incident",INCIDENT),("road_broad",None)]),
    ("BASE + school_500m + elderly_500m",
     [("struct",STRUCT),("poi",POI),("demo",DEMO_500),
      ("school",SCHOOL_500),("incident",INCIDENT),("road_broad",None)]),
    ("BASE + school_1km + elderly_1km",
     [("struct",STRUCT),("poi",POI),("demo",DEMO_500),
      ("school",SCHOOL_1KM),("incident",INCIDENT),("road_broad",None)]),
    ("ALT 1km: struct+POI+demo_1km+school_1km+incident+road",
     [("struct",STRUCT),("poi",POI),("demo",DEMO_1KM),
      ("school",SCHOOL_1KM),("incident",INCIDENT),("road_broad",None)]),
]

all_res = {}
for outcome in ["y_morn_F","y_mid_S","y_F_to_S"]:
    df = morn_F if outcome=="y_F_to_S" else merged
    for label, blocks in specs:
        key = f"{outcome}_{label[:20]}"
        all_res[key] = run_spec(df, outcome, blocks, label)

print(f"\n\n{'='*92}\n  SCHOOL / ELDERLY SUMMARY\n{'='*92}")
print(f"  {'Outcome':40s} {'Spec':40s} {'N':>5s} {'PR²':>7s} {'AUC':>6s}")
for outcome,label in OUTCOME_LABELS.items():
    for spec_label, _ in specs:
        key = f"{outcome}_{spec_label[:20]}"
        r = all_res[key]
        print(f"  {label:40s} {spec_label[:40]:40s} {r['n']:>5d} {r['pr2']:>7.4f} {r['auc']:>6.4f}")

# Save the most useful CSVs
for outcome in ["y_morn_F","y_mid_S","y_F_to_S"]:
    for spec_label, _ in specs:
        key = f"{outcome}_{spec_label[:20]}"
        suffix = ("base" if "BASE =" in spec_label else
                  "school500" if "BASE + school_500m" in spec_label else
                  "school1km" if "BASE + school_1km" in spec_label else
                  "alt1km")
        pd.DataFrame(all_res[key]["rows"]).to_csv(
            f"{OUT}/preS8_logit_{outcome}_{suffix}.csv", index=False)
print("\nCSVs saved -> preS8_logit_*_(base|school500|school1km|alt1km).csv")
