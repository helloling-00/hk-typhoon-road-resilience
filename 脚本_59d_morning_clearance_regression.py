"""
Morning clearance (08:30) — where did CONGESTED roads become FAST on Sep 23?
Focus: roads that are normally Slow on workday mornings but cleared on Sep 23.

Outcome: improvement = sep23_dev - ctrl_mean_dev (continuous, positive = faster)
Key predictors: school density, office density, employee ratio, student ratio
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

DATA = "/Users/helloling/workspace/thesis/data"
OUT  = "/Users/helloling/workspace/thesis"
DEV_HI, DEV_LO = 0.03, -0.03

CTRL_DATES = ["2025-09-16", "2025-09-26", "2025-09-29", "2025-09-30",
              "2025-10-02", "2025-10-06", "2025-10-08", "2025-10-09"]

# ─── Load ─────────────────────────────────────────────────────────────────
print("Loading...", flush=True)
ts = pd.read_parquet(f"{DATA}/yagiasha_road_timeseries.parquet")
ts["ds"] = pd.to_datetime(ts["dt"]).dt.strftime("%Y-%m-%d")

# Per-road: mean deviation on control workdays at slot 17
ctrl = ts[(ts["ds"].isin(CTRL_DATES)) & (ts["slot"] == 17)].copy()
ctrl_agg = ctrl.groupby("road_id")["dev"].agg(["mean", "std", "count"]).reset_index()
ctrl_agg.columns = ["road_id", "ctrl_dev_mean", "ctrl_dev_std", "ctrl_n"]
# Also compute F/S/N rates
ctrl["state"] = ctrl["dev"].apply(lambda d: "F" if d > DEV_HI else ("S" if d < DEV_LO else "N"))
ctrl_state = ctrl.groupby("road_id")["state"].value_counts(normalize=True).unstack(fill_value=0)
ctrl_state.columns = ["ctrl_F_rate", "ctrl_N_rate", "ctrl_S_rate"]
ctrl_agg = ctrl_agg.merge(ctrl_state.reset_index(), on="road_id", how="left")

# Sep 23 slot 17
sep23 = ts[(ts["ds"] == "2025-09-23") & (ts["slot"] == 17)].copy()
sep23["state"] = sep23["dev"].apply(lambda d: "F" if d > DEV_HI else ("S" if d < DEV_LO else "N"))

# ─── Merge features ──────────────────────────────────────────────────────
rt = pd.read_parquet(f"{DATA}/regression_table.parquet")
feat_cols = [
    "road_id", "road_category",
    "intersection_degree", "dist_to_coast_m",
    "work_density", "education_density", "retail_density", "food_drink_density",
    "recreation_density", "medical_density", "transport_density",
    "tourism_density", "finance_density", "civic_density",
    "incident_count_500m",
    "population_total_500m", "population_density_500m",
    "median_income_500m",
    "working_pop_ratio_500m",
    "ratio_学生_500m", "ratio_雇员_500m",
    "ratio_age_0_14_500m", "ratio_age_25_44_500m",
    "ratio_age_65plus_500m", "ratio_退休人士_500m",
]
rt_sub = rt[feat_cols].drop_duplicates("road_id")

df = sep23[["road_id", "dev", "state"]].rename(columns={"dev": "sep23_dev", "state": "sep23_state"})
df = df.merge(ctrl_agg, on="road_id", how="inner")
df = df.merge(rt_sub, on="road_id", how="inner")

# Improvement: how much faster on Sep 23 vs normal
df["improvement"] = df["sep23_dev"] - df["ctrl_dev_mean"]

# Drop rows missing demographics
df = df.dropna(subset=["ratio_雇员_500m", "ratio_学生_500m",
                        "population_density_500m", "median_income_500m"])

for c in ["incident_count_500m", "population_density_500m"]:
    df[f"log_{c}"] = np.log1p(df[c])

cat_map = {
    "motorway": "highway", "motorway_link": "highway",
    "trunk": "highway", "trunk_link": "highway",
    "primary": "arterial", "primary_link": "arterial",
    "secondary": "arterial", "secondary_link": "arterial",
    "tertiary": "local", "tertiary_link": "local",
    "street": "local", "service": "local",
}
df["road_broad"] = df["road_category"].map(cat_map)

print(f"  Analysis sample: {len(df)} roads")
print(f"  Mean ctrl deviation: {df['ctrl_dev_mean'].mean():.4f}")
print(f"  Mean Sep23 deviation: {df['sep23_dev'].mean():.4f}")
print(f"  Mean improvement:     {df['improvement'].mean():.4f}")
print()

# ─── Descriptive: improvement by school & office quartile ────────────────
for var, label in [
    ("education_density", "School density"),
    ("work_density", "Office density"),
    ("ratio_学生_500m", "Student ratio"),
    ("ratio_雇员_500m", "Employee ratio"),
]:
    df["q"] = pd.qcut(df[var], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    grp = df.groupby("q", observed=False).agg(
        n=("improvement", "count"),
        ctrl_dev=("ctrl_dev_mean", "mean"),
        sep23_dev=("sep23_dev", "mean"),
        improvement=("improvement", "mean"),
        ctrl_S_rate=("ctrl_S_rate", "mean"),
        sep23_F_rate=("sep23_state", lambda x: (x == "F").mean()),
    )
    print(f"=== {label} ===")
    print(grp.round(4).to_string())
    print()

# ─── Key insight: normally Slow roads — how did they do? ────────────────
df["normally_S"] = (df["ctrl_S_rate"] >= 0.5).astype(int)
df["became_F"] = ((df["normally_S"] == 1) & (df["sep23_state"] == "F")).astype(int)

ns = df[df["normally_S"] == 1]
print(f"  Roads normally S (ctrl S rate >= 0.5): {len(ns)}")
print(f"    of which became F on Sep 23: {ns['became_F'].sum()} ({ns['became_F'].mean():.1%})")
print(f"    mean improvement: {ns['improvement'].mean():.4f}")

for var, label in [
    ("education_density", "School density"),
    ("ratio_学生_500m", "Student ratio"),
    ("work_density", "Office density"),
    ("ratio_雇员_500m", "Employee ratio"),
]:
    ns["q"] = pd.qcut(ns[var], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    grp = ns.groupby("q", observed=False).agg(
        n=("improvement", "count"),
        became_F=("became_F", "mean"),
        improvement=("improvement", "mean"),
    )
    print(f"\n  === Normally-S roads: improvement by {label} ===")
    print(grp.round(4).to_string())

# ─── Regression ──────────────────────────────────────────────────────────
STRUCT = ["intersection_degree", "dist_to_coast_m"]
INCIDENT = ["log_incident_count_500m"]

def run_ols(df, y_col, X_var_blocks, label):
    X_parts = []
    for block_name, block_vars in X_var_blocks:
        if block_name == "road_broad":
            X_parts.append(
                pd.get_dummies(df["road_broad"], prefix="road", drop_first=True).astype(float)
            )
        else:
            X_parts.append(df[block_vars].astype(float))
    X = pd.concat(X_parts, axis=1)
    y = df[y_col].values.astype(float)

    valid = X.notna().all(axis=1) & (~np.isnan(y))
    X = X[valid]; y = y[valid.values]
    X = X.loc[:, X.nunique() > 1]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    X_sm = sm.add_constant(X_scaled)

    model = sm.OLS(y, X_sm.astype(float))
    result = model.fit()

    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    print(f"  N = {len(y):,}  |  mean(y) = {y.mean():.4f}  |  R² = {result.rsquared:.4f}  |  adj R² = {result.rsquared_adj:.4f}")
    print()
    print(f"  {'Variable':32s} {'Coef':>8s} {'SE':>8s} {'t':>8s} {'p':>8s}")
    print(f"  {'─'*32} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    rows = []
    for var in result.params.index:
        coef = result.params[var]
        se = result.bse[var]
        t = result.tvalues[var]
        pval = result.pvalues[var]
        sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        print(f"  {var:32s} {coef:>8.4f} {se:>8.4f} {t:>8.4f} {pval:>8.4f}  {sig}")
        rows.append({"variable": var, "coef": coef, "se": se, "t": t, "p": pval})

    return {"result": result, "r2": result.rsquared, "adj_r2": result.rsquared_adj,
            "n": len(y), "rows": rows}

# Model 1: improvement on ALL roads
print("\n" + "═"*80)
print("  MODEL 1: improvement = sep23_dev - ctrl_dev (ALL roads)")
print("═"*80)

r1 = run_ols(df, "improvement", [
    ("struct", STRUCT), ("incident", INCIDENT), ("road_broad", None),
    ("key", ["education_density", "work_density", "ratio_学生_500m", "ratio_雇员_500m",
             "log_population_density_500m", "median_income_500m"]),
], "OLS: improvement ~ school + office + student + employee")

# Model 2: improvement on normally-S roads only
print("\n" + "═"*80)
print("  MODEL 2: improvement on normally-S roads (ctrl S rate >= 0.5)")
print("═"*80)

r2 = run_ols(ns, "improvement", [
    ("struct", STRUCT), ("incident", INCIDENT), ("road_broad", None),
    ("key", ["education_density", "work_density", "ratio_学生_500m", "ratio_雇员_500m",
             "log_population_density_500m", "median_income_500m"]),
], "OLS: improvement (normally-S roads) ~ school + office + student + employee")

# Model 3: became_F (binary) on normally-S roads
print("\n" + "═"*80)
print("  MODEL 3: became_F (logit) on normally-S roads")
print("═"*80)

def run_logit(df, y_col, X_var_blocks, label):
    X_parts = []
    for block_name, block_vars in X_var_blocks:
        if block_name == "road_broad":
            X_parts.append(
                pd.get_dummies(df["road_broad"], prefix="road", drop_first=True).astype(float)
            )
        else:
            X_parts.append(df[block_vars].astype(float))
    X = pd.concat(X_parts, axis=1)
    y = df[y_col].values.astype(int)

    valid = X.notna().all(axis=1)
    X = X[valid]; y = y[valid.values]
    X = X.loc[:, X.nunique() > 1]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    X_sm = sm.add_constant(X_scaled)

    model = sm.Logit(y, X_sm.astype(float))
    result = model.fit(disp=False, maxiter=500)

    from sklearn.metrics import roc_auc_score
    ll_null, ll_model = result.llnull, result.llf
    mcfadden_r2 = 1 - ll_model / ll_null
    auc = roc_auc_score(y, result.predict(X_sm))
    n1 = y.sum()

    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    print(f"  N = {len(y):,}  |  y=1: {n1:,} ({n1/len(y)*100:.1f}%)  |  McFadden R² = {mcfadden_r2:.4f}  |  AUC = {auc:.4f}")
    print()
    print(f"  {'Variable':32s} {'Coef':>8s} {'SE':>8s} {'OR':>8s} {'p':>8s} {'95% CI low':>10s} {'95% CI high':>10s}")
    print(f"  {'─'*32} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*10}")

    rows = []
    for var in result.params.index:
        coef = result.params[var]
        se = result.bse[var]
        pval = result.pvalues[var]
        or_val = np.exp(coef)
        ci_low = np.exp(coef - 1.96*se)
        ci_high = np.exp(coef + 1.96*se)
        sig = "***" if pval<0.01 else ("**" if pval<0.05 else ("*" if pval<0.1 else ""))
        print(f"  {var:32s} {coef:>8.4f} {se:>8.4f} {or_val:>8.4f} {pval:>8.4f} {ci_low:>10.4f} {ci_high:>10.4f}  {sig}")
        rows.append({"variable": var, "coef": coef, "se": se, "OR": or_val,
                     "p": pval, "ci_95_low": ci_low, "ci_95_high": ci_high})

    return {"result": result, "pseudo_r2": mcfadden_r2, "auc": auc,
            "n": len(y), "bal_pct": n1/len(y)*100, "rows": rows}

r3 = run_logit(ns, "became_F", [
    ("struct", STRUCT), ("incident", INCIDENT), ("road_broad", None),
    ("key", ["education_density", "work_density", "ratio_学生_500m", "ratio_雇员_500m",
             "log_population_density_500m", "median_income_500m"]),
], "Logit: became_F on normally-S roads ~ school + office + student + employee")

# ─── Coefficient plot ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

target_vars = ["education_density", "work_density", "ratio_学生_500m", "ratio_雇员_500m",
               "log_population_density_500m", "median_income_500m"]
var_labels = ["School\ndensity", "Office\ndensity", "Student\nratio", "Employee\nratio",
              "Pop\ndensity", "Median\nincome"]

for ax, (r, title, is_logit) in zip(axes, [
    (r1, "All roads\n(improvement)", False),
    (r2, "Normally-S roads\n(improvement)", False),
    (r3, "Normally-S roads\n(became F, logit)", True),
]):
    res = r["result"]
    coefs, err_low, err_high = [], [], []
    for var in target_vars:
        if var in res.params.index:
            c = res.params[var]
            se = res.bse[var]
            if is_logit:
                coefs.append(np.exp(c))  # OR
                err_low.append(np.exp(c) - np.exp(c - 1.96*se))
                err_high.append(np.exp(c + 1.96*se) - np.exp(c))
            else:
                coefs.append(c)
                err_low.append(1.96*se)
                err_high.append(1.96*se)
        else:
            coefs.append(np.nan); err_low.append(np.nan); err_high.append(np.nan)

    colors = ["#d62728" if c > (1 if is_logit else 0) else "#2ca02c" for c in coefs]
    y_pos = range(len(var_labels))
    ax.barh(y_pos, coefs, xerr=[err_low, err_high], color=colors, alpha=0.82, height=0.6, capsize=3)
    if is_logit:
        ax.axvline(1, color="black", lw=0.7, ls="--")
        ax.set_xlabel("Odds Ratio", fontsize=9)
    else:
        ax.axvline(0, color="black", lw=0.7, ls="--")
        ax.set_xlabel("Coefficient", fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(var_labels, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")

fig.suptitle("Morning Clearance on Sep 23: Which Roads Improved?\n(Ragasa Sep 23, S3, slot 17 at 08:30)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
out_plot = f"{OUT}/图59d_morning_improvement.png"
fig.savefig(out_plot, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\n  saved -> {out_plot}")

# ─── Save CSVs ───────────────────────────────────────────────────────────
for r, fname in [(r1, "morning_improve_all"), (r2, "morning_improve_normS"),
                 (r3, "morning_becameF_normS")]:
    pd.DataFrame(r["rows"]).to_csv(f"{OUT}/preS8_{fname}.csv", index=False)
print("  All CSVs saved")

print("\nDone.")
