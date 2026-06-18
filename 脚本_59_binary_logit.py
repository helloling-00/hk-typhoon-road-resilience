"""
Binary logistic regressions for pre-S8 road-state analysis.
Three models:
  1. Morning clearance: F vs not-F at 08:30
  2. Midday congestion: S vs not-S at 13:00
  3. Reversal: F→S vs F→F (transition conditional on morning-F)

All predictors standardized (mean=0, std=1). Outputs for each model:
  McFadden R², AUC, N, class balance, coefficients/OR/p/95%CI table.
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
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

# ─── Load ──────────────────────────────────────────────────────────────────
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

# ─── Build analysis dataset ────────────────────────────────────────────────
s17 = sep23[sep23["slot"]==17][["road_id","dev"]].rename(columns={"dev":"dev_morn"})
s26 = sep23[sep23["slot"]==26][["road_id","dev"]].rename(columns={"dev":"dev_mid"})
both = s17.merge(s26, on="road_id", how="inner")
both["state_morn"] = both["dev_morn"].apply(classify)
both["state_mid"]  = both["dev_mid"].apply(classify)
both["trans"] = both["state_morn"] + "→" + both["state_mid"]

merged = both.merge(rt_sub, on="road_id", how="inner")

# Log-transform incident counts (reduces skew)
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

# ─── Define predictor variables ────────────────────────────────────────────
X_vars = [
    "intersection_degree", "dist_to_coast_m",
    "work_density", "education_density", "retail_density", "food_drink_density",
    "recreation_density", "medical_density", "transport_density",
    "tourism_density", "finance_density", "civic_density",
    "log_incident_count_500m", "log_severe_incident_500m",
]

# ─── Helper function ───────────────────────────────────────────────────────
def run_binary_logit(df, y_col, title, ref_label="0"):
    """Fit binary logit, print full diagnostics, return results."""
    # Build X with dummies
    X_broad = pd.get_dummies(df["road_broad"], prefix="road", drop_first=True)
    X = pd.concat([df[X_vars], X_broad], axis=1).astype(float)
    y = df[y_col].values.astype(int)

    # Drop NA
    valid = X.notna().all(axis=1)
    X = X[valid]
    y = y[valid.values]

    # Standardize
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Fit
    X_sm = sm.add_constant(X_scaled)
    model = sm.Logit(y, X_sm.astype(float))
    result = model.fit(disp=False, maxiter=500)

    # McFadden R²
    ll_null = result.llnull
    ll_model = result.llf
    mcfadden_r2 = 1 - ll_model / ll_null

    # AUC
    y_pred_prob = result.predict(X_sm)
    auc = roc_auc_score(y, y_pred_prob)

    # Class balance
    n1 = y.sum()
    n0 = len(y) - n1
    pct1 = n1 / len(y) * 100

    # ─── Print output ──────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"  N = {len(y):,}")
    print(f"  Class balance: y=1 → {n1:,} ({pct1:.1f}%)   y=0 → {n0:,} ({100-pct1:.1f}%)")
    print(f"  All predictors standardized (mean=0, std=1)")
    print(f"  McFadden pseudo R² = {mcfadden_r2:.4f}")
    print(f"  AUC = {auc:.4f}")
    print()

    # Coefficient table
    print(f"  {'Variable':30s} {'Coef':>8s} {'SE':>8s} {'OR':>8s} {'p':>8s} {'95% CI low':>11s} {'95% CI high':>11s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*11} {'-'*11}")

    rows_for_csv = []
    for var in result.params.index:
        coef = result.params[var]
        se = result.bse[var]
        pval = result.pvalues[var]
        or_val = np.exp(coef)
        ci_low = np.exp(coef - 1.96 * se)
        ci_high = np.exp(coef + 1.96 * se)
        sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))

        print(f"  {var:30s} {coef:>8.4f} {se:>8.4f} {or_val:>8.4f} {pval:>8.4f} {ci_low:>11.4f} {ci_high:>11.4f}  {sig}")
        rows_for_csv.append({"variable": var, "coef": coef, "se": se, "OR": or_val,
                             "p": pval, "ci_95_low": ci_low, "ci_95_high": ci_high})

    return result, mcfadden_r2, auc, len(y), pct1, rows_for_csv, y_pred_prob, y, X_scaled

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 1: Morning clearance — F vs not-F at 08:30
# ═══════════════════════════════════════════════════════════════════════════
merged["y_morn_F"] = (merged["state_morn"] == "F").astype(int)
r1, r2_1, auc1, n1, bal1, csv1, prob1, y1, X1s = run_binary_logit(
    merged, "y_morn_F",
    "MODEL 1: Morning clearance  —  F vs not-F at 08:30  (F=1, N/S=0)"
)

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 2: Midday congestion — S vs not-S at 13:00
# ═══════════════════════════════════════════════════════════════════════════
merged["y_mid_S"] = (merged["state_mid"] == "S").astype(int)
r2, r2_2, auc2, n2, bal2, csv2, prob2, y2, X2s = run_binary_logit(
    merged, "y_mid_S",
    "MODEL 2: Midday congestion  —  S vs not-S at 13:00  (S=1, F/N=0)"
)

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 3: Reversal — F→S vs F→F (only roads that were F at 08:30)
# ═══════════════════════════════════════════════════════════════════════════
morn_F = merged[merged["state_morn"] == "F"].copy()
morn_F["y_F_to_S"] = (morn_F["trans"] == "F→S").astype(int)
r3, r2_3, auc3, n3, bal3, csv3, prob3, y3, X3s = run_binary_logit(
    morn_F, "y_F_to_S",
    "MODEL 3: Reversal  —  F→S vs F→F  (conditional on being Faster at 08:30)"
)

# ─── Coefficient plot: 3 models side by side ──────────────────────────────
print("\n\nPlotting coefficient comparison...", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(22, 9), sharey=True)
model_names = [
    "Model 1: F vs not-F at 08:30\n(Morning clearance)",
    "Model 2: S vs not-S at 13:00\n(Midday congestion)",
    "Model 3: F→S vs F→F\n(Reversal, conditional on morning-F)"
]
results_list = [r1, r2, r3]

for ax, result, mname in zip(axes, results_list, model_names):
    # Drop const for plotting
    params = result.params.drop("const", errors="ignore")
    bse = result.bse.drop("const", errors="ignore")

    vals = params.values
    ses = bse.values
    cis = 1.96 * ses
    labels = params.index.tolist()

    colors = ["#d62728" if v > 0 else "#2ca02c" for v in vals]
    y_pos = range(len(labels))
    ax.barh(y_pos, vals, xerr=cis, color=colors, alpha=0.82, height=0.6)
    ax.axvline(0, color="black", lw=0.7)
    if ax == axes[0]:
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8.5)
    else:
        ax.set_yticks([])
    ax.set_title(mname, fontsize=11, fontweight="bold")
    ax.set_xlabel("Coefficient (std. predictors)", fontsize=9)

fig.suptitle("Binary Logistic Regressions: Predictors of Road State under Pre-S8 (Ragasa Sep 23)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
out_plot = f"{OUT}/图58c_binary_logit_coefs.png"
fig.savefig(out_plot, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  saved -> {out_plot}")

# ─── AUC curves ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (prob, y_true, mname, auc_val, n_val, bal_val) in zip(
    axes,
    [(prob1, y1, model_names[0], auc1, n1, bal1),
     (prob2, y2, model_names[1], auc2, n2, bal2),
     (prob3, y3, model_names[2], auc3, n3, bal3)]
):
    fpr, tpr, _ = roc_curve(y_true, prob)
    ax.plot(fpr, tpr, lw=2, color="#d62728")
    ax.plot([0, 1], [0, 1], lw=1, ls="--", color="grey")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#d62728")
    ax.set_xlabel("False Positive Rate", fontsize=9)
    if ax == axes[0]:
        ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_title(f"AUC = {auc_val:.3f}   N = {n_val:,}   y=1: {bal_val:.1f}%",
                 fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

fig.suptitle("ROC Curves: Pre-S8 Road-State Logistic Models  (Ragasa Sep 23)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out_roc = f"{OUT}/图58d_roc_curves.png"
fig.savefig(out_roc, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  saved -> {out_roc}")

# ─── Save CSVs ─────────────────────────────────────────────────────────────
for fname, rows in [("preS8_logit_morn_F.csv", csv1),
                     ("preS8_logit_mid_S.csv", csv2),
                     ("preS8_logit_F_to_S.csv", csv3)]:
    pd.DataFrame(rows).to_csv(f"{OUT}/{fname}", index=False)
    print(f"  saved -> {fname}")

print("\nDone.")
