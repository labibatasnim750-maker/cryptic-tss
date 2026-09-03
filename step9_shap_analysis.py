"""
STEP 9 — SHAP Interpretability Analysis

Explains which sequence features drive the model's predictions, using
SHapley Additive exPlanations (Lundberg & Lee, 2017).

This addresses Objective 5. A tool intended to guide experimental redesign
must be able to say *why* a region was flagged, not merely that it was —
otherwise a researcher has no basis for deciding what to recode.

Two levels of explanation are produced:
cReads : best_model.pkl, feature_cols.pkl, features_with_motifs.csv
Writes: five PNG figures + shap_feature_importance.csv
"""

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")          # write files without opening windows
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



# CONFIGURATION


FEATURES_FILE = "features_with_motifs.csv"
MODEL_FILE    = "best_model.pkl"
RANDOM_SEED   = 42

# SHAP computes exact values for tree models, but plotting thousands of points
# is slow and the figures become unreadable. A subsample is used for the
# visualisations; importance rankings are stable well below this size.
SAMPLE_SIZE   = 1000

DPI           = 150

np.random.seed(RANDOM_SEED)



# 1. LOAD MODEL AND DATA


print("=" * 66)
print("STEP 9: SHAP Interpretability Analysis")
print("=" * 66)

model        = joblib.load(MODEL_FILE)
feature_cols = joblib.load("feature_cols.pkl")
model_name   = joblib.load("best_model_name.pkl")

df = pd.read_csv(FEATURES_FILE)
X  = df[feature_cols]
y  = df["label"].values

print(f"Model                    : {model_name}")
print(f"Feature file             : {FEATURES_FILE}")
print(f"Sequences                : {len(df)}")
print(f"Features                 : {len(feature_cols)}")



# 2. UNWRAP THE PIPELINE

# best_model.pkl is a Pipeline: StandardScaler followed by XGBoost. SHAP's
# TreeExplainer must be given the tree model itself, and must see the data in
# the same scaled space the trees were fitted on — so the scaler is applied
# manually here and the classifier extracted.

scaler     = model.named_steps["scaler"]
classifier = model.named_steps["clf"]

X_scaled = pd.DataFrame(
    scaler.transform(X),
    columns = feature_cols,
)

print(f"Pipeline unwrapped       : scaler + {type(classifier).__name__}")



# 3. COMPUTE SHAP VALUES

# TreeExplainer is exact for tree ensembles and fast — this is the reason
# XGBoost was chosen for deployment over the Voting Ensemble in Step 7.
#
# A SHAP value is the contribution of one feature to one prediction, measured
# in log-odds. Positive pushes towards "promoter", negative towards
# "non-promoter". For any single sequence, the SHAP values across all features
# sum exactly to the difference between that prediction and the average
# prediction — which is what makes the decomposition trustworthy.

print("\nComputing SHAP values (exact, TreeExplainer)...")

explainer   = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_scaled)

print(f"✓ SHAP matrix shape      : {shap_values.shape}")



# 4. GLOBAL FEATURE IMPORTANCE

# Importance = mean absolute SHAP value across all sequences. This measures how
# much a feature moves predictions on average, regardless of direction.

mean_abs = np.abs(shap_values).mean(axis=0)


def feature_group(name):
    """Assign each feature to a category so groups can be compared."""
    if name.startswith("motif_"):
        return "Motif (Step 6b)"
    if name.startswith("pwm_score"):
        return "PWM (fixed position)"
    if name.startswith("3mer_"):
        return "3-mer"
    if name.startswith("4mer_"):
        return "4-mer"
    if name == "gc_content":
        return "GC content"
    return "Other"


importance = pd.DataFrame({
    "feature":    feature_cols,
    "importance": mean_abs,
    "group":      [feature_group(f) for f in feature_cols],
}).sort_values("importance", ascending=False).reset_index(drop=True)

importance["rank"]         = importance.index + 1
importance["pct_of_total"] = (100 * importance["importance"]
                              / importance["importance"].sum())

importance.to_csv("shap_feature_importance.csv", index=False)

print("\n" + "=" * 66)
print("TOP 20 MOST INFLUENTIAL FEATURES")
print("=" * 66)
print(f"{'Rank':<6}{'Feature':<28}{'Group':<22}{'Importance':>10}")
print("-" * 66)
for _, r in importance.head(20).iterrows():
    print(f"{r['rank']:<6}{r['feature']:<28}{r['group']:<22}"
          f"{r['importance']:>10.4f}")



# 5. DID THE MOTIF FEATURES EARN THEIR PLACE?

# Step 6b raised held-out PR-AUC from 0.8650 to 0.9466. This asks where that
# gain came from: 18 motif features against 323 of everything else.

print("\n" + "=" * 66)
print("IMPORTANCE BY FEATURE GROUP")
print("=" * 66)

grouped = importance.groupby("group").agg(
    n_features   = ("feature",      "count"),
    total_imp    = ("importance",   "sum"),
    mean_imp     = ("importance",   "mean"),
    pct_of_total = ("pct_of_total", "sum"),
).sort_values("pct_of_total", ascending=False)

print(f"{'Group':<24}{'N':>5}{'% of total':>12}{'Mean per feature':>19}")
print("-" * 66)
for grp, r in grouped.iterrows():
    print(f"{grp:<24}{int(r['n_features']):>5}"
          f"{r['pct_of_total']:>11.1f}%{r['mean_imp']:>19.4f}")

motif_rows = importance[importance["group"] == "Motif (Step 6b)"]
motif_pct  = motif_rows["pct_of_total"].sum()
motif_n    = len(motif_rows)
best_motif = motif_rows.iloc[0] if motif_n else None

print("-" * 66)
print(f"Motif features are {motif_n}/{len(feature_cols)} of the feature set "
      f"({100*motif_n/len(feature_cols):.1f}%)")
print(f"but account for {motif_pct:.1f}% of total feature importance.")
if best_motif is not None:
    print(f"Highest-ranked motif feature: {best_motif['feature']} "
          f"(rank {best_motif['rank']} overall)")

# Features that contribute essentially nothing
dead = (importance["importance"] < 1e-6).sum()
print(f"\nFeatures with ~zero influence: {dead} "
      f"({100*dead/len(feature_cols):.1f}% of the feature set)")



# 6. SUBSAMPLE FOR PLOTTING


n_sample = min(SAMPLE_SIZE, len(X_scaled))
idx      = np.random.choice(len(X_scaled), n_sample, replace=False)

X_plot    = X_scaled.iloc[idx]
shap_plot = shap_values[idx]

print(f"\nPlotting on {n_sample} randomly sampled sequences...")



# 7. FIGURE 1 — BEESWARM SUMMARY

# Each dot is one sequence. Horizontal position is that feature's SHAP value
# (its effect on that prediction); colour is the feature's value (red high,
# blue low). This shows both how much a feature matters and in which direction.

shap.summary_plot(shap_plot, X_plot, max_display=20, show=False)
plt.title("SHAP summary — top 20 features", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig("shap_01_beeswarm.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("✓ shap_01_beeswarm.png")



# 8. FIGURE 2 — BAR CHART OF MEAN IMPORTANCE


shap.summary_plot(shap_plot, X_plot, plot_type="bar",
                  max_display=20, show=False)
plt.title("Mean |SHAP value| — top 20 features", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig("shap_02_importance_bar.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("✓ shap_02_importance_bar.png")



# 9. FIGURE 3 — IMPORTANCE BY GROUP


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

colours = ["#028090" if g == "Motif (Step 6b)" else "#94a3b8"
           for g in grouped.index]

ax1.barh(range(len(grouped)), grouped["pct_of_total"], color=colours)
ax1.set_yticks(range(len(grouped)))
ax1.set_yticklabels(grouped.index)
ax1.invert_yaxis()
ax1.set_xlabel("% of total feature importance")
ax1.set_title("Share of total importance by group")

ax2.barh(range(len(grouped)), grouped["mean_imp"], color=colours)
ax2.set_yticks(range(len(grouped)))
ax2.set_yticklabels(grouped.index)
ax2.invert_yaxis()
ax2.set_xlabel("Mean importance per feature")
ax2.set_title("Importance per individual feature")

fig.suptitle("Motif features (teal) vs all others", fontsize=13)
plt.tight_layout()
plt.savefig("shap_03_group_importance.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("✓ shap_03_group_importance.png")



# 10. FIGURE 4 — DEPENDENCE PLOTS FOR THE TOP MOTIF FEATURES
# These show how a feature's SHAP value changes with its actual value, which
# is where the biology becomes readable: e.g. whether promoter risk really does
# fall away as the spacer deviates from the optimal 17 bp.
# The x-axis uses the ORIGINAL unscaled values so the numbers are interpretable
# (a spacer of 17 bp, not 0.3 standard deviations). Row order is identical,
# since both X and X_scaled are indexed by the same `idx`.

top_motifs = motif_rows.head(4)["feature"].tolist()

if top_motifs:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, feat in zip(axes.ravel(), top_motifs):
        j = feature_cols.index(feat)
        ax.scatter(X.iloc[idx][feat], shap_plot[:, j],
                   s=8, alpha=0.35, c="#028090")
        ax.axhline(0, color="#64748b", lw=0.8, ls="--")
        ax.set_xlabel(f"{feat}  (original units)")
        ax.set_ylabel("SHAP value (log-odds)")
        ax.set_title(feat, fontsize=11)

    # Hide any unused panels if fewer than 4 motif features exist
    for ax in axes.ravel()[len(top_motifs):]:
        ax.set_visible(False)

    fig.suptitle("How motif features drive predictions\n"
                 "(above zero pushes towards 'promoter')",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("shap_04_motif_dependence.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("✓ shap_04_motif_dependence.png")



# 11. FIGURE 5 — LOCAL EXPLANATION FOR ONE SEQUENCE

# This is the form of output the final tool needs. For a specific flagged
# region, it names the features responsible — telling a researcher what to
# change rather than only that something is wrong.
#
# Uses the public shap.waterfall_plot API. The older waterfall_legacy function
# is an internal that is deprecated in current SHAP versions.

print("\nGenerating waterfall plot for a single prediction...")

probs     = model.predict_proba(X)[:, 1]
confident = np.where((y == 1) & (probs > 0.9))[0]
example   = int(confident[0]) if len(confident) else int(np.argmax(probs))

plt.figure(figsize=(10, 8))
shap.waterfall_plot(
    shap.Explanation(
        values        = shap_values[example],
        base_values   = explainer.expected_value,
        data          = X.iloc[example].values,   # unscaled, for readability
        feature_names = feature_cols,
    ),
    max_display = 15,
    show        = False,
)
plt.title(f"Why sequence '{df.iloc[example]['name']}' was predicted a promoter\n"
          f"(model probability {probs[example]:.3f})",
          fontsize=12, pad=15)
plt.tight_layout()
plt.savefig("shap_05_single_prediction.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("✓ shap_05_single_prediction.png")


# 12. THE SAME EXPLANATION AS TEXT

print("\n" + "=" * 66)
print("WORKED EXAMPLE — A SINGLE PREDICTION EXPLAINED")
print("=" * 66)
print(f"Sequence      : {df.iloc[example]['name']}")
print(f"True label    : {'promoter' if y[example] == 1 else 'non-promoter'}")
print(f"Predicted     : {probs[example]:.4f} probability of being a promoter")
print(f"\nBase value (average prediction, log-odds): "
      f"{explainer.expected_value:.4f}")
print("\nLargest contributions:")
print(f"  {'Feature':<28}{'Value':>10}{'SHAP':>10}  Effect")
print("  " + "-" * 60)

contrib = pd.DataFrame({
    "feature": feature_cols,
    "value":   X.iloc[example].values,
    "shap":    shap_values[example],
})
contrib["abs"] = contrib["shap"].abs()

for _, r in contrib.nlargest(10, "abs").iterrows():
    direction = "→ promoter" if r["shap"] > 0 else "→ non-promoter"
    print(f"  {r['feature']:<28}{r['value']:>10.3f}"
          f"{r['shap']:>+10.3f}  {direction}")

print(f"\n  {'Sum of all SHAP values':<28}{'':>10}"
      f"{shap_values[example].sum():>+10.3f}")
print(f"  Base + sum = "
      f"{explainer.expected_value + shap_values[example].sum():.3f} log-odds, "
      f"which is the model's output for this sequence.")



# SUMMARY
print("\n" + "=" * 66)
print("STEP 9 COMPLETE")
print("=" * 66)
print(f"Model explained          : {model_name}")
print(f"Sequences analysed       : {len(df)}")
print(f"Top feature overall      : {importance.iloc[0]['feature']}")
print(f"Motif features           : {motif_n} features, "
      f"{motif_pct:.1f}% of total importance")
print("\nFiles written:")
for f in ["shap_01_beeswarm.png",
          "shap_02_importance_bar.png",
          "shap_03_group_importance.png",
          "shap_04_motif_dependence.png",
          "shap_05_single_prediction.png",
          "shap_feature_importance.csv"]:
    print(f"  {f}")