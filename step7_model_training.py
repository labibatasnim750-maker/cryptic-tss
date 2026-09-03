"""
STEP 7 — Model Training & Comparison

Trains and compares six classifiers on the extracted sequence features,
using 5-fold stratified cross-validation on an 80% training split, then
reports a final honest evaluation on a held-out 20% test set.

Model selection is made among BASE ALGORITHMS only. The Voting Ensemble is
a meta-ensemble built from three of the base models, so it is reported for
comparison but excluded from selection.

To run on a different feature file, change FEATURES_FILE below.
    python step7_model_training.py
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.ensemble        import (RandomForestClassifier,
                                     GradientBoostingClassifier,
                                     VotingClassifier)
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import SVC
from sklearn.metrics         import (average_precision_score, roc_auc_score,
                                     f1_score, matthews_corrcoef,
                                     confusion_matrix, classification_report)
import xgboost as xgb


# CONFIGURATION

FEATURES_FILE = "features_with_motifs.csv"      # change to "features_with_motifs.csv" later
MODEL_FILE    = "best_model.pkl"
RANDOM_SEED   = 42
TEST_SIZE     = 0.20
N_FOLDS       = 5



# 1. LOAD DATA

print("=" * 66)
print("STEP 7: Model Training & Comparison")
print("=" * 66)

df = pd.read_csv(FEATURES_FILE)

meta_cols    = ["name", "strand", "sequence", "label", "source"]
feature_cols = [c for c in df.columns if c not in meta_cols]

X = df[feature_cols].values
y = df["label"].values

print(f"Feature file             : {FEATURES_FILE}")
print(f"Sequences                : {len(df)}")
print(f"Features                 : {len(feature_cols)}")
print(f"Positives / Negatives    : {(y == 1).sum()} / {(y == 0).sum()}")

# With an imbalanced dataset, the PR-AUC a random classifier achieves equals
# the proportion of positives — NOT 0.5. Every score below is judged against this.
baseline_prauc = (y == 1).mean()
print(f"Random baseline PR-AUC   : {baseline_prauc:.4f}")



# 2. HOLD OUT A TEST SET
# This 20% is locked away and never used for cross-validation, model selection,
# or tuning. It is opened once, at the end, to estimate performance on sequences
# the model has genuinely never seen. stratify=y preserves the class ratio.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = TEST_SIZE,
    stratify     = y,
    random_state = RANDOM_SEED,
)

print(f"\nTrain set ({int((1-TEST_SIZE)*100)}%)          : {len(X_train):>5} "
      f"({(y_train == 1).sum()} pos / {(y_train == 0).sum()} neg)")
print(f"Test set  ({int(TEST_SIZE*100)}%, held out) : {len(X_test):>5} "
      f"({(y_test == 1).sum()} pos / {(y_test == 0).sum()} neg)")

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()



# 3. DEFINE MODELS
# Every model is wrapped in a Pipeline with StandardScaler. Putting the scaler
# inside the pipeline means it is refitted within each CV fold, so no statistics
# leak from validation folds into training.

def make_models(spw):
    """Build a fresh, unfitted set of model pipelines."""

    lr = LogisticRegression(
        class_weight = "balanced",
        max_iter     = 2000,
        random_state = RANDOM_SEED,
    )

    svm = SVC(
        kernel       = "rbf",
        class_weight = "balanced",
        random_state = RANDOM_SEED,
    )

    rf = RandomForestClassifier(
        n_estimators = 400,
        class_weight = "balanced",
        random_state = RANDOM_SEED,
        n_jobs       = -1,
    )

    gb = GradientBoostingClassifier(
        n_estimators  = 300,
        max_depth     = 3,
        learning_rate = 0.05,
        random_state  = RANDOM_SEED,
    )

    xg = xgb.XGBClassifier(
        n_estimators     = 400,
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = spw,
        eval_metric      = "logloss",
        random_state     = RANDOM_SEED,
        verbosity        = 0,
        n_jobs           = -1,
    )

    # Meta-ensemble: averages the predicted probabilities of three base models.
    vote = VotingClassifier(
        estimators = [("lr", lr), ("rf", rf), ("xgb", xg)],
        voting     = "soft",
        n_jobs     = -1,
    )

    wrap = lambda clf: Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    return {
        "Logistic Regression": wrap(lr),
        "SVM (RBF)":           wrap(svm),
        "Random Forest":       wrap(rf),
        "Gradient Boosting":   wrap(gb),
        "XGBoost":             wrap(xg),
        "Voting Ensemble":     wrap(vote),
    }


# The Voting Ensemble is a meta-ensemble over three of the base models, not a
# base algorithm in its own right. Comparing it head-to-head with base models
# is not like-for-like, so it is reported but excluded from selection.
MODEL_TYPE = {
    "Logistic Regression": "base",
    "SVM (RBF)":           "base",
    "Random Forest":       "base",
    "Gradient Boosting":   "base",
    "XGBoost":             "base",
    "Voting Ensemble":     "meta",
}

models = make_models(scale_pos_weight)


# 4. CROSS-VALIDATION ON THE TRAINING SET
# 5-fold stratified CV gives five scores per metric rather than a single number,
# so performance can be reported as mean ± standard deviation. Every sequence is
# used for validation exactly once.
#
# Scorer strings are used rather than make_scorer(..., needs_proba=True) because
# that argument was removed in scikit-learn 1.7+.

scoring = {
    "pr_auc":  "average_precision",   # PRIMARY — robust under class imbalance
    "roc_auc": "roc_auc",
    "f1":      "f1",
    "mcc":     "matthews_corrcoef",
}

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

print("\n" + "=" * 66)
print(f"{N_FOLDS}-FOLD CROSS-VALIDATION (on the training set)")
print("=" * 66)
print("SVM is the slowest — allow a few minutes.\n")

results = {}

for name, pipe in models.items():
    print(f"  Training: {name} ...", flush=True)
    out = cross_validate(pipe, X_train, y_train,
                         cv=cv, scoring=scoring, n_jobs=-1)
    results[name] = out
    print(f"    PR-AUC {out['test_pr_auc'].mean():.4f}  |  "
          f"ROC-AUC {out['test_roc_auc'].mean():.4f}  |  "
          f"F1 {out['test_f1'].mean():.4f}  |  "
          f"MCC {out['test_mcc'].mean():.4f}")



# 5. COMPARISON TABLE

print("\n" + "=" * 66)
print(f"MODEL COMPARISON  (mean ± sd across {N_FOLDS} folds)")
print("=" * 66)

hdr = (f"{'Model':<21}{'Type':>6}{'PR-AUC':>16}{'ROC-AUC':>16}"
       f"{'F1':>16}{'MCC':>16}")
print(hdr)
print("-" * len(hdr))

for name in models:
    r = results[name]
    print(f"{name:<21}{MODEL_TYPE[name]:>6}"
          f"{r['test_pr_auc'].mean():>8.4f}±{r['test_pr_auc'].std():.3f}"
          f"{r['test_roc_auc'].mean():>8.4f}±{r['test_roc_auc'].std():.3f}"
          f"{r['test_f1'].mean():>8.4f}±{r['test_f1'].std():.3f}"
          f"{r['test_mcc'].mean():>8.4f}±{r['test_mcc'].std():.3f}")

print("-" * len(hdr))
print(f"{'Random baseline':<21}{'—':>6}{baseline_prauc:>16.4f}"
      f"{0.5:>16.4f}{'—':>16}{0.0:>16.4f}")



# 6. MODEL SELECTION

BASE_MODELS  = [m for m in results if MODEL_TYPE[m] == "base"]

DEPLOY_MODEL = max(BASE_MODELS,
                   key=lambda m: results[m]["test_pr_auc"].mean())
deploy_cv    = results[DEPLOY_MODEL]["test_pr_auc"].mean()

top_overall  = max(results, key=lambda m: results[m]["test_pr_auc"].mean())
top_cv       = results[top_overall]["test_pr_auc"].mean()

print("\n" + "=" * 66)
print("MODEL SELECTION")
print("=" * 66)
print(f"Selected (base algorithms) : {DEPLOY_MODEL}")
print(f"Cross-validated PR-AUC     : {deploy_cv:.4f}")
print(f"Improvement over baseline  : {deploy_cv - baseline_prauc:+.4f}")

if top_overall != DEPLOY_MODEL:
    gap = top_cv - deploy_cv
    sd  = results[top_overall]["test_pr_auc"].std()
    print(f"\nHighest overall            : {top_overall} "
          f"({top_cv:.4f}, meta-ensemble)")
    print(f"Gap to selected model      : {gap:+.4f}  "
          f"({gap / sd:.2f} SD — not meaningful)")

# How much does model choice actually matter here?
spread = (max(results[m]["test_pr_auc"].mean() for m in BASE_MODELS) -
          min(results[m]["test_pr_auc"].mean() for m in BASE_MODELS))
print(f"\nSpread across base models  : {spread:.4f}")
print("A small spread indicates performance is limited by the features,")
print("not by model capacity.")



# 7. HELD-OUT TEST SET EVALUATION
# The test set has influenced no decision up to this point. These are the
# numbers to quote in the dissertation.

print("\n" + "=" * 66)
print("HELD-OUT TEST SET EVALUATION")
print("=" * 66)

eval_pipe = make_models(scale_pos_weight)[DEPLOY_MODEL]
eval_pipe.fit(X_train, y_train)

y_pred = eval_pipe.predict(X_test)

# Most classifiers expose predict_proba; SVC without probability=True does not,
# so fall back to its decision_function (both are valid ranking scores).
if hasattr(eval_pipe, "predict_proba"):
    y_score = eval_pipe.predict_proba(X_test)[:, 1]
else:
    y_score = eval_pipe.decision_function(X_test)

test_scores = {
    "PR-AUC":  average_precision_score(y_test, y_score),
    "ROC-AUC": roc_auc_score(y_test, y_score),
    "F1":      f1_score(y_test, y_pred),
    "MCC":     matthews_corrcoef(y_test, y_pred),
}

print(f"Model                    : {DEPLOY_MODEL}\n")
for k, v in test_scores.items():
    print(f"  {k:<22} : {v:.4f}")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("\nConfusion matrix:")
print("                        Predicted")
print("                     Neg        Pos")
print(f"  Actual  Neg   {tn:>8}   {fp:>8}")
print(f"  Actual  Pos   {fn:>8}   {tp:>8}")
print(f"\n  True positives  (promoters correctly found) : {tp}")
print(f"  False negatives (promoters missed)          : {fn}")
print(f"  False positives (false alarms)              : {fp}")
print("\n  For this application a false negative is the costlier error: an")
print("  undetected cryptic promoter reaches synthesis and fails in the lab.")

print("\nClassification report:")
print(classification_report(y_test, y_pred,
                            target_names=["Non-promoter", "Promoter"],
                            digits=3))


# 8. RETRAIN ON ALL DATA AND SAVE
# Performance has now been measured honestly, so the deployed model is refitted
# on 100% of the data to benefit from every available example. This is the model
# used for SHAP (Step 9) and construct validation (Step 10).

print("Retraining on the full dataset ...")

final_model = make_models(scale_pos_weight)[DEPLOY_MODEL]
final_model.fit(X, y)

joblib.dump(final_model,  MODEL_FILE)
joblib.dump(feature_cols, "feature_cols.pkl")
joblib.dump(DEPLOY_MODEL, "best_model_name.pkl")
print(f"✓ Saved: {MODEL_FILE}  ({DEPLOY_MODEL}, trained on {len(X)} sequences)")
print(f"✓ Saved: feature_cols.pkl / best_model_name.pkl")

# Comparison table, for the dissertation
pd.DataFrame([{
    "model":        name,
    "model_type":   MODEL_TYPE[name],
    "pr_auc_mean":  results[name]["test_pr_auc"].mean(),
    "pr_auc_std":   results[name]["test_pr_auc"].std(),
    "roc_auc_mean": results[name]["test_roc_auc"].mean(),
    "roc_auc_std":  results[name]["test_roc_auc"].std(),
    "f1_mean":      results[name]["test_f1"].mean(),
    "f1_std":       results[name]["test_f1"].std(),
    "mcc_mean":     results[name]["test_mcc"].mean(),
    "mcc_std":      results[name]["test_mcc"].std(),
    "is_deployed":  name == DEPLOY_MODEL,
} for name in models]).to_csv("model_comparison.csv", index=False)

# Held-out test results
pd.DataFrame([{
    "feature_file": FEATURES_FILE,
    "model":        DEPLOY_MODEL,
    "n_features":   len(feature_cols),
    **test_scores,
    "tn": tn, "fp": fp, "fn": fn, "tp": tp,
}]).to_csv("holdout_test_results.csv", index=False)

print("✓ Saved: model_comparison.csv / holdout_test_results.csv")



# SUMMARY
print("\n" + "=" * 66)
print("STEP 7 COMPLETE")
print("=" * 66)
print(f"Feature file             : {FEATURES_FILE}  ({len(feature_cols)} features)")
print(f"Models compared          : {len(models)} "
      f"({len(BASE_MODELS)} base + 1 meta-ensemble)")
print(f"Deployed model           : {DEPLOY_MODEL}")
print(f"Cross-validated PR-AUC   : {deploy_cv:.4f}")
print(f"Held-out test PR-AUC     : {test_scores['PR-AUC']:.4f}")
print(f"Random baseline PR-AUC   : {baseline_prauc:.4f}")
print("\n✓ Ready for Step 8: Benchmarking against BPROM & Promoter 2.0")