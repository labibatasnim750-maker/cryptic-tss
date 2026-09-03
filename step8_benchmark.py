"""
STEP 8 — Benchmarking Against Rule-Based Baselines

The original plan specified BPROM and Promoter 2.0 as comparators. Neither
proved usable:

  BPROM         Hosted on the Softberry web server, which was unreachable
                during the project period. The tool is closed-source with no
                local distribution.

  Promoter 2.0  Models eukaryotic RNA polymerase II promoters (TATA box,
                GC box, CCAAT box, initiator elements). It has no model of
                bacterial sigma factor recognition, so benchmarking against
                it would demonstrate only that a eukaryotic tool performs
                poorly on bacterial sequences.

Two rule-based baselines are therefore implemented directly:

  1. PWM scanner       Scores the best -10 and -35 matches using the same
                       Shultzaberger et al. (2007) matrices. This is the
                       method BPROM implements underneath — position weight
                       matrix scoring of both consensus elements combined
                       into a single discriminant.

  2. Consensus match   The simplest possible rule: summed edit distance to
                       TATAAT and TTGACA. No learned parameters at all.

All three methods are evaluated on the identical held-out test split used in
Step 7, so the comparison is like-for-like.

Reads : features_with_motifs.csv, feature_cols.pkl, best_model_name.pkl
Writes: benchmark_results.csv, benchmark_pr_curves.png
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             f1_score, matthews_corrcoef,
                             precision_recall_curve, confusion_matrix)
import xgboost as xgb


# CONFIGURATION

FEATURES_FILE = "features_with_motifs.csv"

# These must match Step 7 exactly, or the test split will differ and the
# comparison will not be like-for-like.
RANDOM_SEED   = 42
TEST_SIZE     = 0.20

TSS_POS       = 59
SEARCH_10     = (40, 53)
SEARCH_35     = (16, 32)

CONSENSUS_10  = "TATAAT"
CONSENSUS_35  = "TTGACA"

# Shultzaberger et al. (2007), columns ordered A, T, G, C
PWM_10 = np.array([
    [ 0.61,  0.27, -1.82, -1.08],
    [ 1.03, -0.64, -1.28, -0.99],
    [-1.45,  1.04, -1.28,  0.14],
    [ 1.03, -0.64, -1.28, -0.99],
    [-1.45,  1.04, -1.28,  0.14],
    [-0.48,  0.88, -1.28, -0.34],
])
PWM_35 = np.array([
    [-1.08,  0.88, -1.28, -0.34],
    [-1.45,  1.04, -1.28,  0.14],
    [-1.28, -1.28,  1.16, -1.28],
    [ 1.03, -0.64, -1.28, -0.99],
    [-0.34, -0.34, -1.28,  0.88],
    [ 1.03, -0.64, -1.28, -0.99],
])
NUC_IDX = {"A": 0, "T": 1, "G": 2, "C": 3}



# 1. LOAD DATA AND REPRODUCE THE STEP 7 SPLIT
# Splitting an index array with identical parameters and seed reproduces the
# exact partition Step 7 used, so all methods are scored on the same 1,175
# sequences.

print("=" * 66)
print("STEP 8: Benchmarking Against Rule-Based Baselines")
print("=" * 66)

df = pd.read_csv(FEATURES_FILE)
y  = df["label"].values

idx_train, idx_test = train_test_split(
    np.arange(len(df)),
    test_size    = TEST_SIZE,
    stratify     = y,
    random_state = RANDOM_SEED,
)

train_df = df.iloc[idx_train].reset_index(drop=True)
test_df  = df.iloc[idx_test].reset_index(drop=True)
y_train  = y[idx_train]
y_test   = y[idx_test]

print(f"Sequences                : {len(df)}")
print(f"Training split           : {len(train_df)} "
      f"({(y_train == 1).sum()} pos / {(y_train == 0).sum()} neg)")
print(f"Held-out test set        : {len(test_df)} "
      f"({(y_test == 1).sum()} pos / {(y_test == 0).sum()} neg)")
print(f"Split reproduces Step 7  : seed {RANDOM_SEED}, stratified")



# 2. PWM SCANNER

def pwm_score(hexamer, pwm):
    """Sum of PWM log-likelihood scores across a 6 bp window."""
    total = 0.0
    for i, nuc in enumerate(hexamer):
        j = NUC_IDX.get(nuc)
        if j is not None:
            total += pwm[i, j]
    return total


def best_pwm_in_region(seq, pwm, lo, hi):
    """Highest PWM score achieved by any 6 bp window within [lo, hi]."""
    best = -np.inf
    for p in range(lo, hi + 1):
        hexamer = seq[p:p + 6]
        if len(hexamer) < 6:
            break
        best = max(best, pwm_score(hexamer, pwm))
    return best


def pwm_classifier_score(seq):
    """
    Combined PWM score: the sum of the best -10 and best -35 matches.
    Higher means more promoter-like.
    """
    seq = seq.upper()
    return (best_pwm_in_region(seq, PWM_10, *SEARCH_10)
            + best_pwm_in_region(seq, PWM_35, *SEARCH_35))


print("\nScoring with the PWM classifier...")

pwm_train = np.array([pwm_classifier_score(s) for s in train_df["sequence"]])
pwm_test  = np.array([pwm_classifier_score(s) for s in test_df["sequence"]])

# The decision threshold is selected on the TRAINING set only, then applied
# unchanged to the test set. Selecting it on the test set would inflate the
# baseline's apparent performance and make the comparison unfair.
candidates = np.linspace(pwm_train.min(), pwm_train.max(), 200)
pwm_thresh = candidates[int(np.argmax(
    [f1_score(y_train, (pwm_train >= t).astype(int)) for t in candidates]
))]
pwm_pred = (pwm_test >= pwm_thresh).astype(int)

print(f"✓ Threshold selected on training set: {pwm_thresh:.3f}")



# 3. NAIVE CONSENSUS-MATCH BASELINE

def hamming(a, b):
    return sum(1 for x, z in zip(a, b) if x != z)


def consensus_score(seq):
    """
    Negative summed edit distance to both consensus hexamers, so that higher
    is more promoter-like and the direction matches the other scorers.
    """
    seq = seq.upper()
    best10 = min((hamming(seq[p:p + 6], CONSENSUS_10)
                  for p in range(SEARCH_10[0], SEARCH_10[1] + 1)
                  if len(seq[p:p + 6]) == 6), default=6)
    best35 = min((hamming(seq[p:p + 6], CONSENSUS_35)
                  for p in range(SEARCH_35[0], SEARCH_35[1] + 1)
                  if len(seq[p:p + 6]) == 6), default=6)
    return -(best10 + best35)


print("Scoring with the consensus-match baseline...")

cons_train = np.array([consensus_score(s) for s in train_df["sequence"]])
cons_test  = np.array([consensus_score(s) for s in test_df["sequence"]])

cands      = np.unique(cons_train)
cons_thresh = cands[int(np.argmax(
    [f1_score(y_train, (cons_train >= t).astype(int)) for t in cands]
))]
cons_pred = (cons_test >= cons_thresh).astype(int)

print(f"✓ Threshold selected on training set: {cons_thresh:.3f}")


# 4. THE TRAINED MODEL
# IMPORTANT: best_model.pkl cannot be used here. At the end of Step 7 the
# deployed model is refitted on all 5,873 sequences, so it has already seen
# every sequence in this test set. Evaluating it here would be training-set
# evaluation and would report perfect scores.
#
# A fresh model is trained on the training split only, using the identical
# configuration from Step 7, so that all three methods are scored on data
# none of them has seen.

print("\nTraining a fresh XGBoost model on the training split only...")

feature_cols = joblib.load("feature_cols.pkl")
model_name   = joblib.load("best_model_name.pkl")

X_train = train_df[feature_cols].values
X_test  = test_df[feature_cols].values

spw = (y_train == 0).sum() / (y_train == 1).sum()

ml_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", xgb.XGBClassifier(
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
    ))
])

ml_model.fit(X_train, y_train)

ml_scores = ml_model.predict_proba(X_test)[:, 1]
ml_pred   = ml_model.predict(X_test)

print(f"✓ Fresh {model_name} trained on {len(X_train)} sequences "
      f"(test set held out)")


# 5. EVALUATE ALL THREE

def evaluate(name, scores, preds):
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    return {
        "method":    name,
        "pr_auc":    average_precision_score(y_test, scores),
        "roc_auc":   roc_auc_score(y_test, scores),
        "f1":        f1_score(y_test, preds),
        "mcc":       matthews_corrcoef(y_test, preds),
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall":    tp / (tp + fn) if (tp + fn) else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


results  = [
    evaluate("Consensus match",            cons_test, cons_pred),
    evaluate("PWM scanner",                pwm_test,  pwm_pred),
    evaluate(f"{model_name} (this work)",  ml_scores, ml_pred),
]
baseline = (y_test == 1).mean()

print("\n" + "=" * 66)
print(f"BENCHMARK RESULTS (held-out test set, n = {len(y_test)})")
print("=" * 66)
print(f"{'Method':<26}{'PR-AUC':>9}{'ROC-AUC':>9}{'F1':>9}{'MCC':>9}")
print("-" * 66)
for r in results:
    print(f"{r['method']:<26}{r['pr_auc']:>9.4f}{r['roc_auc']:>9.4f}"
          f"{r['f1']:>9.4f}{r['mcc']:>9.4f}")
print("-" * 66)
print(f"{'Random baseline':<26}{baseline:>9.4f}{0.5:>9.4f}"
      f"{'—':>9}{0.0:>9.4f}")

print("\nError breakdown:")
print(f"{'Method':<26}{'TP':>7}{'FP':>7}{'FN':>7}{'Precision':>11}{'Recall':>9}")
print("-" * 66)
for r in results:
    print(f"{r['method']:<26}{r['tp']:>7}{r['fp']:>7}{r['fn']:>7}"
          f"{r['precision']:>11.3f}{r['recall']:>9.3f}")


# Sanity check
# The fresh model should reproduce the Step 7 held-out score of ~0.9466.
# A value near 1.0 would indicate the test set had leaked into training.

ml = results[2]
print("\n" + "=" * 66)
print("SANITY CHECK")
print("=" * 66)
print(f"XGBoost PR-AUC here      : {ml['pr_auc']:.4f}")
print(f"Step 7 held-out PR-AUC   : 0.9466 (expected)")
if ml["pr_auc"] > 0.99:
    print("⚠ WARNING: near-perfect score suggests test data leaked into "
          "training.")
elif abs(ml["pr_auc"] - 0.9466) < 0.02:
    print("✓ Matches Step 7 — no leakage, comparison is valid.")
else:
    print("Note: differs from Step 7. Check that FEATURES_FILE matches the "
          "file used there.")


# Improvement over the baselines

for base in (results[0], results[1]):
    print("\n" + "=" * 66)
    print(f"IMPROVEMENT OVER: {base['method'].upper()}")
    print("=" * 66)
    print(f"PR-AUC           : {base['pr_auc']:.4f} → {ml['pr_auc']:.4f}  "
          f"({ml['pr_auc'] - base['pr_auc']:+.4f})")
    print(f"MCC              : {base['mcc']:.4f} → {ml['mcc']:.4f}  "
          f"({ml['mcc'] - base['mcc']:+.4f})")
    print(f"Promoters missed : {base['fn']} → {ml['fn']}  "
          f"({ml['fn'] - base['fn']:+d})")
    print(f"False alarms     : {base['fp']} → {ml['fp']}  "
          f"({ml['fp'] - base['fp']:+d})")


# 6. PRECISION-RECALL CURVES

plt.figure(figsize=(8, 6))

for (name, scores), colour in zip(
        [("Consensus match",             cons_test),
         ("PWM scanner",                 pwm_test),
         (f"{model_name} (this work)",   ml_scores)],
        ["#cbd5e1", "#94a3b8", "#028090"]):
    prec, rec, _ = precision_recall_curve(y_test, scores)
    ap = average_precision_score(y_test, scores)
    plt.plot(rec, prec, color=colour, lw=2,
             label=f"{name}  (PR-AUC {ap:.3f})")

plt.axhline(baseline, color="#e07b2a", ls="--", lw=1.2,
            label=f"Random baseline ({baseline:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-recall on the held-out test set")
plt.legend(loc="lower left")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("benchmark_pr_curves.png", dpi=150, bbox_inches="tight")
plt.close()

pd.DataFrame(results).to_csv("benchmark_results.csv", index=False)

print("\n✓ Saved: benchmark_pr_curves.png")
print("✓ Saved: benchmark_results.csv")

print("\n" + "=" * 66)
print("STEP 8 COMPLETE — Objective 4 addressed")
print("=" * 66)
print("For the write-up: BPROM and CNNProm are both hosted on the Softberry")
print("server, which was unreachable during the project period; neither is")
print("distributed for local installation. Promoter 2.0 models eukaryotic")
print("Pol II promoters and is not applicable to bacterial sigma factor")
print("recognition. Rule-based baselines were therefore implemented directly")
print("using the same Shultzaberger et al. (2007) matrices.")