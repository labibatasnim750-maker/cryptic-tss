##(WRONG) Step-6 FEATURE EXTRACTION
import pandas as pd
import numpy as np
from itertools import product

# LOAD DATASET
print("=" * 55)
print("STEP 6: Feature Extraction")
print("=" * 55)

df = pd.read_csv("final_dataset.csv")
print(f"Sequences loaded         : {len(df)}")
print(f"Sequence length          : {df['sequence'].str.len().unique()}")


# FEATURE 1 — GC CONTENT

def gc_content(seq):
    seq = seq.upper()
    gc  = seq.count("G") + seq.count("C")
    return gc / len(seq)

print("\nExtracting GC content...")
df["gc_content"] = df["sequence"].apply(gc_content)
print(f"✓ GC content extracted")
print(f"  Mean (positives) : {df[df['label']==1]['gc_content'].mean():.3f}")
print(f"  Mean (negatives) : {df[df['label']==0]['gc_content'].mean():.3f}")


# FEATURE 2 — SIGMA-70 PWM SCORES

# Confirmed from lacZp1 analysis:
#   TSS is at position 59 (0-indexed) in the 81 bp window
#   -10 element starts at position 47  (59 - 12)
#   -35 element starts at position 24  (59 - 35)
#
# PWM values from Shultzaberger et al. (2007)
# Columns = A, T, G, C

TSS_POS = 59
POS_10  = TSS_POS - 12   # = 47
POS_35  = TSS_POS - 35   # = 24

print(f"\nPWM positions (confirmed from lacZp1):")
print(f"  TSS at position  : {TSS_POS}")
print(f"  -10 starts at    : {POS_10}")
print(f"  -35 starts at    : {POS_35}")

PWM_10 = np.array([
    # A       T       G       C
    [ 0.61,  0.27,  -1.82,  -1.08],   # position 1
    [ 1.03, -0.64,  -1.28,  -0.99],   # position 2
    [-1.45,  1.04,  -1.28,   0.14],   # position 3
    [ 1.03, -0.64,  -1.28,  -0.99],   # position 4
    [-1.45,  1.04,  -1.28,   0.14],   # position 5
    [-0.48,  0.88,  -1.28,  -0.34],   # position 6
])

PWM_35 = np.array([
    # A       T       G       C
    [-1.08,  0.88,  -1.28,  -0.34],   # position 1
    [-1.45,  1.04,  -1.28,   0.14],   # position 2
    [-1.28, -1.28,   1.16,  -1.28],   # position 3
    [ 1.03, -0.64,  -1.28,  -0.99],   # position 4
    [-0.34, -0.34,  -1.28,   0.88],   # position 5
    [ 1.03, -0.64,  -1.28,  -0.99],   # position 6
])

NUC_IDX = {"A": 0, "T": 1, "G": 2, "C": 3}

def pwm_score(seq, pwm, start_pos):
    """
    Score a 6 bp window against a 6-position PWM.
    Returns sum of log-likelihood scores.
    """
    score = 0.0
    for i in range(6):
        pos = start_pos + i
        if pos >= len(seq):
            break
        nuc = seq[pos].upper()
        idx = NUC_IDX.get(nuc, None)
        if idx is not None:
            score += pwm[i, idx]
    return score

print("\nExtracting PWM scores...")
df["pwm_score_10"] = df["sequence"].apply(
    lambda s: pwm_score(s, PWM_10, POS_10)
)
df["pwm_score_35"] = df["sequence"].apply(
    lambda s: pwm_score(s, PWM_35, POS_35)
)
print(f"✓ PWM scores extracted")
print(f"  -10 score mean (positives) : "
      f"{df[df['label']==1]['pwm_score_10'].mean():.3f}")
print(f"  -10 score mean (negatives) : "
      f"{df[df['label']==0]['pwm_score_10'].mean():.3f}")
print(f"  -35 score mean (positives) : "
      f"{df[df['label']==1]['pwm_score_35'].mean():.3f}")
print(f"  -35 score mean (negatives) : "
      f"{df[df['label']==0]['pwm_score_35'].mean():.3f}")

# FEATURE 3 — K-MER FREQUENCIES
def get_all_kmers(k):
    bases = ["A", "T", "G", "C"]
    return ["".join(p) for p in product(bases, repeat=k)]

def kmer_frequencies(seq, k):
    seq    = seq.upper()
    kmers  = get_all_kmers(k)
    counts = {kmer: 0 for kmer in kmers}
    total  = len(seq) - k + 1
    for i in range(total):
        window = seq[i:i+k]
        if window in counts:
            counts[window] += 1
    return {kmer: count / total for kmer, count in counts.items()}

print("\nExtracting 3-mer frequencies (64 features)...")
kmer3_list = get_all_kmers(3)
kmer3_df   = pd.DataFrame(
    df["sequence"].apply(lambda s: kmer_frequencies(s, 3)).tolist(),
    columns=[f"3mer_{k}" for k in kmer3_list]
)
print(f"✓ 3-mer features extracted  : {kmer3_df.shape[1]} columns")

print("Extracting 4-mer frequencies (256 features)...")
kmer4_list = get_all_kmers(4)
kmer4_df   = pd.DataFrame(
    df["sequence"].apply(lambda s: kmer_frequencies(s, 4)).tolist(),
    columns=[f"4mer_{k}" for k in kmer4_list]
)
print(f"✓ 4-mer features extracted  : {kmer4_df.shape[1]} columns")


# COMBINE ALL FEATURES
print("\nCombining all features...")

meta    = df[["name", "strand", "sequence",
              "label", "source"]].reset_index(drop=True)
scalar  = df[["gc_content",
              "pwm_score_10",
              "pwm_score_35"]].reset_index(drop=True)

feature_df = pd.concat(
    [meta, scalar, kmer3_df, kmer4_df],
    axis=1
)

print(f"✓ All features combined")
print(f"  Total rows               : {len(feature_df)}")
print(f"  Total columns            : {feature_df.shape[1]}")
print(f"  Feature columns only     : "
      f"{feature_df.shape[1] - len(meta.columns)}")

# SANITY CHECKS
print(f"\nSanity checks:")
print(f"  Any NaN values           : {feature_df.isna().sum().sum()}")
print(f"  Positives (label=1)      : {(feature_df['label']==1).sum()}")
print(f"  Negatives (label=0)      : {(feature_df['label']==0).sum()}")

# PWM sanity check:
# Positives should have HIGHER PWM scores than negatives
# because they contain real sigma-70 binding sites
pos_10 = feature_df[feature_df["label"]==1]["pwm_score_10"].mean()
neg_10 = feature_df[feature_df["label"]==0]["pwm_score_10"].mean()
pos_35 = feature_df[feature_df["label"]==1]["pwm_score_35"].mean()
neg_35 = feature_df[feature_df["label"]==0]["pwm_score_35"].mean()

print(f"\n  PWM sanity check (positives should score higher):")
print(f"  -10 score: positives={pos_10:.3f}, negatives={neg_10:.3f}  "
      f"{'✓ PASS' if pos_10 > neg_10 else '✗ FAIL — check PWM positions'}")
print(f"  -35 score: positives={pos_35:.3f}, negatives={neg_35:.3f}  "
      f"{'✓ PASS' if pos_35 > neg_35 else '✗ FAIL — check PWM positions'}")

# SAVE
feature_df.to_csv("features.csv", index=False)
print(f"\n✓ Saved: features.csv")

# FINAL SUMMARY
print("\n" + "=" * 55)
print("FINAL SUMMARY")
print("=" * 55)
print(f"Total sequences          : {len(feature_df)}")
print(f"Features per sequence    : "
      f"{feature_df.shape[1] - len(meta.columns)}")
print(f"  GC content             : 1")
print(f"  PWM scores             : 2  (-10 and -35)")
print(f"  3-mer frequencies      : 64")
print(f"  4-mer frequencies      : 256")
print(f"  TOTAL                  : 323")
print(f"\n✓ Feature extraction complete.")
print(f"✓ Ready for Step 7: Model Training.")

#########################
#########################
#(Running to diagnose NaN values i.e. where they're coming from)
import pandas as pd

df = pd.read_csv("features.csv")

print("=" * 55)
print("NaN DIAGNOSIS")
print("=" * 55)
print(f"Total NaN values         : {df.isna().sum().sum()}")
print(f"Total rows               : {len(df)}")
print(f"Total columns            : {df.shape[1]}")

# Check which column types have NaNs
nan_by_col = df.isna().sum()
nan_cols   = nan_by_col[nan_by_col > 0]

print(f"\nColumns with NaN values  : {len(nan_cols)}")
print(f"\nFirst 10 columns with NaNs:")
print(nan_cols.head(10))

# Check if NaNs are in metadata or feature columns
meta_cols    = ["name", "strand", "sequence", "label", "source"]
feature_cols = [c for c in df.columns if c not in meta_cols]

meta_nans    = df[meta_cols].isna().sum().sum()
feature_nans = df[feature_cols].isna().sum().sum()

print(f"\nNaNs in metadata columns : {meta_nans}")
print(f"NaNs in feature columns  : {feature_nans}")

# Check which feature types have NaNs
gc_nans   = df["gc_content"].isna().sum()
pwm_nans  = df[["pwm_score_10", "pwm_score_35"]].isna().sum().sum()
kmer_nans = df[[c for c in df.columns
                if c.startswith("3mer_") or
                   c.startswith("4mer_")]].isna().sum().sum()

print(f"\nNaNs in gc_content       : {gc_nans}")
print(f"NaNs in PWM scores       : {pwm_nans}")
print(f"NaNs in k-mer columns    : {kmer_nans}")

# Check a few rows that have NaNs
print(f"\nFirst 3 rows with NaNs:")
nan_rows = df[df.isna().any(axis=1)]
print(nan_rows[["name", "strand", "label",
                "gc_content", "pwm_score_10",
                "pwm_score_35"]].head(3))

# Check if NaNs are concentrated in positives or negatives
print(f"\nNaN rows by label:")
print(nan_rows["label"].value_counts())
#########################
#########################





##(CORRECTED) Step-6 FEATURE EXTRACTION
import pandas as pd
import numpy as np
from itertools import product

#LOAD DATASET
print("=" * 55)
print("STEP 6: Feature Extraction")
print("=" * 55)

df = pd.read_csv("final_dataset.csv")
df = df.reset_index(drop=True)   # ← critical: ensure clean 0-based index
print(f"Sequences loaded         : {len(df)}")
print(f"Sequence length          : {df['sequence'].str.len().unique()}")


# FEATURE 1 — GC CONTENT
def gc_content(seq):
    seq = seq.upper()
    gc  = seq.count("G") + seq.count("C")
    return gc / len(seq)

print("\nExtracting GC content...")
gc_values = [gc_content(s) for s in df["sequence"]]
print(f"✓ GC content extracted")


# FEATURE 2 — SIGMA-70 PWM SCORES

# Confirmed from lacZp1 analysis:
# TSS at position 59, -10 starts at 47, -35 starts at 24

TSS_POS = 59
POS_10  = TSS_POS - 12   # = 47
POS_35  = TSS_POS - 35   # = 24

PWM_10 = np.array([
    # A       T       G       C
    [ 0.61,  0.27,  -1.82,  -1.08],
    [ 1.03, -0.64,  -1.28,  -0.99],
    [-1.45,  1.04,  -1.28,   0.14],
    [ 1.03, -0.64,  -1.28,  -0.99],
    [-1.45,  1.04,  -1.28,   0.14],
    [-0.48,  0.88,  -1.28,  -0.34],
])

PWM_35 = np.array([
    # A       T       G       C
    [-1.08,  0.88,  -1.28,  -0.34],
    [-1.45,  1.04,  -1.28,   0.14],
    [-1.28, -1.28,   1.16,  -1.28],
    [ 1.03, -0.64,  -1.28,  -0.99],
    [-0.34, -0.34,  -1.28,   0.88],
    [ 1.03, -0.64,  -1.28,  -0.99],
])

NUC_IDX = {"A": 0, "T": 1, "G": 2, "C": 3}

def pwm_score(seq, pwm, start_pos):
    score = 0.0
    for i in range(6):
        pos = start_pos + i
        if pos >= len(seq):
            break
        nuc = seq[pos].upper()
        idx = NUC_IDX.get(nuc, None)
        if idx is not None:
            score += pwm[i, idx]
    return score

print("Extracting PWM scores...")
pwm10_values = [pwm_score(s, PWM_10, POS_10) for s in df["sequence"]]
pwm35_values = [pwm_score(s, PWM_35, POS_35) for s in df["sequence"]]
print(f"✓ PWM scores extracted")


# FEATURE 3 — K-MER FREQUENCIES

def get_all_kmers(k):
    bases = ["A", "T", "G", "C"]
    return ["".join(p) for p in product(bases, repeat=k)]

def kmer_frequencies(seq, k):
    seq    = seq.upper()
    kmers  = get_all_kmers(k)
    counts = {kmer: 0 for kmer in kmers}
    total  = len(seq) - k + 1
    for i in range(total):
        window = seq[i:i+k]
        if window in counts:
            counts[window] += 1
    return [counts[kmer] / total for kmer in kmers]

print("Extracting 3-mer frequencies (64 features)...")
kmer3_list   = get_all_kmers(3)
kmer3_values = [kmer_frequencies(s, 3) for s in df["sequence"]]
print(f"✓ 3-mer features extracted")

print("Extracting 4-mer frequencies (256 features)...")
kmer4_list   = get_all_kmers(4)
kmer4_values = [kmer_frequencies(s, 4) for s in df["sequence"]]
print(f"✓ 4-mer features extracted")


# COMBINE ALL FEATURES INTO ONE DATAFRAME

print("\nCombining all features...")

# Build feature matrix as a plain numpy array first
# then assign column names — avoids all index alignment issues
n = len(df)

# Scalar features: gc, pwm10, pwm35
scalar_arr = np.column_stack([
    gc_values,
    pwm10_values,
    pwm35_values,
])

# k-mer arrays
kmer3_arr = np.array(kmer3_values)   # shape (5873, 64)
kmer4_arr = np.array(kmer4_values)   # shape (5873, 256)

# Stack everything horizontally
feature_arr = np.hstack([scalar_arr, kmer3_arr, kmer4_arr])

# Column names
scalar_cols = ["gc_content", "pwm_score_10", "pwm_score_35"]
kmer3_cols  = [f"3mer_{k}" for k in kmer3_list]
kmer4_cols  = [f"4mer_{k}" for k in kmer4_list]
feature_cols = scalar_cols + kmer3_cols + kmer4_cols

# Build final dataframe
feature_df = pd.DataFrame(feature_arr, columns=feature_cols)

# Add metadata columns
feature_df.insert(0, "source",   df["source"].values)
feature_df.insert(0, "label",    df["label"].values)
feature_df.insert(0, "sequence", df["sequence"].values)
feature_df.insert(0, "strand",   df["strand"].values)
feature_df.insert(0, "name",     df["name"].values)

print(f"✓ All features combined")
print(f"  Total rows               : {len(feature_df)}")
print(f"  Total columns            : {feature_df.shape[1]}")
print(f"  Feature columns only     : {len(feature_cols)}")

# SANITY CHECKS
print(f"\nSanity checks:")
print(f"  Any NaN values           : {feature_df.isna().sum().sum()}")
print(f"  Positives (label=1)      : {(feature_df['label']==1).sum()}")
print(f"  Negatives (label=0)      : {(feature_df['label']==0).sum()}")

pos_10 = feature_df[feature_df["label"]==1]["pwm_score_10"].mean()
neg_10 = feature_df[feature_df["label"]==0]["pwm_score_10"].mean()
pos_35 = feature_df[feature_df["label"]==1]["pwm_score_35"].mean()
neg_35 = feature_df[feature_df["label"]==0]["pwm_score_35"].mean()
pos_gc = feature_df[feature_df["label"]==1]["gc_content"].mean()
neg_gc = feature_df[feature_df["label"]==0]["gc_content"].mean()

print(f"\n  Biological sanity checks (all should PASS):")
print(f"  GC content : pos={pos_gc:.3f}, neg={neg_gc:.3f}  "
      f"{'✓ PASS' if pos_gc < neg_gc else '✗ FAIL'}")
print(f"  PWM -10    : pos={pos_10:.3f}, neg={neg_10:.3f}  "
      f"{'✓ PASS' if pos_10 > neg_10 else '✗ FAIL'}")
print(f"  PWM -35    : pos={pos_35:.3f}, neg={neg_35:.3f}  "
      f"{'✓ PASS' if pos_35 > neg_35 else '✗ FAIL'}")

# SAVE
feature_df.to_csv("features.csv", index=False)
print(f"\n✓ Saved: features.csv")

# FINAL SUMMARY
print("\n" + "=" * 55)
print("FINAL SUMMARY")
print("=" * 55)
print(f"Total sequences          : {len(feature_df)}")
print(f"Features per sequence    : {len(feature_cols)}")
print(f"  GC content             : 1")
print(f"  PWM scores             : 2")
print(f"  3-mer frequencies      : 64")
print(f"  4-mer frequencies      : 256")
print(f"  TOTAL                  : 323")
print(f"\n✓ Feature extraction complete.")
print(f"✓ Ready for Step 7: Model Training.")

##########
import pandas as pd

df = pd.read_csv("features.csv")

print(f"NaN values before fix    : {df.isna().sum().sum()}")
print(f"\nWhich columns have NaNs:")
print(df.isna().sum()[df.isna().sum() > 0])

# Fix: fill missing names with a placeholder
df["name"] = df["name"].fillna("unknown")

# Confirm no NaNs remain
print(f"\nNaN values after fix     : {df.isna().sum().sum()}")

# Save
df.to_csv("features.csv", index=False)
print(f"✓ Saved: features.csv")
##########
#double checking if no NaN values are left
import pandas as pd

df = pd.read_csv("features.csv")
print(f"Total NaNs: {df.isna().sum().sum()}")  # Should be 0
print(f"Total rows: {len(df)}")
print(f"Total cols: {df.shape[1]}")
####################