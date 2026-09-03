"""
STEP 6b — Motif Feature Extraction

Adds explicit sigma-70 motif features on top of features.csv.

Implements the suggestion from the supervisor meeting: rather than relying on
k-mers to learn the -10 and -35 consensus motifs implicitly, encode them
directly as consensus presence, edit distance, and spacer geometry.

Reads : features.csv              (323 features)
Writes: features_with_motifs.csv  (340 features)

The input file is never modified, so the two feature sets can be compared.
"""

import pandas as pd
import numpy as np

# CONFIGURATION

INPUT  = "features.csv"
OUTPUT = "features_with_motifs.csv"

# Confirmed empirically from lacZp1 in Step 6:
#   RegulonDB 81 bp windows run from -59 to +21 relative to the TSS,
#   so the TSS (+1) sits at index 59.
TSS_POS = 59

CONSENSUS_10 = "TATAAT"     # canonical -10 element
CONSENSUS_35 = "TTGACA"     # canonical -35 element

# Canonical start positions within the window
CANON_10 = TSS_POS - 12     # = 47
CANON_35 = TSS_POS - 35     # = 24

# Ranges to scan for each element (inclusive start positions).
#   -10: must end before the TSS, so the last valid start is 53.
#   -35: wide enough to accommodate spacers of roughly 15-21 bp.
SEARCH_10 = (40, 53)
SEARCH_35 = (16, 32)

# PWM matrices from Shultzaberger et al. (2007), columns ordered A, T, G, C
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


# HELPER FUNCTIONS

def hamming(a, b):
    """
    Count mismatched positions between two equal-length strings.

    Used as the edit distance to consensus: 0 means a perfect match to
    TATAAT or TTGACA, 6 means every position differs.
    """
    return sum(1 for x, y in zip(a, b) if x != y)


def pwm_score(hexamer, pwm):
    """Sum the PWM log-likelihood scores for a 6 bp sequence."""
    total = 0.0
    for i, nuc in enumerate(hexamer):
        idx = NUC_IDX.get(nuc)
        if idx is not None:          # unknown bases (N) contribute nothing
            total += pwm[i, idx]
    return total


def scan_region(seq, consensus, pwm, lo, hi, canonical):
    """
    Slide a 6 bp window across positions lo..hi and find the best match.

    Returns four values:
        best_h    - lowest Hamming distance to the consensus found
        best_h_p  - position where that match occurred
        best_p    - highest PWM score found
        best_p_p  - position where that score occurred

    Where several positions tie, the one nearest the canonical position is
    preferred, since a motif at the expected spacing is more plausible than
    a chance match elsewhere in the window.
    """
    best_h,  best_h_p = 7,       canonical
    best_p,  best_p_p = -np.inf, canonical

    for p in range(lo, hi + 1):
        hexamer = seq[p:p + 6]
        if len(hexamer) < 6:
            break

        h = hamming(hexamer, consensus)
        if h < best_h or (h == best_h and
                          abs(p - canonical) < abs(best_h_p - canonical)):
            best_h, best_h_p = h, p

        s = pwm_score(hexamer, pwm)
        if s > best_p or (s == best_p and
                          abs(p - canonical) < abs(best_p_p - canonical)):
            best_p, best_p_p = s, p

    return best_h, best_h_p, best_p, best_p_p


# LOAD

print("=" * 66)
print("STEP 6b: Motif Feature Extraction")
print("=" * 66)

df = pd.read_csv(INPUT)
print(f"Input file               : {INPUT}")
print(f"Sequences                : {len(df)}")
print(f"Existing columns         : {df.shape[1]}")
print(f"\nMotif positions (from Step 6):")
print(f"  TSS at index           : {TSS_POS}")
print(f"  Canonical -10 start    : {CANON_10}   (scanning {SEARCH_10[0]}-{SEARCH_10[1]})")
print(f"  Canonical -35 start    : {CANON_35}   (scanning {SEARCH_35[0]}-{SEARCH_35[1]})")



# EXTRACT MOTIF FEATURES


print("\nExtracting motif features...")

rows = []

for seq in df["sequence"]:
    seq = str(seq).upper()

    # Best matches found by scanning the search regions
    h10, p10, pwm10, _ = scan_region(seq, CONSENSUS_10, PWM_10,
                                     *SEARCH_10, CANON_10)
    h35, p35, pwm35, _ = scan_region(seq, CONSENSUS_35, PWM_35,
                                     *SEARCH_35, CANON_35)

    # Matches at the fixed canonical positions
    # These use positional knowledge rather than searching, so they capture
    # whether the motif sits exactly where a canonical promoter would put it.
    fixed_h10 = hamming(seq[CANON_10:CANON_10 + 6], CONSENSUS_10)
    fixed_h35 = hamming(seq[CANON_35:CANON_35 + 6], CONSENSUS_35)

    # Spacer geometry
    # Bases between the end of the -35 element and the start of the -10.
    # Optimal spacing is 17 +/- 1 bp; large deviations weaken the promoter.
    spacer = p10 - (p35 + 6)

    # Extended -10 element
    # A TG dinucleotide immediately upstream of the -10 hexamer enhances
    # promoter activity and can compensate for a weak or absent -35 element.
    ext10 = 1 if (p10 >= 2 and seq[p10 - 2:p10] == "TG") else 0

    # Discriminator
    # The region between the -10 element and the TSS. AT-richness here assists
    # DNA melting during open complex formation, so low GC is favourable.
    disc = seq[p10 + 6:TSS_POS]
    disc_gc = (disc.count("G") + disc.count("C")) / len(disc) if disc else 0.0

    rows.append({
        # Edit distance to consensus (0 = perfect match)
        "motif_hamming_10":       h10,
        "motif_hamming_35":       h35,
        "motif_hamming_sum":      h10 + h35,
        "motif_hamming_10_fixed": fixed_h10,
        "motif_hamming_35_fixed": fixed_h35,

        # Exact consensus present anywhere in the search region
        "motif_exact_10":         1 if h10 == 0 else 0,
        "motif_exact_35":         1 if h35 == 0 else 0,
        "motif_exact_both":       1 if (h10 == 0 and h35 == 0) else 0,

        # Best PWM score found by scanning, vs the fixed-position score in Step 6
        "motif_pwm_best_10":      pwm10,
        "motif_pwm_best_35":      pwm35,
        "motif_pwm_best_sum":     pwm10 + pwm35,

        # Where the best matches were found
        "motif_pos_10":           p10,
        "motif_pos_35":           p35,

        # Spacer geometry
        "motif_spacer":           spacer,
        "motif_spacer_optimal":   1 if 16 <= spacer <= 18 else 0,
        "motif_spacer_dev":       abs(spacer - 17),

        # Additional biological elements
        "motif_extended_10":      ext10,
        "motif_discriminator_gc": disc_gc,
    })

motif_df = pd.DataFrame(rows)
print(f"✓ Extracted {motif_df.shape[1]} motif features")



# COMBINE AND SAVE


out = pd.concat([df.reset_index(drop=True), motif_df], axis=1)

print(f"\nColumns before           : {df.shape[1]}")
print(f"Columns after            : {out.shape[1]}")
print(f"Any NaN values           : {out.isna().sum().sum()}")

out.to_csv(OUTPUT, index=False)
print(f"✓ Saved: {OUTPUT}")



# DISCRIMINATION CHECK

# A feature is only useful if it differs between the two classes. Anything with
# a near-zero difference here will not help the model, and that is worth
# knowing before training rather than after.

print("\n" + "=" * 66)
print("DISCRIMINATION CHECK")
print("=" * 66)
print(f"{'Feature':<26}{'Positives':>11}{'Negatives':>11}"
      f"{'Diff':>10}{'Expected':>9}")
print("-" * 66)

# What direction of difference each feature should show if it is working
expected = {
    "motif_hamming_10":       "lower",
    "motif_hamming_35":       "lower",
    "motif_hamming_sum":      "lower",
    "motif_hamming_10_fixed": "lower",
    "motif_hamming_35_fixed": "lower",
    "motif_exact_10":         "higher",
    "motif_exact_35":         "higher",
    "motif_exact_both":       "higher",
    "motif_pwm_best_10":      "higher",
    "motif_pwm_best_35":      "higher",
    "motif_pwm_best_sum":     "higher",
    "motif_pos_10":           "—",
    "motif_pos_35":           "—",
    "motif_spacer":           "—",
    "motif_spacer_optimal":   "higher",
    "motif_spacer_dev":       "lower",
    "motif_extended_10":      "higher",
    "motif_discriminator_gc": "lower",
}

pos_mask = out["label"] == 1
n_pass, n_checked = 0, 0

for col in motif_df.columns:
    p    = out.loc[pos_mask,  col].mean()
    n    = out.loc[~pos_mask, col].mean()
    diff = p - n
    exp  = expected[col]

    if exp == "lower":
        mark = "✓" if diff < 0 else "✗"
        n_checked += 1
        n_pass    += (diff < 0)
    elif exp == "higher":
        mark = "✓" if diff > 0 else "✗"
        n_checked += 1
        n_pass    += (diff > 0)
    else:
        mark = " "

    print(f"{col:<26}{p:>11.3f}{n:>11.3f}{diff:>+10.3f}{exp:>8} {mark}")

print("-" * 66)
print(f"Features behaving as expected: {n_pass}/{n_checked}")
print("\n'lower' means positives should score below negatives (closer to")
print("consensus); 'higher' means the reverse. Features marked '—' have no")
print("expected direction and are included for the model to use as context.")

print("\n" + "=" * 66)
print("NEXT STEP")
print("=" * 66)
print("In step7_model_training.py, change the configuration line to:")
print(f'    FEATURES_FILE = "{OUTPUT}"')
print("then re-run Step 7 and compare held-out PR-AUC against 0.8650.")