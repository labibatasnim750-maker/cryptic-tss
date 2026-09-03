"""
STEP 6c — RNA / DNA Thermodynamic Features

Adds folding-energy features on top of features_with_motifs.csv using
ViennaRNA (Lorenz et al., 2011).

Every feature so far encodes sequence *composition* or motif *geometry*.
These encode *thermodynamics* — how readily a region melts or folds — which
is a genuinely different kind of signal.

Biological rationale
The 81 bp window runs from -59 to +21 relative to the TSS at index 59.
Only the downstream portion is ever transcribed, so folding the whole window
as RNA would be meaningless. Instead, several distinct regions are computed:

  1. Transcript (+1 to +21)   RNA secondary structure of the nascent mRNA.
                              Strong hairpins near the 5' end can impede
                              promoter escape.

  2. Discriminator (-10 to +1) The region RNA polymerase must melt to form
                              the open complex. AT-rich, low-stability
                              sequence here favours melting.

  3. Promoter core (-40 to +1) Duplex stability across both consensus
                              elements and the spacer. Easier-to-melt DNA
                              is associated with stronger promoters.

  4. Whole window             Overall stability, as a coarse baseline.

DNA duplex stability is approximated with nearest-neighbour stacking energies
(SantaLucia 1998), since ViennaRNA models RNA rather than DNA duplexes.

Reads : features_with_motifs.csv  (341 features)
Writes: features_with_rna.csv     (341 + 14 features)
"""

import pandas as pd
import numpy as np
import RNA


# CONFIGURATION


INPUT   = "features_with_motifs.csv"
OUTPUT  = "features_with_rna.csv"

TSS_POS = 59          # confirmed empirically in Step 6

# Region boundaries within the 81 bp window (start inclusive, end exclusive)
REGION_TRANSCRIPT    = (TSS_POS,      81)        # +1 to +21
REGION_DISCRIMINATOR = (TSS_POS - 10, TSS_POS)   # -10 to +1
REGION_PROMOTER_CORE = (TSS_POS - 40, TSS_POS)   # -40 to +1
REGION_UPSTREAM      = (0,            TSS_POS)   # -59 to +1

# Nearest-neighbour DNA duplex free energies, kcal/mol at 37 C, 1 M NaCl.
# From SantaLucia (1998) PNAS 95:1460. Values are for the dinucleotide step
# read 5'->3' on the top strand; more negative means a more stable duplex.
NN_DNA = {
    "AA": -1.00, "AT": -0.88, "AC": -1.44, "AG": -1.28,
    "TA": -0.58, "TT": -1.00, "TC": -1.30, "TG": -1.45,
    "CA": -1.45, "CT": -1.28, "CC": -1.84, "CG": -2.17,
    "GA": -1.30, "GT": -1.44, "GC": -2.24, "GG": -1.84,
}



# HELPER FUNCTIONS


def rna_mfe(dna_seq):
    """
    Minimum free energy of the RNA secondary structure formed by the
    transcript of this DNA sequence.

    DNA is transcribed to RNA, so T becomes U before folding. More negative
    means more stable structure. Returns 0.0 for sequences too short to fold.
    """
    if len(dna_seq) < 4:
        return 0.0
    rna = dna_seq.upper().replace("T", "U")
    _, mfe = RNA.fold(rna)
    return float(mfe)


def rna_paired_fraction(dna_seq):
    """
    Fraction of bases predicted to be paired in the MFE structure.

    ViennaRNA returns dot-bracket notation where '.' is unpaired and
    '(' or ')' is paired. A high paired fraction means a highly structured
    transcript, which can impede promoter escape.
    """
    if len(dna_seq) < 4:
        return 0.0
    rna = dna_seq.upper().replace("T", "U")
    structure, _ = RNA.fold(rna)
    paired = sum(1 for c in structure if c in "()")
    return paired / len(structure)


def dna_duplex_energy(seq):
    """
    Approximate DNA duplex free energy by summing nearest-neighbour
    stacking energies over all dinucleotide steps.

    More negative means a more stable, harder-to-melt duplex. RNA polymerase
    must melt the promoter region to initiate, so less stable DNA here is
    biologically favourable.

    Returns total energy and energy normalised per base pair, since the
    regions compared have different lengths.
    """
    seq = seq.upper()
    steps = [seq[i:i + 2] for i in range(len(seq) - 1)]
    total = sum(NN_DNA.get(s, 0.0) for s in steps)
    per_bp = total / len(steps) if steps else 0.0
    return total, per_bp



# LOAD


print("=" * 66)
print("STEP 6c: RNA / DNA Thermodynamic Features")
print("=" * 66)

df = pd.read_csv(INPUT)

print(f"Input file               : {INPUT}")
print(f"Sequences                : {len(df)}")
print(f"Existing columns         : {df.shape[1]}")
print(f"ViennaRNA version        : {RNA.__version__}")

print(f"\nRegions analysed (TSS at index {TSS_POS}):")
print(f"  Transcript             : {REGION_TRANSCRIPT[0]}-{REGION_TRANSCRIPT[1]}"
      f"   (+1 to +21, folded as RNA)")
print(f"  Discriminator          : {REGION_DISCRIMINATOR[0]}-{REGION_DISCRIMINATOR[1]}"
      f"   (-10 to +1, melted during initiation)")
print(f"  Promoter core          : {REGION_PROMOTER_CORE[0]}-{REGION_PROMOTER_CORE[1]}"
      f"   (-40 to +1, spans both consensus elements)")



# EXTRACT THERMODYNAMIC FEATURES

# RNA folding is the slow part — roughly a few thousand sequences per minute.

print(f"\nComputing thermodynamic features for {len(df)} sequences...")
print("RNA folding is computationally intensive — allow 1-3 minutes.\n")

rows = []

for i, seq in enumerate(df["sequence"]):
    if i > 0 and i % 1000 == 0:
        print(f"  {i}/{len(df)} sequences processed...", flush=True)

    seq = str(seq).upper()

    transcript = seq[REGION_TRANSCRIPT[0]:REGION_TRANSCRIPT[1]]
    discrim    = seq[REGION_DISCRIMINATOR[0]:REGION_DISCRIMINATOR[1]]
    core       = seq[REGION_PROMOTER_CORE[0]:REGION_PROMOTER_CORE[1]]
    upstream   = seq[REGION_UPSTREAM[0]:REGION_UPSTREAM[1]]

    # RNA secondary structure of the nascent transcript
    mfe_transcript    = rna_mfe(transcript)
    paired_transcript = rna_paired_fraction(transcript)

    # RNA folding of the whole window, as a coarse baseline
    mfe_window        = rna_mfe(seq)
    paired_window     = rna_paired_fraction(seq)

    # DNA duplex stability by region
    dna_disc_tot,  dna_disc_bp  = dna_duplex_energy(discrim)
    dna_core_tot,  dna_core_bp  = dna_duplex_energy(core)
    dna_up_tot,    dna_up_bp    = dna_duplex_energy(upstream)
    dna_win_tot,   dna_win_bp   = dna_duplex_energy(seq)

    rows.append({
        # RNA secondary structure — nascent transcript
        "rna_mfe_transcript":       mfe_transcript,
        "rna_paired_transcript":    paired_transcript,
        "rna_mfe_transcript_perbp": mfe_transcript / len(transcript)
                                    if transcript else 0.0,

        # RNA secondary structure — whole window (baseline)
        "rna_mfe_window":           mfe_window,
        "rna_paired_window":        paired_window,

        # DNA duplex stability — discriminator (must melt to initiate)
        "dna_energy_discriminator":      dna_disc_tot,
        "dna_energy_discriminator_perbp": dna_disc_bp,

        # DNA duplex stability — promoter core
        "dna_energy_core":          dna_core_tot,
        "dna_energy_core_perbp":    dna_core_bp,

        # DNA duplex stability — upstream region
        "dna_energy_upstream":      dna_up_tot,
        "dna_energy_upstream_perbp": dna_up_bp,

        # DNA duplex stability — whole window
        "dna_energy_window":        dna_win_tot,
        "dna_energy_window_perbp":  dna_win_bp,

        # Difference between core and upstream stability. A promoter core that
        # is easier to melt than its surroundings is mechanistically favourable.
        "dna_energy_core_vs_up":    dna_core_bp - dna_up_bp,
    })

rna_df = pd.DataFrame(rows)
print(f"\n✓ Extracted {rna_df.shape[1]} thermodynamic features")



# COMBINE AND SAVE


out = pd.concat([df.reset_index(drop=True), rna_df], axis=1)

print(f"\nColumns before           : {df.shape[1]}")
print(f"Columns after            : {out.shape[1]}")
print(f"Any NaN values           : {out.isna().sum().sum()}")

out.to_csv(OUTPUT, index=False)
print(f"✓ Saved: {OUTPUT}")


# DISCRIMINATION CHECK

# A feature only helps if it differs between classes. Checking now avoids
# spending five minutes retraining on features that carry no signal.
#
# Expected directions:
#   Promoters are AT-rich, so their DNA duplex energies should be LESS
#   negative (less stable, easier to melt) than non-promoters — i.e. HIGHER.
#   RNA MFE expectations are weaker and mainly exploratory.

print("\n" + "=" * 66)
print("DISCRIMINATION CHECK")
print("=" * 66)
print(f"{'Feature':<34}{'Positives':>11}{'Negatives':>11}{'Diff':>10}")
print("-" * 66)

pos_mask = out["label"] == 1

summary = []
for col in rna_df.columns:
    p    = out.loc[pos_mask,  col].mean()
    n    = out.loc[~pos_mask, col].mean()
    diff = p - n

    # Standardised effect size, so features on different scales are comparable
    sd     = out[col].std()
    cohens = diff / sd if sd > 0 else 0.0

    summary.append((col, abs(cohens)))
    print(f"{col:<34}{p:>11.3f}{n:>11.3f}{diff:>+10.3f}")

print("-" * 66)

print("\nEffect sizes (|difference| in standard deviations):")
print(f"{'Feature':<34}{'Effect':>10}   Strength")
print("-" * 66)
for col, eff in sorted(summary, key=lambda t: -t[1]):
    if eff >= 0.8:
        strength = "large"
    elif eff >= 0.5:
        strength = "medium"
    elif eff >= 0.2:
        strength = "small"
    else:
        strength = "negligible"
    print(f"{col:<34}{eff:>10.3f}   {strength}")

n_useful = sum(1 for _, e in summary if e >= 0.2)
print("-" * 66)
print(f"Features with at least a small effect: {n_useful}/{len(summary)}")

if n_useful == 0:
    print("\nNo feature separates the classes meaningfully. Retraining is")
    print("unlikely to improve performance — which is itself a reportable")
    print("result: thermodynamic signal adds nothing beyond composition here.")


# NEXT STEP
print("\n" + "=" * 66)
print("NEXT STEP")
print("=" * 66)
print("In step7_model_training.py, change the configuration line to:")
print(f'    FEATURES_FILE = "{OUTPUT}"')
print("then re-run Step 7 and compare held-out PR-AUC against 0.9466.")
print("\nBefore re-running, preserve the current results:")
print("    mv holdout_test_results.csv holdout_motifs_341features.csv")
print("    mv model_comparison.csv     model_comparison_341features.csv")
