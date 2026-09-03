import pandas as pd
df = pd.read_csv("/Users/labibatasnim/Downloads/Thesis Work/e_coli_promoters.csv")

print("=" * 50)
print("STEP 3: Cleaning & filtering promoters")
print("=" * 50)
print(f"Rows loaded              : {len(df)}")
print(f"Columns                  : {df.columns.tolist()}")
print(f"\nSigma factors present:")
print(df["sigma_factor"].value_counts())
print(f"\nMissing sequences        : {df['promoter_sequence'].isna().sum()}")
print(f"Sequence lengths present : {df['promoter_sequence'].str.len().unique()}")


##STEP-3 FILTERING PROMOTERS
import pandas as pd

# LOAD
df = pd.read_csv("e_coli_promoters.csv")
print(f"Rows loaded                        : {len(df)}")

# FILTER 1: Drop missing sequences
df = df.dropna(subset=["promoter_sequence", "tss", "strand"])
print(f"After dropping missing sequences   : {len(df)}")

# FILTER 2: Keep only Sigma70 promoters
# This catches "Sigma70" and any combo containing "Sigma70"
df_sigma70 = df[df["sigma_factor"].str.contains("Sigma70", na=False)].copy()
print(f"After keeping Sigma70 only         : {len(df_sigma70)}")

# FILTER 3: Remove duplicate TSS positions
df_sigma70 = df_sigma70.drop_duplicates(subset=["tss", "strand"])
print(f"After removing duplicate TSS       : {len(df_sigma70)}")

# FILTER 4: Keep only valid DNA sequences
valid = df_sigma70["promoter_sequence"].str.upper().str.fullmatch(r"[ATGCN]+")
df_sigma70 = df_sigma70[valid].copy()
print(f"After removing invalid sequences   : {len(df_sigma70)}")

# FILTER 5: Keep only sequences that are exactly 81 bp
df_sigma70 = df_sigma70[
    df_sigma70["promoter_sequence"].str.len() == 81
].copy()
print(f"After keeping only 81 bp sequences : {len(df_sigma70)}")

# ADD LABEL
df_sigma70["label"]    = 1
df_sigma70["source"]   = "RegulonDB_Sigma70"
df_sigma70["sequence"] = df_sigma70["promoter_sequence"].str.upper()

# KEEP ONLY NEEDED COLUMNS
positives = df_sigma70[[
    "id", "name", "sigma_factor", "tss",
    "strand", "sequence", "label", "source"
]].copy()

# SAVE
positives.to_csv("positives.csv", index=False)
print(f"\n✓ Saved positives.csv")

# SUMMARY
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Final positive examples  : {len(positives)}")
print(f"Sequence length          : {positives['sequence'].str.len().unique()}")
print(f"Label values             : {positives['label'].unique()}")
print(f"Strand distribution      :")
print(positives["strand"].value_counts())
print(f"\nFirst 3 rows:")
print(positives[["name", "tss", "strand", "label"]].head(3))


##STEP-4 EXTRACT NEGATIVE EXAMPLES
import pandas as pd
import random

# SETTINGS
CDS_CSV = "cds.csv"
OUTPUT_CSV = "negatives.csv"
WINDOW_SIZE = 81
RANDOM_SEED = 42
N_POSITIVES = 1959
TARGET_NEG = N_POSITIVES * 2  # 2:1 ratio = 3,918 negatives

random.seed(RANDOM_SEED)

# LOAD FILES
print("=" * 55)
print("STEP 4: Extracting negative examples from CDS regions")
print("=" * 55)

cds = pd.read_csv(CDS_CSV)
print(f"CDS rows loaded                    : {len(cds)}")

# CLEAN
cds = cds.dropna(subset=["dna_sequence", "strand", "posleft", "posright"])
print(f"After dropping missing rows        : {len(cds)}")

cds["seq_len"] = cds["dna_sequence"].str.len()
cds = cds[cds["seq_len"] >= WINDOW_SIZE + 10].copy()
print(f"After length filter                : {len(cds)}")

# LOAD KNOWN TSS POSITIONS
positives = pd.read_csv("positives.csv")
known_tss = set(positives["tss"].dropna().astype(int))
print(f"Known TSS positions to avoid       : {len(known_tss)}")
print(f"Target negatives to generate       : {TARGET_NEG}")


# HELPER: CHECK IF A WINDOW OVERLAPS ANY KNOWN TSS
def overlaps_tss(win_start, win_end, known_tss):
    """
    Returns True if ANY known TSS falls inside the window
    [win_start, win_end] on the genome.

    win_start : left-most genomic coordinate of the window
    win_end   : right-most genomic coordinate of the window
    """
    for tss in known_tss:
        if win_start <= tss <= win_end:
            return True
    return False


# SAMPLE WINDOWS FROM CDS REGIONS
records = []

for _, row in cds.iterrows():
    seq = str(row["dna_sequence"]).upper()
    strand = str(row["strand"])
    name = str(row["name"])
    seq_len = len(seq)
    pos_left = int(row["posleft"])
    pos_right = int(row["posright"])
    max_start = seq_len - WINDOW_SIZE

    if max_start < 1:
        continue

    n_samples = min(2, max_start)
    start_positions = random.sample(range(0, max_start), n_samples)

    for start in start_positions:
        window = seq[start: start + WINDOW_SIZE]

        # CORRECTED GENOMIC COORDINATES
        # Always store as (win_left, win_right) = the smaller and larger
        # genomic coordinates regardless of strand.
        # For plus/forward strand: coding sequence runs left to right
        # For minus/reverse strand: coding sequence runs right to left
        if strand in ["+", "forward"]:
            win_left = pos_left + start
            win_right = win_left + WINDOW_SIZE - 1
        else:
            # Minus strand: index 0 in dna_sequence = posright on genome
            # So index `start` corresponds to posright - start on the genome
            win_right = pos_right - start
            win_left = win_right - WINDOW_SIZE + 1

        # CORRECTED TSS OVERLAP CHECK
        # Check whether ANY known TSS falls anywhere inside this window
        # not just within 40 bp of one edge
        if overlaps_tss(win_left, win_right, known_tss):
            continue

        # Skip invalid characters
        if not all(c in "ATGCN" for c in window):
            continue

        # Skip low complexity windows
        if len(set(window)) < 3:
            continue

        records.append({
            "name": name,
            "win_left": win_left,
            "win_right": win_right,
            "strand": strand,
            "sequence": window,
            "label": 0,
            "source": "CDS_RegulonDB",
        })

print(f"Windows extracted before trimming  : {len(records)}")

# TRIM TO TARGET
neg_df = pd.DataFrame(records)

if len(neg_df) > TARGET_NEG:
    neg_df = neg_df.sample(
        n=TARGET_NEG,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)
elif len(neg_df) < TARGET_NEG:
    print(f"⚠ Warning: only {len(neg_df)} negatives generated.")
    print(f"  Target was {TARGET_NEG}. Will proceed with what we have.")

print(f"Final negative count               : {len(neg_df)}")

# SAVE
neg_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Saved: {OUTPUT_CSV}")

# SUMMARY
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"Total negative examples  : {len(neg_df)}")
print(f"Sequence length check    : {neg_df['sequence'].str.len().unique()}")
print(f"Label check              : {neg_df['label'].unique()}")
print(f"Source                   : {neg_df['source'].unique()}")
print(f"\nStrand distribution:")
print(neg_df["strand"].value_counts())
print(f"\nFirst 3 rows:")
print(neg_df[["name", "win_left", "win_right", "strand", "label"]].head(3))


##########
import pandas as pd

pos = pd.read_csv("positives.csv")
neg = pd.read_csv("negatives.csv")

print("positives.csv columns:")
print(pos.columns.tolist())
print(f"Rows: {len(pos)}")
print(pos.head(2))

print("\nnegatives.csv columns:")
print(neg.columns.tolist())
print(f"Rows: {len(neg)}")
print(neg.head(2))
##########

##STEP-5 COMBINING POSITIVE & NEGATIVE DATASETS
import pandas as pd

# LOAD
print("=" * 55)
print("STEP 5: Combining positives and negatives")
print("=" * 55)

pos = pd.read_csv("positives.csv")
neg = pd.read_csv("negatives.csv")

print(f"Positives loaded         : {len(pos)}")
print(f"Negatives loaded         : {len(neg)}")

# STANDARDISE COLUMNS
# Both files have sequence, label, source, strand, name
# We keep only the shared columns so they stack cleanly
pos_clean = pos[["name", "strand", "sequence", "label", "source"]].copy()
neg_clean = neg[["name", "strand", "sequence", "label", "source"]].copy()

print(f"\npos_clean columns        : {pos_clean.columns.tolist()}")
print(f"neg_clean columns        : {neg_clean.columns.tolist()}")

# COMBINE
dataset = pd.concat([pos_clean, neg_clean], ignore_index=True)

# SHUFFLE
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nAfter combining and shuffling:")
print(f"Total examples           : {len(dataset)}")
print(f"Positives (label=1)      : {(dataset['label'] == 1).sum()}")
print(f"Negatives (label=0)      : {(dataset['label'] == 0).sum()}")
print(f"Sequence length check    : {dataset['sequence'].str.len().unique()}")
print(f"Any missing sequences    : {dataset['sequence'].isna().sum()}")
print(f"Any duplicate sequences  : {dataset['sequence'].duplicated().sum()}")

# REMOVE DUPLICATES
before  = len(dataset)
dataset = dataset.drop_duplicates(subset=["sequence"]).reset_index(drop=True)
after   = len(dataset)
print(f"Duplicates removed       : {before - after}")
print(f"Final dataset size       : {len(dataset)}")

# VALIDATE
# Double check labels are correct after deduplication
assert set(dataset["label"].unique()) == {0, 1}, "Unexpected label values found"
assert dataset["sequence"].str.len().nunique() == 1, "Sequences are not all same length"
assert dataset["sequence"].isna().sum() == 0, "Missing sequences found"
print(f"\n✓ All validation checks passed")

# SAVE AS CSV
dataset.to_csv("final_dataset.csv", index=False)
print(f"✓ Saved: final_dataset.csv")

# SAVE AS FASTA
with open("final_dataset.fasta", "w") as f:
    for i, row in dataset.iterrows():
        header = (f">{row['name']} | "
                  f"label:{row['label']} | "
                  f"strand:{row['strand']} | "
                  f"source:{row['source']}")
        f.write(header + "\n")
        f.write(str(row["sequence"]) + "\n")

print(f"✓ Saved: final_dataset.fasta")

# FINAL SUMMARY
print("\n" + "=" * 55)
print("FINAL SUMMARY")
print("=" * 55)
print(f"Positive examples        : {(dataset['label']==1).sum()}")
print(f"Negative examples        : {(dataset['label']==0).sum()}")
print(f"Total dataset size       : {len(dataset)}")
print(f"Class ratio (neg:pos)    : "
      f"{(dataset['label']==0).sum() / (dataset['label']==1).sum():.1f} : 1")
print(f"\nStrand distribution:")
print(dataset["strand"].value_counts())
print(f"\nSource distribution:")
print(dataset["source"].value_counts())
print(f"\nFirst 3 rows:")
print(dataset[["name", "strand", "label", "source"]].head(3))
print(f"\n✓ Dataset preparation complete.")
print(f"✓ Ready for Step 6: Feature Extraction.")

#########################
#########################
import pandas as pd

raw = pd.read_csv("e_coli_promoters.csv")

# Get lacZp1 - the most well-characterised E. coli promoter
# Its -10 element is TATAAT and -35 element is TTTACA
# We know exactly where these should appear relative to TSS
lac = raw[raw["name"] == "lacZp1"].iloc[0]
seq = lac["promoter_sequence"].upper()

print(f"lacZp1 sequence:")
print(seq)
print(f"\nLength: {len(seq)}")

# Search for the known -10 element of lacZp1: TATAAT
pos_10 = seq.find("TATAAT")
print(f"\nTATAAT (-10 element) found at position: {pos_10}")

# Search for the known -35 element of lacZp1: TTTACA
pos_35 = seq.find("TTTACA")
print(f"TTTACA (-35 element) found at position: {pos_35}")

# If TSS is at position X, then:
# -10 element starts at X - 12 (approximately)
# -35 element starts at X - 37 (approximately)
# So TSS = pos_10 + 12
if pos_10 != -1:
    tss_estimate = pos_10 + 12
    print(f"\nEstimated TSS position   : {tss_estimate}")
    print(f"Bases around TSS         : ...{seq[tss_estimate-2:tss_estimate+3]}...")

# Print sequence with position markers every 10 bp
print(f"\nSequence with positions:")
print("".join(str(i % 10) for i in range(len(seq))))
print(seq)
print("0         1         2         3         4         5         6         7         8")
print("0123456789012345678901234567890123456789012345678901234567890123456789012345678901")
#########################
#########################
import pandas as pd

raw = pd.read_csv("e_coli_promoters.csv")
lac = raw[raw["name"] == "lacZp1"].iloc[0]
seq = lac["promoter_sequence"].upper()

print(f"Full sequence:")
print(seq)
print()

# lacZp1 well-known elements:
# -35 element : TTTACA at around position -35 relative to TSS
# -10 element : TATGTT (lacZp1 has a slightly degenerate -10)
#               or look for TATAAT nearby

# The -35 TTTACA is at position 24
pos_35 = 24
print(f"-35 element (TTTACA) at position : {pos_35}")
print(f"Sequence there                   : {seq[pos_35:pos_35+6]}")

# Standard spacer between -35 and -10 is 17 bp
# So -10 element starts at:
pos_10_expected = pos_35 + 6 + 17
print(f"\n-10 element expected at position : {pos_10_expected}")
print(f"Sequence there                   : {seq[pos_10_expected:pos_10_expected+6]}")

# TSS (+1) is approximately 10-12 bp after the -10 element
tss_expected = pos_10_expected + 6 + 6
print(f"\nTSS expected around position     : {tss_expected}")
print(f"Base at TSS                      : {seq[tss_expected]}")

# Print annotated sequence
print(f"\nAnnotated sequence:")
print(seq)
markers = [" "] * len(seq)
for i in range(6):
    markers[pos_35 + i]        = "3"   # -35 element
    markers[pos_10_expected+i] = "1"   # -10 element
markers[tss_expected] = "T"            # TSS
print("".join(markers))
print("3 = -35 element | 1 = -10 element | T = TSS")

# What position is the TSS?
print(f"\nConclusion:")
print(f"  -35 starts at position : {pos_35}")
print(f"  -10 starts at position : {pos_10_expected}")
print(f"  TSS at position        : {tss_expected}")
print(f"  Upstream of TSS        : {tss_expected} bp")
print(f"  Downstream of TSS      : {len(seq) - tss_expected - 1} bp")
#########################
#########################
