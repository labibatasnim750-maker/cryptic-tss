#!/usr/bin/env python3
"""
cryptic_tss.py — Cryptic Promoter Scanner for Synthetic DNA


Scans a bacterial DNA sequence and reports where unintended sigma-70
promoter activity is likely, so problem regions can be recoded before
the construct is synthesised.

QUICK START

    python cryptic_tss.py myconstruct.fasta

That is all that is needed. Everything else has a sensible default.

MORE OPTIONS

    python cryptic_tss.py myconstruct.fasta --both-strands
    python cryptic_tss.py myconstruct.fasta --sensitive
    python cryptic_tss.py --demo          # check the tool works

WHAT YOU GET
A summary printed to the screen, plus these files:

    <name>_report.txt          plain-English findings and suggested edits
    <name>_risk_profile.png    risk plotted along the sequence
    <name>_risk_profile.csv    risk score at every position
    <name>_hotspots.csv        flagged regions only

All positions reported are coordinates in the sequence you supplied,
including reverse-strand hits, so they can be used directly.

Author : Labiba Tasnim Zeba
Project: MSc Bioinformatics, University of Bristol
"""

import argparse
import os
import sys
import time
from itertools import product

__version__ = "1.1"

# Where this script lives. Model files are looked up here rather than in the
# caller's working directory, so the tool runs from anywhere once the .pkl
# files sit alongside it.
HERE = os.path.dirname(os.path.abspath(__file__))


# DEPENDENCY CHECK

# Checked before anything else so a missing package produces one clear
# instruction rather than a traceback partway through a long scan.

def check_dependencies():
    missing = []
    for module, install_name in [("numpy", "numpy"),
                                 ("pandas", "pandas"),
                                 ("joblib", "joblib"),
                                 ("matplotlib", "matplotlib"),
                                 ("xgboost", "xgboost")]:
        try:
            __import__(module)
        except ImportError:
            missing.append(install_name)

    if missing:
        print("ERROR: required packages are not installed.\n")
        print("  Install everything this tool needs with:")
        print(f"      pip install -r {os.path.join(HERE, 'requirements.txt')}\n")
        print("  Or just the missing ones:")
        print(f"      pip install {' '.join(missing)}\n")
        if "xgboost" in missing and sys.platform == "darwin":
            print("  On macOS, xgboost also needs the OpenMP runtime:")
            print("      brew install libomp\n")
        sys.exit(1)


check_dependencies()

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



# MODEL PARAMETERS

# These must match the training pipeline exactly. They are grouped here so a
# model for a different sigma factor can be supported by adding an entry
# rather than editing code throughout the file.

WINDOW_SIZE = 81
TSS_POS     = 59              # index of the TSS within an 81 bp window

SIGMA_FACTORS = {
    "70": {
        "consensus_10": "TATAAT",
        "consensus_35": "TTGACA",
        "canon_10":     TSS_POS - 12,     # 47
        "canon_35":     TSS_POS - 35,     # 24
        "search_10":    (40, 53),
        "search_35":    (16, 32),
        "default_model": "best_model.pkl",
        "description":  "housekeeping genes, primary sigma factor",
    },
    # To add sigma-32 or sigma-38 later: train a model following Steps 3-7
    # with the sigma factor filter changed, then add an entry here with that
    # factor's consensus sequences and the path to the saved model.
}

# Shultzaberger et al. (2007) position weight matrices, columns A, T, G, C
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

# Risk bands used throughout the output. Chosen so that "HIGH" corresponds
# roughly to the precision the model achieved on held-out data (0.86).
RISK_BANDS = [
    (0.80, "HIGH",     "Very likely to drive unintended transcription"),
    (0.50, "MODERATE", "Promoter-like; worth reviewing"),
    (0.30, "LOW",      "Weak signal; probably harmless"),
    (0.00, "MINIMAL",  "No meaningful promoter signal"),
]
BAND_ORDER = [b[1] for b in RISK_BANDS]


def risk_band(score):
    for threshold, label, description in RISK_BANDS:
        if score >= threshold:
            return label, description
    return "MINIMAL", ""



# FEATURE EXTRACTION
# Reimplements Steps 6 and 6b to work on raw sequence rather than a CSV of
# pre-cut windows. Feature names and ordering must match the trained model
# exactly; this is verified at startup against feature_cols.pkl.

def all_kmers(k):
    return ["".join(p) for p in product("ATGC", repeat=k)]


KMER3     = all_kmers(3)
KMER4     = all_kmers(4)
KMER3_IDX = {k: i for i, k in enumerate(KMER3)}
KMER4_IDX = {k: i for i, k in enumerate(KMER4)}

FEATURE_ORDER = (
    ["gc_content", "pwm_score_10", "pwm_score_35"]
    + [f"3mer_{k}" for k in KMER3]
    + [f"4mer_{k}" for k in KMER4]
    + ["motif_hamming_10", "motif_hamming_35", "motif_hamming_sum",
       "motif_hamming_10_fixed", "motif_hamming_35_fixed",
       "motif_exact_10", "motif_exact_35", "motif_exact_both",
       "motif_pwm_best_10", "motif_pwm_best_35", "motif_pwm_best_sum",
       "motif_pos_10", "motif_pos_35",
       "motif_spacer", "motif_spacer_optimal", "motif_spacer_dev",
       "motif_extended_10", "motif_discriminator_gc"]
)


def pwm_score(hexamer, pwm):
    total = 0.0
    for i, nuc in enumerate(hexamer):
        j = NUC_IDX.get(nuc)
        if j is not None:
            total += pwm[i, j]
    return total


def kmer_freqs(seq, k, kmer_idx, n_kmers):
    counts = np.zeros(n_kmers)
    total  = len(seq) - k + 1
    for i in range(total):
        j = kmer_idx.get(seq[i:i + k])
        if j is not None:
            counts[j] += 1
    return counts / total


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def scan_region(seq, consensus, pwm, lo, hi, canonical):
    """
    Find the best consensus match and best PWM score within [lo, hi].

    Ties break towards the canonical position, since a motif at the expected
    spacing is more plausible than a chance match elsewhere in the window.
    """
    best_h, best_h_p = 7, canonical
    best_p, best_p_p = -np.inf, canonical

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


def extract_features(window, sigma):
    """Compute all 341 features for one 81 bp window."""
    seq = window.upper()

    gc    = (seq.count("G") + seq.count("C")) / len(seq)
    pwm10 = pwm_score(seq[sigma["canon_10"]:sigma["canon_10"] + 6], PWM_10)
    pwm35 = pwm_score(seq[sigma["canon_35"]:sigma["canon_35"] + 6], PWM_35)
    k3    = kmer_freqs(seq, 3, KMER3_IDX, len(KMER3))
    k4    = kmer_freqs(seq, 4, KMER4_IDX, len(KMER4))

    h10, p10, best_pwm10, _ = scan_region(
        seq, sigma["consensus_10"], PWM_10,
        *sigma["search_10"], sigma["canon_10"])
    h35, p35, best_pwm35, _ = scan_region(
        seq, sigma["consensus_35"], PWM_35,
        *sigma["search_35"], sigma["canon_35"])

    fixed_h10 = hamming(seq[sigma["canon_10"]:sigma["canon_10"] + 6],
                        sigma["consensus_10"])
    fixed_h35 = hamming(seq[sigma["canon_35"]:sigma["canon_35"] + 6],
                        sigma["consensus_35"])

    spacer = p10 - (p35 + 6)
    ext10  = 1 if (p10 >= 2 and seq[p10 - 2:p10] == "TG") else 0

    disc    = seq[p10 + 6:TSS_POS]
    disc_gc = (disc.count("G") + disc.count("C")) / len(disc) if disc else 0.0

    motif = np.array([
        h10, h35, h10 + h35, fixed_h10, fixed_h35,
        1 if h10 == 0 else 0,
        1 if h35 == 0 else 0,
        1 if (h10 == 0 and h35 == 0) else 0,
        best_pwm10, best_pwm35, best_pwm10 + best_pwm35,
        p10, p35,
        spacer,
        1 if 16 <= spacer <= 18 else 0,
        abs(spacer - 17),
        ext10,
        disc_gc,
    ])

    return np.concatenate([[gc, pwm10, pwm35], k3, k4, motif])


# INPUT HANDLING

def looks_like_sequence(text):
    """
    Decide whether a string is DNA rather than a filename.

    Lets the user paste a sequence directly without needing a flag, which is
    the common case when checking a short part they have just designed.
    """
    stripped = text.strip().upper().replace("\n", "").replace(" ", "")
    if len(stripped) < 20:
        return False
    return all(c in "ATGCUN" for c in stripped)


def read_fasta(path):
    """Read a FASTA file, returning [(name, sequence), ...]."""
    records, name, seq = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(seq)))
                header = line[1:].split()
                name, seq = (header[0] if header else "seq"), []
            else:
                seq.append(line)
    if name is not None:
        records.append((name, "".join(seq)))
    return records


def read_plain(path):
    """Read a file containing only sequence, with no FASTA header."""
    with open(path) as f:
        seq = "".join(line.strip() for line in f if not line.startswith(">"))
    name = os.path.splitext(os.path.basename(path))[0]
    return [(name, seq)]


def load_input(arg):
    """
    Work out what the user gave us and return [(name, sequence), ...].

    Accepts a FASTA file, a plain sequence file, or a pasted sequence.
    """
    if looks_like_sequence(arg):
        clean = arg.strip().upper().replace("\n", "").replace(" ", "")
        return [("pasted_sequence", clean)]

    if not os.path.exists(arg):
        print(f"ERROR: cannot find '{arg}'\n")
        print("  Check the filename and path. If you meant to paste a")
        print("  sequence directly, it must be at least 20 bases and")
        print("  contain only A, T, G, C.\n")
        candidates = [f for f in sorted(os.listdir("."))
                      if f.lower().endswith((".fa", ".fasta", ".fna",
                                             ".txt", ".seq"))]
        if candidates:
            print("  Sequence files in the current directory:")
            for f in candidates[:12]:
                print(f"      {f}")
        else:
            print("  No sequence files found in the current directory.")
        print()
        sys.exit(1)

    with open(arg) as f:
        first = f.readline()

    records = read_fasta(arg) if first.startswith(">") else read_plain(arg)

    if not records:
        sys.exit(f"ERROR: no sequences found in '{arg}'")

    return records


def clean_sequence(seq):
    """Uppercase, convert RNA to DNA, and mark unknown characters as N."""
    seq = seq.upper().replace("U", "T")
    return "".join(c if c in "ATGC" else "N" for c in seq)


def reverse_complement(seq):
    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    return "".join(comp[c] for c in reversed(seq))


def to_forward(pos, seq_len, strand):
    """
    Convert a coordinate on the scanned strand back to the input sequence.

    Reverse-strand scanning runs on the reverse complement, so index p there
    corresponds to index (len - 1 - p) in the sequence the user supplied.
    Every position shown to the user goes through here, so a reported
    coordinate can always be used directly against their own FASTA.
    """
    if strand == "+":
        return pos
    return seq_len - 1 - pos


def to_forward_span(start, end, seq_len, strand):
    """Convert a coordinate range, keeping start <= end after any flip."""
    a = to_forward(start, seq_len, strand)
    b = to_forward(end, seq_len, strand)
    return (a, b) if a <= b else (b, a)



# SCANNING


def choose_step_size(length, requested):
    """
    Pick a step size that keeps runtime reasonable on long sequences.

    Scanning every position on a 100 kb plasmid takes several minutes. Since
    adjacent windows overlap by 80 of 81 bases their scores are almost
    identical, so stepping by a few bases loses very little resolution.
    """
    if requested is not None:
        return requested, False
    if length <= 20_000:
        return 1, False
    if length <= 100_000:
        return 3, True
    return 5, True


def scan_sequence(seq, model, sigma, step=1, show_progress=True):
    """
    Slide an 81 bp window along the sequence and score every position.

    The reported position is the index corresponding to index 59 of the
    window, so a score at position p answers: "if transcription started
    here, how promoter-like is the surrounding context?"

    Positions returned are on the strand being scanned; conversion back to
    input coordinates happens in the reporting layer.
    """
    starts = list(range(0, len(seq) - WINDOW_SIZE + 1, step))
    if not starts:
        raise ValueError(
            f"Sequence is {len(seq)} bp but at least {WINDOW_SIZE} bp "
            f"is needed to scan.")

    # Time a small sample so a long scan can warn the user up front rather
    # than appearing to hang.
    if show_progress and len(starts) > 2000:
        t0 = time.time()
        for i in starts[:200]:
            extract_features(seq[i:i + WINDOW_SIZE], sigma)
        rate = 200 / max(time.time() - t0, 1e-6)
        est  = len(starts) / rate
        if est > 20:
            print(f"  Scanning {len(starts):,} positions "
                  f"(about {est/60:.1f} minutes)...")
        else:
            print(f"  Scanning {len(starts):,} positions...")
    elif show_progress:
        print(f"  Scanning {len(starts):,} positions...")

    feats, positions, skipped = [], [], 0

    for n, i in enumerate(starts):
        window = seq[i:i + WINDOW_SIZE]
        if "N" in window:
            skipped += 1
            continue
        feats.append(extract_features(window, sigma))
        positions.append(i + TSS_POS)

        if show_progress and n and n % 10_000 == 0:
            print(f"    {n:,} / {len(starts):,}", flush=True)

    if not feats:
        raise ValueError("Nothing could be scanned — every window contained "
                         "ambiguous bases (N).")

    if skipped and show_progress:
        print(f"  ({skipped:,} windows skipped: ambiguous bases)")

    scores = model.predict_proba(np.array(feats))[:, 1]
    return pd.DataFrame({"position": positions, "risk_score": scores})


def find_hotspots(profile, threshold, min_gap=25):
    """
    Collapse runs of high-scoring positions into discrete regions.

    Adjacent windows overlap heavily, so one real promoter produces dozens of
    consecutive high scores. Without clustering the output would list the same
    site many times over.
    """
    high = profile[profile["risk_score"] >= threshold]
    if high.empty:
        return pd.DataFrame(columns=["start", "end", "peak_position",
                                     "peak_score", "width"])

    high = high.sort_values("position").reset_index(drop=True)
    groups, current = [], [high.iloc[0]]

    for _, row in high.iloc[1:].iterrows():
        if row["position"] - current[-1]["position"] <= min_gap:
            # Two promoters can sit inside one run of above-threshold
            # positions. Split when the profile dips well below the peaks
            # on either side, so a strong site is never hidden inside a
            # wider region reported under another site's coordinates.
            between = profile[(profile["position"] > current[-1]["position"]) &
                              (profile["position"] < row["position"])]
            trough   = between["risk_score"].min() if not between.empty else None
            shoulder = min(current[-1]["risk_score"], row["risk_score"])
            if trough is not None and trough < shoulder * 0.6:
                groups.append(current)
                current = [row]
            else:
                current.append(row)
        else:
            groups.append(current)
            current = [row]
    groups.append(current)

    out = []
    for g in groups:
        g_df = pd.DataFrame(g)
        peak = g_df.loc[g_df["risk_score"].idxmax()]
        out.append({
            "start":         int(g_df["position"].min()),
            "end":           int(g_df["position"].max()),
            "peak_position": int(peak["position"]),
            "peak_score":    float(peak["risk_score"]),
            "width":         int(g_df["position"].max()
                                 - g_df["position"].min() + 1),
        })

    return (pd.DataFrame(out)
            .sort_values("peak_score", ascending=False)
            .reset_index(drop=True))


def band_counts(hotspots):
    """
    Count regions per risk band across ALL hotspots.

    Counted from the hotspot table rather than from the detailed findings,
    which are capped for readability — otherwise the headline count and the
    breakdown beside it would disagree on long sequences.
    """
    counts = {}
    for _, hs in hotspots.iterrows():
        band, _ = risk_band(hs["peak_score"])
        counts[band] = counts.get(band, 0) + 1
    return counts


# INTERPRETATION
# Turns a numerical score into something a wet-lab user can act on. This is
# the part that makes the tool useful rather than merely accurate.

def describe_hotspot(seq, peak_position, score, sigma, seq_len, strand="+"):
    """
    Explain in plain English why a region was flagged, and what to change.

    Returns a dict of findings rather than formatted text so the same content
    can go to both the console and the report file. All coordinates exposed
    to the user are in the input sequence's frame; the bare scan position is
    kept so a result can be traced back.
    """
    start  = peak_position - TSS_POS
    window = seq[start:start + WINDOW_SIZE]

    if len(window) < WINDOW_SIZE:
        return None

    h10, p10, _, _ = scan_region(window, sigma["consensus_10"], PWM_10,
                                 *sigma["search_10"], sigma["canon_10"])
    h35, p35, _, _ = scan_region(window, sigma["consensus_35"], PWM_35,
                                 *sigma["search_35"], sigma["canon_35"])

    spacer  = p10 - (p35 + 6)
    elem_10 = window[p10:p10 + 6]
    elem_35 = window[p35:p35 + 6]
    gc      = (window.count("G") + window.count("C")) / len(window)

    reasons = []
    if h10 == 0:
        reasons.append(f"perfect -10 element ({elem_10})")
    elif h10 <= 1:
        reasons.append(f"near-perfect -10 element ({elem_10}, "
                       f"{h10} mismatch)")
    elif h10 <= 2:
        reasons.append(f"recognisable -10 element ({elem_10}, "
                       f"{h10} mismatches)")

    if h35 == 0:
        reasons.append(f"perfect -35 element ({elem_35})")
    elif h35 <= 1:
        reasons.append(f"near-perfect -35 element ({elem_35}, "
                       f"{h35} mismatch)")
    elif h35 <= 2:
        reasons.append(f"recognisable -35 element ({elem_35}, "
                       f"{h35} mismatches)")

    # Spacing only means something if there are two recognisable elements to
    # space apart. Crediting "optimal spacing" between two poor matches gives
    # the geometry weight it has not earned, and can leave spacing as the only
    # stated reason for a region whose motifs are both weak.
    if h10 <= 2 and h35 <= 2:
        if spacer == 17:
            reasons.append("optimal 17 bp spacing between elements")
        elif 16 <= spacer <= 18:
            reasons.append(f"near-optimal {spacer} bp spacing "
                           f"(17 is ideal)")
        else:
            reasons.append(f"{spacer} bp spacing (17 is ideal, so this "
                           f"weakens the match)")

    if gc < 0.42:
        reasons.append(f"AT-rich context ({gc:.0%} GC), which favours "
                       f"DNA melting")

    if not reasons:
        reasons.append("a combination of weaker sequence signals; see the "
                       "risk profile for context")

    band, band_desc = risk_band(score)

    # Target whichever element is the stronger match, since that is what the
    # model is responding to. The -10 is usually the more effective target,
    # but not when the -35 is the element actually carrying the signal.
    if h35 < h10:
        target, elem, tpos = "-35", elem_35, start + p35
        consensus = sigma["consensus_35"]
    else:
        target, elem, tpos = "-10", elem_10, start + p10
        consensus = sigma["consensus_10"]

    # Everything the user acts on is expressed in their own coordinates.
    fwd_pos = to_forward(peak_position, seq_len, strand)
    fwd_10_a, fwd_10_b = to_forward_span(start + p10, start + p10 + 5,
                                         seq_len, strand)
    fwd_35_a, fwd_35_b = to_forward_span(start + p35, start + p35 + 5,
                                         seq_len, strand)
    fwd_t_a, fwd_t_b   = to_forward_span(tpos, tpos + 5, seq_len, strand)

    strand_note = ("" if strand == "+" else
                   " on the reverse strand (the element as written here is "
                   "the reverse complement of your sequence at these "
                   "coordinates)")

    suggestion = (f"Recode positions {fwd_t_a}-{fwd_t_b} "
                  f"(the {target} element, currently {elem}){strand_note}. "
                  f"In a coding region, use synonymous codons that "
                  f"disrupt the {consensus} match — increasing GC here "
                  f"is usually effective.")

    return {
        "strand":        strand,
        "position":      fwd_pos,
        "scan_position": peak_position,
        "score":         score,
        "band":          band,
        "band_desc":     band_desc,
        "window":        window,
        "elem_10":       elem_10,
        "elem_10_pos":   fwd_10_a,
        "elem_10_end":   fwd_10_b,
        "elem_10_mm":    h10,
        "elem_35":       elem_35,
        "elem_35_pos":   fwd_35_a,
        "elem_35_end":   fwd_35_b,
        "elem_35_mm":    h35,
        "spacer":        spacer,
        "gc":            gc,
        "reasons":       reasons,
        "target":        target,
        "target_pos":    fwd_t_a,
        "target_end":    fwd_t_b,
        "suggestion":    suggestion,
    }


# REPORTING

def write_report(path, name, seq, profile, hotspots, findings,
                 threshold, sigma_name, both_strands, strand, seq_len):
    """Write the plain-English report a lab user would actually read."""
    sigma_cfg   = SIGMA_FACTORS[sigma_name]
    strand_word = "reverse" if strand == "-" else "forward"
    both_note   = " (both scanned; see the other report)" if both_strands else ""
    ctx_label   = ("reverse strand, 5'->3'" if strand == "-"
                   else "as supplied")

    L = []
    L.append("=" * 72)
    L.append("CRYPTIC PROMOTER SCAN REPORT")
    L.append("=" * 72)
    L.append("")
    L.append(f"Sequence        : {name}")
    L.append(f"Length          : {seq_len:,} bp")
    L.append(f"Sigma factor    : sigma-{sigma_name} "
             f"({sigma_cfg['description']})")
    L.append(f"Strand scanned  : {strand_word}{both_note}")
    L.append(f"Positions scored: {len(profile):,}")
    L.append(f"Threshold       : {threshold}")
    L.append(f"Tool version    : {__version__}")
    L.append(f"Date            : {time.strftime('%Y-%m-%d %H:%M')}")
    L.append("")
    L.append("All positions below are coordinates in the sequence you")
    L.append("supplied, counting from 0.")
    if strand == "-":
        L.append("Reverse-strand hits have been converted back to your")
        L.append("coordinates, so they can be used directly.")
    L.append("")

    L.append("-" * 72)
    L.append("SUMMARY")
    L.append("-" * 72)
    L.append("")

    if hotspots.empty:
        L.append("No regions exceeded the risk threshold.")
        L.append("")
        L.append(f"The highest score anywhere in the sequence was "
                 f"{profile['risk_score'].max():.3f}, below the threshold "
                 f"of {threshold}.")
        L.append("")
        L.append("This construct shows no strong evidence of cryptic "
                 "sigma-70 promoter activity.")
    else:
        counts = band_counts(hotspots)

        L.append(f"{len(hotspots)} region(s) flagged:")
        L.append("")
        for band in BAND_ORDER:
            if band in counts:
                L.append(f"    {counts[band]:>3}  {band}")
        L.append("")
        if counts.get("HIGH"):
            L.append("Regions marked HIGH are very likely to drive unintended")
            L.append("transcription and should be recoded before synthesis.")

    L.append("")

    if findings:
        L.append("-" * 72)
        L.append("FLAGGED REGIONS")
        L.append("-" * 72)

        if len(findings) < len(hotspots):
            L.append("")
            L.append(f"Showing the {len(findings)} highest-scoring of "
                     f"{len(hotspots)} regions. The full list is in the "
                     f"hotspots CSV.")

        for n, f in enumerate(findings, 1):
            L.append("")
            L.append(f"[{n}]  Position {f['position']:,}     "
                     f"risk {f['score']:.3f}     {f['band']}")
            L.append("")
            L.append(f"     {f['band_desc']}")
            L.append("")
            L.append("     Why this region was flagged:")
            for r in f["reasons"]:
                L.append(f"       - {r}")
            L.append("")
            L.append("     Promoter architecture found:")
            mm35 = "mismatch" if f["elem_35_mm"] == 1 else "mismatches"
            mm10 = "mismatch" if f["elem_10_mm"] == 1 else "mismatches"
            L.append(f"       -35 element   {f['elem_35']}  "
                     f"at positions {f['elem_35_pos']:,}-{f['elem_35_end']:,}  "
                     f"({f['elem_35_mm']} {mm35} from "
                     f"{sigma_cfg['consensus_35']})")
            L.append(f"       spacer        {f['spacer']} bp  "
                     f"(optimal is 17)")
            L.append(f"       -10 element   {f['elem_10']}  "
                     f"at positions {f['elem_10_pos']:,}-{f['elem_10_end']:,}  "
                     f"({f['elem_10_mm']} {mm10} from "
                     f"{sigma_cfg['consensus_10']})")
            L.append(f"       TSS           position {f['position']:,}")
            L.append("")
            L.append("     Suggested action:")
            for line in _wrap(f["suggestion"], 62):
                L.append(f"       {line}")
            L.append("")
            L.append(f"     Sequence context ({ctx_label}):")
            L.append(f"       {f['window']}")

    L.append("")
    L.append("-" * 72)
    L.append("NOTES")
    L.append("-" * 72)
    L.append("")
    L.append("Scores are the model's estimated probability that a given")
    L.append("position acts as a sigma-70 transcription start site. On")
    L.append("held-out test data the model achieved PR-AUC 0.947, with")
    L.append("precision 0.86 and recall 0.89 at the default threshold.")
    L.append("")
    L.append("Predictions are computational and should be confirmed")
    L.append("experimentally (for example by RNA-seq or RT-PCR) before")
    L.append("firm conclusions are drawn.")
    L.append("")

    with open(path, "w") as fh:
        fh.write("\n".join(L))


def _wrap(text, width):
    """Simple word wrap, avoiding a textwrap import for one use."""
    words, lines, current = text.split(), [], ""
    for w in words:
        if len(current) + len(w) + 1 > width:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}".strip()
    if current:
        lines.append(current)
    return lines


def plot_profile(profile, hotspots, threshold, name, outpath):
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(profile["position"], profile["risk_score"],
            color="#028090", lw=0.8)
    ax.fill_between(profile["position"], 0, profile["risk_score"],
                    color="#028090", alpha=0.15)
    ax.axhline(threshold, color="#e07b2a", ls="--", lw=1.2,
               label=f"Threshold ({threshold})")

    for _, hs in hotspots.head(15).iterrows():
        ax.axvspan(hs["start"], hs["end"], color="#e07b2a", alpha=0.18)
        ax.annotate(f"{hs['peak_position']:,}",
                    xy=(hs["peak_position"], hs["peak_score"]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=7.5, color="#b45309")

    ax.set_xlabel("Position in sequence (bp)")
    ax.set_ylabel("Cryptic promoter risk")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{name}   —   {len(hotspots)} region(s) above threshold",
                 fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


# DEMO
def make_demo_sequence(seed=42):
    """
    Build a test sequence with a canonical promoter at a known position.

    Used to confirm the tool is working: if it cannot find a promoter that
    was deliberately planted, something is wrong with the installation.
    """
    rng = np.random.default_rng(seed)
    bg  = rng.choice(list("ATGC"), size=1500, p=[0.2, 0.2, 0.3, 0.3])
    seq = list("".join(bg))

    promoter = ("TTGACA"                # -35
                "GCTAGCTAGCTAGCTAG"     # 17 bp spacer
                "TATAAT"                # -10
                "ATATTAAGA")            # discriminator, ending at the TSS
    at = 700
    seq[at:at + len(promoter)] = list(promoter)
    return "".join(seq), at + len(promoter) - 1


# MODEL LOADING

def resolve_model_paths(args, sigma):
    """
    Find the model and its feature list.

    Looked for next to this script first, then in the working directory, so
    the tool works when copied elsewhere as long as the two .pkl files travel
    with it. feature_cols.pkl is always taken from beside the model, since a
    model and a feature list from different training runs must never be mixed.
    """
    if args.model:
        model_path = os.path.abspath(args.model)
        if not os.path.exists(model_path):
            print(f"\nERROR: model file '{args.model}' not found.\n")
            sys.exit(1)
    else:
        candidates = []
        for d in (HERE, os.getcwd()):
            c = os.path.join(d, sigma["default_model"])
            if c not in candidates:
                candidates.append(c)
        model_path = next((c for c in candidates if os.path.exists(c)), None)

        if model_path is None:
            print(f"\nERROR: model file '{sigma['default_model']}' not "
                  f"found.\n")
            print("  Looked in:")
            for c in candidates:
                print(f"      {c}")
            print("\n  This file is produced by step7_model_training.py.")
            print("  Either copy it (and feature_cols.pkl) next to this")
            print("  script, or point at it with:")
            print("      --model /path/to/best_model.pkl\n")
            sys.exit(1)

    cols_path = os.path.join(os.path.dirname(model_path), "feature_cols.pkl")
    if not os.path.exists(cols_path):
        print(f"\nERROR: feature_cols.pkl not found.\n")
        print(f"  It must sit alongside the model file:")
        print(f"      {cols_path}\n")
        print("  Both files come from the same training run and cannot be")
        print("  mixed between runs.\n")
        sys.exit(1)

    return model_path, cols_path



# MAIN

def main():
    p = argparse.ArgumentParser(
        prog="cryptic_tss.py",
        description="Find cryptic sigma-70 promoters in synthetic DNA "
                    "before you build it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples
--------
  Scan a construct                 python cryptic_tss.py myplasmid.fasta
  Check both strands               python cryptic_tss.py myplasmid.fasta --both-strands
  Catch weaker signals             python cryptic_tss.py myplasmid.fasta --sensitive
  Paste a sequence directly        python cryptic_tss.py ATGCGTAAGCTTGACA...
  Save results elsewhere           python cryptic_tss.py myplasmid.fasta -o results/
  Check the tool is working        python cryptic_tss.py --demo

FASTA files, plain-sequence files, and pasted sequences are all accepted.
All reported positions are coordinates in the sequence you supplied.
        """)

    p.add_argument("input", nargs="?",
                   help="FASTA file, sequence file, or a pasted DNA sequence")
    p.add_argument("-o", "--output", default=".", metavar="DIR",
                   help="where to save results (default: here)")
    p.add_argument("-t", "--threshold", type=float, default=0.5, metavar="N",
                   help="flag regions scoring above this, 0-1 (default: 0.5)")
    p.add_argument("--sensitive", action="store_true",
                   help="lower the threshold to 0.3 to catch weaker signals")
    p.add_argument("--strict", action="store_true",
                   help="raise the threshold to 0.8 for high-confidence only")
    p.add_argument("--both-strands", action="store_true",
                   help="also scan the reverse complement")
    p.add_argument("--step", type=int, default=None, metavar="N",
                   help="scan every N bases (default: automatic)")
    p.add_argument("--max-regions", type=int, default=20, metavar="N",
                   help="how many regions to describe in detail "
                        "(default: 20)")
    p.add_argument("--sigma", default="70", choices=list(SIGMA_FACTORS),
                   help="sigma factor to scan for (default: 70)")
    p.add_argument("--model", default=None, metavar="FILE",
                   help="use a specific model file")
    p.add_argument("--quiet", action="store_true",
                   help="print less")
    p.add_argument("--demo", action="store_true",
                   help="run on a test sequence to check the tool works")
    p.add_argument("--version", action="version",
                   version=f"cryptic_tss.py {__version__}")

    args = p.parse_args()

    if not args.demo and not args.input:
        p.print_help()
        print("\nERROR: give a sequence file, or use --demo to test the tool.")
        sys.exit(1)

    if not 0 <= args.threshold <= 1:
        sys.exit(f"ERROR: --threshold must be between 0 and 1 "
                 f"(got {args.threshold}).")

    if args.step is not None and args.step < 1:
        sys.exit(f"ERROR: --step must be 1 or more (got {args.step}).")

    # --sensitive and --strict are friendlier ways to set the threshold
    if args.sensitive and args.strict:
        sys.exit("ERROR: --sensitive and --strict cannot be used together.")

    threshold = args.threshold
    if args.sensitive:
        threshold = 0.3
    elif args.strict:
        threshold = 0.8

    sigma = SIGMA_FACTORS[args.sigma]
    quiet = args.quiet

    if not quiet:
        print()
        print("=" * 72)
        print(f"  CRYPTIC PROMOTER SCANNER  v{__version__}")
        print("=" * 72)

    # Load the model
    model_path, cols_path = resolve_model_paths(args, sigma)
    model        = joblib.load(model_path)
    trained_cols = list(joblib.load(cols_path))

    # Features are passed to the model as a bare array, so a mismatch in
    # ordering would silently feed the wrong values to the wrong inputs.
    if trained_cols != FEATURE_ORDER:
        print(f"\nERROR: this model does not match the scanner.\n")
        print(f"  Model file     : {model_path}")
        print(f"  Model expects  : {len(trained_cols)} features")
        print(f"  Scanner builds : {len(FEATURE_ORDER)} features\n")
        if len(trained_cols) != len(FEATURE_ORDER):
            print("  Retrain with FEATURES_FILE = 'features_with_motifs.csv' "
                  "in step7_model_training.py.\n")
        else:
            diff = [c for c in FEATURE_ORDER if c not in trained_cols][:5]
            if diff:
                print(f"  Not in the model: {', '.join(diff)}")
            print("  The feature list and the model are from different "
                  "runs.\n")
        sys.exit(1)

    # Collect input
    if args.demo:
        demo_seq, true_tss = make_demo_sequence()
        records = [("demo_sequence", demo_seq)]
        if not quiet:
            print(f"\nDemo mode — a canonical promoter has been planted at "
                  f"position {true_tss}")
            print("in an otherwise random 1,500 bp sequence. The scanner "
                  "should find it.")
    else:
        records = load_input(args.input)

    os.makedirs(args.output, exist_ok=True)

    # Scan
    for name, raw in records:
        seq     = clean_sequence(raw)
        seq_len = len(seq)

        if not quiet:
            print()
            print("-" * 72)
            print(f"  {name}   ({seq_len:,} bp)")
            print("-" * 72)

        if seq_len < WINDOW_SIZE:
            print(f"  Skipped — sequences must be at least "
                  f"{WINDOW_SIZE} bp.")
            continue

        step, auto = choose_step_size(seq_len, args.step)
        if auto and not quiet:
            print(f"  Long sequence: scanning every {step} bases to keep "
                  f"this quick.")
            print(f"  Use --step 1 for full resolution.")

        strands = [("", "+", seq)]
        if args.both_strands:
            strands = [("_forward", "+", seq),
                       ("_reverse", "-", reverse_complement(seq))]

        for suffix, strand, strand_seq in strands:
            if args.both_strands and not quiet:
                print(f"\n  {'Forward' if strand == '+' else 'Reverse'} "
                      f"strand")

            profile  = scan_sequence(strand_seq, model, sigma,
                                     step=step, show_progress=not quiet)
            hotspots = find_hotspots(profile, threshold)

            findings = []
            for _, hs in hotspots.head(args.max_regions).iterrows():
                d = describe_hotspot(strand_seq, int(hs["peak_position"]),
                                     float(hs["peak_score"]), sigma,
                                     seq_len, strand)
                if d:
                    findings.append(d)

            # Written outputs carry the user's own coordinates. The scanned
            # position is kept alongside so a result can be traced back.
            out_profile = profile.copy()
            out_profile["strand"] = strand
            out_profile["scan_position"] = out_profile["position"]
            out_profile["position"] = [
                to_forward(pos, seq_len, strand)
                for pos in out_profile["scan_position"]]
            out_profile = out_profile[["position", "risk_score",
                                       "strand", "scan_position"]]

            out_hot = hotspots.copy()
            if not out_hot.empty:
                spans = [to_forward_span(a, b, seq_len, strand)
                         for a, b in zip(out_hot["start"], out_hot["end"])]
                out_hot["start"] = [s[0] for s in spans]
                out_hot["end"]   = [s[1] for s in spans]
                out_hot["peak_position"] = [
                    to_forward(pos, seq_len, strand)
                    for pos in out_hot["peak_position"]]
                out_hot["strand"] = strand

            stem = os.path.join(args.output, f"{name}{suffix}")
            out_profile.to_csv(f"{stem}_risk_profile.csv", index=False)
            out_hot.to_csv(f"{stem}_hotspots.csv", index=False)
            plot_profile(profile, hotspots, threshold,
                         f"{name}{suffix}", f"{stem}_risk_profile.png")
            write_report(f"{stem}_report.txt", f"{name}{suffix}", strand_seq,
                         profile, hotspots, findings, threshold,
                         args.sigma, args.both_strands, strand, seq_len)

            # Console summary: the primary output
            print()
            if hotspots.empty:
                print(f"  No regions above {threshold}.")
                print(f"  Highest score anywhere: "
                      f"{profile['risk_score'].max():.3f}")
                print()
                print("  This construct looks clear of strong cryptic "
                      "sigma-70 promoters.")
            else:
                counts = band_counts(hotspots)
                parts  = [f"{counts[b]} {b}" for b in BAND_ORDER
                          if b in counts]
                print(f"  {len(hotspots)} region(s) flagged: "
                      f"{', '.join(parts)}")
                print()

                for n, f in enumerate(findings[:5], 1):
                    marker = "!!" if f["band"] == "HIGH" else "  "
                    print(f"  {marker} [{n}] position {f['position']:,}  "
                          f"risk {f['score']:.3f}  ({f['band']})")
                    print(f"        {f['reasons'][0]}")
                    print(f"        -35 {f['elem_35']} .. "
                          f"{f['spacer']} bp .. "
                          f"-10 {f['elem_10']}")
                    print(f"        -> recode positions "
                          f"{f['target_pos']:,}-{f['target_end']:,} "
                          f"({f['target']})")
                    print()

                if len(findings) > 5:
                    print(f"  ...and {len(hotspots) - 5} more. "
                          f"See the report for the top {len(findings)}.")
                    print()

            # Demo check
            if args.demo:
                near = profile[(profile["position"] >= true_tss - 15) &
                               (profile["position"] <= true_tss + 15)]
                print("  " + "=" * 60)
                if not near.empty:
                    best = near.loc[near["risk_score"].idxmax()]
                    print(f"  DEMO CHECK")
                    print(f"    Promoter planted at  : {true_tss}")
                    print(f"    Best score nearby    : "
                          f"{best['risk_score']:.3f} "
                          f"at position {int(best['position'])}")
                    if best["risk_score"] >= threshold:
                        print(f"    Result               : PASS — "
                              f"the scanner found it")
                    else:
                        print(f"    Result               : FAIL — "
                              f"scored below threshold")
                        print(f"    Something may be wrong with the model "
                              f"or feature extraction.")
                else:
                    print(f"  DEMO CHECK: FAIL — no positions scored "
                          f"near {true_tss}")
                print("  " + "=" * 60)

            print(f"\n  Results saved:")
            print(f"    {stem}_report.txt          <- start here")
            print(f"    {stem}_risk_profile.png")
            print(f"    {stem}_risk_profile.csv")
            print(f"    {stem}_hotspots.csv")

    if not quiet:
        print()
        print("=" * 72)
        print()


if __name__ == "__main__":
    main()