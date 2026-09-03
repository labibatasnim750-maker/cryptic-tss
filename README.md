# cryptic_tss

Find cryptic sigma-70 promoters in synthetic DNA before you build it.

Short, degenerate sigma factor recognition motifs occur by chance roughly
every few hundred base pairs of any sufficiently long sequence. In native
DNA most are suppressed by chromosomal context; in synthetic constructs
that context is absent, so latent promoter-like motifs are free to initiate
unwanted transcription. This tool scans a sequence and reports where that
is likely, so problem regions can be recoded before synthesis.

---

## Install

```bash
git clone https://github.com/labibatasnim750-maker/cryptic-tss.git
cd cryptic-tss
pip install -r requirements.txt
```

On macOS, `xgboost` also needs the OpenMP runtime, which pip cannot supply:

```bash
brew install libomp
```

Check it works:

```bash
python cryptic_tss.py --demo
```

This plants a canonical promoter at a known position in 1,500 bp of random
sequence and confirms the scanner finds it. You should see `Result: PASS`.

---

## Use

```bash
python cryptic_tss.py myconstruct.fasta
```

FASTA files, plain sequence files, and sequences pasted straight onto the
command line are all accepted.

| Option | Effect |
|---|---|
| `--both-strands` | also scan the reverse complement |
| `--sensitive` | lower the threshold to 0.3 |
| `--strict` | raise it to 0.8 |
| `-o DIR` | write results elsewhere |
| `-t N` | set the threshold directly, 0–1 |
| `--step N` | scan every N bases (default: automatic) |
| `--max-regions N` | regions described in detail (default: 20) |
| `--demo` | run the self-test |

**All reported positions are coordinates in the sequence you supplied,
counting from 0.** Reverse-strand hits are converted back for you.

---

## Output

| File | Contents |
|---|---|
| `<name>_report.txt` | plain-English findings and suggested edits — **start here** |
| `<name>_risk_profile.png` | risk plotted along the sequence |
| `<name>_risk_profile.csv` | risk score at every scanned position |
| `<name>_hotspots.csv` | flagged regions only |

For each flagged region the report gives the promoter architecture found
(−35 element, spacer, −10 element, TSS), why it was flagged, and which six
bases to recode. The suggestion targets whichever consensus element is the
stronger match.

---

## The model

Gradient-boosted trees (XGBoost) trained on 5,873 labelled 81 bp windows
from RegulonDB v12.0 — 1,955 experimentally confirmed sigma-70 promoters
and 3,918 negatives sampled from protein-coding regions.

Each window is described by 341 features: GC content, position weight matrix
scores against the Shultzaberger et al. (2007) matrices, 3-mer and 4-mer
frequencies, and 18 explicit motif features covering consensus edit
distance, element position, spacer geometry, the extended −10 motif, and
discriminator AT-richness.

### Performance

Held-out test set of 1,175 sequences (391 positive, 784 negative), used in
neither training nor model selection.

| Method | PR-AUC | ROC-AUC | F1 | MCC | Precision | Recall |
|---|---|---|---|---|---|---|
| **This model** | **0.9466** | **0.9658** | **0.8726** | **0.8080** | 0.861 | 0.885 |
| Consensus match | 0.7319 | 0.8784 | 0.7378 | 0.5929 | 0.642 | 0.867 |
| PWM scanner | 0.6756 | 0.8149 | 0.6527 | 0.4594 | 0.598 | 0.719 |
| Random baseline | 0.3328 | 0.5000 | — | 0.0000 | — | — |

The random baseline is 0.333 rather than 0.5 because the dataset is 2:1
negative to positive.

Against consensus matching the model recovers 7 additional true promoters
but eliminates 133 false positives. Its advantage lies in discriminating
motif *context* rather than motif *presence*.

---

## Limitations

- **Sigma-70 only.** Promoters recognised by sigma-32, -38, -54 or -28 are
  not detected.
- **Trained on *E. coli* K-12.** Generalisation elsewhere is untested.
- **Promoters with a weak −35 element may be missed** — the model treats
  −35 recognition as approximately necessary.
- **Predictions are computational.** Confirm by RNA-seq or RT-PCR.
- Sequences over 20 kb are scanned with a step of 3–5 bases; `--step 1`
  forces full resolution.
- `best_model.pkl` was saved with XGBoost 3.3.0 and scikit-learn 1.9.0 and
  may not load under substantially older versions.

---

## Repository contents

```
cryptic_tss.py       the scanner
best_model.pkl       trained XGBoost model
feature_cols.pkl     feature names and ordering, from the same training run
requirements.txt     dependencies
scripts/             analysis pipeline (dataset construction through SHAP)
```

`best_model.pkl` and `feature_cols.pkl` must come from the same training run
and sit in the same directory. The scanner verifies this at startup and
refuses to run on a mismatch.

---

## Citation

Zeba LT. *Machine Learning-Based Prediction of Cryptic Transcriptional Start
Sites in Synthetic DNA Constructs of Bacteria.* MSc Bioinformatics
dissertation, University of Bristol, 2026. Supervised by Prof. Thomas E.
Gorochowski.

### Key references

- Shultzaberger RK, Chen Z, Lewis KA, Schneider TD. Anatomy of *Escherichia
  coli* σ70 promoters. *Nucleic Acids Res.* 2007;35(3):771-88.
- Salgado H, et al. RegulonDB v12.0. *Nucleic Acids Res.* 2024;52(D1):D255-64.
- Browning DF, Busby SJW. Local and global regulation of transcription
  initiation in bacteria. *Nat Rev Microbiol.* 2016;14(10):638-50.
- Chen T, Guestrin C. XGBoost: a scalable tree boosting system. *KDD '16*;785-94.

