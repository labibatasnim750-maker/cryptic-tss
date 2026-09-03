import pandas as pd

df = pd.read_csv("features_with_rna.csv")

rna_cols = [c for c in df.columns
            if c.startswith("rna_") or c.startswith("dna_energy")]

print("Correlation with gc_content:")
print("-" * 52)
for c in sorted(rna_cols,
                key=lambda x: -abs(df[x].corr(df["gc_content"]))):
    r = df[c].corr(df["gc_content"])
    flag = "  ← redundant" if abs(r) > 0.9 else ""
    print(f"{c:<34}{r:>+8.3f}{flag}")