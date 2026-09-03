#!/usr/bin/env python3
"""Objective 6 — validate cryptic_tss.py against the Promoter Calculator
on a matched synonymous pair (sfGFP vs transcriptionally neutralised)."""

import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "felipe_out"

mine_sf = pd.read_csv(f"{OUT}/sfGFP_risk_profile.csv")
mine_no = pd.read_csv(f"{OUT}/noTXsfGFP_risk_profile.csv")
pc_sf   = pd.read_csv("sfGFP_preds.csv",     index_col=0)
pc_no   = pd.read_csv("noTXsfGFP_preds.csv", index_col=0)

df = (mine_sf[["position","risk_score"]].rename(columns={"risk_score":"risk_sf"})
      .merge(mine_no[["position","risk_score"]].rename(columns={"risk_score":"risk_no"}),
             on="position")
      .merge(pc_sf[["TSS","Tx_rate"]].rename(columns={"TSS":"position","Tx_rate":"tx_sf"}),
             on="position")
      .merge(pc_no[["TSS","Tx_rate"]].rename(columns={"TSS":"position","Tx_rate":"tx_no"}),
             on="position"))

print(f"\nPositions scored by both tools: {len(df)}")
assert len(df) > 300, "coordinate frames do not line up - stop and check"

# TEST A: do the two tools agree on where promoters are?
rho, p = stats.spearmanr(df.risk_sf, df.tx_sf)
print(f"\nTEST A - agreement on sfGFP")
print(f"  Spearman rho = {rho:.3f}  (p = {p:.2e})")
for n in (10, 25, 50):
    shared = len(set(df.nlargest(n,"tx_sf").position) &
                 set(df.nlargest(n,"risk_sf").position))
    print(f"  top {n:>3}: {shared:>2}/{n} shared ({shared/n:.0%})")

# TEST B: does the model SEE the neutralisation?
df["risk_drop"] = df.risk_sf - df.risk_no
df["tx_ratio"]  = df.tx_sf / df.tx_no

knocked   = df[(df.tx_sf > 1000) & (df.tx_ratio > 2)]
untouched = df[df.tx_ratio.between(0.95, 1.05)]

print(f"\nTEST B - detecting the neutralisation")
print(f"  neutralised sites: {len(knocked)}   untouched sites: {len(untouched)}")
print(f"  mean risk drop, neutralised : {knocked.risk_drop.mean():+.3f}")
print(f"  mean risk drop, untouched   : {untouched.risk_drop.mean():+.3f}")
u, pv = stats.mannwhitneyu(knocked.risk_drop, untouched.risk_drop,
                           alternative="greater")
print(f"  Mann-Whitney U, one-sided   : p = {pv:.2e}")

# TEST C: the known clusters, one by one
print(f"\nTEST C - sites the Promoter Calculator says were targeted")
for _, r in df.nlargest(15, "tx_sf").sort_values("position").iterrows():
    print(f"  pos {int(r.position):>3}  PC {r.tx_sf:>7.0f} -> {r.tx_no:>6.0f} "
          f"({r.tx_ratio:>5.1f}x)   model {r.risk_sf:.3f} -> {r.risk_no:.3f}   "
          f"{'FLAGGED' if r.risk_sf >= 0.5 else 'missed '}")

# Figure
fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
ax[0].plot(df.position, df.risk_sf, color="#c0392b", lw=.9, label="sfGFP")
ax[0].plot(df.position, df.risk_no, color="#2980b9", lw=.9, label="noTX")
ax[0].axhline(.5, color="grey", ls="--", lw=.8)
ax[0].set_ylabel("Model risk"); ax[0].legend(); ax[0].grid(alpha=.2)
ax[1].plot(df.position, df.tx_sf, color="#c0392b", lw=.9)
ax[1].plot(df.position, df.tx_no, color="#2980b9", lw=.9)
ax[1].set_yscale("log"); ax[1].set_ylabel("Promoter Calculator\nTx rate")
ax[1].grid(alpha=.2)
ax[2].plot(df.position, df.risk_drop, color="#8e44ad", lw=.9)
ax[2].axhline(0, color="grey", lw=.8)
ax[2].set_ylabel("Risk drop"); ax[2].set_xlabel("Position (bp)")
ax[2].grid(alpha=.2)
for _, r in knocked.iterrows():
    for a in ax: a.axvline(r.position, color="#f39c12", alpha=.12, lw=2)
plt.tight_layout(); plt.savefig("felipe_validation.png", dpi=150)
df.to_csv("felipe_comparison.csv", index=False)
print("\nWritten: felipe_validation.png, felipe_comparison.csv\n")
