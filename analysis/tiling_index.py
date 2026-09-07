import os
from pathlib import Path
ROOT = Path(os.environ.get('SPARSE_CODING_ROOT', Path(__file__).resolve().parents[1]))
"""P8: tiling index -- is a sparse population built from LOCAL (tiling) or
GLOBAL (compact) neurons?

For each K-sparse model, for each neuron compute its tuning breadth across the
15 conditions:
  - receptive-field width : number of conditions above a fraction of the max
  - activation entropy   : -sum p log p over normalized condition tuning
Broad tuning (high width/entropy) = compact/global substrate; narrow = local
tiling.

Tests: PS -> mean tuning breadth; tuning breadth -> NNR.
Outputs: revisedana/outputs/p8_tiling_index.json
"""
import sys, os, json, time
import numpy as np
import torch


import analysis_utils as au

t0 = time.time()
OUT = os.path.join(str(ROOT / "outputs"), "p8_tiling_index.json")

X, cat_known, y = au.load_stringer_data()
Ks = au.K_VALS
SEEDS = au.SEEDS
cats0 = cat_known - 1
NCATS = len(np.unique(cats0))
THRESH = 0.2  # fraction of max to count as "active"


def tuning_breadth(R):
    """Per-neuron tuning breadth across conditions. R: (C, dim)."""
    C = R.shape[0]
    Rt = R - R.min(0, keepdims=True)
    widths, entropies = [], []
    for j in range(R.shape[1]):
        t = Rt[:, j]
        if t.max() <= 0:
            widths.append(0); entropies.append(0.0); continue
        p = t / t.sum()
        ent = -np.sum(p * np.log(p + 1e-12)) / np.log(C)   # 0..1
        width = float((t >= THRESH * t.max()).sum()) / C   # fraction of conditions
        widths.append(width); entropies.append(ent)
    return float(np.mean(widths)), float(np.mean(entropies))


rows = []
for k in Ks:
    for s in SEEDS:
        m = au.load_k_gradient_model(k, s)
        with torch.no_grad():
            logits, h_sparse, h2 = m(X)
            H = h_sparse.cpu().numpy()
        R = au.build_condition_mean_matrix(H, cat_known)
        ps = float(np.nanmean([au.population_sparseness(H[i]) for i in range(len(H))]))
        nnr = au.condition_nnr(R, k_nn=2)
        width, ent = tuning_breadth(R)
        rows.append(dict(k=k, seed=s, ps=ps, nnr=nnr,
                         rf_width=width, entropy=ent))
        print(f"  K={k:3d} s={s:4d}: PS={ps:.3f} rf_width={width:.3f} entropy={ent:.3f}")

from scipy.stats import spearmanr
ps = np.array([r["ps"] for r in rows]); nnr = np.array([r["nnr"] for r in rows])
w = np.array([r["rf_width"] for r in rows]); e = np.array([r["entropy"] for r in rows])
print("\n=== across 40 models ===")
for a, b, lab in [("ps", "w", "PS -> receptive-field width (broad if +: compact/global)"),
                  ("ps", "e", "PS -> tuning entropy (broad if +)"),
                  ("w", "nnr", "rf width -> NNR (broad neurons -> clustered if -)")]:
    r, p = spearmanr(eval(a), eval(b))
    print(f"  {lab}: r={r:+.3f} p={p:.2e}")

json.dump(rows, open(OUT, "w"), indent=1)
print(f"Saved: {OUT}")
print(f"Time: {time.time()-t0:.1f}s")
