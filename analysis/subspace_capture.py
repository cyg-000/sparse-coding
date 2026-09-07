import os
from pathlib import Path
ROOT = Path(os.environ.get('SPARSE_CODING_ROOT', Path(__file__).resolve().parents[1]))
"""P7: subspace capture (SAE greedy method, Bhalla et al. 2026).

Question: is a high-PS population a COMPACT representation (few units reconstruct
the condition geometry) or a LOCAL TILING (many units needed)?

For each K-sparse model: greedily reconstruct the centered condition-mean
geometry using units (basis atoms), plotting fraction of units retained vs
variance explained.  Units are orthogonal basis vectors, so the greedy reduces
to sorting units by their variance contribution (column variance of the centered
condition means).

Metrics:
  frac_units_90 : fraction of units needed to explain 90% of geometry variance
                  (compact = small; tiling = large)
  gini_units    : concentration of geometry variance across units
  PCA comparison: fewest orthogonal directions (optimal lower bound)

Outputs: revisedana/outputs/p7_subspace_capture.json
"""
import sys, os, json, time
import numpy as np
import torch


import analysis_utils as au

t0 = time.time()
OUT = os.path.join(str(ROOT / "outputs"), "p7_subspace_capture.json")

X, cat_known, y = au.load_stringer_data()
Ks = au.K_VALS
SEEDS = au.SEEDS
cats0 = cat_known - 1
NCATS = len(np.unique(cats0))


def capture_metrics(H, var_threshold=0.90):
    R = au.build_condition_mean_matrix(H, cat_known)      # (C, dim)
    Xc = R - R.mean(0, keepdims=True)
    total_ss = float(np.sum(Xc**2))
    col_var = np.sum(Xc**2, axis=0)                        # per-unit variance contribution
    order = np.argsort(col_var)[::-1]                      # units by importance
    cum = np.cumsum(col_var[order]) / total_ss if total_ss > 0 else np.zeros(len(col_var))
    # fraction of units to reach var_threshold (interpolate)
    idx = np.searchsorted(cum, var_threshold)
    frac_90 = (idx + 1) / len(col_var)
    # gini of the variance distribution (concentration)
    x = np.sort(col_var)
    n = len(x); gini = (2 * np.sum(np.arange(1, n + 1) * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))
    # PCA: optimal orthogonal directions
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    s2 = s**2
    cum_pca = np.cumsum(s2) / s2.sum() if s2.sum() > 0 else np.zeros(len(s2))
    idx_pca = np.searchsorted(cum_pca, var_threshold)
    frac_pca_90 = (idx_pca + 1) / len(s2)
    return dict(frac_units_90=float(frac_90), gini_units=float(gini),
                frac_pca_90=float(frac_pca_90), n_active=float((col_var > 0).sum()))


rows = []
for k in Ks:
    for s in SEEDS:
        m = au.load_k_gradient_model(k, s)
        with torch.no_grad():
            logits, h_sparse, h2 = m(X)
            H = h_sparse.cpu().numpy()
        ps = float(np.nanmean([au.population_sparseness(H[i]) for i in range(len(H))]))
        cm = capture_metrics(H)
        cm.update(k=k, seed=s, ps=ps)
        rows.append(cm)
        print(f"  K={k:3d} s={s:4d}: PS={ps:.3f} frac_units_90={cm['frac_units_90']:.3f} "
              f"gini={cm['gini_units']:.3f} frac_pca_90={cm['frac_pca_90']:.3f}")

from scipy.stats import spearmanr
ps = np.array([r["ps"] for r in rows])
fu = np.array([r["frac_units_90"] for r in rows])
gn = np.array([r["gini_units"] for r in rows])
fp = np.array([r["frac_pca_90"] for r in rows])
print("\n=== across 40 models ===")
for a, b, lab in [("ps", "fu", "PS -> fraction units for 90% geometry (compact if <0)"),
                  ("ps", "gn", "PS -> gini of geometry variance (concentrated if >0)"),
                  ("ps", "fp", "PS -> PCA fraction (optimal lower bound)")]:
    r, p = spearmanr(eval(a), eval(b))
    print(f"  {lab}: r={r:+.3f} p={p:.2e}")

json.dump(rows, open(OUT, "w"), indent=1)
print(f"Saved: {OUT}")
print(f"Time: {time.time()-t0:.1f}s")
