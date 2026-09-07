import os
from pathlib import Path
ROOT = Path(os.environ.get('SPARSE_CODING_ROOT', Path(__file__).resolve().parents[1]))
"""P6: separate support from magnitude (r_c = z_c . a_c).

Two counterfactuals on the K-sparse condition means:
  A) SHUFFLE MAGNITUDE: keep the active support (which units are top-K) fixed,
     but replace the activation values with random magnitudes.
  B) RANDOMIZE SUPPORT: keep the magnitude values (the top-K activations), but
     assign them to RANDOM units.

If magnitude encodes the category structure (P3's 'shared substrate + magnitude
encoding'), then shuffling magnitude should break clustering more than
randomizing support. If support is key, the opposite.

Outputs: revisedana/outputs/p6_support_vs_magnitude.json
"""
import sys, os, json, time
import numpy as np
import torch


import analysis_utils as au

t0 = time.time()
OUT = os.path.join(str(ROOT / "outputs"), "p6_support_vs_magnitude.json")

X, cat_known, y = au.load_stringer_data()
Ks = au.K_VALS
SEEDS = au.SEEDS
cats0 = cat_known - 1
NCATS = len(np.unique(cats0))


def nnr_of_R(R):
    R = np.maximum(R, 0)
    return au.condition_nnr(R, k_nn=2)


def counterfactual(R, k, mode, seed=0):
    """R: (C, dim) condition means. mode: 'mag' shuffle magnitude, 'sup' randomize support."""
    rng = np.random.default_rng(seed)
    C, D = R.shape
    R_out = R.copy()
    for c in range(C):
        row = R[c]
        topk = np.argsort(row)[-k:]                      # active support (units)
        vals = row[topk]                                 # magnitudes
        if mode == "mag":
            # keep support, randomize magnitudes (from the global magnitude distribution)
            new_vals = rng.choice(np.maximum(R, 0).ravel(), size=k, replace=True)
            R_out[c, topk] = new_vals
        elif mode == "sup":
            # keep magnitudes, randomize WHICH units carry them
            new_units = rng.choice(D, size=k, replace=False)
            R_out[c] = 0
            R_out[c, new_units] = vals
    return np.maximum(R_out, 0)


rows = []
for k in Ks:
    for s in SEEDS:
        m = au.load_k_gradient_model(k, s)
        with torch.no_grad():
            logits, h_sparse, h2 = m(X)
            H = h_sparse.cpu().numpy()
        R = au.build_condition_mean_matrix(H, cat_known)
        ps = float(np.nanmean([au.population_sparseness(H[i]) for i in range(len(H))]))
        nnr_orig = nnr_of_R(R)
        # multiple seeds for robustness
        nnr_mag, nnr_sup = [], []
        for seed in range(5):
            nnr_mag.append(nnr_of_R(counterfactual(R, k, "mag", seed)))
            nnr_sup.append(nnr_of_R(counterfactual(R, k, "sup", seed)))
        rows.append(dict(k=k, seed=s, ps=ps, nnr_orig=nnr_orig,
                         nnr_mag=float(np.mean(nnr_mag)), nnr_sup=float(np.mean(nnr_sup))))
        print(f"  K={k:3d} s={s:4d}: PS={ps:.3f} orig_NNR={nnr_orig:.3f} "
              f"shuffle_mag_NNR={np.mean(nnr_mag):.3f} random_sup_NNR={np.mean(nnr_sup):.3f}")

# ---- aggregate ----
from scipy.stats import spearmanr
ps = np.array([r["ps"] for r in rows])
o = np.array([r["nnr_orig"] for r in rows])
mg = np.array([r["nnr_mag"] for r in rows])
sp = np.array([r["nnr_sup"] for r in rows])
print("\n=== across 40 models ===")
print(f"  original NNR:      mean={o.mean():.3f}")
print(f"  shuffle-mag NNR:   mean={mg.mean():.3f}  (breakage={mg.mean()-o.mean():+.3f})")
print(f"  random-support NNR: mean={sp.mean():.3f}  (breakage={sp.mean()-o.mean():+.3f})")
for a, b, lab in [("ps", "o", "PS -> orig NNR"),
                  ("ps", "mg", "PS -> shuffle-mag NNR"),
                  ("ps", "sp", "PS -> random-support NNR"),
                  ("mg", "o", "shuffle-mag vs orig (clustering retained?)"),
                  ("sp", "o", "random-support vs orig")]:
    r, p = spearmanr(eval(a), eval(b))
    print(f"  {lab}: r={r:+.3f} p={p:.2e}")

json.dump(rows, open(OUT, "w"), indent=1)
print(f"Saved: {OUT}")
print(f"Time: {time.time()-t0:.1f}s")
