import os
from pathlib import Path
ROOT = Path(os.environ.get('SPARSE_CODING_ROOT', Path(__file__).resolve().parents[1]))
"""P9 (targeted ablation): shared vs unique units -- is the shared substrate
causally responsible for the clustered geometry?

In the K-sparse model, for each condition its active support = top-K units.
  - "shared" units: active for MANY conditions (frequent hubs)
  - "unique" units: active for FEW conditions
Ablate (zero) the top fraction of shared vs unique vs random units, and measure
the resulting NNR (clustering) and PS-LCR distortion.  If deleting shared units
breaks geometry most, overlapping sparse supports causally produce clustering.

Outputs: revisedana/outputs/p9_shared_ablation.json
"""
import sys, os, json, time
import numpy as np
import torch


import analysis_utils as au

t0 = time.time()
OUT = os.path.join(str(ROOT / "outputs"), "p9_shared_ablation.json")

X, cat_known, y = au.load_stringer_data()
Ks = au.K_VALS
SEEDS = au.SEEDS
cats0 = cat_known - 1
NCATS = len(np.unique(cats0))


def hub_frequency(R, k):
    """How many conditions is each unit in the top-K of?"""
    topk = np.argsort(R, axis=1)[:, -k:]
    freq = np.zeros(R.shape[1], dtype=int)
    for c in range(R.shape[0]):
        freq[topk[c]] += 1
    return freq


def nnr_of(R):
    return au.condition_nnr(np.maximum(R, 0), k_nn=2)


def ablate(R, units):
    R2 = R.copy(); R2[:, units] = 0
    return R2


def geometry_retained(R0, R1):
    """R^2 of the original geometry variance recoverable after ablation."""
    denom = float(np.sum(R0**2))
    return 1.0 - float(np.sum((R0 - R1)**2)) / denom if denom > 0 else 0.0


rows = []
for k in Ks:
    for s in SEEDS:
        m = au.load_k_gradient_model(k, s)
        with torch.no_grad():
            logits, h_sparse, h2 = m(X)
            H = h_sparse.cpu().numpy()
        R = au.build_condition_mean_matrix(H, cat_known)
        ps = float(np.nanmean([au.population_sparseness(H[i]) for i in range(len(H))]))
        nnr_orig = nnr_of(R)
        freq = hub_frequency(R, k)
        active = np.where(freq > 0)[0]
        n_abl = max(1, int(0.25 * len(active)))
        shared = active[np.argsort(freq[active])[::-1][:n_abl]]
        unique = active[np.argsort(freq[active])[:n_abl]]
        rng = np.random.default_rng(0)
        rand = rng.choice(active, size=n_abl, replace=False)
        g_shared = geometry_retained(R, ablate(R, shared))
        g_unique = geometry_retained(R, ablate(R, unique))
        g_rand = geometry_retained(R, ablate(R, rand))
        rows.append(dict(k=k, seed=s, ps=ps, nnr_orig=nnr_orig,
                         geom_shared=g_shared, geom_unique=g_unique, geom_random=g_rand,
                         n_ablated=n_abl, n_active=len(active)))
        print(f"  K={k:3d} s={s:4d}: PS={ps:.3f} geom_retained: shared={g_shared:.3f} "
              f"unique={g_unique:.3f} random={g_rand:.3f}")

from scipy.stats import spearmanr
ps = np.array([r["ps"] for r in rows])
sh = np.array([r["geom_shared"] for r in rows])
un = np.array([r["geom_unique"] for r in rows])
rd = np.array([r["geom_random"] for r in rows])
print("\n=== across 40 models (geometry retained after ablating 25% of active units) ===")
print(f"  ablate SHARED:  {sh.mean():.3f}  (LOW = shared hubs carry geometry)")
print(f"  ablate UNIQUE:  {un.mean():.3f}")
print(f"  ablate RANDOM:  {rd.mean():.3f}")
for a, b, lab in [("sh", "un", "shared vs unique retained (shared LOSES more if <)"),
                  ("sh", "rd", "shared vs random retained"),
                  ("ps", "sh", "PS -> retained after ablate-shared")]:
    r, p = spearmanr(eval(a), eval(b))
    print(f"  {lab}: r={r:+.3f} p={p:.2e}")

json.dump(rows, open(OUT, "w"), indent=1)
print(f"Saved: {OUT}")
print(f"Time: {time.time()-t0:.1f}s")
