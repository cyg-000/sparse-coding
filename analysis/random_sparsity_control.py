import os
from pathlib import Path
ROOT = Path(os.environ.get('SPARSE_CODING_ROOT', Path(__file__).resolve().parents[1]))
"""Exp1: Random Sparsity Control — structured top-K vs random-K.
Tests whether random-K masking produces the same NNR as structured top-K.
If random-K does NOT reproduce the PS-NNR mapping, structured sparsity is necessary.

Output: cache/cache_random_sparsity.json
"""
import json, numpy as np, os, time, sys
import torch

from analysis_utils import (load_k_gradient_model, load_stringer_data,
    population_sparseness, compute_ps_nnr_from_H, save_cache,
    K_VALS, SEEDS, PERT_SEED, BOTTLENECK_DIM)

t0 = time.time()
CACHE_FILE = str(ROOT / "outputs" / "cache_random_sparsity.json")
if os.path.exists(CACHE_FILE):
    print(f"Cache exists: {CACHE_FILE}")
    sys.exit(0)

N_RANDOM_REPS = 5
rng = np.random.default_rng(PERT_SEED)

print("Loading data...")
X, cat_known, _ = load_stringer_data()
print(f"Data: {X.shape[0]} images, {len(np.unique(cat_known))} categories")

all_results = []

for k_val in K_VALS:
    for seed in SEEDS:
        model = load_k_gradient_model(k_val, seed)
        with torch.no_grad():
            logits, h_sparse, h2 = model(X)
            H = h_sparse.cpu().numpy()      # top-K masked
            H2 = h2.cpu().numpy()            # pre-mask ReLU

        # Standard top-K
        ps_topk, nnr_topk, _ = compute_ps_nnr_from_H(H, cat_known)

        # Random-K: per image, select k random indices from available active neurons
        ps_rand_vals = np.zeros(N_RANDOM_REPS)
        nnr_rand_vals = np.zeros(N_RANDOM_REPS)

        for rep in range(N_RANDOM_REPS):
            H_rand = np.zeros_like(H2)
            for i in range(H2.shape[0]):
                active = np.where(H2[i] > 0)[0]
                n_active = len(active)
                if n_active <= k_val:
                    H_rand[i, active] = H2[i, active]
                else:
                    idx = rng.choice(active, size=k_val, replace=False)
                    H_rand[i, idx] = H2[i, idx]
            ps_r, nnr_r, _ = compute_ps_nnr_from_H(H_rand, cat_known)
            ps_rand_vals[rep] = ps_r
            nnr_rand_vals[rep] = nnr_r

        all_results.append({
            "k": k_val, "seed": seed,
            "ps_topk": float(ps_topk), "nnr_topk": float(nnr_topk),
            "ps_rand_mean": float(np.mean(ps_rand_vals)),
            "ps_rand_std": float(np.std(ps_rand_vals)),
            "nnr_rand_mean": float(np.mean(nnr_rand_vals)),
            "nnr_rand_std": float(np.std(nnr_rand_vals)),
            "delta_nnr": float(np.mean(nnr_rand_vals) - nnr_topk),
        })

        print(f"  K={k_val:3d} S={seed:4d}  topK: PS={ps_topk:.3f} NNR={nnr_topk:.4f}  "
              f"rand NNR={np.mean(nnr_rand_vals):.4f}+/-{np.std(nnr_rand_vals):.4f}  "
              f"delta={all_results[-1]['delta_nnr']:+.4f}")

# Aggregate
from collections import defaultdict
from scipy.stats import spearmanr
k_sum = defaultdict(lambda: {"topk_nnr": [], "rand_nnr": []})
for r in all_results:
    k_sum[r["k"]]["topk_nnr"].append(r["nnr_topk"])
    k_sum[r["k"]]["rand_nnr"].append(r["nnr_rand_mean"])

all_ps_topk = np.array([r["ps_topk"] for r in all_results])
all_nnr_topk = np.array([r["nnr_topk"] for r in all_results])
all_nnr_rand = np.array([r["nnr_rand_mean"] for r in all_results])

r_topk, p_topk = spearmanr(all_ps_topk, all_nnr_topk)
r_rand, p_rand = spearmanr(all_ps_topk, all_nnr_rand)

print(f"\n  Top-K:    r(PS,NNR) = {r_topk:+.4f}, p={p_topk:.4f}")
print(f"  Random-K: r(PS,NNR) = {r_rand:+.4f}, p={p_rand:.4f}")
print(f"  Mean delta NNR = {np.mean([r['delta_nnr'] for r in all_results]):+.4f}")

save_cache({
    "metadata": {"experiment": "random_sparsity", "n_reps": N_RANDOM_REPS,
                 "perturbation_seed": PERT_SEED, "date": time.strftime("%Y-%m-%d %H:%M:%S")},
    "results": all_results,
    "summary": {
        "per_k": [{"k": k, "topk_nnr_mean": float(np.mean(v["topk_nnr"])),
                   "rand_nnr_mean": float(np.mean(v["rand_nnr"])),
                   "delta_nnr": float(np.mean(v["rand_nnr"])-np.mean(v["topk_nnr"]))}
                  for k, v in sorted(k_sum.items())],
        "spearman_topk": {"r": float(r_topk), "p": float(p_topk)},
        "spearman_random": {"r": float(r_rand), "p": float(p_rand)},
        "delta_r": float(abs(r_topk) - abs(r_rand)),
        "structured_necessary": abs(r_topk) > abs(r_rand) + 0.1,
    }
}, CACHE_FILE)
print(f"Time: {time.time()-t0:.1f}s")
