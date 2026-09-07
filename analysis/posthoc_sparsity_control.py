import os
from pathlib import Path
ROOT = Path(os.environ.get('SPARSE_CODING_ROOT', Path(__file__).resolve().parents[1]))
"""Exp2: Entropy-Matched Thresholding — post-hoc K-sparsification vs trained.
Tests whether a dense-trained model, when post-hoc thresholded to match trained
K-sparse sparsity levels, reproduces the same PS-NNR relationship.

Output: cache/cache_entropy_control.json
"""
import json, numpy as np, os, time, sys
import torch

from analysis_utils import (load_k_gradient_model, load_stringer_data,
    compute_ps_nnr_from_H, save_cache, K_VALS, SEEDS)

t0 = time.time()
CACHE_FILE = str(ROOT / "outputs" / "cache_entropy_control.json")
if os.path.exists(CACHE_FILE):
    print(f"Cache exists: {CACHE_FILE}")
    sys.exit(0)

print("Loading data...")
X, cat_known, _ = load_stringer_data()

# Use K=256 models (fully dense bottleneck) as the dense source
K_SOURCE = 256
all_results = []

# Also load trained K-sparse baseline from main cache for comparison
baseline = json.load(open(str(ROOT / "outputs" / "k_gradient_all.json")))

for seed in SEEDS:
    model = load_k_gradient_model(K_SOURCE, seed)
    with torch.no_grad():
        logits, h_sparse, h2 = model(X)
        H2 = h2.cpu().numpy()  # pre-mask ReLU, shape (n_images, 256)

    # Baseline: full K=256 (no thresholding)
    ps_base, nnr_base, _ = compute_ps_nnr_from_H(H2, cat_known)

    # Post-hoc threshold to each target K
    for target_k in [128, 64, 32, 16, 8, 4, 2]:
        H_thresh = np.zeros_like(H2)
        for i in range(H2.shape[0]):
            row = H2[i]
            # Keep top target_k values, zero rest
            if target_k >= len(row):
                H_thresh[i] = row
            else:
                thr = np.partition(row, -target_k)[-target_k]
                H_thresh[i] = row * (row >= thr)
        ps_th, nnr_th, _ = compute_ps_nnr_from_H(H_thresh, cat_known)

        all_results.append({
            "seed": seed, "target_k": target_k,
            "ps": float(ps_th), "nnr": float(nnr_th),
            "sparsity": float((H_thresh == 0).mean()),
        })

    print(f"  seed={seed}: base PS={ps_base:.3f} NNR={nnr_base:.4f}")

# Aggregate across seeds
from collections import defaultdict
k_sum = defaultdict(lambda: {"ps": [], "nnr": []})
for r in all_results:
    k_sum[r["target_k"]]["ps"].append(r["ps"])
    k_sum[r["target_k"]]["nnr"].append(r["nnr"])

# Trained baseline per-K
trained_sum = defaultdict(lambda: {"ps": [], "nnr": []})
for r in baseline:
    trained_sum[r["k"]]["ps"].append(r["ps"])
    trained_sum[r["k"]]["nnr"].append(r["nnr"])

from scipy.stats import spearmanr
# Post-hoc: compute r across target K means
k_ordered = sorted(k_sum.keys())
ph_ps_means = [np.mean(k_sum[k]["ps"]) for k in k_ordered]
ph_nnr_means = [np.mean(k_sum[k]["nnr"]) for k in k_ordered]
r_ph, p_ph = spearmanr(ph_ps_means, ph_nnr_means)

# Trained: compute r across K means
trained_k = sorted(trained_sum.keys())
tr_ps_means = [np.mean(trained_sum[k]["ps"]) for k in trained_k]
tr_nnr_means = [np.mean(trained_sum[k]["nnr"]) for k in trained_k]
r_tr, p_tr = spearmanr(tr_ps_means, tr_nnr_means)

print(f"\n  Post-hoc threshold: r(PS,NNR) = {r_ph:+.4f}, p={p_ph:.4f}")
print(f"  Trained from scratch: r(PS,NNR) = {r_tr:+.4f}, p={p_tr:.4f}")
print(f"  Delta r = {abs(r_tr)-abs(r_ph):+.4f}")

save_cache({
    "metadata": {"experiment": "entropy_control", "source_k": K_SOURCE,
                 "seeds": SEEDS, "date": time.strftime("%Y-%m-%d %H:%M:%S")},
    "results": all_results,
    "per_k_posthoc": {str(k): {"ps_mean": float(np.mean(v["ps"])), "nnr_mean": float(np.mean(v["nnr"]))}
                      for k, v in sorted(k_sum.items())},
    "per_k_trained": {str(k): {"ps_mean": float(np.mean(v["ps"])), "nnr_mean": float(np.mean(v["nnr"]))}
                      for k, v in sorted(trained_sum.items())},
    "summary": {
        "r_posthoc": float(r_ph), "p_posthoc": float(p_ph),
        "r_trained": float(r_tr), "p_trained": float(p_tr),
        "delta_r": float(abs(r_tr) - abs(r_ph)),
        "falsified": abs(r_tr) > abs(r_ph) + 0.1,
    }
}, CACHE_FILE)
print(f"Time: {time.time()-t0:.1f}s")
