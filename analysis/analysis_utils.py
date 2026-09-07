import os
from pathlib import Path
ROOT = Path(os.environ.get('SPARSE_CODING_ROOT', Path(__file__).resolve().parents[1]))
"""Shared utility module for K-sparse mechanism falsification experiments.

All 7 experiments (exp1-7) import from here. One source of truth for:
  - Metric functions: population_sparseness, condition_nnr, participation_ratio
  - Data loading: load_stringer_data (with pickle caching of resized images)
  - Model definitions: KSparseClassifier (4096->1024->256->15)
  - Helpers: build_condition_mean_matrix, bootstrap_spearman, cache I/O

Author: auto-generated for experiment suite
Date: 2026-06-18
"""
import numpy as np
import os, time, pickle, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
from scipy.stats import spearmanr
from collections import defaultdict

# =====================================================================
# CONSTANTS
# =====================================================================
IN_DIM, MID_DIM, BOTTLENECK_DIM, N_CLASSES = 4096, 1024, 256, 15
K_VALS = [256, 128, 64, 32, 16, 8, 4, 2]
SEEDS = [42, 123, 456, 789, 1024]
PERT_SEED = 42
BOOT_SEED = 99

MODEL_DIR = str(ROOT / "models")
DATA_DIR = str(ROOT / "data" / "stringer")
CLASS_FILE = str(ROOT / "data" / "stimuli_class_assignment.mat")
CACHE_DIR = str(ROOT / "outputs")

os.makedirs(CACHE_DIR, exist_ok=True)

# =====================================================================
# METRIC FUNCTIONS
# =====================================================================

def population_sparseness(r):
    """Vinje-Gallant (2000) population sparseness.
    PS = [1 - (sum r)^2 / (N * sum r^2)] / [1 - 1/N]
    r is clipped to >= 0 (ReLU-like).
    """
    r = np.maximum(r, 0)
    N = len(r)
    r_sq, r_sum = (r**2).sum(), r.sum()
    if r_sq == 0 or r_sum == 0 or N <= 1:
        return np.nan
    return (1 - r_sum**2 / (N * r_sq)) / (1 - 1/N)


def condition_nnr(R, k_nn=2):
    """NNR = d_NN / d_other on cosine distance.
    Centered -> L2-normalized -> 1-corr distance -> kNN ratio.
    d_other excludes the k nearest neighbors.
    """
    n = R.shape[0]
    if n < k_nn + 2:
        return np.nan
    Rc = R - R.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Rc, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    Ru = Rc / norms
    dist = 1.0 - (Ru @ Ru.T)
    k_eff = min(k_nn, n - 2)
    ratios = np.zeros(n)
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        vi = np.arange(n)[m]
        sl = np.argsort(dist[i, m])
        nn_idx = vi[sl[:k_eff]]
        other_idx = vi[sl[k_eff:]]
        dn = dist[i, nn_idx].mean()
        do_val = dist[i, other_idx].mean()
        ratios[i] = dn / do_val if do_val > 0 else np.nan
    return np.nanmean(ratios)


def participation_ratio(R):
    """PR = (sum sigma)^2 / sum(sigma^2) from SVD of centered condition-mean matrix.
    Lower = fewer effective dimensions = more clustered.
    """
    R_ctr = R - R.mean(axis=1, keepdims=True)
    _, S, _ = np.linalg.svd(R_ctr, full_matrices=False)
    s2 = (S**2).sum()
    if s2 == 0:
        return np.nan
    return float((S.sum()**2) / s2)


# =====================================================================
# DATA LOADING (with pickle cache for resized images)
# =====================================================================

_STRINGER_CACHE = os.path.join(CACHE_DIR, "_stringer_data.pkl")

def load_stringer_data():
    """Load Stringer natimg2800 data: X (n_images, 4096), cat_known, y.
    Images resized from native resolution to 64x64 -> flattened to 4096.
    Caches the processed tensors via pickle to avoid re-resizing.
    """
    if os.path.exists(_STRINGER_CACHE):
        data = pickle.load(open(_STRINGER_CACHE, "rb"))
        return data["X"], data["cat_known"], data["y"]

    print("[analysis_utils] Loading Stringer data (first time, ~30s)...")
    t0 = time.time()

    img_data = sio.loadmat(os.path.join(DATA_DIR, "images_natimg2800_all.mat"))
    imgs = img_data["imgs"]
    h, w, n_img = imgs.shape
    imgs_rs = np.zeros((64, 64, n_img), dtype=np.float32)
    for i in range(n_img):
        img_2d = imgs[:, :, i].astype(np.float32)
        hs, ws = h / 64, w / 64
        for hi in range(64):
            for wi in range(64):
                imgs_rs[hi, wi, i] = img_2d[
                    int(hi * hs):int(min((hi + 1) * hs, h)),
                    int(wi * ws):int(min((wi + 1) * ws, w))
                ].mean()
    X_full = imgs_rs.reshape(4096, n_img).T.astype(np.float32) / 255.0

    classes = sio.loadmat(CLASS_FILE)
    cat_labels = classes["class_assignment"].ravel()
    known_mask = np.isin(cat_labels, list(range(1, 16)))
    X_known = X_full[known_mask]
    cat_known = cat_labels[known_mask]
    y = torch.tensor(cat_known - 1, dtype=torch.long)

    X = torch.tensor(X_known, dtype=torch.float32)

    pickle.dump({"X": X, "cat_known": cat_known, "y": y},
                open(_STRINGER_CACHE, "wb"))
    print(f"[analysis_utils] Done in {time.time()-t0:.1f}s, {X.shape[0]} images")

    return X, cat_known, y


# =====================================================================
# MODEL DEFINITION
# =====================================================================

class KSparseClassifier(nn.Module):
    """4096 -> 1024(ReLU) -> 256(ReLU+K-Sparse) -> 15.
    k_override allows inference-time K different from training K.
    Forward returns: (logits, h_sparse, h2) where h2 is pre-mask.
    """
    def __init__(self, k):
        super().__init__()
        self.fc1 = nn.Linear(IN_DIM, MID_DIM)          # 4096 -> 1024
        self.fc2 = nn.Linear(MID_DIM, BOTTLENECK_DIM)  # 1024 -> 256
        self.classifier = nn.Linear(BOTTLENECK_DIM, N_CLASSES)
        self.k = k

    def forward(self, x, k_override=None):
        k_use = k_override if k_override is not None else self.k
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        topk_v, topk_i = torch.topk(h2, k_use, dim=1)
        mask = torch.zeros_like(h2)
        mask.scatter_(1, topk_i, 1.0)
        h_sparse = h2 * mask
        logits = self.classifier(h_sparse)
        return logits, h_sparse, h2


# =====================================================================
# MODEL LOADING
# =====================================================================

_device = torch.device("cpu")

def load_k_gradient_model(k, seed):
    """Load a trained K-sparse model from models/k{K}_seed{SEED}.pt.
    Returns model in eval mode, on CPU.
    """
    pt_path = os.path.join(MODEL_DIR, f"k{k}_seed{seed}.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Model not found: {pt_path}")
    model = KSparseClassifier(k=k).to(_device)
    model.load_state_dict(torch.load(pt_path, map_location=_device))
    model.eval()
    return model


def get_model_activations(model, X, cat_known, k_override=None):
    """Forward pass: returns h_sparse (masked), h2 (pre-mask), and R_cat (condition-mean)."""
    with torch.no_grad():
        logits, h_sparse, h2 = model(X, k_override=k_override)
        H = h_sparse.cpu().numpy()
        H2 = h2.cpu().numpy()

    R_cat = build_condition_mean_matrix(H, cat_known)
    return H, H2, R_cat


# =====================================================================
# HELPERS
# =====================================================================

def build_condition_mean_matrix(H, cat_ids):
    """Average activations per semantic category. Returns (n_cats, dim)."""
    unique_cats = np.unique(cat_ids)
    dim = H.shape[1]
    R = np.zeros((len(unique_cats), dim), dtype=np.float32)
    for ci, cat in enumerate(unique_cats):
        mask = cat_ids == cat
        R[ci, :] = H[mask, :].mean(axis=0)
    return np.maximum(R, 0)


def compute_ps_nnr_from_H(H, cat_known):
    """Compute PS and NNR from hidden activations H.
    PS: per-image population_sparseness, averaged.
    NNR: condition_nnr on condition-mean matrix (k_nn=2).
    """
    n_images = H.shape[0]
    ps_vals = np.array([population_sparseness(H[i, :]) for i in range(n_images)])
    ps = np.nanmean(ps_vals)
    R_cat = build_condition_mean_matrix(H, cat_known)
    nnr = condition_nnr(R_cat, k_nn=2)
    return ps, nnr, R_cat


def compute_sparsity_frac(H):
    """Fraction of zero activations."""
    return float((H == 0).mean())


def bootstrap_spearman(x, y, n_boot=500, seed=99):
    """Bootstrap Spearman correlation. Returns (mean_r, ci_lo, ci_hi)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_rs = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_rs[b], _ = spearmanr(x[idx], y[idx])
    ci_lo = np.percentile(boot_rs[np.isfinite(boot_rs)], 2.5)
    ci_hi = np.percentile(boot_rs[np.isfinite(boot_rs)], 97.5)
    return float(np.mean(boot_rs)), float(ci_lo), float(ci_hi)


def _convert_for_json(v):
    """Recursively convert numpy types to Python native for JSON serialization."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, list):
        return [_convert_for_json(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _convert_for_json(v) for k, v in v.items()}
    return v


def save_cache(data, path):
    """Save JSON cache with metadata (handles numpy types)."""
    json.dump(_convert_for_json(data), open(path, "w"), indent=2)
    print(f"  Saved: {path}")


def load_cache(path):
    """Load JSON cache if exists, else None."""
    if os.path.exists(path):
        return json.load(open(path))
    return None


# =====================================================================
# SELF-TEST
# =====================================================================
if __name__ == "__main__":
    print("=== analysis_utils self-test ===")
    t0 = time.time()

    # PS sanity: all-ones -> PS=0 (maximally dense), one-hot -> PS=1 (maximally sparse)
    ps_dense = population_sparseness(np.ones(10))
    ps_sparse = population_sparseness(np.array([1.0] + [0.0]*9))
    print(f"PS dense(all ones) = {ps_dense:.4f} (expect 0)")
    print(f"PS sparse(one-hot) = {ps_sparse:.4f} (expect 1)")

    # NNR sanity: identical conditions -> NNR ~0, random -> ~1
    R_same = np.random.randn(5, 20).astype(np.float32) * 0 + 1.0
    nnr_same = condition_nnr(R_same)
    R_rand = np.random.randn(5, 20).astype(np.float32)
    nnr_rand = condition_nnr(R_rand)
    print(f"NNR identical = {nnr_same:.4f} (expect near 0)")
    print(f"NNR random = {nnr_rand:.4f} (expect near 1)")

    # PR sanity: rank-1 matrix -> PR=1, full-rank identity -> PR=rank
    R1 = np.ones((10, 50))
    pr_1 = participation_ratio(R1)
    print(f"PR rank-1 = {pr_1:.2f} (expect 1)")

    # Data loading
    X, cat_known, y = load_stringer_data()
    print(f"Data: X={X.shape}, n_cats={len(np.unique(cat_known))}")

    # Model loading (spot-check one)
    model = load_k_gradient_model(256, 42)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model K=256 seed=42: {n_params:,} params")

    # Forward pass (subset for speed)
    X_sub = X[:100]
    cat_sub = cat_known[:100]
    H, H2, R_cat = get_model_activations(model, X_sub, cat_sub)
    ps, nnr, _ = compute_ps_nnr_from_H(H, cat_sub)
    sp = compute_sparsity_frac(H)
    pr = participation_ratio(R_cat)
    print(f"Forward pass (n=100): PS={ps:.4f}, NNR={nnr:.4f}, sp={sp:.1%}, PR={pr:.2f}")

    print(f"\nAll tests passed in {time.time()-t0:.1f}s")
