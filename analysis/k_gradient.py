import os
from pathlib import Path
ROOT = Path(os.environ.get('SPARSE_CODING_ROOT', Path(__file__).resolve().parents[1]))
r"""K-Sparse Gradient: Causal effect of PS on representational clustering.

Architecture: 4096 → 1024(GELU → K-Sparse) → 512(ReLU) → 15
K values: 1024(full), 512, 256, 128, 64, 32, 16, 8
5 fixed seeds each → 40 runs total
Silhouette on 15 semantic categories → PS×Sil curve overlay with neural data
"""
import numpy as np, time, os, random
import torch, torch.nn as nn, torch.nn.functional as F
import scipy.io as sio
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
t0 = time.time()

device = torch.device('cpu')
SEEDS = [42, 123, 456, 789, 1024]
K_VALUES = [1024, 512, 256, 128, 64, 32, 16, 8]
OUT_DIR = str(ROOT / "outputs" / "models")
os.makedirs(OUT_DIR, exist_ok=True)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# ========================a====================================
# 1. Load Stringer data (full 15 categories)
# ============================================================
print("Loading Stringer data...")
data_dir = str(ROOT / "data" / "stringer")
img_data = sio.loadmat(os.path.join(data_dir, "images_natimg2800_all.mat"))
imgs = img_data['imgs']; h, w, n_img = imgs.shape
imgs_rs = np.zeros((64, 64, n_img), dtype=np.float32)
for i in range(n_img):
    img_2d = imgs[:, :, i].astype(np.float32)
    hs, ws = h / 64, w / 64
    for hi in range(64):
        for wi in range(64):
            imgs_rs[hi, wi, i] = img_2d[int(hi*hs):int(min((hi+1)*hs,h)), int(wi*ws):int(min((wi+1)*ws,w))].mean()
X_full = imgs_rs.reshape(4096, n_img).T.astype(np.float32) / 255.0

class_file = str(ROOT / "data" / "stimuli_class_assignment.mat")
classes = sio.loadmat(class_file)
cat_labels = classes['class_assignment'].ravel()
cat_names = [str(n[0]) for n in classes['class_names'].ravel()]
KNOWN_CATS = list(range(1, 16))
known_mask = np.isin(cat_labels, KNOWN_CATS)

X_known = X_full[known_mask]
cat_known = cat_labels[known_mask]
print(f"Images: {X_known.shape[0]}, dim: 4096, classes: {len(KNOWN_CATS)}")

X = torch.tensor(X_known, dtype=torch.float32).to(device)
y = torch.tensor(cat_known - 1, dtype=torch.long).to(device)

# ============================================================
# 2. Model
# ============================================================
IN_DIM, MID_DIM, BOTTLENECK_DIM, N_CLASSES = 4096, 1024, 256, 15
N_EPOCHS, BATCH_SIZE, LR = 150, 128, 1e-3
K_VALUES = [256, 128, 64, 32, 16, 8, 4, 2]

class KSparseClassifier(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.fc1 = nn.Linear(IN_DIM, MID_DIM)         # 4096→1024
        self.fc2 = nn.Linear(MID_DIM, BOTTLENECK_DIM)  # 1024→256 bottleneck
        self.classifier = nn.Linear(BOTTLENECK_DIM, N_CLASSES)
        self.k = k

    def forward(self, x):
        h1 = F.relu(self.fc1(x))              # ReLU: 4096→1024
        h2 = F.relu(self.fc2(h1))             # ReLU: 1024→256 bottleneck
        topk_v, topk_i = torch.topk(h2, self.k, dim=1)
        mask = torch.zeros_like(h2); mask.scatter_(1, topk_i, 1.0)
        h_sparse = h2 * mask                   # K-Sparse on bottleneck
        logits = self.classifier(h_sparse)
        return logits, h_sparse, h2

# ============================================================
# 3. Metrics
# ============================================================
def population_sparseness(r):
    N = len(r); r = np.maximum(r, 0)
    r_sq, r_sum = (r**2).sum(), r.sum()
    if r_sq == 0 or r_sum == 0 or N <= 1: return np.nan
    return (1 - r_sum**2 / (N * r_sq)) / (1 - 1/N)

def condition_silhouette(R, k=2):
    n = R.shape[0]
    if n < 6: return np.nan
    Rc = R - R.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Rc, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    Ru = Rc / norms; corr = Ru @ Ru.T; dist = 1 - corr
    k_eff = min(k, n-2); ratios = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        valid_idx = np.arange(n)[mask]
        sorted_local = np.argsort(dist[i, mask])
        nn_idx = valid_idx[sorted_local[:k_eff]]
        other_idx = valid_idx[sorted_local[k_eff:]]
        d_nn = dist[i, nn_idx].mean()
        d_other = dist[i, other_idx].mean()
        ratios[i] = d_nn / d_other if d_other > 0 else np.nan
    return np.nanmean(ratios)

# ============================================================
# 4. Train all (K, seed) combinations
# ============================================================
all_results = []

for k_val in K_VALUES:
    for seed in SEEDS:
        set_seed(seed)
        model = KSparseClassifier(k=k_val).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        n = X.shape[0]

        for epoch in range(N_EPOCHS):
            model.train()
            total_loss, correct = 0, 0
            perm = torch.randperm(n)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]; xb, yb = X[idx], y[idx]
                opt.zero_grad()
                logits, h1, h2 = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                correct += (logits.argmax(1) == yb).sum().item()

        model.eval()
        with torch.no_grad():
            logits, H1, H2 = model(X)
            acc = (logits.argmax(1) == y).float().mean().item() * 100
            H1_np = H1.cpu().numpy()

        # PS on bottleneck (1024-dim GELU+K-Sparse)
        ps_vals = [population_sparseness(H1_np[i,:]) for i in range(H1_np.shape[0])]
        ps_mean = np.nanmean(ps_vals)
        sparsity = (H1_np == 0).mean()

        # Silhouette on 15 categories
        unique_cats = np.unique(cat_known)
        R_cat = np.zeros((len(unique_cats), BOTTLENECK_DIM), dtype=np.float32)
        for ci, cat in enumerate(unique_cats):
            mask = cat_known == cat
            R_cat[ci, :] = H1_np[mask, :].mean(axis=0)
        R_cat = np.maximum(R_cat, 0)
        sil = condition_silhouette(R_cat, k=2)

        all_results.append({
            'k': k_val, 'seed': seed,
            'ps': ps_mean, 'sil': sil, 'sparsity': sparsity, 'acc': acc,
            'k_frac': k_val / BOTTLENECK_DIM
        })

    # Per-K summary
    k_results = [r for r in all_results if r['k'] == k_val]
    ps_vals_k = [r['ps'] for r in k_results]
    sil_vals_k = [r['sil'] for r in k_results]
    acc_vals_k = [r['acc'] for r in k_results]
    print(f"  K={k_val:4d} ({k_val/BOTTLENECK_DIM*100:5.1f}%): "
          f"PS={np.mean(ps_vals_k):.4f}±{np.std(ps_vals_k):.4f}, "
          f"Sil={np.mean(sil_vals_k):.4f}±{np.std(sil_vals_k):.4f}, "
          f"Acc={np.mean(acc_vals_k):.1f}±{np.std(acc_vals_k):.1f}%")

# ============================================================
# 5. Aggregate per K (mean±std across seeds)
# ============================================================
k_summary = []
for k_val in K_VALUES:
    k_res = [r for r in all_results if r['k'] == k_val]
    k_summary.append({
        'k': k_val, 'k_frac': k_val / MID_DIM,
        'ps_mean': np.mean([r['ps'] for r in k_res]),
        'ps_std': np.std([r['ps'] for r in k_res]),
        'sil_mean': np.mean([r['sil'] for r in k_res]),
        'sil_std': np.std([r['sil'] for r in k_res]),
        'acc_mean': np.mean([r['acc'] for r in k_res]),
        'acc_std': np.std([r['acc'] for r in k_res]),
    })

ps_means = np.array([s['ps_mean'] for s in k_summary])
sil_means = np.array([s['sil_mean'] for s in k_summary])
r_k, p_k = spearmanr(ps_means, sil_means)
print(f"\nK-gradient PS×Sil: Spearman r = {r_k:+.4f}, p = {p_k:.4f}")

# ============================================================
# 6. Neural data reference
# ============================================================
neural_ps = np.array([0.4991,0.5058,0.5111,0.5265,0.5299,0.5360,0.5669,0.5723,0.5760,
                      0.5823,0.5962,0.5967,0.6212,0.6280,0.6321,0.6327,0.6382,0.6475,
                      0.6575,0.6662,0.6776,0.6971,0.7043,0.7057,0.7114,0.7131,0.7212,
                      0.7285,0.7356,0.7361,0.7389,0.7511,0.7536,0.7548,0.7588,0.7623,
                      0.7657,0.7731,0.7756,0.7791,0.7818,0.7907,0.7945,0.7998,0.8110,
                      0.8285,0.8369,0.8371,0.8410,0.8568,0.8617,0.8715,0.8801,0.8979])
neural_sil = np.array([0.571,0.621,0.542,0.540,0.500,0.415,0.517,0.616,0.615,0.540,0.600,
                       0.324,0.567,0.251,0.475,0.417,0.484,0.554,0.418,0.275,0.317,0.299,
                       0.447,0.447,0.451,0.621,0.463,0.550,0.434,0.329,0.441,0.502,0.469,
                       0.342,0.449,0.423,0.390,0.403,0.381,0.406,0.505,0.394,0.356,0.336,
                       0.508,0.494,0.310,0.304,0.223,0.434,0.309,0.262,0.304,0.307])

# ============================================================
# 7. Figure: K-gradient × Neural overlay
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))

# Neural data
ax.scatter(neural_ps, neural_sil, c='gray', alpha=0.35, s=25, label='Neural (Steinmetz, 54 regions)')

# K-gradient curve with error bands
k_fracs = np.array([s['k_frac'] for s in k_summary])
ps_v = np.array([s['ps_mean'] for s in k_summary])
sil_v = np.array([s['sil_mean'] for s in k_summary])
ps_e = np.array([s['ps_std'] for s in k_summary])
sil_e = np.array([s['sil_std'] for s in k_summary])

# Color by PS value
colors = plt.cm.plasma(np.linspace(0.2, 0.95, len(k_summary)))
for i, s in enumerate(k_summary):
    ax.errorbar(s['ps_mean'], s['sil_mean'],
                xerr=s['ps_std'], yerr=s['sil_std'],
                fmt='o', color=colors[i], markersize=10, capsize=3,
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)
    ax.annotate(f"K={s['k']}", (s['ps_mean']+0.005, s['sil_mean']-0.02),
                fontsize=7, ha='left', color='#333')

# Connect K values with a line to show the causal curve
sort_idx = np.argsort(ps_v)
ax.plot(ps_v[sort_idx], sil_v[sort_idx], '-', color='#7c3aed', linewidth=2, alpha=0.6,
        label=f'K-gradient (r={r_k:+.3f}, p={p_k:.3f})')

# Combined fit
all_ps = np.concatenate([neural_ps, ps_v])
all_sil = np.concatenate([neural_sil, sil_v])
r_comb, _ = spearmanr(all_ps, all_sil)
z = np.polyfit(all_ps, all_sil, 1)
x_line = np.linspace(0.45, 1.0, 100)
ax.plot(x_line, np.polyval(z, x_line), 'k--', linewidth=1.5, alpha=0.4,
        label=f'Combined (r={r_comb:+.3f})')

# Annotations
ax.annotate('DENSE\n256 units active\nweakest clustering',
            xy=(ps_v[0], sil_v[0]), xytext=(ps_v[0]+0.08, sil_v[0]+0.04),
            fontsize=8, arrowprops=dict(arrowstyle='->', color='gray'))
ax.annotate('SPARSE\nonly top-2 active\nstrongest clustering',
            xy=(ps_v[-1], sil_v[-1]), xytext=(ps_v[-1]-0.12, sil_v[-1]-0.08),
            fontsize=8, arrowprops=dict(arrowstyle='->', color='gray'))

ax.set_xlabel('Population Sparseness (PS)', fontsize=13)
ax.set_ylabel('Silhouette (clustering)', fontsize=13)
ax.set_title('Causal Effect of PS on Representational Clustering\nReLU+K-Sparse Gradient (Biological Constraint)', fontsize=14)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0.43, 1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'k_gradient_relu.png'), dpi=150)
print(f"\nFigure saved: k_gradient_relu.png")

# ============================================================
# 8. Figure: Per-seed scatter
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6.5))
seed_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for si, seed in enumerate(SEEDS):
    seed_res = [r for r in all_results if r['seed'] == seed]
    seed_ps = [r['ps'] for r in seed_res]
    seed_sil = [r['sil'] for r in seed_res]
    ax.plot(seed_ps, seed_sil, 'o-', color=seed_colors[si], markersize=6,
            alpha=0.7, label=f'Seed {seed}')
    for i, r in enumerate(seed_res):
        ax.annotate(f"K={r['k']}", (r['ps']+0.002, r['sil']-0.01),
                    fontsize=5.5, color=seed_colors[si], alpha=0.7)

ax.scatter(neural_ps, neural_sil, c='gray', alpha=0.25, s=15, zorder=0, label='Neural')
ax.set_xlabel('PS', fontsize=12)
ax.set_ylabel('Silhouette', fontsize=12)
ax.set_title('K-Gradient: Per-Seed Curves (5 seeds × 8 K values)', fontsize=13)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'per_seed_curves_relu.png'), dpi=150)
print("Saved: per_seed_curves_relu.png")

# ============================================================
# 9. Summary table
# ============================================================
print(f"\n{'='*60}")
print("FINAL K-GRADIENT RESULTS")
print(f"{'='*60}")
print(f"{'K':>5s} {'K%':>6s} {'PS':>8s} {'Sil':>8s} {'Acc':>8s}")
print("-" * 45)
for s in k_summary:
    print(f"{s['k']:5d} {s['k_frac']:5.1%} {s['ps_mean']:7.4f}±{s['ps_std']:.4f} {s['sil_mean']:7.4f}±{s['sil_std']:.4f} {s['acc_mean']:6.1f}%")

print(f"\nK-gradient r = {r_k:+.4f}, p = {p_k:.4f}")
print(f"Neural r = -0.59 (Steinmetz)")
print(f"Combined r = {r_comb:+.4f}")
print(f"\nTotal time: {time.time()-t0:.0f}s ({len(K_VALUES)*len(SEEDS)} runs)")
