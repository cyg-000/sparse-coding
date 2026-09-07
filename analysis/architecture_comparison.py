import os
from pathlib import Path
ROOT = Path(os.environ.get('SPARSE_CODING_ROOT', Path(__file__).resolve().parents[1]))
r"""5-tier encoding continuum v2 — CLASSIFIER, not autoencoder.

Key insight: in biological systems, sparse coding serves CATEGORIZATION
(different stimuli activate different neuron subsets). A classifier directly
optimizes for category separation, making it a better model of the computation
that produces the PS×Silhouette relationship in the brain.

Architecture: 4096→1024→512→15, with sparsity control at the 512-dim hidden layer.
"""
import numpy as np, time, os
import torch, torch.nn as nn, torch.nn.functional as F
import scipy.io as sio
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
t0 = time.time()

device = torch.device('cpu')
print(f"Device: {device}")

# ============================================================
# 1. LOAD DATA
# ============================================================
data_dir = str(ROOT / "data" / "stringer")
class_file = str(ROOT / "data" / "stimuli_class_assignment.mat")

print("Loading images...")
img_data = sio.loadmat(os.path.join(data_dir, "images_natimg2800_all.mat"))
imgs = img_data['imgs']
h, w, n_img = imgs.shape

# Resize to 64x64
print("Resizing to 64x64...")
imgs_rs = np.zeros((64, 64, n_img), dtype=np.float32)
for i in range(n_img):
    img_2d = imgs[:, :, i].astype(np.float32)
    h_step, w_step = h / 64, w / 64
    for hi in range(64):
        hs, he = int(hi * h_step), int(min((hi + 1) * h_step, h))
        for wi in range(64):
            ws, we = int(wi * w_step), int(min((wi + 1) * w_step, w))
            imgs_rs[hi, wi, i] = img_2d[hs:he, ws:we].mean()

X_full = imgs_rs.reshape(4096, n_img).T.astype(np.float32) / 255.0

classes = sio.loadmat(class_file)
cat_labels = classes['class_assignment'].ravel()
KNOWN_CATS = list(range(1, 16))

known_mask = np.isin(cat_labels, KNOWN_CATS)
X_known = X_full[known_mask]
cat_known = cat_labels[known_mask]

X = torch.tensor(X_known, dtype=torch.float32).to(device)
y = torch.tensor(cat_known - 1, dtype=torch.long).to(device)  # 0-14
n_images = X.shape[0]
print(f"Images: {n_images}, dim: 4096, classes: {len(KNOWN_CATS)}")

# ============================================================
# 2. MODEL DEFINITIONS
# ============================================================
IN_DIM, MID_DIM, HIDDEN_DIM, N_CLASSES = 4096, 1024, 512, 15

# --- Tier 1: Dense (GELU = smooth, fewer dead units) ---
class DenseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IN_DIM, MID_DIM)
        self.fc2 = nn.Linear(MID_DIM, HIDDEN_DIM)
        self.classifier = nn.Linear(HIDDEN_DIM, N_CLASSES)

    def forward(self, x):
        h1 = F.gelu(self.fc1(x))
        h = F.gelu(self.fc2(h1))
        logits = self.classifier(h)
        return logits, h

# --- Tier 2: Attention (GELU + attention = compute-sparse, represent-dense) ---
class AttentionClassifier(nn.Module):
    def __init__(self, n_groups=16):
        super().__init__()
        self.fc1 = nn.Linear(IN_DIM, MID_DIM)
        self.n_groups = n_groups
        self.group_dim = MID_DIM // n_groups
        self.attn = nn.MultiheadAttention(self.group_dim, num_heads=4, batch_first=True)
        self.fc2 = nn.Linear(MID_DIM, HIDDEN_DIM)
        self.classifier = nn.Linear(HIDDEN_DIM, N_CLASSES)

    def forward(self, x):
        h1 = F.gelu(self.fc1(x))
        h1_g = h1.view(-1, self.n_groups, self.group_dim)
        h1_a, _ = self.attn(h1_g, h1_g, h1_g)
        h = F.gelu(self.fc2(h1_a.reshape(-1, MID_DIM)))
        logits = self.classifier(h)
        return logits, h

# --- Tier 3: MoE ---
class MoE_Linear(nn.Module):
    def __init__(self, in_dim, out_dim, n_experts=4, top_k=2):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.router = nn.Linear(in_dim, n_experts)
        self.experts = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(n_experts)])

    def forward(self, x):
        B = x.shape[0]
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_w, topk_idx = torch.topk(router_probs, self.top_k, dim=-1)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
        out = torch.zeros(B, self.experts[0].out_features, device=x.device)
        for k in range(self.top_k):
            e_idx = topk_idx[:, k]; w = topk_w[:, k].unsqueeze(-1)
            for e in range(self.n_experts):
                m = e_idx == e
                if m.any(): out[m] += w[m] * self.experts[e](x[m])
        return out

class MoEClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IN_DIM, MID_DIM)
        self.moe = MoE_Linear(MID_DIM, HIDDEN_DIM, n_experts=4, top_k=2)
        self.classifier = nn.Linear(HIDDEN_DIM, N_CLASSES)

    def forward(self, x):
        h1 = F.gelu(self.fc1(x))
        h = F.gelu(self.moe(h1))  # GELU = dense rep, MoE = sparse computation
        logits = self.classifier(h)
        return logits, h

# --- Tier 4: L1 on hidden (ReLU = naturally sparse, L1 enhances) ---
class L1Classifier(nn.Module):
    def __init__(self, l1_lambda=0.005):
        super().__init__()
        self.fc1 = nn.Linear(IN_DIM, MID_DIM)
        self.fc2 = nn.Linear(MID_DIM, HIDDEN_DIM)
        self.classifier = nn.Linear(HIDDEN_DIM, N_CLASSES)
        self.l1_lambda = l1_lambda

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h1))
        logits = self.classifier(h)
        return logits, h

    def l1_loss(self, h):
        return self.l1_lambda * h.abs().mean()

# --- Tier 5: K-Sparse (ReLU + top-k, maximal sparsity) ---
class KSparseClassifier(nn.Module):
    def __init__(self, k=48):
        super().__init__()
        self.fc1 = nn.Linear(IN_DIM, MID_DIM)
        self.fc2 = nn.Linear(MID_DIM, HIDDEN_DIM)
        self.classifier = nn.Linear(HIDDEN_DIM, N_CLASSES)
        self.k = k

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h_pre = self.fc2(h1)
        topk_v, topk_i = torch.topk(h_pre, self.k, dim=1)
        mask = torch.zeros_like(h_pre); mask.scatter_(1, topk_i, 1.0)
        h = F.relu(h_pre) * mask
        logits = self.classifier(h)
        return logits, h

# ============================================================
# 3. TRAINING
# ============================================================
N_EPOCHS, BATCH_SIZE, LR = 100, 128, 1e-3

def train_classifier(model, X, y, name, use_l1=False):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    n = X.shape[0]

    for epoch in range(N_EPOCHS):
        model.train()
        total_loss, correct = 0, 0
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb, yb = X[idx], y[idx]
            opt.zero_grad()
            logits, h = model(xb)
            loss = criterion(logits, yb)
            if use_l1 and hasattr(model, 'l1_loss'):
                loss = loss + model.l1_loss(h)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == yb).sum().item()

        if (epoch + 1) % 30 == 0:
            acc = correct / n * 100
            print(f"    [{name}] Epoch {epoch+1:3d}/{N_EPOCHS}: loss={total_loss:.4f}, acc={acc:.1f}%")

    # Final accuracy
    model.eval()
    with torch.no_grad():
        logits, _ = model(X)
        final_acc = (logits.argmax(1) == y).float().mean().item() * 100
    print(f"    [{name}] Final accuracy: {final_acc:.1f}%")

    return model

# ============================================================
# 4. METRICS
# ============================================================
def population_sparseness(r):
    r = np.maximum(r, 0)
    N = len(r); r_sq, r_sum = (r**2).sum(), r.sum()
    if r_sq == 0 or r_sum == 0 or N <= 1: return np.nan
    return (1 - r_sum**2 / (N * r_sq)) / (1 - 1/N)

def condition_silhouette(R, k=2):
    n = R.shape[0]
    if n < 6: return np.nan
    Rc = R - R.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Rc, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    Ru = Rc / norms
    corr = Ru @ Ru.T
    dist = 1 - corr
    k_eff = min(k, n-2)
    ratios = np.zeros(n)
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

def compute_metrics(model, X, cat_ids):
    model.eval()
    with torch.no_grad():
        _, H_all = model(X)
        H_all = H_all.cpu().numpy()

    unique_cats = np.unique(cat_ids)
    R_cat = np.zeros((len(unique_cats), HIDDEN_DIM), dtype=np.float32)
    for ci, cat in enumerate(unique_cats):
        mask = cat_ids == cat
        R_cat[ci, :] = H_all[mask, :].mean(axis=0)
    R_cat = np.maximum(R_cat, 0)

    ps_cat_means = []
    for cat in unique_cats:
        mask = cat_ids == cat
        ps_vals = [population_sparseness(H_all[i,:]) for i in np.where(mask)[0]]
        ps_cat_means.append(np.nanmean(ps_vals))
    ps_mean = np.nanmean(ps_cat_means)

    sil = condition_silhouette(R_cat, k=2)
    sp_frac = (H_all == 0).mean()

    return ps_mean, sil, sp_frac, H_all, R_cat

# ============================================================
# 5. RUN ALL TIERS
# ============================================================
cat_np = cat_known

configs = [
    ("T1: Dense (GELU, h=512)",       DenseClassifier(),                 False),
    ("T2: Attention (GELU, h=512)",    AttentionClassifier(),             False),
    ("T3: MoE (GELU, top-2/4, h=512)", MoEClassifier(),                  False),
    ("T4: L1 (ReLU, l=0.005, h=512)", L1Classifier(l1_lambda=0.005),    True),
    ("T5: K-Sparse (ReLU, k=48, h=512)", KSparseClassifier(k=48),       False),
]

all_results = []

for name, model, use_l1 in configs:
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    t_train = time.time()
    model = train_classifier(model, X, y, name, use_l1=use_l1)
    train_time = time.time() - t_train

    ps_mean, sil, sp_frac, H, R_cat = compute_metrics(model, X, cat_np)
    print(f"  PS = {ps_mean:.4f}, Sil = {sil:.4f}, Sparsity = {sp_frac:.1%}")
    all_results.append({'name': name, 'ps': float(ps_mean), 'sil': float(sil),
                        'sparsity_frac': float(sp_frac), 'train_time': float(train_time)})

# Save architecture comparison to cache
import json
CACHE_FILE = str(ROOT / "outputs" / "cache_architecture_comparison.json")
json.dump(all_results, open(CACHE_FILE, "w"), indent=2)
print(f"  Saved: {CACHE_FILE}")

# ============================================================
# 6. RESULTS
# ============================================================
print(f"\n{'='*60}")
print("FINAL: CLASSIFIER-BASED ENCODING CONTINUUM")
print(f"{'='*60}")

model_ps = np.array([r['ps'] for r in all_results])
model_sil = np.array([r['sil'] for r in all_results])

print(f"\n{'Model':>30s}  {'PS':>8s}  {'Sil':>8s}  {'Sparsity':>8s}")
print("-" * 60)
for r in all_results:
    print(f"{r['name']:>30s}  {r['ps']:>7.4f}  {r['sil']:>7.4f}  {r['sparsity_frac']:>7.1%}")

r_model, p_model = spearmanr(model_ps, model_sil)
print(f"\nModel PS vs Sil:  Spearman r = {r_model:+.4f}, p = {p_model:.4f}")
print(f"Neural PS vs Sil: Spearman r = -0.59, p < 0.0001")

# Neural data
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

# Plot
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(neural_ps, neural_sil, c='gray', alpha=0.4, s=25, label='Neural (Steinmetz, 54 regions)')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, r in enumerate(all_results):
    ax.scatter(r['ps'], r['sil'], c=colors[i], s=200, marker='s', edgecolors='black',
               linewidth=2, zorder=5, label=r['name'])

all_ps = np.concatenate([neural_ps, model_ps])
all_sil = np.concatenate([neural_sil, model_sil])
r_combined, _ = spearmanr(all_ps, all_sil)
z = np.polyfit(all_ps, all_sil, 1)
x_line = np.linspace(all_ps.min(), all_ps.max(), 100)
ax.plot(x_line, np.polyval(z, x_line), 'k--', linewidth=1.5, alpha=0.5,
        label=f'Combined fit')

ax.set_xlabel('Population Sparseness (PS)', fontsize=13)
ax.set_ylabel('Silhouette (clustering)', fontsize=13)
ax.set_title('PS vs Representational Clustering: Neural & Classifier Models', fontsize=14)
ax.legend(fontsize=7.5, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(str(ROOT / "figures" / "output" / "architecture_comparison.png"), dpi=150)
print(f"\nFigure saved: ps_sil_continuum.png")
print(f"Total time: {time.time()-t0:.0f}s")
