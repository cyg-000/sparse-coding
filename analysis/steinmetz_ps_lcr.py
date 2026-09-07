import os
from pathlib import Path
ROOT = Path(os.environ.get('SPARSE_CODING_ROOT', Path(__file__).resolve().parents[1]))
"""Steinmetz: PS × Pattern Separation v4 — FIXED windows, vectorized (aligned with IBL).

Fixes vs v3:
1. Behavior conditions: choice (L/R/N) × outcome (C/E only, exclude fb=0) — matches Methods 3×2=6
2. Fixed time windows: behav [0, 500ms] from go_cue; stim [25, 250ms] from stim onset
   (no RT confound — previously variable go_cue→response_times)
3. Firing rate (Hz): trial×cluster count matrix (IBL style), then divide by window duration
4. Saves to cache_ps3.json (per-region + pooled + bootstrap CI)
5. Min cond check: n_cond ≥ k+2 (k=2→4, k=3→5)
Stimulus note: binned contrast tests sensory evidence geometry, not object identity.
"""
import numpy as np, tarfile, os, json, time
from collections import defaultdict, Counter
from scipy.stats import spearmanr

t0 = time.time()
DATA_DIR = str(ROOT / "data" / "steinmetz")
OUT_DIR = str(ROOT / "outputs" / "steinmetz_extracted")
CACHE_FILE = str(ROOT / "outputs" / "cache_ps3.json")

SESSIONS = [
    "Richards_2017-10-30", "Richards_2017-11-01",
    "Forssmann_2017-11-05", "Hench_2017-06-15",
    "Forssmann_2017-11-02", "Hench_2017-06-16",
]
MIN_UNITS = 20; MIN_TRIALS = 5
BEHAV_WIN = (0.0, 0.5)       # [0, 500ms] from go_cue
STIM_WIN = (0.025, 0.250)    # [25, 250ms] from stim onset

REGION_CATEGORY = {}
for r in ['CA1','CA2','CA3','DG','SUB','POST','CA','HPF','ProS','HATA']:
    REGION_CATEGORY[r] = 'Hippocampus'
for r in ['OLF','PIR','EP','EPd','EPv','TT','NLOT','COA','TR','PAA','MOB','AOB','LOT']:
    REGION_CATEGORY[r] = 'Olfactory'
for r in ['TH','LGd','LP','LD','PO','POL','MG','MD','CL','VAL','PT','SPF','RT','VPL','VPM',
           'VPMpc','VPLpc','VM','RE','RH','CM','PCN','IMD','PF','SMT','PR']:
    REGION_CATEGORY[r] = 'Thalamus'
for r in ['CP','ACB','OT','STR','FS','GPe','GPi','SNr','SNc','VTA','ICj']:
    REGION_CATEGORY[r] = 'Striatum'
for r in ['VISp','VISl','VISal','VISam','VISpm','VISrl','VISa','VIS','AUD','SSp','SSs','SS',
           'MOp','MOs','MO','ORB','ORBm','ORBl','PL','ILA','RSP','ACA','GU','PTLp','TEa',
           'PERI','ECT','AI','FRP','DP','TTd']:
    REGION_CATEGORY[r] = 'Cortex'
for r in ['MB','MRN','SC','SCig','SCm','SCs','SCsg','IC','PAG','APN','RN','NB','SNr','SNc',
           'PRT','DTN','LTN','PPN','DR','CLI']:
    REGION_CATEGORY[r] = 'Midbrain'
for r in ['BLA','BMA','MEA','CEA','IA','AAA','PA']:
    REGION_CATEGORY[r] = 'Amygdala'
for r in ['LS','LSc','LSr','MS','NDB','BAC','SF','SH']:
    REGION_CATEGORY[r] = 'Septum'
for r in ['LH','ZI','HY','MM','PH','PVH','ARH','VMH','DMH','LHA','MPN','MPO','AHN']:
    REGION_CATEGORY[r] = 'Hypothalamus'


def pop_sparseness(r):
    N = len(r); r = np.maximum(r, 0)
    s2, s = (r**2).sum(), r.sum()
    if s2 == 0 or s == 0 or N <= 1: return np.nan
    return (1 - s**2/(N*s2)) / (1 - 1/N)


def lifetime_sparseness(R):
    """Kurtosis-based lifetime sparseness (across-condition response distribution per neuron)."""
    nc, nn = R.shape; ls = np.zeros(nn)
    for ni in range(nn):
        r = np.maximum(R[:, ni], 0)
        if r.sum() == 0: ls[ni] = np.nan; continue
        rn = (r - r.mean()) / (r.std() + 1e-10)
        ls[ni] = np.mean(rn**4)
    return float(np.nanmean(ls))

def gini_coefficient(r):
    rs = np.sort(np.maximum(r, 0))
    if rs.sum() == 0 or len(rs) < 2: return np.nan
    cum = np.cumsum(rs) / rs.sum()
    return float(1 - 2 * cum[:-1].sum() / (len(rs) - 1))

def normalized_entropy(r):
    r = np.maximum(r, 0)
    if r.sum() == 0: return np.nan
    p = r / r.sum(); p = p[p > 0]
    H = -np.sum(p * np.log(p)); Hm = np.log(len(r))
    return float(1 - H / Hm) if Hm > 0 else np.nan

def participation_ratio(r):
    r = np.maximum(r, 0)
    s1 = r.sum(); s2 = (r**2).sum()
    if s2 == 0 or len(r) < 1: return np.nan
    return float((s1**2 / s2) / len(r))

def cv_of_response(r):
    r = np.maximum(r, 0)
    if r.mean() == 0: return np.nan
    return float(r.std() / r.mean())


def rdm_variance(R):
    Rc = R - R.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(Rc, axis=1, keepdims=True); nrm[nrm==0]=1e-10; Ru = Rc/nrm
    d = 1 - (Ru @ Ru.T)
    t = np.triu_indices(R.shape[0], k=1)
    return float(np.var(d[t]))


def sep_silhouette(R, k=3):
    """d_NN / d_other on corr-distance. d_other excludes kNN. Requires n_cond ≥ k+2."""
    n = R.shape[0]
    if n < k + 2: return np.nan, np.nan
    Rc = R - R.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(Rc, axis=1, keepdims=True); nrm[nrm==0]=1e-10; Ru = Rc/nrm
    d = 1 - (Ru @ Ru.T)
    ke = min(k, n-2); rr = np.zeros(n)
    for i in range(n):
        m = np.ones(n, dtype=bool); m[i]=False
        vi = np.arange(n)[m]; sl = np.argsort(d[i,m])
        nn_ = vi[sl[:ke]]; ot = vi[sl[ke:]]
        dn = d[i,nn_].mean(); do = d[i,ot].mean()
        rr[i] = dn/do if do>0 else np.nan
    return float(np.nanmean(rr)), float(np.nanstd(rr))


def bin_contrast(c):
    if c == 0 or np.isnan(c): return 0
    elif c <= 0.25: return 1
    elif c <= 0.5: return 2
    else: return 3


def extract(sn):
    tar_path = os.path.join(DATA_DIR, sn + ".tar")
    sd = os.path.join(OUT_DIR, sn)
    need = ['clusters.peakChannel.npy','channels.brainLocation.tsv',
            'spikes.times.npy','spikes.clusters.npy',
            'trials.goCue_times.npy','trials.response_times.npy',
            'trials.visualStim_times.npy',
            'trials.visualStim_contrastLeft.npy','trials.visualStim_contrastRight.npy',
            'trials.feedbackType.npy','trials.response_choice.npy']
    if os.path.exists(sd):
        ex = set(os.listdir(sd))
        if all(f in ex for f in need):
            return {f: os.path.join(sd, f) for f in need}
    os.makedirs(sd, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        for m in tar.getmembers():
            fn = m.name.split('/')[-1]
            if fn in need:
                tar.extract(m, sd)
                src = os.path.join(sd, m.name); dst = os.path.join(sd, fn)
                if src != dst and os.path.exists(src):
                    if os.path.exists(dst): os.remove(dst)
                    os.rename(src, dst)
    return {f: os.path.join(sd, f) for f in need}


# ============================================================
# MAIN
# ============================================================
all_results = []

for sn in SESSIONS:
    print(f"\n{'='*60}\nProcessing: {sn}")
    files = extract(sn)
    if len(files) < 10:
        print(f"  SKIP: {len(files)}/10 files"); continue

    pc = np.load(files['clusters.peakChannel.npy']).ravel().astype(int)
    n_cl = len(pc)
    with open(files['channels.brainLocation.tsv']) as f:
        ll = f.read().strip().split('\n')
    crg = np.array([line.split('\t')[3] if len(line.split('\t'))>=4 else 'root' for line in ll[1:]])
    cr = np.array([crg[pc[i]-1] if 0<=pc[i]-1<len(crg) else 'root' for i in range(n_cl)])

    st = np.load(files['spikes.times.npy']).ravel()
    sc = np.load(files['spikes.clusters.npy']).ravel().astype(int)
    gc = np.load(files['trials.goCue_times.npy']).ravel()
    rt = np.load(files['trials.response_times.npy']).ravel()
    stim_t = np.load(files['trials.visualStim_times.npy']).ravel()
    cL = np.load(files['trials.visualStim_contrastLeft.npy']).ravel()
    cR = np.load(files['trials.visualStim_contrastRight.npy']).ravel()
    fb = np.load(files['trials.feedbackType.npy']).ravel()
    choice = np.load(files['trials.response_choice.npy']).ravel()
    nt = len(gc)

    # ---- Behavior conditions: choice(L/R/N) × outcome(C/E) = 6 ----
    # Exclude fb=0 (no feedback) to match Methods
    behav_lab = []; behav_valid = np.ones(nt, dtype=bool)
    for ti, (cl, fv) in enumerate(zip(choice, fb)):
        cs = 'R' if cl>0 else ('L' if cl<0 else 'N')
        if fv > 0: os_ = 'C'
        elif fv < 0: os_ = 'E'
        else: behav_valid[ti] = False; continue
        behav_lab.append(f"{cs}_{os_}")
    ba = np.array([bl for bl in behav_lab if bl])  # only valid
    # Need to remap indices for only valid trials
    behav_lab_full = np.full(nt, '', dtype=object)
    for ti in range(nt):
        if behav_valid[ti]:
            behav_lab_full[ti] = f"{'R' if choice[ti]>0 else ('L' if choice[ti]<0 else 'N')}_{'C' if fb[ti]>0 else 'E'}"
    ba_ids, ba_map = np.unique([bl for bl in behav_lab_full if bl], return_inverse=True)
    bc = Counter(ba_map)
    valid_behav = [c for c, cnt in bc.items() if cnt >= MIN_TRIALS]
    n_behav = len(valid_behav)
    # Map from full-trial indices
    behav_assign = np.full(nt, -1, dtype=int)
    for ti in range(nt):
        if behav_valid[ti]:
            behav_assign[ti] = np.where(ba_ids == behav_lab_full[ti])[0][0]

    # ---- Stimulus conditions ----
    stim_lab = np.array([f"L{bin_contrast(cl)}_R{bin_contrast(cr)}" for cl,cr in zip(cL,cR)])
    si_ids, si_map = np.unique(stim_lab, return_inverse=True)
    sc_ = Counter(si_map)
    valid_stim = [c for c, cnt in sc_.items() if cnt >= MIN_TRIALS]
    n_stim_cond = len(valid_stim)

    print(f"  Trials: {nt}, Clusters: {n_cl}")
    print(f"  Behav conds: {n_behav} (excl {(~behav_valid).sum()} no-fb trials)")
    print(f"  Stim conds:  {n_stim_cond}")

    # ---- Vectorized spike counting (IBL style) ----
    # Build trial×cluster count matrix for each window
    si = np.argsort(st); sts = st[si]; scs = sc[si]

    # Behavioral window [0, 500ms] from go_cue
    tc_behav = np.zeros((nt, n_cl), dtype=np.float32)
    for ti in range(nt):
        i0 = np.searchsorted(sts, gc[ti] + BEHAV_WIN[0])
        i1 = np.searchsorted(sts, gc[ti] + BEHAV_WIN[1])
        if i1 > i0:
            cw = scs[i0:i1]; v = (cw >= 0) & (cw < n_cl)
            if v.any(): np.add.at(tc_behav[ti], cw[v], 1)
    # Convert to Hz: count / 0.5s
    tc_behav /= BEHAV_WIN[1]

    # Stimulus window [25, 250ms] from stim onset
    tc_stim = np.zeros((nt, n_cl), dtype=np.float32)
    stim_dur = STIM_WIN[1] - STIM_WIN[0]
    for ti in range(nt):
        i0 = np.searchsorted(sts, stim_t[ti] + STIM_WIN[0])
        i1 = np.searchsorted(sts, stim_t[ti] + STIM_WIN[1])
        if i1 > i0:
            cw = scs[i0:i1]; v = (cw >= 0) & (cw < n_cl)
            if v.any(): np.add.at(tc_stim[ti], cw[v], 1)
    tc_stim /= stim_dur

    # Group by region
    ru = defaultdict(list)
    for ci in range(n_cl):
        if cr[ci] != 'root': ru[cr[ci]].append(ci)

    for reg, ul in sorted(ru.items()):
        if len(ul) < MIN_UNITS: continue
        nu = len(ul); cat = REGION_CATEGORY.get(reg, 'Other')
        ua = np.array(ul, dtype=int)
        res = {'session':sn, 'region':reg, 'category':cat, 'n_units':nu}

        # ---- Behavioral conditions ----
        if n_behav >= 4:  # k=2 → min 4
            Rb = np.zeros((n_behav, nu), dtype=np.float32)
            for vi, ci in enumerate(valid_behav):
                mask = (behav_assign == ci) & behav_valid
                Rb[vi, :] = tc_behav[mask, :][:, ua].mean(axis=0)
            Rb = np.maximum(Rb, 0)
            ps_b = np.nanmean([pop_sparseness(Rb[ci,:]) for ci in range(Rb.shape[0])])
            sep_b, sep_b_std = sep_silhouette(Rb, k=2)
            rdm_b = rdm_variance(Rb)
            # Specificity controls
            ls_b = lifetime_sparseness(Rb)
            gini_b = np.nanmean([gini_coefficient(Rb[ci,:]) for ci in range(Rb.shape[0])])
            ent_b = np.nanmean([normalized_entropy(Rb[ci,:]) for ci in range(Rb.shape[0])])
            pr_b = np.nanmean([participation_ratio(Rb[ci,:]) for ci in range(Rb.shape[0])])
            cv_b = np.nanmean([cv_of_response(Rb[ci,:]) for ci in range(Rb.shape[0])])
            res.update({'ps_behav':float(ps_b), 'sep_behav':sep_b,
                        'sep_behav_std':sep_b_std, 'rdmvar_behav':rdm_b,
                        'ls_behav':float(ls_b), 'gini_behav':float(gini_b),
                        'ent_behav':float(ent_b), 'pr_behav':float(pr_b),
                        'cv_behav':float(cv_b),
                        'n_cond_behav':n_behav})

        # ---- Stimulus conditions ----
        if n_stim_cond >= 5:  # k=3 → min 5
            Rs = np.zeros((n_stim_cond, nu), dtype=np.float32)
            for vi, ci in enumerate(valid_stim):
                mask = si_map == ci
                Rs[vi, :] = tc_stim[mask, :][:, ua].mean(axis=0)
            Rs = np.maximum(Rs, 0)
            ps_s = np.nanmean([pop_sparseness(Rs[ci,:]) for ci in range(Rs.shape[0])])
            sep_s, sep_s_std = sep_silhouette(Rs, k=3)
            rdm_s = rdm_variance(Rs)
            # Specificity controls
            ls_s = lifetime_sparseness(Rs)
            gini_s = np.nanmean([gini_coefficient(Rs[ci,:]) for ci in range(Rs.shape[0])])
            ent_s = np.nanmean([normalized_entropy(Rs[ci,:]) for ci in range(Rs.shape[0])])
            pr_s = np.nanmean([participation_ratio(Rs[ci,:]) for ci in range(Rs.shape[0])])
            cv_s = np.nanmean([cv_of_response(Rs[ci,:]) for ci in range(Rs.shape[0])])
            res.update({'ps_stim':float(ps_s), 'sep_stim':sep_s,
                        'sep_stim_std':sep_s_std, 'rdmvar_stim':rdm_s,
                        'ls_stim':float(ls_s), 'gini_stim':float(gini_s),
                        'ent_stim':float(ent_s), 'pr_stim':float(pr_s),
                        'cv_stim':float(cv_s),
                        'n_cond_stim':n_stim_cond})

        all_results.append(res)

    n_reg = sum(1 for r in all_results if r['session'] == sn)
    # Print
    for r in sorted([x for x in all_results if x['session']==sn], key=lambda x: x.get('ps_behav', x.get('ps_stim', 0))):
        pv = r.get('ps_behav', r.get('ps_stim', np.nan))
        sv = r.get('sep_behav', r.get('sep_stim', np.nan))
        rv = r.get('rdmvar_behav', r.get('rdmvar_stim', np.nan))
        print(f"    {r['region']:10s} [{r['category']:12s}] n={r['n_units']:4d}  "
              f"PS={pv:.4f}  sep={sv:.3f}  RDMvar={rv:.4f}")
    print(f"  => {n_reg} regions")

# ============================================================
# CORRELATION + BOOTSTRAP
# ============================================================
print(f"\n{'='*60}")
print(f"RESULTS: PS × Pattern Separation (v4 — fixed windows, Hz)")
print(f"{'='*60}")

rng_b = np.random.default_rng(99)
cache_pooled = {}

for ctype, prefix, k in [('behav', 'behav', 2), ('stim', 'stim', 3)]:
    pk = f'ps_{prefix}'; sk = f'sep_{prefix}'; rk = f'rdmvar_{prefix}'
    val = [r for r in all_results if pk in r and sk in r
           and not np.isnan(r[pk]) and not np.isnan(r[sk])]
    if len(val) < 8:
        print(f"\n{ctype}: only {len(val)} valid, skip"); continue

    ps = np.array([r[pk] for r in val])
    sep = np.array([r[sk] for r in val])
    rdm = np.array([r[rk] for r in val])
    nc = np.array([r[f'n_cond_{prefix}'] for r in val])

    print(f"\n--- {ctype} ---")
    print(f"  N={len(val)}, PS [{ps.min():.4f},{ps.max():.4f}] mean={ps.mean():.4f}")
    print(f"  Sep [{sep.min():.3f},{sep.max():.3f}] mean={sep.mean():.3f}")
    print(f"  Conds: [{nc.min()},{nc.max()}]")

    r_sep, p_sep = spearmanr(ps, sep)
    r_rdm, p_rdm = spearmanr(ps, rdm)
    print(f"  PS vs Sep:  r={r_sep:+.4f}  p={p_sep:.4e}")
    print(f"  PS vs RDMvar: r={r_rdm:+.4f}  p={p_rdm:.4e}")

    # Bootstrap CI for PS vs Sep
    n_boot = 500; boot_sep = np.zeros(n_boot); boot_rdm = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng_b.choice(len(ps), size=len(ps), replace=True)
        boot_sep[b], _ = spearmanr(ps[idx], sep[idx])
        boot_rdm[b], _ = spearmanr(ps[idx], rdm[idx])

    ci_s = np.percentile(boot_sep, [2.5, 97.5])
    ci_r = np.percentile(boot_rdm, [2.5, 97.5])
    print(f"  Bootstrap PS vs Sep (500): r={boot_sep.mean():+.4f}  "
          f"95%CI=[{ci_s[0]:+.4f},{ci_s[1]:+.4f}]  "
          f"r>0:{(boot_sep>0).mean():.0%}  r<0:{(boot_sep<0).mean():.0%}")
    print(f"  Bootstrap PS vs RDMvar:   r={boot_rdm.mean():+.4f}  "
          f"95%CI=[{ci_r[0]:+.4f},{ci_r[1]:+.4f}]")

    cache_pooled[ctype] = {
        'n': len(val), 'k': k,
        'ps_range': [round(float(ps.min()),4), round(float(ps.max()),4)],
        'ps_mean': round(float(ps.mean()),4),
        'sep_range': [round(float(sep.min()),4), round(float(sep.max()),4)],
        'sep_mean': round(float(sep.mean()),4),
        'spearman': {
            'ps_vs_sep': {'r': round(float(r_sep),4), 'p': float(f"{p_sep:.4e}")},
            'ps_vs_rdmvar': {'r': round(float(r_rdm),4), 'p': float(f"{p_rdm:.4e}")},
        },
        'bootstrap_500': {
            'ps_vs_sep': {'mean_r': round(float(boot_sep.mean()),4),
                          'ci_95': [round(float(ci_s[0]),4), round(float(ci_s[1]),4)]},
            'ps_vs_rdmvar': {'mean_r': round(float(boot_rdm.mean()),4),
                             'ci_95': [round(float(ci_r[0]),4), round(float(ci_r[1]),4)]},
        },
        'frac_negative_sep': round(float((boot_sep < 0).mean()),4),
    }

# ============================================================
# SAVE
# ============================================================
json.dump({
    "metadata": {
        "script": "steinmetz_ps3.py", "version": "v4",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sessions": SESSIONS,
        "params": {
            "MIN_UNITS": MIN_UNITS, "MIN_TRIALS_PER_COND": MIN_TRIALS,
            "behav_window_ms": [0, 500], "stim_window_ms": [25, 250],
            "units": "firing_rate_Hz",
            "behav_conditions": "choice(L/R/N)×outcome(C/E), excludes fb=0",
            "stim_conditions": "binned_contrast(4/eye)",
            "k_behav": 2, "k_stim": 3,
            "bootstrap": {"n": 500, "seed": 99},
        },
        "notes": "0-indexed spikes.clusters. Fixed windows (no RT confound). "
                 "Hz mean across trials. Stim = sensory evidence geometry.",
    },
    "pooled": cache_pooled,
    "per_region": all_results,
}, open(CACHE_FILE, "w"), indent=2)
print(f"\nSaved: {CACHE_FILE}")
print(f"Time: {time.time()-t0:.1f}s")
