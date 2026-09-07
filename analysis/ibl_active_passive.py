import os
from pathlib import Path
ROOT = Path(os.environ.get('SPARSE_CODING_ROOT', Path(__file__).resolve().parents[1]))
"""P2-m: STRICTLY MATCHED IBL passive vs active (same session, same region).

For every session that has BOTH a passive Gabor block and active task trials
(same probe, same neurons), compute per-region PS, NNR and task-relevant Fisher
redundancy in BOTH blocks, and match regions. This is the Science-style
same-neurons state comparison: does the redundancy flip (and do PS/NNR move)
WITHIN regions, not just between populations?

Outputs: revisedana/outputs/p2_ibl_redundancy_matched.json
"""
import sys, os, json, glob, time
import numpy as np
import pandas as pd


import analysis_utils as au

t0 = time.time()
IBL_DIR = str(ROOT / "data" / "ibl")
ONE_CACHE = str(ROOT / "data" / "ibl_one_cache")
MIN_TRIALS = 5
MIN_UNITS = 20
N_SAMPLE = 50
RIDGE = 1e-3
STIM_WIN = (0.1, 0.3)
OUT = os.path.join(str(ROOT / "outputs"), "p2_ibl_redundancy_matched.json")


def bin_contrast(c):
    """Steinmetz verbatim contrast binning: 0->0, (0,.25]->1, (.25,.5]->2, >.5->3."""
    if c == 0:
        return 0
    if c <= 0.25:
        return 1
    if c <= 0.5:
        return 2
    return 3

with open(str(ROOT / "data" / "ibl_ccf_map.json")) as f:
    id_map = json.load(f)
_p = json.load(open(str(ROOT / "data" / "ibl_passive_results.json")))
_a = json.load(open(str(ROOT / "data" / "ibl_steinmetz_results.json")))
PASSIVE_SESS = set(r["session"] for r in _p)
ACTIVE_SESS = set(r["session"] for r in _a)
BOTH = sorted(PASSIVE_SESS & ACTIVE_SESS)
print(f"matched sessions (both blocks): {len(BOTH)}")


def find_passive_csv(lab, subj, date_str):
    base = os.path.join(ONE_CACHE, lab, 'Subjects', subj, date_str, '001', 'alf')
    main = os.path.join(base, '_ibl_passiveGabor.table.csv')
    if os.path.exists(main):
        return main
    revs = sorted(glob.glob(os.path.join(base, '#*#', '_ibl_passiveGabor.table.csv')))
    return revs[-1] if revs else None


def load_spikes(lab, subj, date_str):
    spike_dir = os.path.join(IBL_DIR, lab, 'Subjects', subj, date_str,
                             '001', 'alf', 'probe00', 'pykilosort')

    def ldp(path, prefix):
        files = glob.glob(os.path.join(path, prefix + '*'))
        return np.load(files[0]).ravel() if files else None

    st = ldp(spike_dir, 'spikes.times'); sc = ldp(spike_dir, 'spikes.clusters')
    cc = ldp(spike_dir, 'clusters.channels'); cbi = ldp(spike_dir, 'channels.brainLocationIds_ccf_2017')
    if any(x is None for x in [st, sc, cc, cbi]):
        return None
    sc = sc.astype(int); cc = cc.astype(int); cbi = cbi.astype(int)
    good = np.ones(len(cc), dtype=bool)
    mf = glob.glob(os.path.join(spike_dir, 'clusters.metrics*'))
    if mf:
        try:
            good = pd.read_parquet(mf[0])['label'].values >= (2.0 / 3.0)
        except Exception:
            pass
    good_idx = np.where(good)[0]
    if len(good_idx) < MIN_UNITS:
        return None
    old_to_new = -np.ones(len(cc), dtype=int); old_to_new[good_idx] = np.arange(len(good_idx))
    sc_good = old_to_new[sc]; m = sc_good >= 0
    st, sc_good = st[m], sc_good[m]
    region = [id_map.get(str(cbi[int(ch)]), f'ID{cbi[int(ch)]}') for ch in cc[good_idx]]
    return dict(st=st, sc=sc_good, region=region, n_good=len(good_idx))


def trial_counts(st, sc, n_cl, onset, win):
    si = np.argsort(st); sts = st[si]; scs = sc[si]
    tc = np.zeros((len(onset), n_cl), dtype=np.float32)
    for ti in range(len(onset)):
        i0 = np.searchsorted(sts, onset[ti] + win[0]); i1 = np.searchsorted(sts, onset[ti] + win[1])
        if i1 > i0:
            cw = scs[i0:i1]; v = (cw >= 0) & (cw < n_cl)
            if v.any():
                np.add.at(tc[ti], cw[v], 1)
    tc /= (win[1] - win[0])
    return tc


def block_metrics(tc, cond, task, ua):
    """PS, NNR, redundancy for one block's region."""
    Xk = tc[:, ua]; R = np.array([Xk[cond == c].mean(0) for c in np.unique(cond) if (cond == c).sum() >= MIN_TRIALS])
    if len(R) < 4 or len(ua) < MIN_UNITS:
        return None
    ps = np.nanmean([au.population_sparseness(R[i]) for i in range(len(R))])
    nnr = au.condition_nnr(R, k_nn=2)
    keep = task >= 0
    Xt, gt = Xk[keep], task[keep]
    if (gt == 1).sum() < MIN_TRIALS or (gt == 0).sum() < MIN_TRIALS:
        red = np.nan
    else:
        d = Xt[gt == 1].mean(0) - Xt[gt == 0].mean(0)
        Sigma, cnt = 0.0, 0
        for c in np.unique(cond):
            Xc = Xt[cond[keep] == c]
            if len(Xc) < MIN_TRIALS: continue
            Xc = Xc - Xc.mean(0)
            Sigma += Xc.T @ Xc / (len(Xc) - 1); cnt += 1
        if cnt == 0:
            red = np.nan
        else:
            Sigma = Sigma / cnt + RIDGE * np.eye(len(ua))
            diag = np.diag(Sigma); Jd = float(np.sum(d**2 / diag))
            if Jd > 0:
                J = float(d @ np.linalg.solve(Sigma, d))
                red = (1.0 - J / Jd) if J > 0 else np.nan
            else:
                red = np.nan
    return dict(ps=float(ps), nnr=float(nnr), red=float(red))


def subsample(idx, n=N_SAMPLE, seed=0):
    rng = np.random.default_rng(seed)
    if len(idx) <= n: return np.array(idx)
    return rng.choice(idx, size=n, replace=False)


matched = []
n_err = 0
for sess_key in BOTH:
    lab, subj, date = sess_key.split('/')
    try:
        sp = load_spikes(lab, subj, date)
        if sp is None:
            n_err += 1; continue
        alf_dir = os.path.join(IBL_DIR, lab, 'Subjects', subj, date, '001', 'alf')
        ru = {}
        for ci, rname in enumerate(sp['region']):
            ru.setdefault(rname, []).append(ci)

        # PASSIVE block
        csv_path = find_passive_csv(lab, subj, date)
        if csv_path is None:
            continue
        gabor = pd.read_csv(csv_path, index_col=0)
        if 'contrast' not in gabor.columns:
            continue
        onset_p = gabor['start'].values if 'start' in gabor.columns else gabor.index.values.astype(float)
        contrast = gabor['contrast'].values.ravel()
        tc_p = trial_counts(sp['st'], sp['sc'], sp['n_good'], onset_p, STIM_WIN)
        cond_p = np.array([str(c) for c in contrast])
        lo, hi = np.percentile(contrast, 40), np.percentile(contrast, 60)
        task_p = np.where(contrast < lo, 0, np.where(contrast > hi, 1, -1))

        # ACTIVE block
        tfs = (glob.glob(os.path.join(alf_dir, '*', '_ibl_trials.table*'), recursive=True) +
               glob.glob(os.path.join(alf_dir, '_ibl_trials.table*')))
        trials = None
        for tf in tfs:
            try:
                trials = pd.read_parquet(tf)
                if len(trials) > 10: break
            except Exception:
                continue
        if trials is None or not all(c in trials.columns for c in ['stimOn_times', 'contrastLeft', 'contrastRight']):
            continue
        onset_a = trials['stimOn_times'].values.ravel()
        crL = trials['contrastLeft'].values.ravel(); crR = trials['contrastRight'].values.ravel()
        tc_a = trial_counts(sp['st'], sp['sc'], sp['n_good'], onset_a, STIM_WIN)
        cln = np.nan_to_num(crL, nan=0); crn = np.nan_to_num(crR, nan=0)
        cond_a = np.array([f"L{bin_contrast(a)}_R{bin_contrast(b)}" for a, b in zip(cln, crn)])
        # task = left-dominant vs right-dominant (NaN -> 0, so one-sided contrasts work)
        task_a = np.where((cln > crn) & (cln > 0), 1, np.where((crn > cln) & (crn > 0), 0, -1))

        for rname, ul in ru.items():
            if len(ul) < MIN_UNITS:
                continue
            ua = subsample(ul)
            bm_p = block_metrics(tc_p, cond_p, task_p, ua)
            bm_a = block_metrics(tc_a, cond_a, task_a, ua)
            if bm_p is None or bm_a is None:
                continue
            if not (np.isfinite(bm_p['red']) and np.isfinite(bm_a['red'])):
                continue
            matched.append({"session": sess_key, "region": rname,
                            "ps_p": bm_p['ps'], "nnr_p": bm_p['nnr'], "red_p": bm_p['red'],
                            "ps_a": bm_a['ps'], "nnr_a": bm_a['nnr'], "red_a": bm_a['red']})
    except Exception:
        n_err += 1
        continue
    if len(matched) % 200 == 0 and len(matched) > 0:
        print(f"  ... {len(matched)} matched region-pairs")

print(f"\nmatched region-pairs: {len(matched)} (errors: {n_err})")

# ---- aggregate ----
if matched:
    import numpy as np
    ps_p = np.array([m['ps_p'] for m in matched]); ps_a = np.array([m['ps_a'] for m in matched])
    nnr_p = np.array([m['nnr_p'] for m in matched]); nnr_a = np.array([m['nnr_a'] for m in matched])
    red_p = np.array([m['red_p'] for m in matched]); red_a = np.array([m['red_a'] for m in matched])
    print(f"\n=== MATCHED passive -> active ===")
    print(f"  PS:    {ps_p.mean():+.3f} -> {ps_a.mean():+.3f}   (dPS={ (ps_a-ps_p).mean():+.4f})")
    print(f"  NNR:   {nnr_p.mean():+.3f} -> {nnr_a.mean():+.3f}   (dNNR={(nnr_a-nnr_p).mean():+.4f})")
    print(f"  Red:   {red_p.mean():+.3f} -> {red_a.mean():+.3f}   (dRed={(red_a-red_p).mean():+.4f}, flip_frac={(red_a>0).mean():.2f} vs passive {(red_p>0).mean():.2f})")
    from scipy.stats import wilcoxon, spearmanr
    for arr, lab in [(red_a - red_p, "dRed"), (ps_a - ps_p, "dPS"), (nnr_a - nnr_p, "dNNR")]:
        try:
            w, p = wilcoxon(arr)
            print(f"  Wilcoxon {lab}: p={p:.2e}")
        except Exception as e:
            print(f"  Wilcoxon {lab}: {e}")
    # law within each state (matched regions)
    rp_law = spearmanr(ps_p, nnr_p); ra_law = spearmanr(ps_a, nnr_a)
    print(f"  law passive r={rp_law[0]:+.3f}  active r={ra_law[0]:+.3f}")
    json.dump({"matched": matched,
               "summary": {"n": len(matched),
                           "ps": {"passive": float(ps_p.mean()), "active": float(ps_a.mean())},
                           "nnr": {"passive": float(nnr_p.mean()), "active": float(nnr_a.mean())},
                           "red": {"passive": float(red_p.mean()), "active": float(red_a.mean()),
                                   "passive_median": float(np.median(red_p)), "active_median": float(np.median(red_a)),
                                   "flip_frac_active": float((red_a > 0).mean()),
                                   "passive_pos_frac": float((red_p > 0).mean())},
                           "law": {"passive_r": float(rp_law[0]), "active_r": float(ra_law[0])},
                           "wilcoxon": {"dRed_p": float(wilcoxon(red_a - red_p)[1]),
                                        "dPS_p": float(wilcoxon(ps_a - ps_p)[1]),
                                        "dNNR_p": float(wilcoxon(nnr_a - nnr_p)[1])}}},
              open(OUT, "w"), indent=1)
    print(f"Saved: {OUT}")
print(f"Time: {time.time()-t0:.1f}s")
