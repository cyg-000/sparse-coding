import os
from pathlib import Path
ROOT = Path(os.environ.get('SPARSE_CODING_ROOT', Path(__file__).resolve().parents[1]))
"""Matched-state redundancy audit on identical stimulus axes.

The previous audit compared passive contrast magnitude with active stimulus
side.  This audit fixes that construct mismatch.  For every session-region it
uses the same neurons and balances trials within side x contrast strata, then
tests two pre-specified axes available in both blocks:

  1. stimulus side/position (left versus right);
  2. contrast magnitude (low versus high).

Two second-order statistics are reported:
  * shrinkage Fisher redundancy, 1 - I_full / I_diagonal;
  * task-axis covariance alignment, a stable inverse-free measure of whether
    residual covariance lies along the discriminating mean axis.

Inference is clustered at session level.  Nothing outside revisedana is
modified.
"""
import collections, collections.abc, glob, json, os, time, zlib
from pathlib import Path

collections.Callable = collections.abc.Callable
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

HERE = Path(__file__).parent
ROOT = HERE.parent
OUT = HERE / "outputs" / "audit_fisher_same_task_exact.json"
IBL_DIR = ROOT / "ibl_data"
ONE = Path(str(ROOT / "data" / "ibl_one_cache"))
WIN = (0.1, 0.3)
MIN_UNITS = 20
MIN_CELL = 3
UNIT_COUNTS = (20, 30)


def population_sparseness(v):
    """Treves-Rolls population sparseness for one non-negative mean vector."""
    v = np.asarray(v, float)
    n = v.size
    ss = float(np.sum(v * v))
    if n <= 1 or ss <= 0:
        return np.nan
    return float((1 - float(v.sum()) ** 2 / (n * ss)) / (1 - 1 / n))


def condition_geometry(X, strata, k=3):
    """PS and LCR from identical matched trials and unit identities."""
    labels = np.unique(strata)
    means = np.vstack([X[strata == s].mean(0) for s in labels])
    if len(means) <= k + 1:
        return None
    ps = float(np.nanmean([population_sparseness(row) for row in means]))
    centered = means - means.mean(1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norm <= 0):
        return None
    unit = centered / norm
    distance = 1 - unit @ unit.T
    ratios = []
    for i in range(len(means)):
        values = np.sort(np.delete(distance[i], i))
        denominator = float(np.mean(values[k:]))
        if denominator > 0:
            ratios.append(float(np.mean(values[:k]) / denominator))
    if not ratios:
        return None
    return ps, float(np.mean(ratios)), int(len(means))


def seedof(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0xffffffff


def contrast_bin(x):
    """Exact contrast code at four-decimal precision (e.g. .25 -> 2500)."""
    return int(round(max(float(x), 0.0) * 10000))


def find_passive(lab, subject, date):
    base = ONE / lab / "Subjects" / subject / date / "001" / "alf"
    main = base / "_ibl_passiveGabor.table.csv"
    if main.exists():
        return main
    revisions = sorted(base.glob("#*#/_ibl_passiveGabor.table.csv"))
    return revisions[-1] if revisions else None


def load_first(path, prefix):
    files = glob.glob(str(path / (prefix + "*")))
    return np.load(files[0]).ravel() if files else None


ID_MAP = json.load(open(ROOT / "ibl_ccf_map.json"))
PASSIVE = json.load(open(ROOT / "ibl_passive_results.json"))
ACTIVE = json.load(open(ROOT / "ibl_steinmetz_results.json"))
SESSIONS = sorted(set(x["session"] for x in PASSIVE) & set(x["session"] for x in ACTIVE))


def load_spikes(lab, subject, date):
    path = IBL_DIR / lab / "Subjects" / subject / date / "001" / "alf" / "probe00" / "pykilosort"
    st = load_first(path, "spikes.times")
    sc = load_first(path, "spikes.clusters")
    cc = load_first(path, "clusters.channels")
    ccf = load_first(path, "channels.brainLocationIds_ccf_2017")
    if any(x is None for x in (st, sc, cc, ccf)):
        return None
    sc, cc, ccf = sc.astype(int), cc.astype(int), ccf.astype(int)
    good = np.ones(len(cc), bool)
    metrics = glob.glob(str(path / "clusters.metrics*"))
    if metrics:
        try:
            good = pd.read_parquet(metrics[0])["label"].values >= (2 / 3)
        except Exception:
            pass
    good_idx = np.where(good)[0]
    if len(good_idx) < MIN_UNITS:
        return None
    remap = -np.ones(len(cc), int)
    remap[good_idx] = np.arange(len(good_idx))
    mapped = remap[sc]
    keep = mapped >= 0
    regions = [ID_MAP.get(str(ccf[int(ch)]), f"ID{ccf[int(ch)]}") for ch in cc[good_idx]]
    return {"st": st[keep], "sc": mapped[keep], "regions": regions, "n": len(good_idx)}


def trial_counts(spikes, onsets):
    order = np.argsort(spikes["st"])
    st, sc = spikes["st"][order], spikes["sc"][order]
    out = np.zeros((len(onsets), spikes["n"]), np.float32)
    for i, onset in enumerate(onsets):
        lo = np.searchsorted(st, onset + WIN[0])
        hi = np.searchsorted(st, onset + WIN[1])
        ids = sc[lo:hi]
        if len(ids):
            np.add.at(out[i], ids, 1)
    return out / (WIN[1] - WIN[0])


def matched_strata(side_p, contrast_p, side_a, contrast_a, seed):
    """Exact matching in side x contrast-bin cells across the two blocks."""
    rng = np.random.default_rng(seed)
    ip, ia = [], []
    used = []
    common_contrasts = sorted(set(contrast_p) & set(contrast_a))
    for side in (0, 1):
        for cb in common_contrasts:
            p = np.where((side_p == side) & (contrast_p == cb))[0]
            a = np.where((side_a == side) & (contrast_a == cb))[0]
            n = min(len(p), len(a))
            if n >= MIN_CELL:
                ip.extend(rng.choice(p, n, False))
                ia.extend(rng.choice(a, n, False))
                used.append((side, cb, n))
    if len(set(x[0] for x in used)) < 2 or len(set(x[1] for x in used if x[1] > 0)) < 2:
        return None
    return np.asarray(ip), np.asarray(ia), used


def residual_covariance(X, strata):
    residuals = []
    for s in np.unique(strata):
        z = X[strata == s]
        if len(z) >= MIN_CELL:
            residuals.append(z - z.mean(0))
    if not residuals:
        return None
    return LedoitWolf().fit(np.vstack(residuals)).covariance_


def crossfit_alignment(X, task, strata, seed):
    """Estimate the task axis and residual covariance on disjoint trials."""
    rng = np.random.default_rng(seed)
    halves = [[], []]
    for s in np.unique(strata):
        ids = np.where(strata == s)[0]
        if len(ids) < 4:
            continue
        ids = rng.permutation(ids)
        cut = len(ids) // 2
        halves[0].extend(ids[:cut])
        halves[1].extend(ids[cut:])
    if not halves[0] or not halves[1]:
        return np.nan
    estimates = []
    for mean_half, cov_half in ((0, 1), (1, 0)):
        im = np.asarray(halves[mean_half], int)
        ic = np.asarray(halves[cov_half], int)
        if min(np.sum(task[im] == 0), np.sum(task[im] == 1)) < 2:
            continue
        d = X[im][task[im] == 1].mean(0) - X[im][task[im] == 0].mean(0)
        # For cross-fitting, two observations in a stratum are sufficient to
        # contribute one independent residual direction.
        residuals = []
        for s in np.unique(strata[ic]):
            z = X[ic][strata[ic] == s]
            if len(z) >= 2:
                residuals.append(z - z.mean(0))
        if not residuals:
            continue
        cov = LedoitWolf().fit(np.vstack(residuals)).covariance_
        diag = np.maximum(np.diag(cov), 1e-12)
        scale = np.sqrt(diag)
        corr = cov / np.outer(scale, scale)
        np.fill_diagonal(corr, 0)
        dz = d / scale
        if np.linalg.norm(dz) > 0:
            u = dz / np.linalg.norm(dz)
            estimates.append(float(u @ corr @ u))
    return float(np.mean(estimates)) if estimates else np.nan


def shuffle_within(task, nuisance, seed):
    """Destroy the tested axis while preserving the orthogonal stimulus factor."""
    out = np.asarray(task).copy()
    rng = np.random.default_rng(seed)
    for value in np.unique(nuisance):
        ids = np.where(nuisance == value)[0]
        out[ids] = rng.permutation(out[ids])
    return out


def metrics(X, task, strata):
    if min(np.sum(task == 0), np.sum(task == 1)) < MIN_CELL:
        return None
    d = X[task == 1].mean(0) - X[task == 0].mean(0)
    cov = residual_covariance(X, strata)
    if cov is None or np.linalg.norm(d) == 0:
        return None
    diag = np.maximum(np.diag(cov), 1e-12)
    j_diag = float(np.sum(d * d / diag))
    try:
        j_full = float(d @ np.linalg.solve(cov, d))
    except np.linalg.LinAlgError:
        return None
    fisher = float(1 - j_full / j_diag) if j_diag > 0 and j_full > 0 else np.nan

    # Correlation-scale off-diagonal covariance projected on the task axis.
    scale = np.sqrt(diag)
    corr = cov / np.outer(scale, scale)
    np.fill_diagonal(corr, 0)
    dz = d / scale
    u = dz / np.linalg.norm(dz)
    align = float(u @ corr @ u)
    return fisher, align


def session_inference(rows, field, seed=91, B=4000):
    sessions = sorted(set(r["session"] for r in rows))
    delta = np.array([np.mean([r[field + "_a"] - r[field + "_p"] for r in rows if r["session"] == s]) for s in sessions])
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(delta, len(delta), True).mean() for _ in range(B)])
    perm = np.array([(delta * rng.choice((-1, 1), len(delta))).mean() for _ in range(B)])
    return {
        "n_pairs": len(rows), "n_sessions": len(sessions),
        "passive_mean": float(np.mean([r[field + "_p"] for r in rows])),
        "active_mean": float(np.mean([r[field + "_a"] for r in rows])),
        "delta_session_mean": float(delta.mean()),
        "session_boot_ci": np.percentile(boot, (2.5, 97.5)).tolist(),
        "session_signflip_p": float((np.abs(perm) >= abs(delta.mean())).mean()),
    }


rows = []
start = time.time()
for si, session in enumerate(SESSIONS):
    lab, subject, date = session.split("/")
    spikes = load_spikes(lab, subject, date)
    passive_file = find_passive(lab, subject, date)
    if spikes is None or passive_file is None:
        continue
    passive = pd.read_csv(passive_file, index_col=0)
    if not all(c in passive for c in ("start", "position", "contrast")):
        continue
    xp = trial_counts(spikes, passive["start"].to_numpy())
    side_p = (passive["position"].to_numpy() > 0).astype(int)
    contrast_p = np.array([contrast_bin(x) for x in passive["contrast"].to_numpy()])

    alf = IBL_DIR / lab / "Subjects" / subject / date / "001" / "alf"
    trial_files = glob.glob(str(alf / "*" / "_ibl_trials.table*")) + glob.glob(str(alf / "_ibl_trials.table*"))
    active = None
    for f in trial_files:
        try:
            candidate = pd.read_parquet(f)
            if len(candidate) > 10 and all(c in candidate for c in ("stimOn_times", "contrastLeft", "contrastRight")):
                active = candidate
                break
        except Exception:
            continue
    if active is None:
        continue
    left = np.nan_to_num(active["contrastLeft"].to_numpy())
    right = np.nan_to_num(active["contrastRight"].to_numpy())
    valid = ((left > 0) ^ (right > 0)) & np.isfinite(active["stimOn_times"].to_numpy())
    if valid.sum() < 12:
        continue
    xa = trial_counts(spikes, active["stimOn_times"].to_numpy()[valid])
    side_a = (left[valid] > right[valid]).astype(int)
    contrast_a = np.array([contrast_bin(x) for x in np.maximum(left[valid], right[valid])])

    match = matched_strata(side_p, contrast_p, side_a, contrast_a, seedof(session, "trials"))
    if match is None:
        continue
    ip, ia, cells = match
    # Combined stratum ID; residual covariance is computed within identical stimuli.
    strata_p = side_p[ip] * 10 + contrast_p[ip]
    strata_a = side_a[ia] * 10 + contrast_a[ia]
    by_region = {}
    for unit, region in enumerate(spikes["regions"]):
        by_region.setdefault(region, []).append(unit)
    for region, units in by_region.items():
        for n_units in UNIT_COUNTS:
            if len(units) < n_units:
                continue
            rng = np.random.default_rng(seedof(session, region, n_units))
            selected = rng.choice(units, n_units, False)
            # First-order geometry uses exactly the same matched trials and
            # selected units as the second-order state comparison.  The
            # condition taxonomy is the shared side x exact-contrast stratum
            # in both states.
            Xp_full, Xa_full = xp[ip][:, selected], xa[ia][:, selected]
            geom_p = condition_geometry(Xp_full, strata_p, k=3)
            geom_a = condition_geometry(Xa_full, strata_a, k=3)
            if geom_p is None or geom_a is None:
                continue
            for axis in ("side", "contrast"):
                if axis == "side":
                    task_p, task_a = side_p[ip], side_a[ia]
                    nuisance_p, nuisance_a = contrast_p[ip], contrast_a[ia]
                else:
                    # Exclude zero contrast; split identically at <=.25 versus >.25.
                    keep_p = contrast_p[ip] > 0
                    keep_a = contrast_a[ia] > 0
                    task_p = (contrast_p[ip][keep_p] > 2500).astype(int)
                    task_a = (contrast_a[ia][keep_a] > 2500).astype(int)
                    nuisance_p, nuisance_a = side_p[ip][keep_p], side_a[ia][keep_a]
                if axis == "side":
                    Xp, Xa = xp[ip][:, selected], xa[ia][:, selected]
                    mp = metrics(Xp, task_p, strata_p)
                    ma = metrics(Xa, task_a, strata_a)
                    cvp = crossfit_alignment(Xp, task_p, strata_p, seedof(session, region, n_units, axis, "p"))
                    cva = crossfit_alignment(Xa, task_a, strata_a, seedof(session, region, n_units, axis, "a"))
                    null_p = shuffle_within(task_p, nuisance_p, seedof(session, region, n_units, axis, "null_p"))
                    null_a = shuffle_within(task_a, nuisance_a, seedof(session, region, n_units, axis, "null_a"))
                    cvnp = crossfit_alignment(Xp, null_p, strata_p, seedof(session, region, n_units, axis, "np"))
                    cvna = crossfit_alignment(Xa, null_a, strata_a, seedof(session, region, n_units, axis, "na"))
                else:
                    Xp, Xa = xp[ip][keep_p][:, selected], xa[ia][keep_a][:, selected]
                    Sp, Sa = strata_p[keep_p], strata_a[keep_a]
                    mp = metrics(Xp, task_p, Sp)
                    ma = metrics(Xa, task_a, Sa)
                    cvp = crossfit_alignment(Xp, task_p, Sp, seedof(session, region, n_units, axis, "p"))
                    cva = crossfit_alignment(Xa, task_a, Sa, seedof(session, region, n_units, axis, "a"))
                    null_p = shuffle_within(task_p, nuisance_p, seedof(session, region, n_units, axis, "null_p"))
                    null_a = shuffle_within(task_a, nuisance_a, seedof(session, region, n_units, axis, "null_a"))
                    cvnp = crossfit_alignment(Xp, null_p, Sp, seedof(session, region, n_units, axis, "np"))
                    cvna = crossfit_alignment(Xa, null_a, Sa, seedof(session, region, n_units, axis, "na"))
                if mp is None or ma is None or not np.all(np.isfinite((*mp, *ma, cvp, cva, cvnp, cvna))):
                    continue
                rows.append({
                    "session": session, "region": region, "axis": axis,
                    "n_units": n_units, "n_trials": int(len(ip)),
                    "fisher_p": mp[0], "fisher_a": ma[0],
                    "align_p": mp[1], "align_a": ma[1],
                    "align_cv_p": cvp, "align_cv_a": cva,
                    "align_cv_null_p": cvnp, "align_cv_null_a": cvna,
                    "ps_p": geom_p[0], "ps_a": geom_a[0],
                    "lcr_p": geom_p[1], "lcr_a": geom_a[1],
                    "n_geometry_conditions": geom_p[2],
                })
    if (si + 1) % 25 == 0:
        print(si + 1, "/", len(SESSIONS), "rows", len(rows))

summary = {}
for axis in ("side", "contrast"):
    summary[axis] = {}
    for n_units in UNIT_COUNTS:
        rr = [r for r in rows if r["axis"] == axis and r["n_units"] == n_units]
        summary[axis][str(n_units)] = {
            "fisher": session_inference(rr, "fisher", seed=100 + n_units),
            "alignment": session_inference(rr, "align", seed=200 + n_units),
            "alignment_crossfit": session_inference(rr, "align_cv", seed=300 + n_units),
            "alignment_crossfit_label_null": session_inference(rr, "align_cv_null", seed=400 + n_units),
        } if rr else None

json.dump({"summary": summary, "rows": rows}, open(OUT, "w"), indent=1)
print(json.dumps(summary, indent=2))
print("saved", OUT, "seconds", time.time() - start)
