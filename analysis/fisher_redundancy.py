"""Shared second-order analysis used by active/passive replication pipelines.

The module separates stimulus-conditioned means from within-stimulus trial
covariance.  It contains no dataset-specific I/O and performs no work on
import, so future datasets can use exactly the same estimators.
"""
from __future__ import annotations
import os
from pathlib import Path
ROOT = Path(os.environ.get('SPARSE_CODING_ROOT', Path(__file__).resolve().parents[1]))

import itertools
from collections import defaultdict

import numpy as np
from sklearn.covariance import LedoitWolf


MIN_CELL = 3


def seedof(*parts):
    import zlib
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0xFFFFFFFF


def residual_covariance(X, strata, min_cell=MIN_CELL):
    residuals = []
    for value in np.unique(strata):
        z = X[strata == value]
        if len(z) >= min_cell:
            residuals.append(z - z.mean(0))
    if not residuals:
        return None
    return LedoitWolf().fit(np.vstack(residuals)).covariance_


def metrics(X, task, strata):
    """Return shrinkage Fisher redundancy and covariance/task-axis alignment."""
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
    scale = np.sqrt(diag)
    corr = cov / np.outer(scale, scale)
    np.fill_diagonal(corr, 0)
    dz = d / scale
    u = dz / np.linalg.norm(dz)
    alignment = float(u @ corr @ u)
    return fisher, alignment


def _split_strata(strata, seed):
    rng = np.random.default_rng(seed)
    halves = [[], []]
    for value in np.unique(strata):
        ids = np.where(strata == value)[0]
        if len(ids) < 4:
            continue
        ids = rng.permutation(ids)
        cut = len(ids) // 2
        halves[0].extend(ids[:cut])
        halves[1].extend(ids[cut:])
    return [np.asarray(x, int) for x in halves]


def crossfit_alignment(X, task, strata, seed):
    """Estimate the task axis and residual covariance on disjoint trials."""
    halves = _split_strata(strata, seed)
    if not len(halves[0]) or not len(halves[1]):
        return np.nan
    estimates = []
    for mean_half, cov_half in ((0, 1), (1, 0)):
        im, ic = halves[mean_half], halves[cov_half]
        if min(np.sum(task[im] == 0), np.sum(task[im] == 1)) < 2:
            continue
        d = X[im][task[im] == 1].mean(0) - X[im][task[im] == 0].mean(0)
        residuals = []
        for value in np.unique(strata[ic]):
            z = X[ic][strata[ic] == value]
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


def clustered_inference(rows, field, cluster="animal", seed=91, B=5000):
    """Equal-weight clustered paired inference on active-minus-passive values."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[cluster]].append(row[field + "_a"] - row[field + "_p"])
    labels = sorted(grouped)
    delta = np.asarray([np.mean(grouped[label]) for label in labels], float)
    rng = np.random.default_rng(seed)
    boot = np.asarray([rng.choice(delta, len(delta), True).mean() for _ in range(B)])
    if len(delta) <= 16:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(delta))))
        perm = (signs * delta).mean(1)
    else:
        perm = np.asarray([(delta * rng.choice((-1, 1), len(delta))).mean() for _ in range(B)])
    observed = float(delta.mean())
    return {
        "n_rows": len(rows),
        "n_clusters": len(labels),
        "cluster": cluster,
        "passive_mean_row_weighted": float(np.mean([r[field + "_p"] for r in rows])),
        "active_mean_row_weighted": float(np.mean([r[field + "_a"] for r in rows])),
        "delta_cluster_mean": observed,
        "cluster_boot_ci": np.percentile(boot, (2.5, 97.5)).tolist(),
        "cluster_signflip_p_two_sided": float((np.abs(perm) >= abs(observed) - 1e-15).mean()),
        "cluster_deltas": {str(k): float(np.mean(grouped[k])) for k in labels},
    }

