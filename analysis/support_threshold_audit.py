import os
from pathlib import Path
ROOT = Path(os.environ.get('SPARSE_CODING_ROOT', Path(__file__).resolve().parents[1]))
"""P14 threshold audit for biological structured-support nulls.

Extends P14 v2 beyond the exact-zero definition of support.  For every
condition, high-response support is defined either by positive mean response
or by the top 10, 20, 30 or 40 percent of units.  Degree-preserving 2x2 edge
swaps retain the number of supported units per condition and the number of
conditions in which each unit is highly active.  The complete response-value
multiset of every condition, and therefore Treves-Rolls PS, is preserved.

Inference:
  * empirical null distribution for the across-region PS-LCR correlation;
  * session-level paired effect of support randomization on mean LCR;
  * session-level paired effect on within-session PS-LCR correlation.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import integrate, stats

# Importing v2 reconstructs the canonical 54 condition-mean matrices.  Its
# 200-permutation provenance output is retained; this audit writes a separate
# canonical threshold-sensitivity file.
import p14_biological_null_model_v2 as base



OUT = ROOT / "outputs" / "p14_biological_support_threshold_audit.json"
RNG = np.random.default_rng(20260827)
SUPPORT_MODES = {
    # The original support definition is the primary audit (5,000 draws).
    # Rank-threshold definitions are sensitivity analyses (1,000 each).
    "positive_mean": (None, 5000),
    "top_10pct": (0.10, 1000),
    "top_20pct": (0.20, 1000),
    "top_30pct": (0.30, 1000),
    "top_40pct": (0.40, 1000),
}


def jzs_bf_t(t_value, n, df=None, rscale=math.sqrt(0.5)):
    if df is None:
        df = n - 1
    t_value = abs(float(t_value))
    if t_value == 0:
        return 1.0

    def integrand(g):
        return ((1 + n * g * rscale**2) ** -0.5
                * (1 + t_value**2 / ((1 + n * g * rscale**2) * df)) ** (-(df + 1) / 2)
                * (2 * math.pi) ** -0.5 * g ** -1.5 * math.exp(-1 / (2 * g)))

    numerator = integrate.quad(integrand, 0, np.inf, epsabs=1e-10, limit=500)[0]
    denominator = (1 + t_value**2 / df) ** (-(df + 1) / 2)
    return float(numerator / denominator)


def support_mask(R, fraction):
    if fraction is None:
        return R > 0
    n_keep = max(1, int(math.ceil(R.shape[1] * fraction)))
    mask = np.zeros_like(R, dtype=bool)
    for i, row in enumerate(R):
        # Stable sorting makes ties deterministic; the randomization concerns
        # unit identity after the observed support has been defined.
        idx = np.argsort(row, kind="mergesort")[-n_keep:]
        mask[i, idx] = True
    return mask


def degree_shuffle(B, rng):
    out = B.copy()
    nrow = out.shape[0]
    n_one = int(out.sum())
    n_zero = int(out.size - n_one)
    # Twenty accepted switches are sufficient to move identity assignments in
    # these very small (4-6 condition) bipartite matrices; many independent
    # chains are used instead of one excessively long chain.
    target = max(1, min(20, 3 * n_one, 3 * n_zero))
    accepted = 0
    for _ in range(target * 20):
        if accepted >= target:
            break
        r1, r2 = rng.choice(nrow, 2, replace=False)
        c10 = np.flatnonzero(out[r1] & ~out[r2])
        c01 = np.flatnonzero(~out[r1] & out[r2])
        if len(c10) and len(c01):
            c1 = int(rng.choice(c10)); c2 = int(rng.choice(c01))
            out[r1, c1] = False; out[r1, c2] = True
            out[r2, c1] = True; out[r2, c2] = False
            accepted += 1
    return out, accepted


def randomized_matrix(R, observed_support, randomized_support, rng):
    out = np.empty_like(R)
    for i in range(R.shape[0]):
        high = R[i, observed_support[i]].copy()
        low = R[i, ~observed_support[i]].copy()
        rng.shuffle(high); rng.shuffle(low)
        out[i, randomized_support[i]] = high
        out[i, ~randomized_support[i]] = low
    return out


def paired_session_summary(real_values, null_mean_values, sessions):
    unique = sorted(set(sessions))
    delta = []
    for session in unique:
        idx = np.array([s == session for s in sessions])
        paired = null_mean_values[idx] - real_values[idx]
        paired = paired[np.isfinite(paired)]
        if paired.size:
            delta.append(float(np.mean(paired)))
    delta = np.asarray(delta)
    mean = float(delta.mean())
    sd = float(delta.std(ddof=1))
    dz = mean / sd if sd > 0 else np.inf
    test = stats.ttest_1samp(delta, 0)
    bf10 = jzs_bf_t(test.statistic, len(delta))
    return {
        "n_sessions": len(delta), "session_deltas": delta.tolist(),
        "mean_delta": mean, "cohen_dz": float(dz),
        "t": float(test.statistic), "p_t": float(test.pvalue),
        "bf10": bf10, "bf01": 1 / bf10,
    }


matrices = base.matrices
ps = np.asarray(base.ps, float)
real = np.asarray(base.real, float)
sessions = np.asarray([x[0] for x in matrices], object)
observed_r = float(stats.spearmanr(ps, real).statistic)
result = {
    "metadata": {
        "n_regions": len(matrices), "n_sessions": len(set(sessions)),
        "n_permutations": {k: v[1] for k, v in SUPPORT_MODES.items()},
        "interpretation_limit": "structured cross-condition neuronal identity; not evidence of learning",
    },
    "real": {"r": observed_r, "mean_lcr": float(real.mean())},
    "support_definitions": {},
}

for mode, (fraction, n_perm) in SUPPORT_MODES.items():
    null_lcr = np.full((n_perm, len(matrices)), np.nan)
    valid = np.zeros((n_perm, len(matrices)), bool)
    max_ps_error = 0.0
    max_row_error = 0
    max_col_error = 0
    for j, (_, _, R) in enumerate(matrices):
        S = support_mask(R, fraction)
        row_degree = S.sum(1)
        col_degree = S.sum(0)
        for b in range(n_perm):
            Sn, accepted = degree_shuffle(S, RNG)
            if accepted == 0:
                continue
            Rn = randomized_matrix(R, S, Sn, RNG)
            valid[b, j] = True
            null_lcr[b, j] = base.condition_nnr(Rn, k_nn=2)
            pn = float(np.mean([base.population_sparseness(row) for row in Rn]))
            max_ps_error = max(max_ps_error, abs(pn - ps[j]))
            max_row_error = max(max_row_error, int(np.max(abs(Sn.sum(1) - row_degree))))
            max_col_error = max(max_col_error, int(np.max(abs(Sn.sum(0) - col_degree))))

    null_r = np.asarray([
        stats.spearmanr(ps[np.isfinite(row)], row[np.isfinite(row)]).statistic
        for row in null_lcr
    ], float)
    null_region_mean = np.nanmean(null_lcr, axis=0)

    # Within-session correlation loss is summarized at the six independent
    # sessions. Sessions with fewer than three valid regions are undefined.
    real_session_r, null_session_r = [], []
    for session in sorted(set(sessions)):
        idx = sessions == session
        keep = idx & np.isfinite(null_region_mean)
        if keep.sum() >= 3:
            real_session_r.append(float(stats.spearmanr(ps[keep], real[keep]).statistic))
            null_session_r.append(float(stats.spearmanr(ps[keep], null_region_mean[keep]).statistic))
    real_session_r = np.asarray(real_session_r)
    null_session_r = np.asarray(null_session_r)
    corr_delta = null_session_r - real_session_r
    corr_test = stats.ttest_1samp(corr_delta, 0)
    corr_bf = jzs_bf_t(corr_test.statistic, len(corr_delta))

    result["support_definitions"][mode] = {
        "fraction": fraction,
        "null_r_mean": float(np.nanmean(null_r)),
        "null_r_95_interval": np.nanquantile(null_r, [0.025, 0.975]).tolist(),
        "n_permutations": n_perm,
        "empirical_p_real_as_or_more_negative": float((1 + np.sum(null_r <= observed_r)) / (n_perm + 1)),
        "null_mean_lcr": float(np.nanmean(null_lcr)),
        "valid_region_fraction": float(np.mean(np.any(valid, axis=0))),
        "valid_draw_fraction": float(np.mean(valid)),
        "preservation": {
            "max_ps_error": max_ps_error,
            "max_row_degree_error": max_row_error,
            "max_column_degree_error": max_col_error,
        },
        "session_level_lcr_disruption": paired_session_summary(real, null_region_mean, sessions),
        "session_level_correlation_attenuation": {
            "n_sessions": int(len(corr_delta)),
            "real_session_r": real_session_r.tolist(),
            "null_session_r": null_session_r.tolist(),
            "mean_null_minus_real": float(corr_delta.mean()),
            "cohen_dz": float(corr_delta.mean() / corr_delta.std(ddof=1)),
            "t": float(corr_test.statistic), "p_t": float(corr_test.pvalue),
            "bf10": corr_bf, "bf01": 1 / corr_bf,
        },
    }
    print(mode, json.dumps(result["support_definitions"][mode], indent=2))

OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
print("Saved", OUT)
