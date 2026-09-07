import os
from pathlib import Path
ROOT = Path(os.environ.get('SPARSE_CODING_ROOT', Path(__file__).resolve().parents[1]))
"""P14 v2: constrained biological nulls for the Steinmetz PS-LCR law.

Null A independently permutes response values across neuronal identities for
each condition. Null B randomizes the binary condition-by-neuron support with
degree-preserving double-edge swaps, retaining both the number of active units
per condition and the number of conditions represented by each neuron; each
condition's original positive response values are then reassigned to its new
support. Both nulls preserve every condition's response-value multiset and PS.
"""

import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy import stats


OUT = ROOT / "outputs" / "p14_biological_null_model_v2.json"
N_PERM = 200
RNG = np.random.default_rng(20260826)
STEIN_EXTRACT = ROOT / "tmp" / "steinmetz_extracted"
SESSIONS = ['Forssmann_2017-11-02', 'Forssmann_2017-11-05',
            'Hench_2017-06-15', 'Hench_2017-06-16',
            'Richards_2017-10-30', 'Richards_2017-11-01']
MIN_TRIALS, MIN_UNITS = 5, 20
BEHAV_WIN = (0.0, 0.5)


def population_sparseness(a):
    a = np.asarray(a, float)
    n = len(a)
    if n <= 1 or np.sum(a*a) <= 0:
        return np.nan
    return (1 - np.sum(a)**2 / (n*np.sum(a*a))) / (1 - 1/n)


def condition_nnr(R, k_nn=2):
    Rc = R - R.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Rc, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    dist = 1 - (Rc/norms) @ (Rc/norms).T
    ratios = []
    for i in range(len(R)):
        idx = np.arange(len(R)) != i
        values = np.sort(dist[i, idx])
        ratios.append(np.mean(values[:k_nn]) / np.mean(values[k_nn:]))
    return float(np.nanmean(ratios))


def load_session(name):
    sd = STEIN_EXTRACT / name
    peak = np.load(sd / 'clusters.peakChannel.npy').ravel().astype(int)
    lines = (sd / 'channels.brainLocation.tsv').read_text().strip().split('\n')
    regions = np.array([line.split('\t')[3] if len(line.split('\t')) >= 4 else 'root'
                        for line in lines[1:]])
    region_by_cluster = [regions[peak[i]-1] if 0 <= peak[i]-1 < len(regions) else 'root'
                         for i in range(len(peak))]
    spike_t = np.load(sd / 'spikes.times.npy').ravel()
    spike_c = np.load(sd / 'spikes.clusters.npy').ravel().astype(int)
    go = np.load(sd / 'trials.goCue_times.npy').ravel()
    choice = np.load(sd / 'trials.response_choice.npy').ravel()
    feedback = np.load(sd / 'trials.feedbackType.npy').ravel()
    condition = np.full(len(go), -1, int)
    for i in range(len(go)):
        if feedback[i] == 0:
            continue
        choice_code = 1 if choice[i] > 0 else (0 if choice[i] < 0 else 2)
        condition[i] = choice_code*2 + (1 if feedback[i] > 0 else 0)
    order = np.argsort(spike_t); spike_t = spike_t[order]; spike_c = spike_c[order]
    counts = np.zeros((len(go), len(region_by_cluster)), np.float32)
    for i, onset in enumerate(go):
        lo = np.searchsorted(spike_t, onset+BEHAV_WIN[0])
        hi = np.searchsorted(spike_t, onset+BEHAV_WIN[1])
        ids = spike_c[lo:hi]
        valid = (ids >= 0) & (ids < counts.shape[1])
        np.add.at(counts[i], ids[valid], 1)
    counts /= BEHAV_WIN[1]-BEHAV_WIN[0]
    region_units = defaultdict(list)
    for i, region in enumerate(region_by_cluster):
        if region != 'root':
            region_units[region].append(i)
    return counts, condition, region_units


def region_means(counts, condition, units):
    units = np.asarray(units)
    rows = [counts[condition == c][:, units].mean(0) for c in np.unique(condition)
            if np.sum(condition == c) >= MIN_TRIALS]
    return np.maximum(np.asarray(rows), 0) if len(rows) >= 4 else None


def degree_preserving_support_null(R, rng):
    support = R > 0
    B = support.copy()
    nrow, ncol = B.shape
    # A capped Markov-chain length is sufficient for these 4-6 condition
    # matrices and keeps the full 54-region null audit reproducible.
    n_one = int(B.sum()); n_zero = int(B.size - n_one)
    target = max(1, min(200, 5*n_one, 5*n_zero))
    accepted = 0
    for _ in range(target * 20):
        if accepted >= target:
            break
        r1, r2 = rng.choice(nrow, 2, replace=False)
        c10 = np.flatnonzero(B[r1] & ~B[r2])
        c01 = np.flatnonzero(~B[r1] & B[r2])
        if len(c10) and len(c01):
            c1 = rng.choice(c10); c2 = rng.choice(c01)
            B[r1, c1] = False; B[r1, c2] = True
            B[r2, c1] = True;  B[r2, c2] = False
            accepted += 1
    Rn = np.zeros_like(R)
    for i in range(nrow):
        values = R[i, support[i]].copy()
        rng.shuffle(values)
        Rn[i, np.flatnonzero(B[i])] = values
    return Rn, accepted


matrices = []
for sn in SESSIONS:
    tc, cond, ru = load_session(sn)
    for reg, units in ru.items():
        if len(units) < MIN_UNITS:
            continue
        R = region_means(tc, cond, units)
        if R is None:
            continue
        matrices.append((sn, reg, R))

n = len(matrices)
ps = np.zeros(n); real = np.zeros(n)
null_ind = np.zeros((N_PERM, n)); null_degree = np.full((N_PERM, n), np.nan)
swaps = np.zeros((N_PERM, n), int)
audit = {"max_ps_error_independent": 0.0, "max_ps_error_degree": 0.0,
         "max_row_degree_error": 0, "max_column_degree_error": 0}

for j, (session, region, R) in enumerate(matrices):
    ps[j] = np.mean([population_sparseness(row) for row in R])
    real[j] = condition_nnr(R, k_nn=2)
    original_row_degree = (R > 0).sum(1)
    original_col_degree = (R > 0).sum(0)
    for b in range(N_PERM):
        Ri = np.vstack([RNG.permutation(row) for row in R])
        null_ind[b, j] = condition_nnr(Ri, k_nn=2)
        pi = np.mean([population_sparseness(row) for row in Ri])
        audit["max_ps_error_independent"] = max(audit["max_ps_error_independent"], abs(pi-ps[j]))

        Rd, accepted = degree_preserving_support_null(R, RNG)
        swaps[b, j] = accepted
        if accepted > 0:
            null_degree[b, j] = condition_nnr(Rd, k_nn=2)
            pd = np.mean([population_sparseness(row) for row in Rd])
            audit["max_ps_error_degree"] = max(audit["max_ps_error_degree"], abs(pd-ps[j]))
            audit["max_row_degree_error"] = max(audit["max_row_degree_error"],
                                                  int(np.max(abs((Rd>0).sum(1)-original_row_degree))))
            audit["max_column_degree_error"] = max(audit["max_column_degree_error"],
                                                     int(np.max(abs((Rd>0).sum(0)-original_col_degree))))


def corr_distribution(arr):
    vals = np.asarray([stats.spearmanr(ps[np.isfinite(row)], row[np.isfinite(row)]).statistic
                       for row in arr])
    observed = float(stats.spearmanr(ps, real).statistic)
    return {"observed_real_r": observed, "null_r_draws": vals.tolist(),
            "null_r_mean": float(np.nanmean(vals)),
            "null_r_95_interval": list(map(float, np.nanquantile(vals, [.025, .975]))),
            "empirical_p_real_as_or_more_negative": float((1 + np.sum(vals <= observed)) / (1 + np.sum(np.isfinite(vals))))}


sessions = np.asarray(sorted(set(x[0] for x in matrices)), object)
by_session = defaultdict(list)
for i, (session, _, _) in enumerate(matrices):
    by_session[session].append(i)
boot_real = []
for _ in range(4000):
    draw = RNG.choice(sessions, len(sessions), replace=True)
    idx = np.concatenate([by_session[s] for s in draw])
    boot_real.append(stats.spearmanr(ps[idx], real[idx]).statistic)

rows = []
for i, (session, region, R) in enumerate(matrices):
    rows.append({"session": session, "region": region, "n_conditions": int(R.shape[0]),
                 "n_units": int(R.shape[1]), "ps": float(ps[i]), "lcr_real": float(real[i]),
                 "lcr_independent_null_mean": float(np.mean(null_ind[:, i])),
                 "lcr_degree_null_mean": float(np.nanmean(null_degree[:, i])),
                 "degree_null_valid_fraction": float(np.mean(swaps[:, i] > 0)),
                 "degree_null_median_swaps": float(np.median(swaps[:, i]))})

result = {
    "metadata": {"n_regions": n, "n_sessions": len(sessions), "n_permutations": N_PERM,
                 "interpretation_limit": "supports structured cross-condition neuronal identity; does not establish learning"},
    "real": {"r": float(stats.spearmanr(ps, real).statistic),
             "session_cluster_boot_ci": list(map(float, np.quantile(boot_real, [.025, .975]))),
             "mean_lcr": float(np.mean(real))},
    "independent_conditionwise_value_shuffle": {**corr_distribution(null_ind),
                                                  "mean_lcr": float(np.mean(null_ind))},
    "degree_preserving_support_shuffle": {**corr_distribution(null_degree),
                                            "mean_lcr": float(np.nanmean(null_degree)),
                                            "valid_fraction": float(np.mean(swaps > 0))},
    "preservation_audit": audit,
    "rows": rows,
}
OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({k:v for k,v in result.items() if k != "rows"}, indent=2))
print(f"Saved {OUT}")
