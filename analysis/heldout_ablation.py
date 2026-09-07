import os
from pathlib import Path
ROOT = Path(os.environ.get('SPARSE_CODING_ROOT', Path(__file__).resolve().parents[1]))
"""P15: repeated bidirectional split-half targeted ablation in Steinmetz.

For every session x region, condition-specific trials are repeatedly split in
half. Unit reuse is defined only in one half and geometry loss is evaluated in
the held-out half; train/test halves are then swapped. Repeats and folds are
averaged within region before session-level inference.
"""

import json
import math
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import integrate, stats
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

HERE = Path(__file__).parent
ROOT = HERE.parent
DATA = ROOT / "tmp" / "steinmetz_extracted"
OUT = HERE / "outputs" / "p15_biological_heldout_ablation.json"

SESSIONS = [
    "Richards_2017-10-30", "Richards_2017-11-01",
    "Forssmann_2017-11-05", "Hench_2017-06-15",
    "Forssmann_2017-11-02", "Hench_2017-06-16",
]
MIN_UNITS = 20
MIN_CONDITION_TRIALS_PER_HALF = 6
N_REPEATS = 50
N_MATCH_DRAWS = 500
N_BOOT = 10000


def seedof(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0xFFFFFFFF


def counts(spike_times, spike_clusters, n_units, onsets):
    order = np.argsort(spike_times)
    spike_times = spike_times[order]
    spike_clusters = spike_clusters[order]
    X = np.zeros((len(onsets), n_units), np.float32)
    for i, onset in enumerate(onsets):
        lo = np.searchsorted(spike_times, onset)
        hi = np.searchsorted(spike_times, onset + .5)
        units = spike_clusters[lo:hi]
        units = units[(units >= 0) & (units < n_units)]
        if len(units):
            np.add.at(X[i], units, 1)
    return X / .5


def rdm(matrix):
    return pdist(np.maximum(matrix, 0), metric="cosine")


def lcr(matrix, k=2):
    matrix = np.maximum(matrix, 0)
    n_conditions = len(matrix)
    if n_conditions < k + 2:
        return np.nan
    centered = matrix - matrix.mean(1, keepdims=True)
    centered /= np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)
    distance = 1 - centered @ centered.T
    values = []
    for i in range(n_conditions):
        row = np.delete(distance[i], i)
        row = np.sort(row)
        denominator = row[k:].mean()
        values.append(row[:k].mean() / denominator if denominator > 0 else np.nan)
    return float(np.nanmean(values))


def recovery(test_means, retained, full_rdm):
    if len(retained) < 2:
        return np.nan
    reduced = rdm(test_means[:, retained])
    if not np.all(np.isfinite(reduced)) or np.std(reduced) == 0:
        return np.nan
    return float(spearmanr(reduced, full_rdm).statistic)


def reuse_score(train_means):
    """Mean normalized co-activation for nearest condition pairs in training."""
    positive = np.maximum(train_means, 0)
    by_unit = positive / np.maximum(positive.sum(0, keepdims=True), 1e-12)
    centered = train_means - train_means.mean(1, keepdims=True)
    centered /= np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-12)
    distance = 1 - centered @ centered.T
    pairs = []
    for i in range(len(train_means)):
        row = distance[i].copy()
        row[i] = np.inf
        pairs.append((i, int(np.argmin(row))))
    normalized = by_unit / np.maximum(by_unit.max(0, keepdims=True), 1e-12)
    return np.mean([np.minimum(normalized[i], normalized[j]) for i, j in pairs], axis=0)


def matched_set(target, candidates, mean_fr, signal_var, n_draws, rng):
    """Match set size and training response magnitude/signal variance."""
    target_feature = np.array([
        np.mean(np.log1p(mean_fr[target])),
        np.mean(np.log1p(signal_var[target])),
    ])
    best = None
    best_error = np.inf
    if len(candidates) < len(target):
        return None, np.nan
    for _ in range(n_draws):
        chosen = rng.choice(candidates, len(target), replace=False)
        feature = np.array([
            np.mean(np.log1p(mean_fr[chosen])),
            np.mean(np.log1p(signal_var[chosen])),
        ])
        error = float(np.sum((feature - target_feature) ** 2))
        if error < best_error:
            best, best_error = chosen, error
    return best, best_error


def fold_effect(train_means, test_means, rng):
    full = rdm(test_means)
    if not np.all(np.isfinite(full)) or np.std(full) == 0:
        return None
    full_lcr = lcr(test_means)
    score = reuse_score(train_means)
    n_units = train_means.shape[1]
    n_remove = max(3, int(round(.25 * n_units)))
    order = np.argsort(score)
    unique = order[:n_remove]
    shared = order[-n_remove:]
    middle = np.setdiff1d(np.arange(n_units), np.r_[shared, unique])
    mean_fr = np.maximum(train_means.mean(0), 0)
    signal_var = np.var(train_means, axis=0)
    random_set, match_error = matched_set(
        shared, middle, mean_fr, signal_var, N_MATCH_DRAWS, rng)
    if random_set is None:
        # Small populations may not have enough middle units; retain strict
        # exclusion of shared units while allowing low-reuse candidates.
        candidates = np.setdiff1d(np.arange(n_units), shared)
        random_set, match_error = matched_set(
            shared, candidates, mean_fr, signal_var, N_MATCH_DRAWS, rng)
    all_units = np.arange(n_units)
    losses = {}
    for key, removed in (("shared", shared), ("random", random_set), ("unique", unique)):
        retained = np.setdiff1d(all_units, removed)
        rec = recovery(test_means, retained, full)
        losses[key] = 1 - rec if np.isfinite(rec) else np.nan
        reduced_lcr = lcr(test_means[:, retained])
        losses["lcr_change_" + key] = reduced_lcr - full_lcr
    losses["shared_minus_random"] = losses["shared"] - losses["random"]
    losses["shared_minus_unique"] = losses["shared"] - losses["unique"]
    losses["random_minus_unique"] = losses["random"] - losses["unique"]
    losses["lcr_shared_minus_random"] = (losses["lcr_change_shared"]
                                          - losses["lcr_change_random"])
    losses["lcr_shared_minus_unique"] = (losses["lcr_change_shared"]
                                          - losses["lcr_change_unique"])
    losses["match_error"] = match_error
    return losses


def jzs_bf_t(t_value, n, df=None, rscale=math.sqrt(.5)):
    if df is None:
        df = n - 1
    t_value = abs(float(t_value))
    if t_value == 0:
        return 1.0

    def integrand(g):
        return ((1 + n * g * rscale**2) ** -.5
                * (1 + t_value**2 / ((1 + n * g * rscale**2) * df)) ** (-(df + 1) / 2)
                * (2 * math.pi) ** -.5 * g ** -1.5 * math.exp(-1 / (2 * g)))

    numerator = integrate.quad(integrand, 0, np.inf, epsabs=1e-10, limit=500)[0]
    denominator = (1 + t_value**2 / df) ** (-(df + 1) / 2)
    return float(numerator / denominator)


def exact_signflip(values):
    values = np.asarray(values, float)
    signs = np.array(np.meshgrid(*[[-1, 1]] * len(values))).T.reshape(-1, len(values))
    null = np.mean(signs * values, axis=1)
    return float(np.mean(np.abs(null) >= abs(values.mean()) - 1e-15))


def summarize(rows, field, seed):
    by_session = defaultdict(list)
    for row in rows:
        if np.isfinite(row[field]):
            by_session[row["session"]].append(row[field])
    session_names = sorted(by_session)
    values = np.asarray([np.mean(by_session[s]) for s in session_names], float)
    test = stats.ttest_1samp(values, 0)
    dz = float(values.mean() / values.std(ddof=1))
    bf10 = jzs_bf_t(test.statistic, len(values))
    rng = np.random.default_rng(seed)
    boot = np.asarray([rng.choice(values, len(values), replace=True).mean()
                       for _ in range(N_BOOT)])
    return {
        "n_sessions": len(values),
        "n_regions": int(sum(np.isfinite(r[field]) for r in rows)),
        "session_names": session_names,
        "session_values": values.tolist(),
        "mean": float(values.mean()),
        "session_cluster_boot_ci": np.quantile(boot, [.025, .975]).tolist(),
        "cohen_dz": dz,
        "t": float(test.statistic),
        "df": int(test.df),
        "p_t": float(test.pvalue),
        "p_exact_signflip": exact_signflip(values),
        "bf10": bf10,
        "bf01": 1 / bf10,
        "positive_sessions": int(np.sum(values > 0)),
        "positive_region_fraction": float(np.mean([r[field] > 0 for r in rows
                                                   if np.isfinite(r[field])])),
    }


region_accumulator = defaultdict(lambda: defaultdict(list))
metadata = {}

for session in SESSIONS:
    directory = DATA / session
    required = [
        "clusters.peakChannel.npy", "channels.brainLocation.tsv",
        "spikes.times.npy", "spikes.clusters.npy", "trials.goCue_times.npy",
        "trials.feedbackType.npy", "trials.response_choice.npy",
    ]
    if not all((directory / f).exists() for f in required):
        continue
    peak_channel = np.load(directory / required[0]).ravel().astype(int)
    lines = (directory / required[1]).read_text().strip().split("\n")
    channel_region = np.array([
        line.split("\t")[3] if len(line.split("\t")) >= 4 else "root"
        for line in lines[1:]
    ])
    regions = np.array([
        channel_region[c - 1] if 0 <= c - 1 < len(channel_region) else "root"
        for c in peak_channel
    ])
    spike_times = np.load(directory / required[2]).ravel()
    spike_clusters = np.load(directory / required[3]).ravel().astype(int)
    cues = np.load(directory / required[4]).ravel()
    feedback = np.load(directory / required[5]).ravel()
    choices = np.load(directory / required[6]).ravel()
    X = counts(spike_times, spike_clusters, len(peak_channel), cues)
    labels = np.array([
        ("R" if c > 0 else ("L" if c < 0 else "N"))
        + ("_C" if f > 0 else "_E") if f != 0 else ""
        for c, f in zip(choices, feedback)
    ])
    valid = [label for label in np.unique(labels)
             if label and np.sum(labels == label) >= 2 * MIN_CONDITION_TRIALS_PER_HALF]
    by_region = defaultdict(list)
    for i, region in enumerate(regions):
        if region != "root":
            by_region[region].append(i)

    for repeat in range(N_REPEATS):
        rng_split = np.random.default_rng(seedof(session, "split", repeat))
        half_a, half_b = {}, {}
        for label in valid:
            ids = rng_split.permutation(np.flatnonzero(labels == label))
            midpoint = len(ids) // 2
            half_a[label], half_b[label] = ids[:midpoint], ids[midpoint:]
        for region, unit_list in by_region.items():
            if len(unit_list) < MIN_UNITS or len(valid) < 4:
                continue
            units = np.asarray(unit_list, int)
            means_a = np.asarray([X[half_a[c]][:, units].mean(0) for c in valid])
            means_b = np.asarray([X[half_b[c]][:, units].mean(0) for c in valid])
            for fold, train, test in (("A_to_B", means_a, means_b),
                                      ("B_to_A", means_b, means_a)):
                rng_match = np.random.default_rng(seedof(session, region, repeat, fold))
                effect = fold_effect(train, test, rng_match)
                if effect is None:
                    continue
                key = (session, region)
                for field, value in effect.items():
                    region_accumulator[key][field].append(value)
                    region_accumulator[key][fold + "__" + field].append(value)

rows = []
for (session, region), fields in sorted(region_accumulator.items()):
    n_folds = len(fields["shared_minus_random"])
    row = {"session": session, "region": region, "n_crossfit_folds": n_folds}
    for field, values in fields.items():
        row[field] = float(np.nanmean(values))
        row[field + "_fold_sd"] = float(np.nanstd(values, ddof=1))
    rows.append(row)

summary = {
    "shared_minus_random": summarize(rows, "shared_minus_random", 1501),
    "shared_minus_unique": summarize(rows, "shared_minus_unique", 1502),
    "random_minus_unique": summarize(rows, "random_minus_unique", 1503),
    "loss_shared": summarize(rows, "shared", 1504),
    "loss_random": summarize(rows, "random", 1505),
    "loss_unique": summarize(rows, "unique", 1506),
    "A_to_B_shared_minus_random": summarize(rows, "A_to_B__shared_minus_random", 1511),
    "B_to_A_shared_minus_random": summarize(rows, "B_to_A__shared_minus_random", 1512),
    "A_to_B_shared_minus_unique": summarize(rows, "A_to_B__shared_minus_unique", 1513),
    "B_to_A_shared_minus_unique": summarize(rows, "B_to_A__shared_minus_unique", 1514),
    "lcr_shared_minus_random": summarize(rows, "lcr_shared_minus_random", 1521),
    "lcr_shared_minus_unique": summarize(rows, "lcr_shared_minus_unique", 1522),
}

output = {
    "metadata": {
        "sessions": SESSIONS,
        "n_repeats": N_REPEATS,
        "folds_per_repeat": 2,
        "selection_fraction": .25,
        "minimum_units": MIN_UNITS,
        "minimum_trials_per_condition_per_half": MIN_CONDITION_TRIALS_PER_HALF,
        "definition": "reuse ranked in training half; RDM loss evaluated only in held-out half",
        "inference": "crossfit folds averaged within region; regions averaged within session; N=6 sessions",
    },
    "summary": summary,
    "rows": rows,
}
OUT.write_text(json.dumps(output, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=2))
print("saved", OUT)
