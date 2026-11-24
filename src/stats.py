"""Statistical tests and effect sizes."""
from __future__ import annotations

from math import comb

import numpy as np
from scipy.stats import rankdata


def conditional_rank_sum_p(a, b) -> float:
    """Two-sided conditional exact rank-sum p value, retaining ties.

    Condition on the pooled observations and the two sample sizes. Equal values
    receive average ranks. Dynamic programming counts every fixed-size label
    assignment by its rank sum, including assignments with identical rank sums.
    The p value is twice the smaller inclusive tail probability, capped at one.
    This is an exhaustive distribution, with no Monte Carlo sampling.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.size == 0 or b.size == 0:
        raise ValueError("Both samples must be nonempty one-dimensional arrays.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Both samples must contain only finite values.")

    # Counting the smaller group's assignments reduces the size of the table.
    if a.size > b.size:
        a, b = b, a
    n = len(a)
    # Average ranks are integers or half-integers; double them for integer sums.
    ranks = np.rint(2 * rankdata(np.concatenate([a, b]), method="average")).astype(int)
    observed = int(ranks[:n].sum())
    counts = [{} for _ in range(n + 1)]
    counts[0][0] = 1
    for i, rank in enumerate(ranks):
        for chosen in range(min(n, i + 1), 0, -1):
            for subtotal, ways in counts[chosen - 1].items():
                total = subtotal + int(rank)
                counts[chosen][total] = counts[chosen].get(total, 0) + ways

    lower = sum(ways for total, ways in counts[n].items() if total <= observed)
    upper = sum(ways for total, ways in counts[n].items() if total >= observed)
    return min(1.0, 2 * min(lower, upper) / comb(len(ranks), n))


def conditional_signed_rank_p(differences) -> float:
    """Two-sided conditional exact Wilcoxon signed-rank p value, retaining ties.

    Zero differences are dropped and equal absolute differences receive average
    ranks. Every assignment of signs to the remaining differences is counted by its
    sum of positive ranks. The p value is twice the smaller inclusive tail
    probability, capped at one.
    """
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1:
        raise ValueError("Differences must be a one-dimensional array.")
    if not np.isfinite(d).all():
        raise ValueError("Differences must contain only finite values.")

    d = d[d != 0]
    if d.size == 0:
        return 1.0
    # Average ranks are integers or half-integers; double them for integer sums.
    ranks = np.rint(2 * rankdata(np.abs(d), method="average")).astype(int)
    observed = int(ranks[d > 0].sum())
    counts = {0: 1}
    for rank in ranks:
        updated = dict(counts)
        for subtotal, ways in counts.items():
            updated[subtotal + int(rank)] = updated.get(subtotal + int(rank), 0) + ways
        counts = updated

    lower = sum(ways for total, ways in counts.items() if total <= observed)
    upper = sum(ways for total, ways in counts.items() if total >= observed)
    return min(1.0, 2 * min(lower, upper) / 2 ** d.size)


def bh_fdr(pvalues) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values, in input order."""
    p = np.asarray(pvalues, float)
    n = p.size
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * n / (np.arange(n) + 1))[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(adjusted, 0, 1)
    return q


def cliffs_delta(a, b) -> float:
    """Cliff's delta of a relative to b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    greater = (a[:, None] > b[None, :]).sum()
    less = (a[:, None] < b[None, :]).sum()
    return (greater - less) / (len(a) * len(b))


def cliffs_delta_ci(a, b, n_bootstrap: int, seed: int) -> tuple[float, float]:
    """Percentile 95% CI of Cliff's delta, resampling animals within each group."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    draws = [
        cliffs_delta(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))])
        for _ in range(n_bootstrap)
    ]
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def sign_flip_test(changes: np.ndarray, n_permutations: int, rng: np.random.Generator) -> float:
    """Test that the mean change vector differs from zero by flipping the sign per animal."""
    observed = np.linalg.norm(changes.mean(axis=0))
    exceed = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=len(changes))[:, None]
        if np.linalg.norm((changes * signs).mean(axis=0)) >= observed:
            exceed += 1
    return (exceed + 1) / (n_permutations + 1)


def group_difference_test(
    changes: np.ndarray, in_group: np.ndarray, n_permutations: int, rng: np.random.Generator
) -> float:
    """Test the distance between two group-mean change vectors by shuffling group labels."""
    observed = np.linalg.norm(changes[in_group].mean(axis=0) - changes[~in_group].mean(axis=0))
    exceed = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(in_group)
        if np.linalg.norm(changes[shuffled].mean(axis=0) - changes[~shuffled].mean(axis=0)) >= observed:
            exceed += 1
    return (exceed + 1) / (n_permutations + 1)
