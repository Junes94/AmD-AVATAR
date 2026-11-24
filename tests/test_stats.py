"""Validate the conditional exact tests against independent permutation calculations."""
import itertools
import unittest

import numpy as np
from scipy.stats import mannwhitneyu, rankdata, wilcoxon

from src.stats import conditional_rank_sum_p, conditional_signed_rank_p


def enumerated_p(a, b):
    """Reference calculation by explicitly enumerating every label assignment."""
    ranks = rankdata(np.concatenate([a, b]), method="average")
    observed = ranks[:len(a)].sum()
    null = np.array([
        ranks[list(indices)].sum()
        for indices in itertools.combinations(range(len(ranks)), len(a))
    ])
    return min(1.0, 2 * min(np.mean(null <= observed), np.mean(null >= observed)))


def enumerated_signed_rank_p(differences):
    """Reference calculation by explicitly enumerating every sign assignment."""
    d = np.asarray(differences, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return 1.0
    ranks = rankdata(np.abs(d), method="average")
    observed = ranks[d > 0].sum()
    null = np.array([ranks[np.array(signs, dtype=bool)].sum() for signs in itertools.product([0, 1], repeat=d.size)])
    return min(1.0, 2 * min(np.mean(null <= observed), np.mean(null >= observed)))


class ConditionalRankSumTests(unittest.TestCase):
    def test_three_zeros_and_one_positive(self):
        self.assertEqual(conditional_rank_sum_p([0, 0], [0, 1]), 1.0)
        self.assertAlmostEqual(mannwhitneyu([0, 0], [0, 1], method="exact").pvalue, 2 / 3)

    def test_two_tied_blocks(self):
        self.assertAlmostEqual(conditional_rank_sum_p([0, 0], [1, 1]), 1 / 3)

    def test_all_equal(self):
        self.assertEqual(conditional_rank_sum_p([0, 0, 0], [0, 0]), 1.0)

    def test_without_ties_matches_mannwhitney(self):
        values = np.arange(7, dtype=float)
        for n in (1, 2, 3):
            for indices in itertools.combinations(range(len(values)), n):
                selected = np.zeros(len(values), dtype=bool)
                selected[list(indices)] = True
                a, b = values[selected], values[~selected]
                self.assertAlmostEqual(
                    conditional_rank_sum_p(a, b),
                    mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue,
                    places=14,
                )

    def test_ties_match_explicit_enumeration(self):
        rng = np.random.default_rng(42)
        for total in range(3, 9):
            for n in range(1, total):
                values = rng.integers(0, 3, size=total)
                a, b = values[:n], values[n:]
                self.assertAlmostEqual(conditional_rank_sum_p(a, b), enumerated_p(a, b), places=14)

    def test_group_and_input_order_invariance(self):
        a, b = np.array([0, 0, 1, 3, 3]), np.array([0, 2, 2])
        expected = conditional_rank_sum_p(a, b)
        self.assertEqual(expected, conditional_rank_sum_p(b, a))
        self.assertEqual(expected, conditional_rank_sum_p(a[::-1], b[::-1]))

    def test_invalid_samples(self):
        for a, b in (([], [1]), ([1], []), ([[1]], [2]), ([np.nan], [1]), ([1], [np.inf])):
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                conditional_rank_sum_p(a, b)


class ConditionalSignedRankTests(unittest.TestCase):
    def test_without_ties_or_zeros_matches_wilcoxon(self):
        for n in range(1, 8):
            for signs in itertools.product([-1, 1], repeat=n):
                d = np.arange(1, n + 1) * np.array(signs)
                self.assertAlmostEqual(
                    conditional_signed_rank_p(d),
                    wilcoxon(d, alternative="two-sided", method="exact").pvalue,
                    places=14,
                )

    def test_ties_and_zeros_match_explicit_enumeration(self):
        rng = np.random.default_rng(42)
        for n in range(1, 11):
            for _ in range(5):
                d = rng.integers(-2, 3, size=n)
                self.assertAlmostEqual(conditional_signed_rank_p(d), enumerated_signed_rank_p(d), places=14)

    def test_all_zero(self):
        self.assertEqual(conditional_signed_rank_p([0, 0, 0]), 1.0)

    def test_sign_and_order_invariance(self):
        d = np.array([0, 1, -1, 2, 2, 3, -0.5])
        expected = conditional_signed_rank_p(d)
        self.assertEqual(expected, conditional_signed_rank_p(-d))
        self.assertEqual(expected, conditional_signed_rank_p(d[::-1]))

    def test_invalid_differences(self):
        for d in ([[1, 2]], [np.nan, 1], [1, np.inf]):
            with self.subTest(d=d), self.assertRaises(ValueError):
                conditional_signed_rank_p(d)


if __name__ == "__main__":
    unittest.main()
