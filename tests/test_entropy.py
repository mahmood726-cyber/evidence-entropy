"""
tests/test_entropy.py — 25 unit tests for EvidenceEntropy
"""
import math
import sys
import os
import csv
import tempfile
import unittest

# Ensure parent directory is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entropy_engine import (
    shannon_entropy,
    kl_divergence,
    mutual_information,
    conditional_entropy,
    normalized_mi,
    build_contingency,
    analyse_domain,
    build_merged_dataset,
    load_review_groups,
    load_scores,
    load_verdicts,
    run_pipeline,
    GRADE_ORDER,
    H_MAX,
    SCORES_PATH,
    VERDICTS_PATH,
    GROUPS_PATH,
)

EPS = 1e-9   # floating-point tolerance for equality checks


# =========================================================================
# T1–T5: shannon_entropy
# =========================================================================

class TestShannonEntropy(unittest.TestCase):

    def test_T1_uniform_six_grades(self):
        """Uniform over 6 grades => log2(6) bits."""
        dist = [1, 1, 1, 1, 1, 1]
        result = shannon_entropy(dist)
        self.assertAlmostEqual(result, math.log2(6), places=10,
                               msg="Uniform-6 should equal log2(6)")

    def test_T2_delta_distribution_zero_entropy(self):
        """All mass on one grade => 0 bits."""
        dist = [0, 0, 0, 6229, 0, 0]
        result = shannon_entropy(dist)
        self.assertAlmostEqual(result, 0.0, places=10,
                               msg="Delta distribution should have zero entropy")

    def test_T3_known_binary_entropy(self):
        """50/50 binary => 1.0 bit."""
        dist = [1, 1]
        result = shannon_entropy(dist)
        self.assertAlmostEqual(result, 1.0, places=10)

    def test_T4_empty_raises(self):
        """Empty list should raise ValueError."""
        with self.assertRaises(ValueError):
            shannon_entropy([])

    def test_T5_single_element(self):
        """Single non-zero element => 0 bits (certainty)."""
        result = shannon_entropy([42])
        self.assertAlmostEqual(result, 0.0, places=10)

    def test_T5b_all_zero_raises(self):
        """All-zero distribution should raise ValueError."""
        with self.assertRaises(ValueError):
            shannon_entropy([0, 0, 0])


# =========================================================================
# T6–T10: kl_divergence
# =========================================================================

class TestKLDivergence(unittest.TestCase):

    def test_T6_identical_distributions_zero(self):
        """D_KL(P||P) = 0."""
        p = [1, 2, 3, 4]
        self.assertAlmostEqual(kl_divergence(p, p), 0.0, places=9)

    def test_T7_known_value(self):
        """
        p = [0.5, 0.5], q = [0.25, 0.75]
        D_KL = 0.5*log2(0.5/0.25) + 0.5*log2(0.5/0.75)
             = 0.5*1 + 0.5*(-0.58496...) = 0.20752...
        """
        p = [1, 1]    # normalised: [0.5, 0.5]
        q = [1, 3]    # normalised: [0.25, 0.75]
        expected = 0.5 * math.log2(0.5 / 0.25) + 0.5 * math.log2(0.5 / 0.75)
        self.assertAlmostEqual(kl_divergence(p, q), expected, places=9)

    def test_T8_asymmetry(self):
        """D_KL(P||Q) != D_KL(Q||P) in general (asymmetric distributions)."""
        # p skewed toward first category, q skewed differently
        p = [7, 1, 2]   # normalised: [0.7, 0.1, 0.2]
        q = [2, 5, 3]   # normalised: [0.2, 0.5, 0.3]
        kl_pq = kl_divergence(p, q)
        kl_qp = kl_divergence(q, p)
        # These are analytically different for these asymmetric distributions
        self.assertGreater(abs(kl_pq - kl_qp), 1e-5,
                           msg=f"Expected D_KL(P||Q) != D_KL(Q||P) but got {kl_pq:.6f} vs {kl_qp:.6f}")

    def test_T9_zero_q_handled(self):
        """P_i > 0, Q_i = 0: stabilised with eps, result is large but finite."""
        p = [1, 0]
        q = [0, 1]
        result = kl_divergence(p, q)
        self.assertGreater(result, 0)
        self.assertTrue(math.isfinite(result))

    def test_T10_length_mismatch_raises(self):
        """Mismatched lengths should raise ValueError."""
        with self.assertRaises(ValueError):
            kl_divergence([1, 2], [1, 2, 3])


# =========================================================================
# T11–T15: mutual_information
# =========================================================================

class TestMutualInformation(unittest.TestCase):

    def test_T11_independent_variables_near_zero(self):
        """
        If rows are proportional (independence), MI ~ 0.
        Contingency: each row is [1, 1] (equal sig/not-sig for every grade).
        """
        contingency = [[1, 1], [1, 1], [1, 1], [1, 1]]
        result = mutual_information(contingency)
        self.assertAlmostEqual(result, 0.0, places=9)

    def test_T12_known_mi_value(self):
        """
        2x2 table with perfect structure allows known calculation.
        [[1,0],[0,1]] => MI = 1.0 bit (XOR perfect dependence).
        """
        contingency = [[1, 0], [0, 1]]
        result = mutual_information(contingency)
        self.assertAlmostEqual(result, 1.0, places=9)

    def test_T13_perfect_dependence_equals_h_y(self):
        """
        Perfect dependence: I(X;Y) = H(Y) when Y is fully determined by X.
        Use a 2x2 table where grade perfectly predicts significance.
        [[a,0],[0,b]] -> I(X;Y) = H(Y) = -p*log2(p) - (1-p)*log2(1-p)
        """
        a, b = 3000, 3229
        contingency = [[a, 0], [0, b]]
        n = a + b
        p = a / n
        h_y = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        result = mutual_information(contingency)
        self.assertAlmostEqual(result, h_y, places=8)

    def test_T14_non_negative(self):
        """MI must always be >= 0."""
        contingency = [[10, 30], [20, 15], [5, 8], [3, 2]]
        result = mutual_information(contingency)
        self.assertGreaterEqual(result, 0.0)

    def test_T15_empty_raises(self):
        """Empty contingency should raise ValueError."""
        with self.assertRaises(ValueError):
            mutual_information([])


# =========================================================================
# T16–T20: conditional_entropy
# =========================================================================

class TestConditionalEntropy(unittest.TestCase):

    def test_T16_independent_equals_h_y(self):
        """
        H(Y|X) = H(Y) when X and Y are independent.
        Independence: all rows have same proportion sig/not-sig.
        """
        # Each row: [1, 1] => P(sig|grade) = 0.5 for all grades
        contingency = [[2, 2], [2, 2], [2, 2]]
        h_cond = conditional_entropy(contingency)
        # H(Y) = -0.5*log2(0.5) - 0.5*log2(0.5) = 1.0 bit
        self.assertAlmostEqual(h_cond, 1.0, places=9)

    def test_T17_perfect_dependence_zero(self):
        """H(Y|X) = 0 when Y is perfectly determined by X."""
        contingency = [[4, 0], [0, 4]]
        h_cond = conditional_entropy(contingency)
        self.assertAlmostEqual(h_cond, 0.0, places=9)

    def test_T18_mi_equals_h_y_minus_h_y_given_x(self):
        """I(X;Y) = H(Y) - H(Y|X) must hold numerically."""
        contingency = [[10, 30], [20, 15], [5, 8], [3, 2]]
        mi_direct = mutual_information(contingency)
        n = sum(contingency[r][c] for r in range(4) for c in range(2))
        col_sums = [sum(contingency[r][c] for r in range(4)) for c in range(2)]
        h_y = shannon_entropy(col_sums)
        h_cond = conditional_entropy(contingency)
        self.assertAlmostEqual(mi_direct, h_y - h_cond, places=8)

    def test_T19_per_domain_computation(self):
        """conditional_entropy on a realistic-size asymmetric table."""
        contingency = [
            [400, 600],  # A+
            [300, 500],  # A
            [200, 200],  # B
            [100,  50],  # C
            [ 50,  20],  # D
            [ 30,  10],  # F
        ]
        h_cond = conditional_entropy(contingency)
        self.assertGreaterEqual(h_cond, 0.0)
        self.assertLessEqual(h_cond, 1.0)   # binary Y => H(Y|X) <= 1 bit

    def test_T20_empty_raises(self):
        """Empty table should raise ValueError."""
        with self.assertRaises(ValueError):
            conditional_entropy([])


# =========================================================================
# T21–T25: Pipeline integration tests
# =========================================================================

class TestPipelineIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load real data once for integration tests."""
        cls.scores   = load_scores(SCORES_PATH)
        cls.verdicts = load_verdicts(VERDICTS_PATH)
        cls.groups   = load_review_groups(GROUPS_PATH)
        cls.merged   = build_merged_dataset(cls.scores, cls.verdicts, cls.groups)
        cls.results  = run_pipeline(SCORES_PATH, VERDICTS_PATH, GROUPS_PATH)

    def test_T21_data_load_sizes(self):
        """Should load ~6229 scores, ~6229 verdicts, ~14 domains."""
        self.assertGreater(len(self.scores),   6000)
        self.assertGreater(len(self.verdicts), 6000)
        self.assertGreater(len(self.groups),     50)

    def test_T22_merge_produces_records(self):
        """Merged dataset should have >= 6000 rows."""
        self.assertGreater(len(self.merged), 6000)

    def test_T23_overall_entropy_within_valid_range(self):
        """H(Trust) must be in [0, log2(6)] = [0, 2.585]."""
        ov = self.results["overall"]
        self.assertGreaterEqual(ov["h_trust"], 0.0)
        self.assertLessEqual(ov["h_trust"], H_MAX + EPS)

    def test_T24_fourteen_domains_found(self):
        """Pipeline must identify exactly 14 Cochrane review domains."""
        domains = self.results["domains"]
        self.assertEqual(len(domains), 14,
                         msg=f"Expected 14 domains, got {len(domains)}: {sorted(domains)}")

    def test_T25_mi_and_nmi_non_negative_and_bounded(self):
        """
        Overall MI >= 0; NMI in [0, 1]; H(Sig|Trust) <= H(Sig).
        Also smoke-tests all per-domain metrics.
        """
        ov = self.results["overall"]
        self.assertGreaterEqual(ov["mi"],  0.0)
        self.assertGreaterEqual(ov["nmi"], 0.0)
        self.assertLessEqual(ov["nmi"], 1.0 + EPS)
        # H(Sig|Trust) <= H(Sig)
        self.assertLessEqual(ov["h_sig_given_trust"], ov["h_sig"] + EPS)
        # Per-domain sanity
        for domain, dm in self.results["domains"].items():
            self.assertGreaterEqual(dm["mi"],  0.0, msg=f"{domain} MI negative")
            self.assertGreaterEqual(dm["nmi"], 0.0, msg=f"{domain} NMI negative")
            self.assertLessEqual(dm["nmi"], 1.0 + EPS, msg=f"{domain} NMI > 1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
