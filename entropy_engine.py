"""
entropy_engine.py
Information-theoretic analysis of meta-analytic trust.

Computes Shannon entropy, KL divergence, mutual information,
and conditional entropy on GRADE distributions vs statistical
significance, stratified by 14 Cochrane review domains.
"""
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRADE_ORDER = ["A+", "A", "B", "C", "D", "F"]
H_MAX = math.log2(6)   # 2.584962500721156 bits  (uniform over 6 grades)

def _model_file(model_name: str, *parts: str) -> str:
    """Resolve sibling model artifacts under either WSL or Windows paths."""
    models_roots = (
        Path(__file__).resolve().parents[1],
        Path("/mnt/c/Models"),
        Path("C:/Models"),
    )
    for root in models_roots:
        candidate = root / model_name / Path(*parts)
        if candidate.exists():
            return str(candidate)
    return str(Path("C:/Models") / model_name / Path(*parts))


SCORES_PATH   = _model_file("EvidenceScore", "results", "scores.csv")
VERDICTS_PATH = _model_file("ActionableEvidence", "results", "verdicts.csv")
GROUPS_PATH   = _model_file("TrustGate", "data", "review_groups.csv")


# ---------------------------------------------------------------------------
# Core information-theoretic functions
# ---------------------------------------------------------------------------

def shannon_entropy(distribution: List[float]) -> float:
    """
    H(X) = -sum(p_i * log2(p_i))

    Parameters
    ----------
    distribution : list of floats
        Probability values.  Need not sum to 1 (will be normalised).
        Zero entries are skipped (0 * log2(0) := 0 by convention).

    Returns
    -------
    float: entropy in bits.

    Raises
    ------
    ValueError if distribution is empty or all-zero.
    """
    if not distribution:
        raise ValueError("distribution must be non-empty")
    total = sum(distribution)
    if total == 0:
        raise ValueError("distribution must contain at least one non-zero value")
    h = 0.0
    for p in distribution:
        if p < 0:
            raise ValueError(f"probabilities must be >= 0, got {p}")
        q = p / total          # normalise
        if q > 0:
            h -= q * math.log2(q)
    return h


def kl_divergence(p: List[float], q: List[float], eps: float = 1e-12) -> float:
    """
    D_KL(P || Q) = sum( P_i * log2(P_i / Q_i) )

    Both distributions are normalised internally.
    Q_i == 0 positions where P_i > 0 are handled by adding eps to Q.

    Parameters
    ----------
    p, q : lists of equal length
    eps  : small constant added to Q to avoid log(0).

    Returns
    -------
    float: KL divergence in bits (>= 0).
    """
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")
    if not p:
        raise ValueError("distributions must be non-empty")
    total_p = sum(p)
    total_q = sum(q)
    if total_p == 0:
        raise ValueError("P must have at least one non-zero value")
    if total_q == 0:
        raise ValueError("Q must have at least one non-zero value")

    kl = 0.0
    for pi, qi in zip(p, q):
        p_norm = pi / total_p
        q_norm = qi / total_q + eps      # stabilise zero denominator
        if p_norm > 0:
            kl += p_norm * math.log2(p_norm / q_norm)
    return max(0.0, kl)   # guard tiny negatives from floating-point


def mutual_information(
    contingency: List[List[float]],
) -> float:
    """
    I(X; Y) = H(Y) - H(Y|X)
             = sum_{x,y} p(x,y) * log2( p(x,y) / (p(x)*p(y)) )

    Parameters
    ----------
    contingency : 2-D list, shape (|X|, |Y|).
        Raw counts or joint probabilities.

    Returns
    -------
    float: mutual information in bits (>= 0).
    """
    if not contingency or not contingency[0]:
        raise ValueError("contingency table must be non-empty")
    rows = len(contingency)
    cols = len(contingency[0])
    total = sum(contingency[r][c] for r in range(rows) for c in range(cols))
    if total == 0:
        raise ValueError("contingency table must not be all-zero")

    row_sums = [sum(contingency[r][c] for c in range(cols)) / total for r in range(rows)]
    col_sums = [sum(contingency[r][c] for r in range(rows)) / total for c in range(cols)]

    mi = 0.0
    for r in range(rows):
        for c in range(cols):
            p_xy = contingency[r][c] / total
            p_x  = row_sums[r]
            p_y  = col_sums[c]
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * math.log2(p_xy / (p_x * p_y))
    return max(0.0, mi)


def conditional_entropy(
    contingency: List[List[float]],
) -> float:
    """
    H(Y | X) = -sum_{x} p(x) * sum_{y} p(y|x) * log2( p(y|x) )

    The first axis (rows) is X; the second axis (cols) is Y.

    Returns
    -------
    float: conditional entropy in bits.
    """
    if not contingency or not contingency[0]:
        raise ValueError("contingency table must be non-empty")
    rows = len(contingency)
    cols = len(contingency[0])
    total = sum(contingency[r][c] for r in range(rows) for c in range(cols))
    if total == 0:
        raise ValueError("contingency table must not be all-zero")

    h_y_given_x = 0.0
    for r in range(rows):
        row_total = sum(contingency[r][c] for c in range(cols))
        if row_total == 0:
            continue
        p_x = row_total / total
        h_row = 0.0
        for c in range(cols):
            p_y_given_x = contingency[r][c] / row_total
            if p_y_given_x > 0:
                h_row -= p_y_given_x * math.log2(p_y_given_x)
        h_y_given_x += p_x * h_row
    return h_y_given_x


def normalized_mi(mi: float, h_trust: float, h_sig: float) -> float:
    """
    NMI = MI / sqrt(H(Trust) * H(Significance))

    Returns 0 if denominator is zero or near-zero.
    """
    denom = math.sqrt(h_trust * h_sig) if h_trust > 0 and h_sig > 0 else 0.0
    if denom < 1e-12:
        return 0.0
    return mi / denom


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _prefix(review_id: str) -> str:
    """Return the numeric prefix of a Cochrane review ID (e.g. 'CD000028')."""
    return review_id.strip()


def load_review_groups(path: str = GROUPS_PATH) -> Dict[str, str]:
    """
    Load review_groups.csv -> {review_id_prefix: review_group}
    """
    mapping: Dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            mapping[row["review_id_prefix"].strip()] = row["review_group"].strip()
    return mapping


def load_scores(path: str = SCORES_PATH) -> Dict[str, dict]:
    """
    Load scores.csv -> {ma_id: {"review_id": ..., "grade": ...}}
    """
    data: Dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            data[row["ma_id"].strip()] = {
                "review_id": row["review_id"].strip(),
                "grade":     row["grade"].strip(),
            }
    return data


def load_verdicts(path: str = VERDICTS_PATH) -> Dict[str, bool]:
    """
    Load verdicts.csv -> {ma_id: significant (bool)}
    """
    data: Dict[str, bool] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            data[row["ma_id"].strip()] = row["significant"].strip() == "True"
    return data


# ---------------------------------------------------------------------------
# Pipeline: merge data and build contingency tables
# ---------------------------------------------------------------------------

def build_merged_dataset(
    scores: Optional[Dict[str, dict]]  = None,
    verdicts: Optional[Dict[str, bool]] = None,
    groups: Optional[Dict[str, str]]   = None,
) -> List[dict]:
    """
    Inner-join scores, verdicts, and domain groups.

    Returns list of dicts:
        {ma_id, review_id, grade, significant (bool), domain}
    """
    if scores is None:
        scores = load_scores()
    if verdicts is None:
        verdicts = load_verdicts()
    if groups is None:
        groups = load_review_groups()

    merged = []
    for ma_id, score_row in scores.items():
        if ma_id not in verdicts:
            continue
        review_id = score_row["review_id"]
        domain = groups.get(review_id, "Other")
        merged.append({
            "ma_id":       ma_id,
            "review_id":   review_id,
            "grade":       score_row["grade"],
            "significant": verdicts[ma_id],
            "domain":      domain,
        })
    return merged


def build_contingency(
    records: List[dict],
    grade_order: List[str] = GRADE_ORDER,
) -> List[List[float]]:
    """
    Build a contingency table: rows = grades, cols = [not-sig, sig].
    """
    table = [[0.0, 0.0] for _ in grade_order]
    grade_idx = {g: i for i, g in enumerate(grade_order)}
    for rec in records:
        gi = grade_idx.get(rec["grade"])
        if gi is None:
            continue
        ci = 1 if rec["significant"] else 0
        table[gi][ci] += 1.0
    return table


# ---------------------------------------------------------------------------
# Per-domain analysis
# ---------------------------------------------------------------------------

def analyse_domain(
    records: List[dict],
    grade_order: List[str] = GRADE_ORDER,
) -> dict:
    """
    Compute all information-theoretic metrics for a set of records
    (a single domain or the full corpus).

    Returns dict with keys:
        n, h_trust, h_sig, h_sig_given_trust, mi, nmi,
        grade_dist, sig_rate, contingency
    """
    n = len(records)
    if n == 0:
        return {
            "n": 0, "h_trust": 0.0, "h_sig": 0.0,
            "h_sig_given_trust": 0.0, "mi": 0.0, "nmi": 0.0,
            "grade_dist": {}, "sig_rate": 0.0, "contingency": [],
        }

    contingency = build_contingency(records, grade_order)

    # Grade distribution (trust)
    grade_counts = [contingency[i][0] + contingency[i][1] for i in range(len(grade_order))]
    h_trust = shannon_entropy(grade_counts) if sum(grade_counts) > 0 else 0.0

    # Significance distribution
    sig_counts  = [sum(contingency[i][0] for i in range(len(grade_order))),
                   sum(contingency[i][1] for i in range(len(grade_order)))]
    h_sig = shannon_entropy(sig_counts) if sum(sig_counts) > 0 else 0.0

    # Conditional entropy H(Sig | Trust)
    h_cond = conditional_entropy(contingency)

    # MI = H(Sig) - H(Sig|Trust)
    mi = mutual_information(contingency)

    # NMI
    nmi = normalized_mi(mi, h_trust, h_sig)

    total_sig = sig_counts[1]
    sig_rate  = total_sig / n if n > 0 else 0.0

    grade_dist = {
        grade_order[i]: int(grade_counts[i]) for i in range(len(grade_order))
    }

    return {
        "n":                  n,
        "h_trust":            h_trust,
        "h_sig":              h_sig,
        "h_sig_given_trust":  h_cond,
        "mi":                 mi,
        "nmi":                nmi,
        "grade_dist":         grade_dist,
        "sig_rate":           sig_rate,
        "contingency":        contingency,
    }


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    scores_path:   str = SCORES_PATH,
    verdicts_path: str = VERDICTS_PATH,
    groups_path:   str = GROUPS_PATH,
) -> dict:
    """
    Full pipeline: load data, compute global + per-domain metrics.

    Returns
    -------
    dict with keys:
        "overall": analyse_domain result for all records
        "domains": {domain_name: analyse_domain result}
        "n_matched": int
    """
    scores   = load_scores(scores_path)
    verdicts = load_verdicts(verdicts_path)
    groups   = load_review_groups(groups_path)

    merged   = build_merged_dataset(scores, verdicts, groups)

    overall  = analyse_domain(merged)

    # Per-domain
    by_domain: Dict[str, List[dict]] = defaultdict(list)
    for rec in merged:
        by_domain[rec["domain"]].append(rec)

    domain_results = {
        domain: analyse_domain(recs) for domain, recs in sorted(by_domain.items())
    }

    return {
        "overall":   overall,
        "domains":   domain_results,
        "n_matched": len(merged),
    }


# ---------------------------------------------------------------------------
# KL: trust-weighted vs naive significance
# ---------------------------------------------------------------------------

def kl_trust_vs_naive(
    merged: List[dict],
    grade_order: List[str] = GRADE_ORDER,
) -> dict:
    """
    Compare P = trust-weighted significance distribution
    with Q = naive (grade-blind) significance distribution.

    Trust-weighted: for each grade g, compute sig_rate(g), weight by p(g).
    Naive: overall sig_rate applied uniformly.

    Returns {"kl_pq": float, "kl_qp": float, "p_dist": list, "q_dist": list}
    """
    contingency = build_contingency(merged, grade_order)
    n = len(merged)
    if n == 0:
        return {"kl_pq": 0.0, "kl_qp": 0.0, "p_dist": [], "q_dist": []}

    total_sig   = sum(contingency[i][1] for i in range(len(grade_order)))
    naive_sig_r = total_sig / n

    # P: per-grade sig rates (trust-weighted empirical)
    grade_counts = [contingency[i][0] + contingency[i][1] for i in range(len(grade_order))]
    p_dist = []
    for i, g in enumerate(grade_order):
        gn = grade_counts[i]
        if gn > 0:
            p_dist.append(contingency[i][1] / gn)
        else:
            p_dist.append(0.0)

    # Q: naive rates (same for every grade)
    q_dist = [naive_sig_r] * len(grade_order)

    return {
        "kl_pq": kl_divergence(p_dist, q_dist),
        "kl_qp": kl_divergence(q_dist, p_dist),
        "p_dist": p_dist,
        "q_dist": q_dist,
    }


# ---------------------------------------------------------------------------
# CLI entry-point (informational)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("Running EvidenceEntropy pipeline...")
    results = run_pipeline()
    ov = results["overall"]
    print(f"N matched  : {results['n_matched']}")
    print(f"H(Trust)   : {ov['h_trust']:.4f} bits  (max={H_MAX:.4f})")
    print(f"H(Sig)     : {ov['h_sig']:.4f} bits")
    print(f"H(Sig|Tr)  : {ov['h_sig_given_trust']:.4f} bits")
    print(f"MI         : {ov['mi']:.4f} bits")
    print(f"NMI        : {ov['nmi']:.4f}")
    print()
    print("Per-domain MI:")
    for domain, dm in results["domains"].items():
        print(f"  {domain:<18} n={dm['n']:5d}  MI={dm['mi']:.4f}  NMI={dm['nmi']:.4f}")
