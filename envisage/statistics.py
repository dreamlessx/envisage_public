"""Paired-test utilities for small-N evaluation.

Reviewer-defense statistics for the Envisage paper. All procedures evaluate
on small samples (rhino N=21, bleph N=27, rhytid N=9), so means alone do not
suffice. This module provides percentile bootstrap confidence intervals,
exact and approximate paired permutation tests, the Wilcoxon signed-rank
test, paired effect sizes (Cohen d_z, Hedges g), the rank-based
non-parametric effect size (Cliff delta), the Shrout-Fleiss intraclass
correlation, and Cohen kappa with a numpy fallback when scikit-learn is
absent.

Reproducibility: every randomized routine accepts an integer seed (default
42) and routes through numpy.random.default_rng. Permutation tests
enumerate all 2**n sign patterns when n is small enough; otherwise they
sample with the Phipson-Smyth correction so the reported p-value never
collapses to zero.

References cited inline in each function docstring.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values,
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    n_iter: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for a single sample.

    Resamples ``values`` with replacement ``n_iter`` times, applies
    ``statistic_fn`` to each resample, and returns the point estimate
    together with the lower and upper percentiles at (1-ci)/2 and
    1-(1-ci)/2.

    Reference: Efron, B. (1979). Bootstrap methods: another look at the
    jackknife. Annals of Statistics, 7(1), 1-26.

    Parameters
    ----------
    values : array_like
        One-dimensional sample. Non-finite entries are not stripped, the
        caller is responsible for that.
    statistic_fn : callable
        Maps a 1-D numpy array to a scalar. Defaults to ``np.mean``.
    n_iter : int
        Number of bootstrap resamples. Default 10000.
    ci : float
        Two-sided coverage in (0, 1). Default 0.95.
    seed : int
        Seed for numpy.random.default_rng.

    Returns
    -------
    point : float
        ``statistic_fn`` applied to the original sample.
    lower : float
        Lower CI bound at percentile (1-ci)/2.
    upper : float
        Upper CI bound at percentile 1-(1-ci)/2.

    Notes
    -----
    For samples with fewer than two elements the routine returns
    (point, point, point) so downstream code can ignore the CI without
    branching.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(statistic_fn(arr))
    if arr.size < 2:
        return (point, point, point)
    rng = np.random.default_rng(seed)
    n = arr.size
    idx = rng.integers(0, n, size=(n_iter, n))
    samples = arr[idx]
    stats_arr = np.empty(n_iter, dtype=float)
    # statistic_fn may not be vectorised across rows; loop in a way that
    # still benefits from numpy's vectorised mean for the default case.
    if statistic_fn is np.mean:
        stats_arr = samples.mean(axis=1)
    elif statistic_fn is np.median:
        stats_arr = np.median(samples, axis=1)
    else:
        for i in range(n_iter):
            stats_arr[i] = float(statistic_fn(samples[i]))
    alpha = 1.0 - ci
    lo_q = (alpha / 2.0) * 100.0
    hi_q = (1.0 - alpha / 2.0) * 100.0
    lower = float(np.percentile(stats_arr, lo_q))
    upper = float(np.percentile(stats_arr, hi_q))
    return (point, lower, upper)


# ---------------------------------------------------------------------------
# Paired permutation
# ---------------------------------------------------------------------------


def paired_permutation(
    a,
    b,
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    n_iter: int = 10000,
    alternative: str = "two-sided",
    seed: int = 42,
) -> tuple[float, float]:
    """Paired sign-flip permutation test on the difference a - b.

    For each pair the sign of (a_i - b_i) is independently flipped, the
    statistic is recomputed, and the p-value is the proportion of
    permutations whose statistic is at least as extreme as the observed
    one. When 2**n <= n_iter every sign pattern is enumerated, giving an
    exact p-value. Otherwise n_iter random sign flips are sampled and the
    Phipson-Smyth (2010) +1 correction is applied to numerator and
    denominator to keep p strictly positive.

    Reference: Pitman, E. J. G. (1937). Significance tests which may be
    applied to samples from any populations. Supplement to the Journal of
    the Royal Statistical Society, 4(1), 119-130. Also: Phipson, B. and
    Smyth, G. K. (2010). Permutation P-values should never be zero.
    Statistical Applications in Genetics and Molecular Biology, 9(1).

    Parameters
    ----------
    a, b : array_like
        Paired samples of equal length.
    statistic_fn : callable
        Maps the difference vector (1-D array) to a scalar. Defaults to
        the mean difference.
    n_iter : int
        Random permutation count when n is too large to enumerate.
    alternative : {"two-sided", "greater", "less"}
        Tail direction. ``"greater"`` tests a > b in the statistic_fn
        sense.
    seed : int
        Seed for numpy.random.default_rng.

    Returns
    -------
    observed : float
        ``statistic_fn(a - b)`` on the unflipped data.
    p_value : float
        Permutation p-value.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("paired_permutation requires equal-length samples")
    diff = a_arr - b_arr
    n = diff.size
    if n == 0:
        return (float("nan"), float("nan"))
    observed = float(statistic_fn(diff))

    def _extreme(stat: np.ndarray) -> np.ndarray:
        if alternative == "two-sided":
            return np.abs(stat) >= abs(observed) - 1e-15
        if alternative == "greater":
            return stat >= observed - 1e-15
        if alternative == "less":
            return stat <= observed + 1e-15
        raise ValueError(f"unknown alternative {alternative!r}")

    if 2 ** n <= n_iter:
        # Enumerate all sign patterns. n is small (<= log2(n_iter), so
        # for n_iter=10000 this triggers when n <= 13).
        n_perm = 2 ** n
        signs = np.empty((n_perm, n), dtype=float)
        for i in range(n_perm):
            for j in range(n):
                signs[i, j] = 1.0 if (i >> j) & 1 == 0 else -1.0
        flipped = signs * diff
        if statistic_fn is np.mean:
            stats_arr = flipped.mean(axis=1)
        else:
            stats_arr = np.array([statistic_fn(flipped[i]) for i in range(n_perm)])
        extreme = _extreme(stats_arr).sum()
        p_value = float(extreme) / float(n_perm)
        return (observed, p_value)

    rng = np.random.default_rng(seed)
    # Random sign flips, +/- 1 with probability 1/2.
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_iter, n))
    flipped = signs * diff
    if statistic_fn is np.mean:
        stats_arr = flipped.mean(axis=1)
    else:
        stats_arr = np.array([statistic_fn(flipped[i]) for i in range(n_iter)])
    extreme = int(_extreme(stats_arr).sum())
    # Phipson-Smyth correction.
    p_value = (extreme + 1) / (n_iter + 1)
    return (observed, float(p_value))


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank
# ---------------------------------------------------------------------------


def paired_wilcoxon(a, b, alternative: str = "two-sided") -> tuple[float, float]:
    """Wilcoxon signed-rank test on paired samples.

    Thin wrapper on scipy.stats.wilcoxon with ``zero_method='wilcox'`` so
    zero-difference pairs are dropped from the rank computation, which is
    the convention used in the original paper.

    Reference: Wilcoxon, F. (1945). Individual comparisons by ranking
    methods. Biometrics Bulletin, 1(6), 80-83.

    Parameters
    ----------
    a, b : array_like
        Paired samples of equal length.
    alternative : {"two-sided", "greater", "less"}
        Passed through to scipy.

    Returns
    -------
    W : float
        The Wilcoxon test statistic.
    p_value : float
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("paired_wilcoxon requires equal-length samples")
    res = stats.wilcoxon(a_arr, b_arr, zero_method="wilcox", alternative=alternative)
    # scipy returns a result object with .statistic and .pvalue.
    return (float(res.statistic), float(res.pvalue))


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def cohens_d_paired(a, b) -> float:
    """Standardized mean difference for paired samples (d_z).

    d_z = mean(a - b) / std(a - b, ddof=1).

    Reference: Cohen, J. (1988). Statistical Power Analysis for the
    Behavioral Sciences (2nd ed.). Lawrence Erlbaum Associates.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    diff = a_arr - b_arr
    if diff.size < 2:
        return float("nan")
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0:
        return float("nan")
    return float(np.mean(diff) / sd)


def cliffs_delta(a, b) -> float:
    """Cliff dominance statistic on two independent samples.

    Vectorized via an outer-difference matrix:
    ((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum())
    divided by (n_a * n_b). Bounded in [-1, 1]; values near zero indicate
    no stochastic dominance.

    Reference: Cliff, N. (1993). Dominance statistics: ordinal analyses to
    answer ordinal questions. Psychological Bulletin, 114(3), 494-509.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.size == 0 or b_arr.size == 0:
        return float("nan")
    gt = (a_arr[:, None] > b_arr[None, :]).sum()
    lt = (a_arr[:, None] < b_arr[None, :]).sum()
    n_pairs = a_arr.size * b_arr.size
    return float((gt - lt) / n_pairs)


def hedges_g(a, b) -> float:
    """Small-sample-corrected paired effect size (Hedges g).

    g = d_z * J where J = 1 - 3 / (4 * df - 1) and df = n - 1. The factor
    J approaches 1 as df grows, so for df < 2 the correction is undefined
    and the function emits a warning and returns the uncorrected d_z.

    Reference: Hedges, L. V. (1981). Distribution theory for Glass's
    estimator of effect size and related estimators. Journal of
    Educational Statistics, 6(2), 107-128.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    diff = a_arr - b_arr
    n = diff.size
    df = n - 1
    d = cohens_d_paired(a_arr, b_arr)
    if df < 2:
        warnings.warn(
            "hedges_g: df < 2, returning uncorrected d_z; correction is undefined"
        )
        return d
    j = 1.0 - 3.0 / (4.0 * df - 1.0)
    return float(d * j)


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------


def icc(ratings, model: str = "2-way-random") -> float:
    """Shrout-Fleiss intraclass correlation coefficient.

    ``ratings`` has shape (n_subjects, n_raters). The mean-square
    decomposition follows Shrout and Fleiss (1979) Table 2.

    Supported models (single-rater forms only):
      - "1-way-random"  -> ICC(1, 1)
      - "2-way-random"  -> ICC(2, 1), absolute agreement (default)
      - "2-way-mixed"   -> ICC(3, 1), consistency

    Reference: Shrout, P. E. and Fleiss, J. L. (1979). Intraclass
    correlations: uses in assessing rater reliability. Psychological
    Bulletin, 86(2), 420-428.
    """
    arr = np.asarray(ratings, dtype=float)
    if arr.ndim != 2:
        raise ValueError("icc expects a 2-D array of shape (n_subjects, n_raters)")
    n, k = arr.shape
    if n < 2 or k < 2:
        return float("nan")
    grand_mean = arr.mean()
    row_means = arr.mean(axis=1)
    col_means = arr.mean(axis=0)

    # Sums of squares.
    ss_total = ((arr - grand_mean) ** 2).sum()
    ss_between_subjects = k * ((row_means - grand_mean) ** 2).sum()
    ss_between_raters = n * ((col_means - grand_mean) ** 2).sum()
    ss_residual = ss_total - ss_between_subjects - ss_between_raters
    ss_within_subjects = ss_total - ss_between_subjects

    msr = ss_between_subjects / (n - 1)  # mean square between subjects (rows)
    msc = ss_between_raters / (k - 1)    # mean square between raters (columns)
    mse = ss_residual / ((n - 1) * (k - 1))  # residual mean square
    msw = ss_within_subjects / (n * (k - 1))  # within-subject mean square (1-way)

    if model == "1-way-random":
        denom = msr + (k - 1) * msw
        if denom == 0.0:
            return float("nan")
        return float((msr - msw) / denom)
    if model == "2-way-random":
        denom = msr + (k - 1) * mse + (k / n) * (msc - mse)
        if denom == 0.0:
            return float("nan")
        return float((msr - mse) / denom)
    if model == "2-way-mixed":
        denom = msr + (k - 1) * mse
        if denom == 0.0:
            return float("nan")
        return float((msr - mse) / denom)
    raise ValueError(f"unknown icc model {model!r}")


def cohens_kappa(rater1, rater2) -> float:
    """Cohen kappa for two raters on categorical labels.

    Tries to import sklearn.metrics.cohen_kappa_score; on ImportError it
    falls back to a numpy implementation that builds the confusion
    matrix, computes observed agreement p_o = trace(C) / n, expected
    agreement p_e = sum(row_marginal * col_marginal), and returns
    (p_o - p_e) / (1 - p_e).

    Reference: Cohen, J. (1960). A coefficient of agreement for nominal
    scales. Educational and Psychological Measurement, 20(1), 37-46.
    """
    r1 = np.asarray(rater1)
    r2 = np.asarray(rater2)
    if r1.shape != r2.shape:
        raise ValueError("cohens_kappa requires equal-length rater arrays")
    if r1.size == 0:
        return float("nan")
    try:
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(r1, r2))
    except ImportError:
        pass
    # Numpy fallback. Build a confusion matrix over the union of labels.
    labels = np.unique(np.concatenate([r1.ravel(), r2.ravel()]))
    label_index = {lab: i for i, lab in enumerate(labels)}
    k = labels.size
    cm = np.zeros((k, k), dtype=float)
    for x, y in zip(r1.ravel(), r2.ravel()):
        cm[label_index[x], label_index[y]] += 1.0
    n = cm.sum()
    if n == 0:
        return float("nan")
    p_o = float(np.trace(cm) / n)
    row_marg = cm.sum(axis=1) / n
    col_marg = cm.sum(axis=0) / n
    p_e = float((row_marg * col_marg).sum())
    if p_e == 1.0:
        return float("nan")
    return float((p_o - p_e) / (1.0 - p_e))
