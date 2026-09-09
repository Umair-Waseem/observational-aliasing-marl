"""Exact-binomial (Clopper-Pearson) intervals and the Fisher exact test.

The response letter for this paper (point R1-S9) states that "the reported
intervals and p-values follow from those records by standard exact-binomial
(Clopper--Pearson) and Fisher computations over the released counts". This
module is that computation, written against the Python standard library alone
so that checking it requires no scientific stack.

WHAT IS COMPUTED

  clopper_pearson(k, n, alpha=0.05)
      The two-sided exact binomial confidence interval, in its standard
      Beta-quantile form:

          lower = 0                              if k == 0
                = BetaInv(alpha/2;  k,   n-k+1)  otherwise
          upper = 1                              if k == n
                = BetaInv(1-alpha/2; k+1, n-k)   otherwise

      Reproduces every interval in Table III of the paper to three decimals:

          5/9 -> [0.212, 0.863]     4/9 -> [0.137, 0.788]
          7/9 -> [0.400, 0.972]     2/9 -> [0.028, 0.600]

  fisher_exact_two_sided(a, b, c, d)
      The two-sided Fisher exact test on the 2x2 table [[a, b], [c, d]],
      by the conventional "sum the tables no more probable than the observed
      one" rule. Row and column margins are held fixed, so each table's
      probability is hypergeometric and `math.comb` gives it exactly.

      Reproduces the paper's de-confound p-value: the arm-fire signature
      (MECH-F84) against transfer, over the nine seeds, is the table
      [[1, 1], [1, 6]] and gives p = 0.4167, printed as 0.417.

IMPLEMENTATION NOTE

`BetaInv` is obtained by bisecting the regularized incomplete beta function
I_x(a,b), which is monotone in x. I_x itself is the standard Lentz continued
fraction with `math.lgamma` for the normalising constant. This is slower than a
library routine and entirely accurate enough: the fixed 200 bisection steps
bracket x to ~1e-60, far below the three decimals the paper prints. The point
of doing it this way is that the reader needs nothing but CPython to check it.
"""

from __future__ import annotations

from math import comb, exp, lgamma, log

__all__ = [
    "betainc_regularized",
    "beta_inverse",
    "clopper_pearson",
    "fisher_exact_two_sided",
]

_MAX_ITER = 300
_EPS = 3.0e-16
_FP_MIN = 1.0e-300
_BISECTIONS = 200


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Lentz evaluation of the continued fraction for the incomplete beta."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FP_MIN:
        d = _FP_MIN
    d = 1.0 / d
    h = d
    for m in range(1, _MAX_ITER + 1):
        m2 = 2 * m
        # even step
        num = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < _FP_MIN:
            d = _FP_MIN
        c = 1.0 + num / c
        if abs(c) < _FP_MIN:
            c = _FP_MIN
        d = 1.0 / d
        h *= d * c
        # odd step
        num = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + num * d
        if abs(d) < _FP_MIN:
            d = _FP_MIN
        c = 1.0 + num / c
        if abs(c) < _FP_MIN:
            c = _FP_MIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def betainc_regularized(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function I_x(a, b), for x in [0, 1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = exp(
        lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x)
    )
    # Use the reflected form where the continued fraction converges faster.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def beta_inverse(p: float, a: float, b: float) -> float:
    """The p-quantile of Beta(a, b): the x with I_x(a, b) = p.

    Found by bisection, which is safe here because I_x is monotone increasing
    in x for every positive a, b.
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(_BISECTIONS):
        mid = 0.5 * (lo + hi)
        if betainc_regularized(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided exact binomial (Clopper-Pearson) interval for k successes of n.

    Returns (lower, upper). The interval is conservative by construction: its
    coverage is at least 1 - alpha, which is why it is the right instrument for
    a nine-seed rate and why the paper reports it rather than a normal
    approximation.
    """
    if not isinstance(k, int) or not isinstance(n, int):
        raise TypeError("k and n must be ints")
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must satisfy 0 <= k <= n")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    lower = 0.0 if k == 0 else beta_inverse(alpha / 2.0, k, n - k + 1)
    upper = 1.0 if k == n else beta_inverse(1.0 - alpha / 2.0, k + 1, n - k)
    return lower, upper


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for the 2x2 table [[a, b], [c, d]].

    Margins are fixed, so the count in the top-left cell is hypergeometric.
    The two-sided p-value is the total probability of every attainable table
    whose probability does not exceed the observed table's.
    """
    for name, value in (("a", a), ("b", b), ("c", c), ("d", d)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an int")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    row1, row2 = a + b, c + d
    col1 = a + c
    total = row1 + row2
    if total == 0:
        raise ValueError("the table must contain at least one observation")

    denominator = comb(total, col1)

    def probability(top_left: int) -> float:
        return comb(row1, top_left) * comb(row2, col1 - top_left) / denominator

    observed = probability(a)
    low = max(0, col1 - row2)
    high = min(row1, col1)
    # The relative tolerance keeps a table that is equiprobable with the
    # observed one from being dropped by floating-point noise.
    return sum(
        probability(x)
        for x in range(low, high + 1)
        if probability(x) <= observed * (1.0 + 1.0e-9)
    )
