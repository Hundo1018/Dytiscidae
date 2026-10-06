"""When a pass share is evidence: exact binomial bounds and the no-model verdict.

From NeutronGym (arXiv 2610.03631, §5.1, ``docs/papers/2610.03631.md``): before a
pass rate is read as capability, the bar faces probes that use no model -- here,
a machine with every actuator held still -- and the decision is made on the
one-sided 95% Clopper-Pearson bound, not on the observed share.  "0 of 218"
reads as zero; its bound is 1.4%, and against 30 elites it may certify nothing.

The paper compared the probe's upper bound with a typed 20% ceiling.  This
project sets thresholds from measurement (CLAUDE.md, "Designing a measurement"
rule 5), so the ceiling is the elites' own *lower* bound on the same bar: a bar
is certified when a still machine clears it measurably less often than the
elites do.  Nothing is typed but the confidence.

Pure Python on purpose (``tests/test_architecture.py``): this is arithmetic on
counts, and it must run in the cheap suites.
"""

from __future__ import annotations

import math

#: Verdicts of :func:`no_model_verdict`, most to least informative.
CERTIFIED = "certified"        # still upper bound < elite lower bound
LEAK = "leak"                  # still machines clear it, not separable from elites
UNDERPOWERED = "underpowered"  # no still machine cleared it, too few to separate
UNMEASURED = "unmeasured"      # no elite clears it: nothing to certify


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed in log space."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    lc = math.lgamma(n + 1)
    terms = [lc - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
             for i in range(k + 1)]
    top = max(terms)
    return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in terms))


def _solve(f, target: float) -> float:
    """The p in [0, 1] where the decreasing function ``f`` crosses ``target``."""
    lo, hi = 0.0, 1.0
    for _ in range(80):                      # 2^-80: far below any share we read
        mid = 0.5 * (lo + hi)
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """One-sided exact bounds ``(lower, upper)`` on a binomial share ``k / n``.

    Each bound is one-sided at ``confidence``: ``upper`` is the largest p with
    P(X <= k | p) >= 1 - confidence, ``lower`` the smallest with
    P(X >= k | p) >= 1 - confidence.  For k = 0 the upper bound is
    ``1 - alpha**(1/n)``; for n = 0 nothing is known and the result is (0, 1).
    """
    if not (0 <= k <= n):
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n == 0:
        return 0.0, 1.0
    alpha = 1.0 - confidence
    upper = 1.0 if k == n else _solve(lambda p: _binom_cdf(k, n, p), alpha)
    lower = 0.0 if k == 0 else _solve(lambda p: _binom_cdf(k - 1, n, p), 1.0 - alpha)
    return lower, upper


def no_model_verdict(still_k: int, still_n: int, elite_k: int, elite_n: int,
                     confidence: float = 0.95) -> str:
    """Whether a bar is earned, from how often still machines and elites clear it.

    ``certified`` needs the still machines' upper bound strictly below the
    elites' lower bound.  A bar no still machine cleared is still only
    ``underpowered`` until the counts can separate the two -- "0 of 12" is not
    a certificate.
    """
    if elite_n == 0 or elite_k == 0:
        return UNMEASURED
    _, still_hi = clopper_pearson(still_k, still_n, confidence)
    elite_lo, _ = clopper_pearson(elite_k, elite_n, confidence)
    if still_hi < elite_lo:
        return CERTIFIED
    return UNDERPOWERED if still_k == 0 else LEAK


def distinct_share(keys) -> tuple[int, int]:
    """``(distinct, total)`` among the passes' keys (NeutronGym's concentration).

    A diagnostic, read by hand: a bar cleared 141 times by 42 distinct designs
    was the paper's degenerate family that the fixed-answer probe missed.
    """
    keys = list(keys)
    return len(set(keys)), len(keys)
