"""Estimators with the properties that are proved in Lean (`RequestProject/Estimator.lean`).

The notebooks report a bare quotient of two ``np.var`` values computed from 15-50 draws of a
*single* gradient component, with no error bars and no pairing between the two circuits.  Three
things are wrong with that and are fixed here:

* ``np.var`` defaults to ``ddof=0``; it equals the unbiased estimator times ``(N-1)/N``
  (Lean: ``CMT.Estimator.naiveVar_eq_sampleVar``), i.e. biased low by 6.7 % at N = 15;
* a ratio of two independent noisy estimates has a much wider distribution than either of them,
  so the two draws must be *paired* and the ratio must carry a confidence interval;
* the sample size must be chosen from the effect one wants to resolve
  (Lean: ``CMT.Estimator.samples_for_confidence``): ~500 draws for a 20 % effect at 95 %
  confidence with the Chebyshev bound, against the 15-50 the notebooks use.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np

__all__ = [
    "sample_variance",
    "naive_variance",
    "mean_confidence_interval",
    "samples_for_confidence",
    "paired_ratio_bootstrap",
    "componentwise_variance",
    "paired_variance_ratio_bootstrap",
]


def sample_variance(xs: Sequence[float]) -> float:
    """Unbiased (``ddof=1``) sample variance -- Lean ``CMT.Estimator.sampleVar_unbiased``."""
    a = np.asarray(xs, dtype=float).ravel()
    if a.size < 2:
        raise ValueError("need at least two samples for an unbiased variance")
    return float(np.var(a, ddof=1))


def naive_variance(xs: Sequence[float]) -> float:
    """``np.var`` with ``ddof=0``, as used in the notebooks; biased low by ``(N-1)/N``."""
    a = np.asarray(xs, dtype=float).ravel()
    return float(np.var(a, ddof=0))


def componentwise_variance(grads: np.ndarray) -> Tuple[float, np.ndarray]:
    """Gradient variance averaged over *all* parameters.

    ``grads`` has shape ``(n_samples, ...)``: one full gradient per random parameter draw.  The
    variance of each component is estimated with ``ddof=1`` over the draws and the components are
    then averaged, which is the quantity the barren-plateau statements are about.  The notebooks
    instead keep ``grad[0, 0, 0]`` only, throwing away all but one of the ``3 * n * L`` components.

    Returns ``(mean_variance, per_component_variances)``.
    """
    g = np.asarray(grads, dtype=float)
    flat = g.reshape(g.shape[0], -1)
    if flat.shape[0] < 2:
        raise ValueError("need at least two parameter draws")
    per = np.var(flat, axis=0, ddof=1)
    return float(per.mean()), per


def mean_confidence_interval(
    xs: Sequence[float], delta: float = 0.05
) -> Dict[str, float]:
    """Error bars for a mean: distribution-free (Chebyshev) and normal-approximation.

    The Chebyshev half width ``sqrt(v / (N * delta))`` is exactly the ``t`` for which
    ``CMT.Estimator.mean_concentration`` gives failure probability ``delta``; it needs no
    distributional assumption.  The normal half width ``1.96 * sqrt(v / N)`` is reported for
    reference and is only asymptotically valid.
    """
    a = np.asarray(xs, dtype=float).ravel()
    n = a.size
    v = sample_variance(a)
    return {
        "mean": float(a.mean()),
        "n": int(n),
        "sample_variance": v,
        "chebyshev_half_width": math.sqrt(v / (n * delta)),
        "normal_half_width": 1.96 * math.sqrt(v / n),
    }


def samples_for_confidence(variance: float, tolerance: float, delta: float = 0.05) -> int:
    """``N >= v / (delta * t^2)`` -- Lean ``CMT.Estimator.samples_for_confidence``."""
    if tolerance <= 0 or not 0 < delta < 1:
        raise ValueError("tolerance must be positive and delta in (0,1)")
    return int(math.ceil(variance / (delta * tolerance ** 2)))


def paired_ratio_bootstrap(
    sparse: Sequence[float],
    dense: Sequence[float],
    resamples: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap interval for the variance ratio from *paired* draws.

    ``sparse[i]`` and ``dense[i]`` must come from the same parameter draw (same seed, same
    parameter vector fed to both circuits).  The bootstrap resamples draw indices, so the pairing
    is preserved and the interval accounts for the correlation between the two estimates.

    Two archived runs of the identical 8-qubit p = 0.05 configuration in the notebooks give 2.27x
    and 1.08x (Lean: ``CMT.Impl.archived_ratio_runs_inconsistent``); an interval that contains 1
    is the honest statement in that situation.
    """
    s = np.asarray(sparse, dtype=float).ravel()
    d = np.asarray(dense, dtype=float).ravel()
    if s.size != d.size:
        raise ValueError("paired samples must have equal length")
    rng = np.random.default_rng(seed)
    point = sample_variance(s) / sample_variance(d)
    idx = rng.integers(0, s.size, size=(resamples, s.size))
    vs = np.var(s[idx], axis=1, ddof=1)
    vd = np.var(d[idx], axis=1, ddof=1)
    good = vd > 0
    ratios = np.sort(vs[good] / vd[good])
    lo = float(np.quantile(ratios, alpha / 2))
    hi = float(np.quantile(ratios, 1 - alpha / 2))
    return {
        "ratio": float(point),
        "ci_low": lo,
        "ci_high": hi,
        "significant": bool(lo > 1.0 or hi < 1.0),
    }


def paired_variance_ratio_bootstrap(
    grads_sparse: np.ndarray,
    grads_dense: np.ndarray,
    resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap interval for the ratio of the *parameter-averaged* gradient variances.

    ``grads_sparse`` and ``grads_dense`` have shape ``(n_samples, ...)`` and row ``i`` of both
    must come from the same parameter draw.  The statistic is
    ``mean_k Var_i[g_ik]`` with ``ddof=1``, and the bootstrap resamples the draws ``i``, keeping
    the pairing.
    """
    s = np.asarray(grads_sparse, dtype=float).reshape(len(grads_sparse), -1)
    d = np.asarray(grads_dense, dtype=float).reshape(len(grads_dense), -1)
    if s.shape[0] != d.shape[0]:
        raise ValueError("paired samples must have equal length")
    rng = np.random.default_rng(seed)

    def stat(a: np.ndarray) -> float:
        return float(np.var(a, axis=0, ddof=1).mean())

    denom = stat(d)
    if denom <= 0:
        raise ValueError("the reference gradient variance is zero; nothing to compare")
    point = stat(s) / denom
    ratios = []
    for _ in range(resamples):
        idx = rng.integers(0, s.shape[0], size=s.shape[0])
        vd = stat(d[idx])
        if vd > 0:  # a degenerate resample (all indices equal) carries no information
            ratios.append(stat(s[idx]) / vd)
    ratios = np.sort(np.asarray(ratios))
    lo = float(np.quantile(ratios, alpha / 2))
    hi = float(np.quantile(ratios, 1 - alpha / 2))
    return {
        "ratio": float(point),
        "ci_low": lo,
        "ci_high": hi,
        "significant": bool(lo > 1.0 or hi < 1.0),
    }
