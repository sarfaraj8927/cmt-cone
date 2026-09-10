"""Data-derived topology extraction, corrected.

Two independent defects of the notebook version are fixed here.

1. **The co-occurrence matrix is not a correlation.**  The notebooks compute
   ``sigmoid(X).T @ sigmoid(X) / M`` on data that already lie in [0, 1].  Without centring, every
   entry is within 1/32 of the rank-one matrix of feature means (Lean:
   ``CMT.Impl.cooc_sigmoid_sub_rankOne_abs_le``), so the statistic essentially ranks pairs by
   brightness; centring removes exactly that rank-one term (Lean:
   ``CMT.Corr.cov_eq_cooc_sub_mean_mul``).  On explicit data the uncentred statistic ranks a pair
   of constant, uninformative pixels *above* a perfectly correlated pair and the covariance ranks
   them the other way round (Lean: ``CMT.Corr.cooc_ranking_differs_from_cov``).  The Pearson
   matrix used here is also invariant under affine renormalisation of the pixel values (Lean:
   ``CMT.Corr.threshold_scale_invariant``), which the sigmoid squashing is not.

2. **Percentile thresholding gives no cluster-size guarantee.**  The corrected Lemma 4.1 needs
   one: its hypothesis is bounded *cluster size*, not bounded degree.  A thresholded graph is
   typically connected, and then the dynamical Lie algebra is the full ``4**n - 1`` -- the same as
   the dense ansatz, so no algebra dimension is removed (Lean:
   ``CMT.finrank_cmtDLA_ge_of_backwardConnected``, ``CMT.Impl.finrank_cmtDLA_cooc_ge``).
   :func:`capacitated_topology` instead adds edges by decreasing correlation and keeps one only if
   the merged cluster stays within a budget ``b`` (Lean: ``CMT.Cluster.cluster``,
   ``CMT.Cluster.cluster_sizeLe``, ``CMT.Cluster.accepted_of_step``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "covariance_matrix",
    "pearson_matrix",
    "cooccurrence_matrix_notebook",
    "capacitated_topology",
    "threshold_topology",
    "cluster_labels",
    "cluster_sizes",
    "dla_upper_bound",
    "dense_dla_lower_bound",
    "complete_edges",
    "describe_topology",
]

Edge = Tuple[int, int]


# ----------------------------------------------------------------------------------------------
# Correlation statistics
# ----------------------------------------------------------------------------------------------

def covariance_matrix(X: np.ndarray) -> np.ndarray:
    """Empirical covariance of the columns of ``X`` (rows = samples).  Lean: ``CMT.Corr.cov``."""
    A = np.asarray(X, dtype=float)
    if A.ndim != 2 or A.shape[0] == 0:
        raise ValueError("X must be a non-empty 2-D array of shape (samples, features)")
    Ac = A - A.mean(axis=0, keepdims=True)
    return (Ac.T @ Ac) / A.shape[0]


def pearson_matrix(X: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix, with constant features given correlation 0.

    A constant pixel carries no information, so 0 is the right answer for it; the uncentred
    statistic of the notebooks ranks a pair of *bright constant* pixels highest.
    """
    C = covariance_matrix(X)
    sd = np.sqrt(np.diag(C))
    out = np.zeros_like(C)
    ok = sd > 0
    denom = np.outer(sd, sd)
    idx = np.outer(ok, ok)
    out[idx] = C[idx] / denom[idx]
    np.fill_diagonal(out, np.where(ok, 1.0, 0.0))
    return out


def cooccurrence_matrix_notebook(X: np.ndarray) -> np.ndarray:
    """The statistic the notebooks use: ``sigmoid(X).T @ sigmoid(X) / M``.

    Kept only so that the two can be compared side by side; it is *not* a correlation.
    """
    A = np.asarray(X, dtype=float)
    S = 1.0 / (1.0 + np.exp(-A))
    return (S.T @ S) / A.shape[0]


# ----------------------------------------------------------------------------------------------
# Topology extraction
# ----------------------------------------------------------------------------------------------

def complete_edges(n: int) -> List[Edge]:
    """All ``n(n-1)/2`` pairs: the hardware-efficient (dense) ansatz."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def threshold_topology(weights: np.ndarray, percentile: float = 80.0) -> List[Edge]:
    """Plain percentile thresholding, as in the notebooks.  No cluster-size guarantee."""
    W = np.asarray(weights, dtype=float)
    n = W.shape[0]
    upper = W[np.triu_indices(n, k=1)]
    if upper.size == 0:
        return []
    tau = float(np.percentile(upper, percentile))
    return [(j, k) for j in range(n) for k in range(j + 1, n) if W[j, k] >= tau]


def capacitated_topology(
    weights: np.ndarray,
    budget: int,
    percentile: Optional[float] = 80.0,
    threshold: Optional[float] = None,
) -> Tuple[List[Edge], np.ndarray, float]:
    """Greedy extraction of a topology whose clusters have at most ``budget`` qubits.

    Candidate edges are the pairs whose weight reaches the threshold (given directly, or as the
    ``percentile`` of the off-diagonal weights, or all pairs when both are ``None``).  They are
    processed by decreasing weight; an edge is kept when its endpoints already lie in the same
    cluster, or when merging their clusters keeps the size within ``budget``.

    This is exactly ``CMT.Cluster.cluster`` / ``CMT.Cluster.accepted``, and it comes with the
    proved guarantees

    * every cluster has at most ``budget`` qubits (``CMT.Cluster.cluster_sizeLe``);
    * ``dim g_CMT <= 2 * K * 4 ** b`` (``CMT.finrank_cmtDLA_le_of_block_count``,
      ``CMT.Cluster.finrank_cmtDLA_accepted_le``);
    * no candidate edge is dropped unless keeping it would exceed the budget
      (``CMT.Cluster.accepted_of_step``).

    Returns ``(edges, labels, tau)``.
    """
    W = np.asarray(weights, dtype=float)
    n = W.shape[0]
    if budget < 1:
        raise ValueError("budget must be at least 1")

    if threshold is None and percentile is not None:
        upper = W[np.triu_indices(n, k=1)]
        threshold = float(np.percentile(upper, percentile)) if upper.size else 0.0
    tau = -np.inf if threshold is None else float(threshold)

    candidates = [
        (W[j, k], j, k)
        for j in range(n)
        for k in range(j + 1, n)
        if W[j, k] >= tau
    ]
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))

    label = list(range(n))
    size = [1] * n
    kept: List[Edge] = []
    for _w, j, k in candidates:
        lj, lk = label[j], label[k]
        if lj == lk:
            kept.append((j, k))
        elif size[lj] + size[lk] <= budget:
            for q in range(n):
                if label[q] == lk:
                    label[q] = lj
            size[lj] += size[lk]
            size[lk] = 0
            kept.append((j, k))
    return kept, np.array(label, dtype=int), tau


def cluster_labels(n: int, edges: Sequence[Edge]) -> np.ndarray:
    """Connected-component label of every qubit under ``edges`` (union-find)."""
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for j, k in edges:
        ra, rb = find(j), find(k)
        if ra != rb:
            parent[rb] = ra
    return np.array([find(q) for q in range(n)], dtype=int)


def cluster_sizes(labels: Sequence[int]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for c in np.asarray(labels).tolist():
        out[c] = out.get(c, 0) + 1
    return out


def dla_upper_bound(labels: Sequence[int]) -> int:
    """``2 * K * 4 ** b`` -- Lean ``CMT.finrank_cmtDLA_le_of_block_count``.

    ``K`` is the number of clusters and ``b`` the largest cluster size.  This is the number the
    paper's Experiment 1 should report next to the sparsity: it is what makes the corrected
    Lemma 4.1, and hence the variance bound of the corrected Theorem 4.1, applicable.
    """
    sizes = cluster_sizes(labels)
    return 2 * len(sizes) * 4 ** max(sizes.values())


def dense_dla_lower_bound(n: int) -> int:
    """``4 ** n - 1``: the DLA of any *connected* topology.

    Lean: ``CMT.finrank_cmtDLA_ge_of_backwardConnected``.
    """
    return 4 ** n - 1


def describe_topology(n: int, edges: Sequence[Edge], labels: Optional[Sequence[int]] = None) -> Dict:
    """Everything Experiment 1 should report about an extracted topology."""
    lab = cluster_labels(n, edges) if labels is None else np.asarray(labels)
    sizes = cluster_sizes(lab)
    n_dense = n * (n - 1) // 2
    return {
        "n_qubits": n,
        "n_edges": len(edges),
        "n_edges_dense": n_dense,
        "sparsity": 1.0 - (len(edges) / n_dense if n_dense else 0.0),
        "n_clusters": len(sizes),
        "max_cluster_size": max(sizes.values()),
        "cluster_sizes": sorted(sizes.values(), reverse=True),
        "dla_upper_bound": dla_upper_bound(lab),
        "dense_dla_lower_bound": dense_dla_lower_bound(n),
        "edges": [tuple(int(v) for v in e) for e in edges],
    }
