"""Data-derived topology extraction for CUDA-Q / NVIDIA Quantum Stack.

This module ports the topology extraction to work with CUDA-Q workflows.
The classical computation (correlation, clustering) remains the same.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cudaq_ansatz import Edge

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
    "describe_topology",
    "complete_edges",
]


# ----------------------------------------------------------------------------------------------
# Correlation statistics
# ----------------------------------------------------------------------------------------------

def covariance_matrix(X: np.ndarray) -> np.ndarray:
    """Empirical covariance of the columns of ``X`` (rows = samples)."""
    A = np.asarray(X, dtype=float)
    if A.ndim != 2 or A.shape[0] == 0:
        raise ValueError("X must be a non-empty 2-D array of shape (samples, features)")
    Ac = A - A.mean(axis=0, keepdims=True)
    return (Ac.T @ Ac) / A.shape[0]


def pearson_matrix(X: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix, with constant features given correlation 0."""
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
    """The statistic the notebooks use: ``sigmoid(X).T @ sigmoid(X) / M``."""
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
    """Plain percentile thresholding, as in the notebooks. No cluster-size guarantee."""
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
    """Greedy extraction of a topology whose clusters have at most ``budget`` qubits."""
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
    """``2 * K * 4 ** b`` -- the proved DLA upper bound for capacitated topology."""
    sizes = cluster_sizes(labels)
    return 2 * len(sizes) * 4 ** max(sizes.values())


def dense_dla_lower_bound(n: int) -> int:
    """``4 ** n - 1``: the DLA of any *connected* topology."""
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


def extract_topology_from_cudaq_data(
    quantum_data: np.ndarray,
    n_qubits: int,
    budget: int,
    percentile: float = 80.0,
) -> Dict:
    """Extract topology from quantum measurement data using CUDA-Q.
    
    Args:
        quantum_data: Measurement outcomes from CUDA-Q (shots x n_qubits)
        n_qubits: Number of qubits
        budget: Maximum cluster size
        percentile: Correlation threshold percentile
    
    Returns:
        Topology description dictionary
    """
    # Convert measurement data to correlation matrix
    # quantum_data is binary (0/1) measurement outcomes
    X = quantum_data.astype(float)
    corr = pearson_matrix(X)
    
    edges, labels, tau = capacitated_topology(corr, budget=budget, percentile=percentile)
    
    return {
        **describe_topology(n_qubits, edges, labels),
        "threshold_tau": tau,
        "correlation_matrix": corr,
    }