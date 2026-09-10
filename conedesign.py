"""Causal-cone topology design: the observable-aware replacement for the sparsity rule.

The first draft of the CMT-QNN paper designed the circuit by a *global* rule: keep only the
strongest correlations, ask for an 80 % reduction of the entangling-gate count, and appeal to
the dynamical Lie algebra (DLA) of the whole topology.  Two things are wrong with that rule and
both are visible in the earlier measurements.

* Sparsity by itself certifies nothing.  A percentile-thresholded graph is normally *connected*,
  and a connected topology generates the full ``4**n - 1`` algebra
  (Lean ``CMT.finrank_cmtDLA_ge_of_backwardConnected``), so an 80 % reduction of the gate count
  can leave the algebraic bound completely vacuous.
* Even when the global algebra is genuinely small, it is the wrong predictor.  Two topologies
  with identical global DLA dimension were measured to differ by up to 7x in gradient variance,
  and two topologies whose global algebras differ by a factor of eleven gave numerically
  identical gradients.

The causal-cone rule fixes both.  For a depth-``L`` circuit the Heisenberg evolution of an
observable supported on ``A`` stays inside the ball of radius ``L`` around ``A`` in the topology
(Lean ``CMT.Cone.suppIn_of_coneReach``), edges outside that ball cannot change it at all
(``CMT.Cone.coneReach_congr_of_agree``, ``CMT.Cone.cone_bound_robust_to_far_edges``), and the
variance bound is controlled by the algebra of the *induced sub-topology on the ball*
(``CMT.Cone.finrank_coneSpan_le``, ``CMT.Cone.gradient_variance_cone_lower_bound``).

So the design rule becomes: **constrain the cone of the readout, not the global gate count.**
:func:`cone_constrained_topology` adds data-derived edges in decreasing order of correlation and
keeps an edge whenever the depth-``L`` cone of the readout register stays within a budget.  Edges
far from the readout are free — which is exactly why the cone rule can keep *more* of the data
structure than the sparsity rule at the same, or a better, trainability guarantee.

Two members of the family are used throughout the experiments:

``cc_sparse``
    the cone rule applied to the thresholded candidate list, so the resulting circuit is also
    sparse and directly comparable with the first draft's CMT-QNN;
``cc_expressive``
    the cone rule applied to *all* pairs, which keeps every edge outside the cone and therefore
    has many more entangling gates than the sparse design while having the same cone.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .cone import ball, cone_dla_dimension, components_in, global_dla_dimension

__all__ = [
    "cone_size",
    "cone_constrained_topology",
    "cone_design_report",
    "gate_reduction",
]

Edge = Tuple[int, int]


def cone_size(n: int, edges: Sequence[Edge], readout: Iterable[int], depth: int) -> int:
    """Number of qubits in the depth-``depth`` causal cone of ``readout``."""
    return len(ball(n, edges, readout, depth))


def gate_reduction(n: int, n_edges: int) -> float:
    """Fraction of the dense entangling-gate count that the topology removes."""
    dense = n * (n - 1) // 2
    return 1.0 - n_edges / dense if dense else 0.0


def cone_constrained_topology(
    weights: np.ndarray,
    depth: int,
    readout: Sequence[int] = (0,),
    cone_budget: int = 2,
    percentile: Optional[float] = None,
    threshold: Optional[float] = None,
    max_edges: Optional[int] = None,
    readout_first: bool = True,
    always_offer_readout_edges: bool = True,
) -> Tuple[List[Edge], float]:
    """Greedy data-derived topology whose depth-``depth`` readout cone stays within budget.

    Candidate pairs are those whose weight reaches the threshold (given directly, or as the
    ``percentile`` of the off-diagonal weights, or *all* pairs when both are ``None``).  They are
    visited by decreasing weight and an edge is accepted exactly when

        ``|ball_depth(readout)| <= cone_budget``

    still holds in the enlarged graph.  Accepting an edge can only grow the ball, so the invariant
    is maintained by construction and the returned topology satisfies the hypothesis of the cone
    bound with the stated budget.

    Returns the edge list and the threshold that was used.
    """
    W = np.abs(np.asarray(weights, dtype=float))
    n = W.shape[0]
    iu = np.triu_indices(n, k=1)
    if threshold is None:
        threshold = float(np.percentile(W[iu], percentile)) if percentile is not None else -np.inf
    ro = set(readout)
    cand = [(float(W[j, k]), (j, k)) for j in range(n) for k in range(j + 1, n)
            if W[j, k] >= threshold
            or (always_offer_readout_edges and (j in ro or k in ro))]
    if readout_first:
        # Edges incident to the readout are the only ones that can consume the cone budget, so
        # they are offered first; otherwise a run of strong edges elsewhere can fill the ball
        # first and leave the readout qubit unentangled, which costs expressivity for nothing.
        cand.sort(key=lambda t: (0 if (t[1][0] in ro or t[1][1] in ro) else 1, -t[0], t[1]))
    else:
        cand.sort(key=lambda t: (-t[0], t[1]))

    edges: List[Edge] = []
    for _, e in cand:
        if max_edges is not None and len(edges) >= max_edges:
            break
        trial = edges + [e]
        if cone_size(n, trial, readout, depth) <= cone_budget:
            edges.append(e)
    return sorted(edges), threshold


def cone_design_report(
    n: int,
    edges: Sequence[Edge],
    readout: Sequence[int],
    depth: int,
    name: str = "",
) -> Dict:
    """Every design quantity the results tables report, for one topology."""
    b = sorted(ball(n, edges, readout, depth))
    comps = components_in(n, edges, range(n))
    return {
        "name": name,
        "n_qubits": n,
        "depth": depth,
        "readout": list(readout),
        "edges": [list(e) for e in edges],
        "n_edges": len(edges),
        "dense_edges": n * (n - 1) // 2,
        "gate_reduction": gate_reduction(n, len(edges)),
        "cone": b,
        "cone_size": len(b),
        "cone_dla_dimension": cone_dla_dimension(n, edges, readout, depth),
        "global_dla_dimension": global_dla_dimension(n, edges),
        "dense_dla_dimension": 4 ** n - 1,
        "cluster_sizes": sorted((len(c) for c in comps), reverse=True),
        "proved_cone_bound": 2 * 4 ** len(b),
    }
