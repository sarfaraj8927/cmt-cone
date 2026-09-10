"""Causal cones of an observable and the algebra that the observable can actually see.

The proved bound of the corrected Theorem 4.1 involves the dimension of the *whole* dynamical
Lie algebra of the topology.  Experiment 5 showed that this is not enough: two topologies with
identical DLA dimension (the path and the star on the same qubit count and edge count) differ by
up to 7x in gradient variance, and two topologies whose algebras differ by a factor of eleven can
give numerically identical gradients.  Both anomalies have the same explanation, and it is a
statement about the *observable*, not about the graph alone.

For a depth-``L`` circuit whose entangling gates live on the edges of ``E``, the Heisenberg
evolution of an observable supported on ``A`` stays supported inside the ball of radius ``L``
around ``A`` in the graph ``E``.  The gradient of that observable therefore depends only on the
sub-topology induced on that ball, and the relevant algebra is the DLA of the induced subgraph --
the *cone algebra* -- not the DLA of ``E``.

This module computes the ball and the cone algebra.  The corresponding machine-checked
statements are

* ``CMT.Cone.suppIn_of_coneReach`` -- the depth-``L`` Heisenberg support is inside the ball;
* ``CMT.Cone.coneReach_congr_of_agree`` -- edges outside the ball cannot change it;
* ``CMT.Cone.finrank_coneSpan_le`` / ``CMT.Cone.gradient_variance_cone_lower_bound`` -- the
  observable-aware variance bound ``delta >= Cmin / (2 * 4 ** |ball|)``;
* ``CMT.Cone.ball_eq_univ_of_connected`` -- the ball is everything once ``L`` reaches the
  eccentricity of ``A``, which is when the global algebra starts to matter.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple

__all__ = [
    "adjacency",
    "ball",
    "components_in",
    "cone_dla_dimension",
    "global_dla_dimension",
    "eccentricity",
    "cone_report",
]

Edge = Tuple[int, int]


def adjacency(n: int, edges: Sequence[Edge]) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = {q: set() for q in range(n)}
    for j, k in edges:
        adj[j].add(k)
        adj[k].add(j)
    return adj


def ball(n: int, edges: Sequence[Edge], support: Iterable[int], radius: int) -> Set[int]:
    """Qubits at graph distance at most ``radius`` from ``support`` in the topology ``E``."""
    adj = adjacency(n, edges)
    cur: Set[int] = set(support)
    for _ in range(radius):
        nxt = set(cur)
        for q in cur:
            nxt |= adj[q]
        if nxt == cur:
            break
        cur = nxt
    return cur


def components_in(n: int, edges: Sequence[Edge], vertices: Iterable[int]) -> List[Set[int]]:
    """Connected components of the subgraph induced on ``vertices``."""
    vs = set(vertices)
    adj = adjacency(n, edges)
    seen: Set[int] = set()
    comps: List[Set[int]] = []
    for v in sorted(vs):
        if v in seen:
            continue
        stack = [v]
        comp: Set[int] = set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            for w in adj[u]:
                if w in vs and w not in comp:
                    stack.append(w)
        seen |= comp
        comps.append(comp)
    return comps


def _dim_from_components(comps: Sequence[Set[int]]) -> int:
    """``sum_components (4 ** |component| - 1)``.

    Exact for the ansatz generator set: every connected component of size ``s`` contributes the
    full ``su(2 ** s)`` (Lean ``CMT.finrank_cmtDLA_ge_of_connected``) and no Pauli string straddles
    two components (Lean ``CMT.finrank_cmtDLA_le_of_blocks``).
    """
    return sum(4 ** len(c) - 1 for c in comps)


def global_dla_dimension(n: int, edges: Sequence[Edge]) -> int:
    """Exact DLA dimension of the topology, from its connected components."""
    return _dim_from_components(components_in(n, edges, range(n)))


def cone_dla_dimension(
    n: int, edges: Sequence[Edge], support: Iterable[int], depth: int
) -> int:
    """Exact DLA dimension of the sub-topology induced on the depth-``depth`` cone of ``support``.

    This is the quantity the observable-aware bound is stated in terms of.
    """
    b = ball(n, edges, support, depth)
    return _dim_from_components(components_in(n, edges, b))


def eccentricity(n: int, edges: Sequence[Edge], support: Iterable[int]) -> int:
    """Smallest radius at which the ball around ``support`` stops growing.

    Below this depth, edges outside the ball provably cannot affect the observable
    (Lean ``CMT.Cone.coneReach_congr_of_agree``); at or above it, the cone bound and the
    global bound coincide on the component containing the support.
    """
    prev = set(support)
    r = 0
    while True:
        nxt = ball(n, edges, support, r + 1)
        if nxt == prev:
            return r
        prev = nxt
        r += 1


def cone_report(
    n: int, edges: Sequence[Edge], support: Iterable[int], depth: int
) -> Dict:
    b = ball(n, edges, support, depth)
    return {
        "n_qubits": n,
        "observable_support": sorted(set(support)),
        "depth": depth,
        "cone": sorted(b),
        "cone_size": len(b),
        "cone_dla_dimension": cone_dla_dimension(n, edges, support, depth),
        "global_dla_dimension": global_dla_dimension(n, edges),
        "eccentricity": eccentricity(n, edges, support),
        "cone_bound_proved": 2 * 4 ** len(b),
        "dense_dla": 4 ** n - 1,
    }
