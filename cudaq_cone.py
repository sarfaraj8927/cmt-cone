"""Causal cones and cone DLA dimension for CUDA-Q / NVIDIA Quantum Stack.

Ported from the PennyLane version with identical functionality.
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
    """``sum_components (4 ** |component| - 1)``."""
    return sum(4 ** len(c) - 1 for c in comps)


def global_dla_dimension(n: int, edges: Sequence[Edge]) -> int:
    """Exact DLA dimension of the topology, from its connected components."""
    return _dim_from_components(components_in(n, edges, range(n)))


def cone_dla_dimension(
    n: int, edges: Sequence[Edge], support: Iterable[int], depth: int
) -> int:
    """Exact DLA dimension of the sub-topology induced on the depth-``depth`` cone of ``support``."""
    b = ball(n, edges, support, depth)
    return _dim_from_components(components_in(n, edges, b))


def eccentricity(n: int, edges: Sequence[Edge], support: Iterable[int]) -> int:
    """Smallest radius at which the ball around ``support`` stops growing."""
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