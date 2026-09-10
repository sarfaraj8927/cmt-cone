"""Experiment 5: an *iso-resource* test of the algebraic mechanism.

Every earlier gradient-variance comparison in this project (and in the notebooks) contrasts a
sparse topology with the dense all-to-all one.  Two things then differ at once: the dimension of
the dynamical Lie algebra (the mechanism of the corrected Theorem 4.1) and the number of gates,
hence the number of depolarizing channels (the mechanism of the corrected Lemma 4.2).  Matching
the noise budget removes the *noise* part of the second effect, but the two circuits still differ
in unitary gate count and in the entangling pattern.

This experiment removes the confound completely.  All topologies compared here have

* the same qubit number ``n``,
* the same number of edges ``|E|`` -- hence the same number of CZ gates, the same circuit depth
  and, at equal ``|E|``, exactly the same number of depolarizing channels per layer
  (``2|E| + n``, Lean ``CMT.NoiseBudget.channels_eight`` for the arithmetic),
* the same number of variational parameters ``3nL``, the same encoding and the same observable,

and differ only in *which* pairs are entangled, i.e. only in the cluster structure and therefore
only in the exact dimension of the dynamical Lie algebra
(``dla.lie_closure_dimension``; bound: ``CMT.finrank_cmtDLA_le_of_block_count``, sharpness:
``CMT.finrank_cmtDLA_ge_of_backwardConnected``).

If the gradient variance tracks the DLA dimension across such a family, the algebraic mechanism is
doing the work; if it does not, the observed advantage of sparse ansätze is a gate-count effect.
This is the experiment that discriminates between the two readings, and neither the notebooks nor
Experiments 2-4 can do it.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .ansatz import channels_per_layer, make_qnode, param_shape
from .dla import lie_closure_dimension
from .stats import componentwise_variance, paired_variance_ratio_bootstrap
from .topology import cluster_labels, cluster_sizes

__all__ = ["isoresource_families", "experiment_isoresource"]

Edge = Tuple[int, int]


def isoresource_families(n_qubits: int) -> Dict[str, List[Edge]]:
    """Topologies on ``n_qubits`` with a common edge count and different cluster structure.

    ``n // 2`` edges are used in every case: the perfect matching (all clusters of size 2, the
    smallest DLA available at that edge count) up to the path and the star (one connected cluster
    of ``n // 2 + 1`` qubits, the largest).
    """
    if n_qubits < 4 or n_qubits % 2 != 0:
        raise ValueError("n_qubits must be even and at least 4")
    m = n_qubits // 2
    matching = [(2 * i, 2 * i + 1) for i in range(m)]
    path = [(i, i + 1) for i in range(m)]
    star = [(0, i + 1) for i in range(m)]
    # a mixed one: a chain of three qubits, then a matching on the rest
    mixed = [(0, 1), (1, 2)] + [(2 * i + 1, 2 * i + 2) for i in range(1, m - 1)]
    fams = {"matching": matching, "mixed": mixed, "path": path, "star": star}
    return {k: v for k, v in fams.items() if len(v) == m}


def experiment_isoresource(
    n_qubits: int = 6,
    topologies: Optional[Dict[str, Sequence[Edge]]] = None,
    n_layers: int = 2,
    noise: float = 0.05,
    n_samples: int = 400,
    seed: int = 0,
    reference: str = "path",
    diff_method: str = "backprop",
    encoding_angle: float = np.pi / 4,
    verbose: bool = True,
) -> Dict:
    """Gradient variance across topologies with identical resources but different DLA dimension.

    The same parameter draws are fed to every circuit (fully paired design), the variance is
    averaged over all ``3nL`` components with ``ddof = 1``, and the ratio of each topology to the
    ``reference`` one carries a paired bootstrap 95 % interval.
    """
    tops = dict(topologies) if topologies is not None else isoresource_families(n_qubits)
    counts = {k: len(v) for k, v in tops.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"topologies must have equal edge counts, got {counts}")
    if reference not in tops:
        raise ValueError(f"reference {reference!r} not among {sorted(tops)}")
    n_edges = next(iter(counts.values()))

    nodes = {k: make_qnode(n_qubits, v, n_layers, noise, 0, "z0", diff_method)
             for k, v in tops.items()}
    rng = np.random.default_rng(seed)
    x = pnp.array(np.full(n_qubits, encoding_angle), requires_grad=False)
    shape = param_shape(n_qubits, n_layers)

    grads: Dict[str, List[np.ndarray]] = {k: [] for k in tops}
    t0 = time.time()
    for i in range(n_samples):
        theta = pnp.array(rng.uniform(0, 2 * np.pi, size=shape), requires_grad=True)
        for k, node in nodes.items():
            grads[k].append(np.asarray(qml.grad(node)(theta, x)))
        if verbose and (i + 1) % max(1, n_samples // 5) == 0:
            print(f"[exp5] n={n_qubits} sample {i + 1}/{n_samples} ({time.time() - t0:.1f}s)")

    arrays = {k: np.array(v) for k, v in grads.items()}
    ref = arrays[reference]
    rows = []
    for k, v in tops.items():
        var, _ = componentwise_variance(arrays[k])
        sizes = sorted(cluster_sizes(cluster_labels(n_qubits, v)).values(), reverse=True)
        boot = paired_variance_ratio_bootstrap(arrays[k], ref, seed=seed)
        rows.append({
            "name": k,
            "edges": [list(e) for e in v],
            "n_edges": len(v),
            "channels_per_layer": channels_per_layer(n_qubits, len(v)),
            "cluster_sizes": sizes,
            "dla_dimension_exact": lie_closure_dimension(n_qubits, v),
            "variance": var,
            "ratio_to_reference": boot["ratio"],
            "ratio_ci95": (boot["ci_low"], boot["ci_high"]),
            "significant": bool(boot["significant"]),
        })
    rows.sort(key=lambda r: r["dla_dimension_exact"])

    dims = np.array([r["dla_dimension_exact"] for r in rows], dtype=float)
    vars_ = np.array([r["variance"] for r in rows], dtype=float)
    out = {
        "experiment": "isoresource",
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "noise": noise,
        "n_samples": n_samples,
        "seed": seed,
        "reference": reference,
        "n_edges_common": n_edges,
        "channels_per_layer_common": channels_per_layer(n_qubits, n_edges),
        "dense_dla_lower_bound": 4 ** n_qubits - 1,
        "rows": rows,
        "seconds": time.time() - t0,
    }
    if len(rows) > 2:
        out["log_log_slope"] = float(np.polyfit(np.log(dims), np.log(vars_), 1)[0])
        out["spearman_like_monotone"] = bool(np.all(np.diff(vars_) <= 0))
    if verbose:
        print(f"[exp5] n={n_qubits}: {n_edges} edges, "
              f"{channels_per_layer(n_qubits, n_edges)} channels/layer for every topology")
        for r in rows:
            print(f"[exp5] {r['name']:9s} clusters {str(r['cluster_sizes']):16s} "
                  f"dim g = {r['dla_dimension_exact']:6d}  Var = {r['variance']:.3e}  "
                  f"ratio to {reference} = {r['ratio_to_reference']:.3f} "
                  f"CI95 ({r['ratio_ci95'][0]:.3f}, {r['ratio_ci95'][1]:.3f})")
        if "log_log_slope" in out:
            print(f"[exp5] log Var vs log dim g: slope {out['log_log_slope']:.3f} "
                  f"(1/dim would give -1)")
    return out
