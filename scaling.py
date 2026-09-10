"""Experiment 6: the iso-resource comparison pushed to a scale where it is a scaling claim.

Experiment 5 compared four topologies at ``n = 6, 8`` and two depths.  That is enough to show
that the algebraic mechanism exists once the gate count is held fixed, but not enough to say how
the effect behaves with ``n``, and it produced two anomalies that the global-DLA bound cannot
explain (equal-dimension topologies differing by up to 7x, and topologies with very different
algebras giving identical gradients).

This module does three things Experiment 5 could not.

1. **Scale.**  Noiseless statevector simulation (``default.qubit``) replaces the density-matrix
   simulation, so the family runs to ``n = 14`` at depths ``L = 2, 4, 6`` with several
   observables and two edge counts.  Barren-plateau statements of the algebraic type are
   noiseless statements, so this is the regime in which the mechanism should be tested; the
   noisy check that the ordering survives depolarizing noise is
   :func:`noise_ordering_check`, run at the sizes where a density matrix is affordable.

2. **Per-parameter resolution.**  The variance of *every* parameter is recorded, not only the
   average, together with the layer it belongs to.  The observable-aware theory predicts the
   variance of a parameter in layer ``l`` of an ``L``-layer circuit from the ball of radius
   ``L - l`` around the observable's support, so the per-parameter data is what actually tests
   it (:func:`parameter_records`).

3. **Two competing predictors.**  Each configuration is scored against the global DLA dimension
   (what the proved bound of the corrected Theorem 4.1 uses) and against the cone dimension of
   the new observable-aware bound (Lean ``CMT.Cone.gradient_variance_cone_lower_bound``).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .cone import ball, cone_dla_dimension, eccentricity, global_dla_dimension

__all__ = [
    "scaling_families",
    "observable_support",
    "parameter_records",
    "experiment_scaling",
    "noise_ordering_check",
]

Edge = Tuple[int, int]


def scaling_families(n: int, n_edges: int) -> Dict[str, List[Edge]]:
    """Topologies on ``n`` qubits with exactly ``n_edges`` edges and different cluster structure.

    ``matching`` (all clusters of size 2, smallest algebra), ``path`` and ``star`` (one connected
    cluster of ``n_edges + 1`` qubits, the largest algebra available at this edge count),
    ``mixed`` (a short chain plus a matching) and ``far_path`` -- a path placed on the *last*
    ``n_edges + 1`` qubits, which has exactly the same algebra as ``path`` but is disjoint from
    the causal cone of an observable on qubit 0 at small depth.  ``far_path`` is the control that
    separates the two predictors.
    """
    if n_edges > n - 1 or n_edges < 1:
        raise ValueError("n_edges must be between 1 and n-1")
    fams: Dict[str, List[Edge]] = {}
    if n_edges <= n // 2:
        fams["matching"] = [(2 * i, 2 * i + 1) for i in range(n_edges)]
    fams["path"] = [(i, i + 1) for i in range(n_edges)]
    fams["star"] = [(0, i + 1) for i in range(n_edges)]
    if n_edges >= 3 and n_edges + 1 <= n:
        far = [(n - 1 - i - 1, n - 1 - i) for i in range(n_edges)]
        fams["far_path"] = [tuple(sorted(e)) for e in far]
    if n_edges >= 3 and 2 + (n_edges - 2) * 2 <= n:
        mixed = [(0, 1), (1, 2)] + [(2 * i + 1, 2 * i + 2) for i in range(1, n_edges - 1)]
        if max(max(e) for e in mixed) < n and len(set(mixed)) == n_edges:
            fams["mixed"] = mixed
    return {k: v for k, v in fams.items() if len(v) == n_edges}


def observable_support(name: str, n: int) -> List[int]:
    if name == "z0":
        return [0]
    if name == "zlast":
        return [n - 1]
    if name == "zmid":
        return [n // 2]
    if name == "z0z1":
        return [0, 1]
    raise ValueError(f"unknown observable {name!r}")


def _observable(name: str, n: int):
    if name == "z0":
        return qml.PauliZ(0)
    if name == "zlast":
        return qml.PauliZ(n - 1)
    if name == "zmid":
        return qml.PauliZ(n // 2)
    if name == "z0z1":
        return qml.PauliZ(0) @ qml.PauliZ(1)
    raise ValueError(f"unknown observable {name!r}")


def _make_node(
    n: int,
    edges: Sequence[Edge],
    n_layers: int,
    observable: str,
    noise: float = 0.0,
    noise_model: str = "depolarizing",
    pad_channels: int = 0,
):
    device = "default.qubit" if noise == 0.0 else "default.mixed"
    dev = qml.device(device, wires=n)
    obs = _observable(observable, n)

    def channel(p, wire):
        if noise_model == "depolarizing":
            qml.DepolarizingChannel(p, wires=wire)
        elif noise_model == "amplitude_damping":
            qml.AmplitudeDamping(p, wires=wire)
        else:
            raise ValueError(f"unknown noise model {noise_model!r}")

    @qml.qnode(dev, diff_method="backprop")
    def circuit(theta, x):
        for q in range(n):
            qml.RY(x[q], wires=q)
        for layer in range(n_layers):
            for (j, k) in edges:
                qml.CZ(wires=[j, k])
                if noise > 0:
                    channel(noise, j)
                    channel(noise, k)
            for q in range(n):
                qml.RX(theta[layer, q, 0], wires=q)
                qml.RY(theta[layer, q, 1], wires=q)
                qml.RZ(theta[layer, q, 2], wires=q)
                if noise > 0:
                    channel(noise, q)
            if noise > 0:
                for i in range(pad_channels):
                    channel(noise, i % n)
        return qml.expval(obs)

    return circuit


def parameter_records(
    n: int,
    edges: Sequence[Edge],
    n_layers: int,
    observable: str,
    per_variance: np.ndarray,
) -> List[Dict]:
    """One record per parameter: its own variance and the cone the theory assigns to it.

    A parameter in layer ``l`` (1-based) of an ``L``-layer circuit is followed by ``L - l``
    further entangling layers, so in the Heisenberg picture the observable can only reach it
    through the ball of radius ``L - l`` around its support.  The theory therefore predicts

    * a variance of exactly zero when the parameter's qubit is outside that ball
      (Lean ``CMT.Cone.coneReach_congr_of_agree``: the whole gradient is unchanged by
      anything outside the ball, and a parameter outside it acts trivially), and
    * otherwise a variance controlled by ``1 / dim g_cone`` with the cone algebra of that radius
      (Lean ``CMT.Cone.gradient_variance_cone_lower_bound``).
    """
    supp = observable_support(observable, n)
    recs: List[Dict] = []
    for layer in range(n_layers):
        radius = n_layers - 1 - layer
        b = ball(n, edges, supp, radius)
        dim = cone_dla_dimension(n, edges, supp, radius)
        for q in range(n):
            for axis in range(3):
                recs.append({
                    "layer": layer,
                    "qubit": q,
                    "axis": axis,
                    "radius": radius,
                    "in_cone": bool(q in b),
                    "cone_size": len(b),
                    "cone_dla_dimension": dim,
                    "variance": float(per_variance[layer, q, axis]),
                })
    return recs


def experiment_scaling(
    n_qubits: int,
    n_layers: int,
    observable: str = "z0",
    n_edges: Optional[int] = None,
    n_samples: int = 300,
    seed: int = 0,
    encoding_angle: float = np.pi / 4,
    noise: float = 0.0,
    verbose: bool = True,
) -> Dict:
    """Gradient variance of a matched-resource family, with per-parameter resolution."""
    m = n_edges if n_edges is not None else n_qubits // 2
    tops = scaling_families(n_qubits, m)
    supp = observable_support(observable, n_qubits)
    rng = np.random.default_rng(seed)
    x = pnp.array(np.full(n_qubits, encoding_angle), requires_grad=False)
    shape = (n_layers, n_qubits, 3)
    nodes = {k: _make_node(n_qubits, v, n_layers, observable, noise) for k, v in tops.items()}

    grads: Dict[str, List[np.ndarray]] = {k: [] for k in tops}
    t0 = time.time()
    thetas = [pnp.array(rng.uniform(0, 2 * np.pi, size=shape), requires_grad=True)
              for _ in range(n_samples)]
    for i, theta in enumerate(thetas):
        for k, node in nodes.items():
            grads[k].append(np.asarray(qml.grad(node)(theta, x)))
        if verbose and (i + 1) % max(1, n_samples // 4) == 0:
            print(f"[exp6] n={n_qubits} L={n_layers} obs={observable} "
                  f"{i + 1}/{n_samples} ({time.time() - t0:.1f}s)", flush=True)

    rows = []
    for k, v in tops.items():
        g = np.array(grads[k])
        per = np.var(g.reshape(g.shape[0], -1), axis=0, ddof=1).reshape(shape)
        rows.append({
            "name": k,
            "edges": [list(e) for e in v],
            "n_edges": len(v),
            "global_dla_dimension": global_dla_dimension(n_qubits, v),
            "cone_dla_dimension": cone_dla_dimension(n_qubits, v, supp, n_layers),
            "cone_size": len(ball(n_qubits, v, supp, n_layers)),
            "eccentricity": eccentricity(n_qubits, v, supp),
            "variance_mean": float(per.mean()),
            "variance_in_cone_mean": float(
                np.mean([r["variance"] for r in parameter_records(
                    n_qubits, v, n_layers, observable, per) if r["in_cone"]])),
            "max_variance_out_of_cone": float(max(
                [r["variance"] for r in parameter_records(
                    n_qubits, v, n_layers, observable, per) if not r["in_cone"]],
                default=0.0)),
            "parameters": parameter_records(n_qubits, v, n_layers, observable, per),
        })
    rows.sort(key=lambda r: r["cone_dla_dimension"])
    out = {
        "experiment": "scaling_isoresource",
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "observable": observable,
        "observable_support": supp,
        "n_edges_common": m,
        "n_samples": n_samples,
        "seed": seed,
        "noise": noise,
        "device": "default.qubit" if noise == 0.0 else "default.mixed",
        "dense_dla": 4 ** n_qubits - 1,
        "rows": rows,
        "seconds": time.time() - t0,
    }
    if verbose:
        for r in rows:
            print(f"[exp6] {r['name']:9s} dim_g={r['global_dla_dimension']:<8d} "
                  f"dim_cone={r['cone_dla_dimension']:<8d} Var={r['variance_mean']:.3e} "
                  f"out-of-cone max={r['max_variance_out_of_cone']:.2e}", flush=True)
    return out


def noise_ordering_check(
    n_qubits: int = 6,
    n_layers: int = 4,
    observable: str = "z0",
    noise: float = 0.05,
    noise_model: str = "depolarizing",
    n_samples: int = 200,
    seed: int = 0,
    verbose: bool = True,
) -> Dict:
    """Does the noiseless ordering of the family survive a noise channel?

    Run with ``noise_model='depolarizing'`` and ``'amplitude_damping'`` this answers the
    "one noise model only" objection at the sizes where a density-matrix simulation is
    affordable.
    """
    m = n_qubits // 2
    tops = scaling_families(n_qubits, m)
    supp = observable_support(observable, n_qubits)
    rng = np.random.default_rng(seed)
    x = pnp.array(np.full(n_qubits, np.pi / 4), requires_grad=False)
    shape = (n_layers, n_qubits, 3)
    nodes = {k: _make_node(n_qubits, v, n_layers, observable, noise, noise_model)
             for k, v in tops.items()}
    grads: Dict[str, List[np.ndarray]] = {k: [] for k in tops}
    t0 = time.time()
    for _ in range(n_samples):
        theta = pnp.array(rng.uniform(0, 2 * np.pi, size=shape), requires_grad=True)
        for k, node in nodes.items():
            grads[k].append(np.asarray(qml.grad(node)(theta, x)))
    rows = []
    for k, v in tops.items():
        g = np.array(grads[k])
        per = np.var(g.reshape(g.shape[0], -1), axis=0, ddof=1)
        rows.append({
            "name": k,
            "edges": [list(e) for e in v],
            "global_dla_dimension": global_dla_dimension(n_qubits, v),
            "cone_dla_dimension": cone_dla_dimension(n_qubits, v, supp, n_layers),
            "variance_mean": float(per.mean()),
        })
    rows.sort(key=lambda r: r["cone_dla_dimension"])
    out = {
        "experiment": "noise_ordering_check",
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "observable": observable,
        "noise": noise,
        "noise_model": noise_model,
        "n_samples": n_samples,
        "seed": seed,
        "rows": rows,
        "cone_order_monotone": bool(
            all(rows[i]["variance_mean"] >= rows[i + 1]["variance_mean"]
                for i in range(len(rows) - 1))),
        "seconds": time.time() - t0,
    }
    if verbose:
        print(f"[noise-check] {noise_model} p={noise}: "
              f"monotone in cone dimension = {out['cone_order_monotone']}", flush=True)
        for r in rows:
            print(f"   {r['name']:9s} dim_cone={r['cone_dla_dimension']:<8d} "
                  f"Var={r['variance_mean']:.3e}", flush=True)
    return out
