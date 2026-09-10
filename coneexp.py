"""The causal-cone experiment suite (Experiments C1-C5).

These are the experiments the first draft's claims are re-measured with, using the causal-cone
design rule in place of the global sparsity rule.  Everything is at the configuration the first
draft reports: eight qubits, four layers, BreastMNIST (and PneumoniaMNIST as the second dataset),
depolarizing noise up to p = 0.10, PennyLane's density-matrix simulator, readout ``<Z_0>``.

* **C1** — the design table: what each rule extracts from the data, and what each rule certifies.
* **C2** — gradient variance against the dense hardware-efficient ansatz across the noise sweep,
  paired draws, bootstrap intervals, in both the unmatched (first-draft) and the matched
  noise-budget protocol.
* **C3** — the two cone-specific predictions: gradients of parameters outside the cone vanish
  exactly, and the cone dimension orders the measured variances where the global dimension
  cannot.
* **C4** — convergence and accuracy, with the honest epochs-to-target statistic reported next to
  the first draft's ratio-of-percentage-reductions statistic.
* **C5** — cone dimension against qubit number, showing the guarantee is uniform in ``n``.

Parallelism: gradient draws are independent, so they are farmed out to a process pool; each
worker rebuilds its own QNode (PennyLane objects are not picklable) and pins BLAS to one thread.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ansatz import cone_matched_padding
from .cone import ball, cone_dla_dimension, global_dla_dimension
from .conedesign import cone_constrained_topology, cone_design_report, gate_reduction
from .data import load_splits
from .stats import componentwise_variance, paired_variance_ratio_bootstrap
from .topology import (
    capacitated_topology,
    complete_edges,
    pearson_matrix,
    threshold_topology,
)

__all__ = [
    "build_family",
    "design_table",
    "gradient_draws",
    "variance_sweep",
    "cone_locality_check",
    "cone_scaling_table",
    "variance_scaling",
]

Edge = Tuple[int, int]

DENSE_NAME = "HEA (dense)"


# ----------------------------------------------------------------------------------------------
# C1: the design family
# ----------------------------------------------------------------------------------------------

def build_family(
    n_qubits: int = 8,
    n_layers: int = 4,
    readout: Sequence[int] = (0,),
    dataset: str = "breastmnist",
    percentile: float = 80.0,
) -> Dict[str, Dict]:
    """The topologies compared throughout, all extracted from the *training* split only.

    ``sparsity-only`` is the first draft's rule (keep the top 20 % of the correlations); it does
    reach the advertised 80 % gate reduction but the resulting graph is connected, so it certifies
    nothing.  ``CMT b=2`` is the cluster rule of the corrected paper.  The two ``CC`` entries are
    the causal-cone rule of this suite: the sparse one is directly comparable with the first
    draft, and the expressive one keeps *every* edge that lies outside the readout cone.
    """
    splits = load_splits(n_qubits, dataset=dataset)
    X_tr = splits["train"][0]
    W = np.abs(pearson_matrix(X_tr))

    fam: Dict[str, Dict] = {}

    fam[DENSE_NAME] = {"edges": complete_edges(n_qubits), "rule": "all pairs"}

    thr = threshold_topology(W, percentile=percentile)
    fam[f"sparsity-only ({percentile:.0f}th pct)"] = {
        "edges": list(thr), "rule": "first draft: keep the strongest correlations"}

    clus, _, _ = capacitated_topology(W, budget=2, percentile=0.0)
    fam["CMT b=2 (cluster rule)"] = {
        "edges": list(clus), "rule": "corrected paper: clusters of at most two qubits"}

    cc_sparse, tau = cone_constrained_topology(
        W, depth=n_layers, readout=readout, cone_budget=2, percentile=percentile)
    fam["CC-sparse (cone rule, k=2)"] = {
        "edges": list(cc_sparse), "rule": "cone rule on the thresholded candidates",
        "threshold": tau}

    cc_expr, _ = cone_constrained_topology(
        W, depth=n_layers, readout=readout, cone_budget=2, percentile=None)
    fam["CC-expressive (cone rule, k=2)"] = {
        "edges": list(cc_expr), "rule": "cone rule on all pairs: every off-cone edge is kept"}

    cc3, _ = cone_constrained_topology(
        W, depth=n_layers, readout=readout, cone_budget=3, percentile=None)
    fam["CC-expressive (cone rule, k=3)"] = {
        "edges": list(cc3), "rule": "cone rule on all pairs with a three-qubit cone"}

    for name, d in fam.items():
        d.update(cone_design_report(n_qubits, d["edges"], readout, n_layers, name))
        d["dataset"] = splits.get("source", dataset)
    return fam


def design_table(
    n_qubits: int = 8,
    n_layers: int = 4,
    readout: Sequence[int] = (0,),
    datasets: Sequence[str] = ("breastmnist", "pneumoniamnist"),
    percentile: float = 80.0,
) -> Dict:
    """Experiment C1: what each design rule extracts, and what it certifies."""
    out: Dict = {"experiment": "C1_design", "n_qubits": n_qubits, "n_layers": n_layers,
                 "readout": list(readout), "rows": {}}
    for ds in datasets:
        fam = build_family(n_qubits, n_layers, readout, ds, percentile)
        out["rows"][ds] = [dict(v) for v in fam.values()]
    return out


# ----------------------------------------------------------------------------------------------
# C2: gradient variance under noise, in parallel
# ----------------------------------------------------------------------------------------------

def _worker(args) -> np.ndarray:
    n, edges, n_layers, noise, pad, seed, count, angle, obs_qubit, pad_wires = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import pennylane as qml
    from pennylane import numpy as pnp

    from .ansatz import make_qnode

    node = make_qnode(n, [tuple(e) for e in edges], n_layers, noise, pad, "z0",
                      pad_wires=tuple(pad_wires))
    if obs_qubit != 0:  # the readout is qubit 0 by construction of the family
        raise ValueError("only the qubit-0 readout is implemented")
    rng = np.random.default_rng(seed)
    x = pnp.array(np.full(n, angle), requires_grad=False)
    out = []
    for _ in range(count):
        theta = pnp.array(rng.uniform(0, 2 * np.pi, size=(n_layers, n, 3)), requires_grad=True)
        out.append(np.asarray(qml.grad(node)(theta, x)))
    return np.asarray(out)


def gradient_draws(
    n_qubits: int,
    edges: Sequence[Edge],
    n_layers: int,
    noise: float,
    pad: int,
    n_samples: int,
    seed: int,
    jobs: int = 1,
    angle: float = np.pi / 4,
    pad_wires: Sequence[int] = (),
) -> np.ndarray:
    """``n_samples`` full gradients of ``<Z_0>`` at random parameter draws.

    The draws are *paired across topologies*: chunk ``c`` always uses seed ``seed + c``, so two
    topologies evaluated with the same ``seed`` and the same chunking see the same parameters.
    """
    jobs = max(1, jobs)
    chunks = [n_samples // jobs + (1 if i < n_samples % jobs else 0) for i in range(jobs)]
    args = [(n_qubits, [list(e) for e in edges], n_layers, noise, pad, seed + i, c, angle, 0,
             tuple(pad_wires)) for i, c in enumerate(chunks) if c > 0]
    if jobs == 1:
        blocks = [_worker(a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            blocks = list(pool.map(_worker, args))
    return np.concatenate(blocks, axis=0)


def variance_sweep(
    family: Dict[str, Dict],
    n_qubits: int = 8,
    n_layers: int = 4,
    noise_values: Sequence[float] = (0.0, 0.01, 0.03, 0.05, 0.10),
    n_samples: int = 150,
    seed: int = 12345,
    match_noise=False,
    jobs: int = 8,
    verbose: bool = True,
) -> Dict:
    """Experiment C2: paired gradient-variance benchmark of every topology against the HEA.

    Three noise protocols are available.

    ``match_noise=False``
        the first draft's protocol: every circuit carries its own ``2|E| + n`` channels per
        layer, so part of the measured advantage is the smaller gate count;
    ``match_noise=True``
        every circuit is padded to the *total* dense channel count
        (Lean ``CMT.NoiseBudget.padded_channels_eq``).  This equalises the budget but not its
        placement, and the padding that lands on the cone qubits is what then dominates;
    ``match_noise="cone"``
        every circuit is padded, *on the qubits of its own causal cone*, up to the number of
        channels the dense circuit applies to those same qubits.  The two circuits then suffer
        the same local decoherence exactly where the gradient of the readout can see it, so the
        remaining difference is algebraic.  This is the control a cone design calls for.
    """
    dense_edges = complete_edges(n_qubits)
    rows = []
    t0 = time.time()
    for p in noise_values:
        grads: Dict[str, np.ndarray] = {}
        pads: Dict[str, int] = {}
        for name, d in family.items():
            edges = [tuple(e) for e in d["edges"]]
            cone = sorted(ball(n_qubits, edges, [0], n_layers))
            pad_wires: Sequence[int] = ()
            if match_noise == "cone":
                pad = cone_matched_padding(n_qubits, edges, cone)
                pad_wires = cone
            elif match_noise:
                pad = 2 * (len(dense_edges) - len(edges))
            else:
                pad = 0
            pads[name] = int(pad)
            grads[name] = gradient_draws(n_qubits, edges, n_layers, p, pad, n_samples,
                                         seed, jobs=jobs, pad_wires=pad_wires)
            if verbose:
                print(f"[C2] p={p} {name:34s} done ({time.time() - t0:.0f}s)", flush=True)
        gd = grads[DENSE_NAME]
        var_dense, _ = componentwise_variance(gd)
        for name, g in grads.items():
            var, per = componentwise_variance(g)
            boot = paired_variance_ratio_bootstrap(g, gd, seed=seed)
            cone = ball(n_qubits, [tuple(e) for e in family[name]["edges"]], [0], n_layers)
            in_cone = np.zeros((n_layers, n_qubits, 3), dtype=bool)
            for q in cone:
                in_cone[:, q, :] = True
            per3 = per.reshape(n_layers, n_qubits, 3)
            rows.append({
                "noise": float(p),
                "topology": name,
                "n_edges": len(family[name]["edges"]),
                "match_noise": match_noise if isinstance(match_noise, str) else bool(match_noise),
                "pad_channels_per_layer": pads[name],
                "cone_size": len(cone),
                "cone_dla_dimension": family[name]["cone_dla_dimension"],
                "global_dla_dimension": family[name]["global_dla_dimension"],
                "variance": var,
                "variance_dense": var_dense,
                "ratio_to_dense": boot["ratio"],
                "ratio_ci95": [boot["ci_low"], boot["ci_high"]],
                "ratio_significant": bool(boot["significant"]),
                "variance_in_cone": float(per3[in_cone].mean()) if in_cone.any() else 0.0,
                "max_variance_out_of_cone": (float(per3[~in_cone].max())
                                             if (~in_cone).any() else 0.0),
            })
            if verbose:
                r = rows[-1]
                print(f"[C2] p={p} {name:34s} Var={var:.3e} ratio={r['ratio_to_dense']:.2f} "
                      f"CI=({boot['ci_low']:.2f},{boot['ci_high']:.2f})", flush=True)
    return {
        "experiment": "C2_variance_sweep",
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_samples": n_samples,
        "seed": seed,
        "match_noise": match_noise if isinstance(match_noise, str) else bool(match_noise),
        "simulator": "default.mixed (density matrix), backprop gradients",
        "rows": rows,
        "seconds": time.time() - t0,
    }


# ----------------------------------------------------------------------------------------------
# C3: the two cone-specific predictions
# ----------------------------------------------------------------------------------------------

def cone_locality_check(
    family: Dict[str, Dict],
    n_qubits: int = 8,
    n_layers: int = 4,
    noise: float = 0.10,
    n_samples: int = 60,
    seed: int = 999,
    jobs: int = 8,
) -> Dict:
    """Experiment C3: off-cone gradients vanish, and the cone dimension orders the variances.

    The first statement is exact (Lean ``CMT.Cone.gradient_vanishes_outside_cone``): a parameter
    carried by a qubit outside the depth-``L`` ball of the readout cannot influence ``<Z_0>`` at
    all, so its gradient is zero to machine precision — under the depolarizing noise as well,
    because the channels are local and therefore respect the same support argument.

    The second is the design rule: sorting the family by cone dimension must sort it by measured
    variance, where sorting by global dimension must not.
    """
    rows = []
    t0 = time.time()
    for name, d in family.items():
        edges = [tuple(e) for e in d["edges"]]
        g = gradient_draws(n_qubits, edges, n_layers, noise, 0, n_samples, seed, jobs=jobs)
        _, per = componentwise_variance(g)
        per3 = per.reshape(n_layers, n_qubits, 3)
        cone = ball(n_qubits, edges, [0], n_layers)
        mask = np.zeros((n_layers, n_qubits, 3), dtype=bool)
        for q in cone:
            mask[:, q, :] = True
        rows.append({
            "topology": name,
            "cone": sorted(cone),
            "cone_size": len(cone),
            "cone_dla_dimension": cone_dla_dimension(n_qubits, edges, [0], n_layers),
            "global_dla_dimension": global_dla_dimension(n_qubits, edges),
            "mean_variance_in_cone": float(per3[mask].mean()),
            "max_variance_out_of_cone": float(per3[~mask].max()) if (~mask).any() else 0.0,
            "max_abs_gradient_out_of_cone": (
                float(np.abs(g.reshape(len(g), -1).reshape(-1, n_layers, n_qubits, 3)[:, ~mask]).max())
                if (~mask).any() else 0.0),
            "n_params_out_of_cone": int((~mask).sum()),
        })
        print(f"[C3] {name:34s} cone={sorted(cone)} "
              f"max |g| off cone = {rows[-1]['max_abs_gradient_out_of_cone']:.2e}", flush=True)

    def concordance(key: str) -> Dict[str, float]:
        pairs = 0
        agree = 0
        ties = 0
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if a[key] == b[key]:
                    ties += 1
                    continue
                pairs += 1
                # smaller algebra dimension must give the larger variance
                if (a[key] < b[key]) == (a["mean_variance_in_cone"] > b["mean_variance_in_cone"]):
                    agree += 1
        return {"pairs": pairs, "correct": agree, "ties": ties,
                "fraction": agree / pairs if pairs else float("nan")}

    return {
        "experiment": "C3_cone_locality",
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "noise": noise,
        "n_samples": n_samples,
        "rows": rows,
        "concordance_cone": concordance("cone_dla_dimension"),
        "concordance_global": concordance("global_dla_dimension"),
        "seconds": time.time() - t0,
    }


# ----------------------------------------------------------------------------------------------
# C5: the guarantee is uniform in the qubit number
# ----------------------------------------------------------------------------------------------

def _clean_worker(args) -> np.ndarray:
    n, edges, n_layers, seed, count, angle = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import pennylane as qml
    from pennylane import numpy as pnp

    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, diff_method="backprop")
    def circuit(theta, x):
        for q in range(n):
            qml.RY(x[q], wires=q)
        for layer in range(n_layers):
            for (j, k) in edges:
                qml.CZ(wires=[j, k])
            for q in range(n):
                qml.RX(theta[layer, q, 0], wires=q)
                qml.RY(theta[layer, q, 1], wires=q)
                qml.RZ(theta[layer, q, 2], wires=q)
        return qml.expval(qml.PauliZ(0))

    rng = np.random.default_rng(seed)
    x = pnp.array(np.full(n, angle), requires_grad=False)
    out = []
    for _ in range(count):
        theta = pnp.array(rng.uniform(0, 2 * np.pi, size=(n_layers, n, 3)), requires_grad=True)
        out.append(np.asarray(qml.grad(circuit)(theta, x)))
    return np.asarray(out)


def variance_scaling(
    qubits: Sequence[int] = (4, 6, 8, 10, 12, 14),
    n_layers: int = 4,
    cone_budget: int = 2,
    n_samples: int = 200,
    seed: int = 2024,
    dataset: str = "breastmnist",
    jobs: int = 8,
) -> Dict:
    """Experiment C6: how the advantage of the cone design grows with the qubit number.

    Noiseless statevector simulation (which is what makes 14 qubits affordable), the data-derived
    cone design against the dense ansatz at the same depth, paired draws and bootstrap intervals.
    The dense ansatz has to lose ground as ``n`` grows because its cone is the whole register,
    while the cone design keeps a two-qubit cone at every ``n``.
    """
    rows = []
    t0 = time.time()
    for n in qubits:
        splits = load_splits(n, dataset=dataset)
        W = np.abs(pearson_matrix(splits["train"][0]))
        cc, _ = cone_constrained_topology(W, depth=n_layers, readout=(0,),
                                          cone_budget=cone_budget, percentile=None)
        dense = complete_edges(n)
        chunks = [n_samples // jobs + (1 if i < n_samples % jobs else 0) for i in range(jobs)]
        grads = {}
        for name, edges in (("dense", dense), ("cone_design", cc)):
            args = [(n, [tuple(e) for e in edges], n_layers, seed + i, c, np.pi / 4)
                    for i, c in enumerate(chunks) if c > 0]
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                blocks = list(pool.map(_clean_worker, args))
            grads[name] = np.concatenate(blocks, axis=0)
        vd, _ = componentwise_variance(grads["dense"])
        vc, _ = componentwise_variance(grads["cone_design"])
        boot = paired_variance_ratio_bootstrap(grads["cone_design"], grads["dense"], seed=seed)
        rep = cone_design_report(n, cc, (0,), n_layers)
        rows.append({
            "n_qubits": n,
            "n_edges_cone_design": len(cc),
            "n_edges_dense": len(dense),
            "cone": rep["cone"],
            "cone_dla_dimension": rep["cone_dla_dimension"],
            "global_dla_dimension": rep["global_dla_dimension"],
            "variance_dense": vd,
            "variance_cone_design": vc,
            "ratio": boot["ratio"],
            "ratio_ci95": [boot["ci_low"], boot["ci_high"]],
        })
        print(f"[C6] n={n:2d} |E|={len(cc):3d} cone={rep['cone']} Var_dense={vd:.3e} "
              f"Var_cone={vc:.3e} ratio={boot['ratio']:.2f} "
              f"({boot['ci_low']:.2f}, {boot['ci_high']:.2f})", flush=True)
    return {"experiment": "C6_variance_scaling", "n_layers": n_layers,
            "cone_budget": cone_budget, "n_samples": n_samples, "dataset": dataset,
            "noise": 0.0, "rows": rows, "seconds": time.time() - t0}


def cone_scaling_table(
    qubits: Sequence[int] = (8, 12, 16, 20, 24, 28, 32),
    n_layers: int = 4,
    cone_budget: int = 2,
    dataset: str = "breastmnist",
) -> Dict:
    """Experiment C5: the design quantities as the qubit number grows.

    Purely combinatorial — no circuit is simulated — so it reaches sizes at which simulation is
    impossible, which is the point: the cone bound does not depend on ``n``
    (Lean ``CMT.Cone.cone_bound_uniform_in_n``), while the dense algebra is ``4**n - 1``.
    """
    rows = []
    for n in qubits:
        splits = load_splits(n, dataset=dataset)
        W = np.abs(pearson_matrix(splits["train"][0]))
        edges, _ = cone_constrained_topology(W, depth=n_layers, readout=(0,),
                                             cone_budget=cone_budget, percentile=None)
        rep = cone_design_report(n, edges, (0,), n_layers)
        rep["gate_reduction"] = gate_reduction(n, len(edges))
        rows.append(rep)
        print(f"[C5] n={n:3d} |E|={len(edges):4d} cone={rep['cone']} "
              f"dim_cone={rep['cone_dla_dimension']} dim_dense={4 ** n - 1}", flush=True)
    return {"experiment": "C5_cone_scaling", "n_layers": n_layers,
            "cone_budget": cone_budget, "dataset": dataset, "rows": rows}
