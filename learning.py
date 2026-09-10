"""Experiment 7: a learning setting in which something actually learns, and the capacity sweep.

The training comparison of Experiment 4 (four qubits, two layers, sixty images, depolarizing
noise) ends at the majority-class rate for both ansätze, so it neither supports nor refutes any
claim about accuracy.  Two things were wrong with it as a test of *learning*: the model had too
little capacity and too little data, and every gradient step was paid for by a density-matrix
simulation, which capped the budget.

This module fixes both.  The circuit is simulated as a statevector with parameter broadcasting
over the whole batch, so a full BreastMNIST split can be used with several layers and a real
optimiser budget; the readout is the vector of all ``<Z_j>`` followed by a trained linear head,
which is the standard variational-classifier construction and has enough capacity to beat the
majority rate.  A logistic regression on the same features is reported next to it, so that
"the model learns" is measured against something.

The same routine answers the expressivity question.  A clustered ansatz provably cannot
correlate qubits in different clusters, so the cluster budget ``b`` is a trade-off knob:
:func:`capacity_sweep` runs the whole family from ``b = 1`` (no entanglement at all) to
``b = n`` (dense), plus the cut-budget relaxation (clusters joined by bridge edges), and reports
for each the exact algebra dimension, the cone dimension of the observable, the measured
gradient variance and the test accuracy.  That is the trainability-versus-performance curve the
paper needs in order to state the cost of the guarantee rather than merely acknowledging it.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .cone import cone_dla_dimension, global_dla_dimension
from .data import load_splits
from .topology import capacitated_topology, cluster_labels, cluster_sizes, complete_edges, pearson_matrix

__all__ = [
    "make_batched_qnode",
    "train_classifier",
    "logistic_baseline",
    "capacity_sweep",
    "bridge_edges",
]

Edge = Tuple[int, int]


def bridge_edges(n: int, edges: Sequence[Edge]) -> List[Edge]:
    """One inter-cluster edge per cluster boundary of the graph ``edges``: the cut relaxation.

    The clusters of ``edges`` are ordered by their smallest qubit and joined in a chain, one
    edge per boundary, so exactly ``K - 1`` edges are added and the result is connected.

    Lean: ``CMT.Conn.clusterBridgeE`` and ``CMT.Conn.cut_edges_restore_full_algebra`` -- these
    ``K - 1`` edges make the graph connected, so the *global* algebra bound is lost entirely
    (it jumps to the full ``4^n - 1``).  The cone bound survives when the bridges are far from
    the observable (``CMT.Cone.cone_bound_robust_to_far_edges``), and the point of the sweep is
    to measure which of the two predicts the gradient.
    """
    labels = cluster_labels(n, edges)
    reps: Dict[int, int] = {}
    for q in range(n):
        c = labels[q]
        if c not in reps:
            reps[c] = q
    order = sorted(reps.values())
    return [(order[i], order[i + 1]) for i in range(len(order) - 1)]


def cluster_block_edges(n: int, budget: int) -> List[Edge]:
    """All pairs inside consecutive blocks of ``budget`` qubits: ``CMT.clusterE``."""
    edges: List[Edge] = []
    for start in range(0, n, budget):
        block = list(range(start, min(start + budget, n)))
        for i in range(len(block)):
            for j in range(i + 1, len(block)):
                edges.append((block[i], block[j]))
    return edges


def make_batched_qnode(n: int, edges: Sequence[Edge], n_layers: int, noise: float = 0.0):
    """``[<Z_0>, ..., <Z_{n-1}>]`` for a whole batch of inputs at once."""
    device = "default.qubit" if noise == 0.0 else "default.mixed"
    dev = qml.device(device, wires=n)

    @qml.qnode(dev, diff_method="backprop")
    def circuit(theta, x):
        for q in range(n):
            qml.RY(x[..., q], wires=q)
        for layer in range(n_layers):
            for (j, k) in edges:
                qml.CZ(wires=[j, k])
                if noise > 0:
                    qml.DepolarizingChannel(noise, wires=j)
                    qml.DepolarizingChannel(noise, wires=k)
            for q in range(n):
                qml.RX(theta[layer, q, 0], wires=q)
                qml.RY(theta[layer, q, 1], wires=q)
                qml.RZ(theta[layer, q, 2], wires=q)
                if noise > 0:
                    qml.DepolarizingChannel(noise, wires=q)
        return [qml.expval(qml.PauliZ(q)) for q in range(n)]

    return circuit


def _accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.sign(np.asarray(pred)) == np.asarray(y)))


def train_classifier(
    n_qubits: int,
    edges: Sequence[Edge],
    n_layers: int,
    data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    epochs: int = 60,
    stepsize: float = 0.05,
    seed: int = 0,
    noise: float = 0.0,
    scale: float = np.pi,
    verbose: bool = False,
) -> Dict:
    """Train the variational classifier with Adam and report held-out accuracy.

    The readout is ``sum_j w_j <Z_j> + c`` with ``w, c`` trained jointly with the circuit
    parameters; the loss is the squared error against labels in ``{-1, +1}``.  The same seed
    gives the same initial parameters to every topology, so the comparison across the sweep is
    paired.
    """
    X_tr, Y_tr = data["train"]
    X_va, Y_va = data["val"]
    X_te, Y_te = data["test"]
    node = make_batched_qnode(n_qubits, edges, n_layers, noise)
    rng = np.random.default_rng(seed)
    theta = pnp.array(rng.uniform(0, 2 * np.pi, size=(n_layers, n_qubits, 3)), requires_grad=True)
    w = pnp.array(rng.normal(0, 0.1, size=n_qubits), requires_grad=True)
    c = pnp.array(0.0, requires_grad=True)

    xt = pnp.array(scale * X_tr, requires_grad=False)
    yt = pnp.array(Y_tr, requires_grad=False)

    def predict(theta, w, c, x):
        z = qml.math.stack(node(theta, x))          # (n_qubits, batch)
        return qml.math.sum(w[:, None] * z, axis=0) + c

    def cost(theta, w, c):
        return qml.math.mean((predict(theta, w, c, xt) - yt) ** 2)

    opt = qml.AdamOptimizer(stepsize=stepsize)
    history = []
    t0 = time.time()
    best = None
    for ep in range(epochs):
        (theta, w, c), loss = opt.step_and_cost(cost, theta, w, c)
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            pv = np.asarray(predict(theta, w, c, pnp.array(scale * X_va, requires_grad=False)))
            acc_va = _accuracy(pv, Y_va)
            history.append({"epoch": ep + 1, "train_loss": float(loss), "val_accuracy": acc_va})
            if best is None or acc_va > best["val_accuracy"]:
                best = {"epoch": ep + 1, "val_accuracy": acc_va,
                        "theta": np.array(theta), "w": np.array(w), "c": float(c)}
            if verbose:
                print(f"    epoch {ep + 1:3d} loss {float(loss):.4f} val acc {acc_va:.3f}",
                      flush=True)
    # evaluate the early-stopped model on the test split
    th = pnp.array(best["theta"], requires_grad=False)
    wb = pnp.array(best["w"], requires_grad=False)
    cb = pnp.array(best["c"], requires_grad=False)
    pte = np.asarray(predict(th, wb, cb, pnp.array(scale * X_te, requires_grad=False)))
    ptr = np.asarray(predict(th, wb, cb, xt))
    maj_te = float(max(np.mean(Y_te == 1), np.mean(Y_te == -1)))
    return {
        "edges": [list(e) for e in edges],
        "n_edges": len(edges),
        "epochs": epochs,
        "seed": seed,
        "final_train_loss": float(history[-1]["train_loss"]),
        "best_val_accuracy": float(best["val_accuracy"]),
        "best_epoch": int(best["epoch"]),
        "train_accuracy": _accuracy(ptr, Y_tr),
        "test_accuracy": _accuracy(pte, Y_te),
        "test_majority_rate": maj_te,
        "beats_majority": bool(_accuracy(pte, Y_te) > maj_te),
        "history": history,
        "seconds": time.time() - t0,
    }


def logistic_baseline(data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict:
    """Logistic regression on the same features, as a reference point for "does it learn"."""
    from sklearn.linear_model import LogisticRegression

    X_tr, Y_tr = data["train"]
    X_te, Y_te = data["test"]
    clf = LogisticRegression(max_iter=2000).fit(X_tr, (Y_tr > 0).astype(int))
    acc = float(clf.score(X_te, (Y_te > 0).astype(int)))
    maj = float(max(np.mean(Y_te == 1), np.mean(Y_te == -1)))
    return {"model": "logistic_regression", "test_accuracy": acc, "test_majority_rate": maj}


def gradient_variance(
    n_qubits: int,
    edges: Sequence[Edge],
    n_layers: int,
    n_samples: int = 150,
    seed: int = 0,
    observable_qubit: int = 0,
) -> float:
    """All-parameter gradient variance of ``<Z_{observable_qubit}>`` (unbiased, noiseless)."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="backprop")
    def circuit(theta, x):
        for q in range(n_qubits):
            qml.RY(x[q], wires=q)
        for layer in range(n_layers):
            for (j, k) in edges:
                qml.CZ(wires=[j, k])
            for q in range(n_qubits):
                qml.RX(theta[layer, q, 0], wires=q)
                qml.RY(theta[layer, q, 1], wires=q)
                qml.RZ(theta[layer, q, 2], wires=q)
        return qml.expval(qml.PauliZ(observable_qubit))

    rng = np.random.default_rng(seed)
    x = pnp.array(np.full(n_qubits, np.pi / 4), requires_grad=False)
    gs = []
    for _ in range(n_samples):
        theta = pnp.array(rng.uniform(0, 2 * np.pi, size=(n_layers, n_qubits, 3)),
                          requires_grad=True)
        gs.append(np.asarray(qml.grad(circuit)(theta, x)).ravel())
    g = np.array(gs)
    return float(np.var(g, axis=0, ddof=1).mean())


def _train_job(args):
    n_qubits, edges, n_layers, data, epochs, stepsize, seed = args
    return train_classifier(n_qubits, edges, n_layers, data, epochs=epochs,
                            stepsize=stepsize, seed=seed)


def capacity_sweep(
    n_qubits: int = 8,
    n_layers: int = 4,
    budgets: Sequence[int] = (1, 2, 4, 8),
    epochs: int = 60,
    seeds: Sequence[int] = (0, 1, 2),
    stepsize: float = 0.05,
    n_train: Optional[int] = None,
    n_eval: Optional[int] = None,
    variance_samples: int = 150,
    include_bridges: bool = True,
    include_dense: bool = True,
    jobs: int = 1,
    dataset: str = "breastmnist",
    verbose: bool = True,
) -> Dict:
    """Trainability against task performance across the cluster-budget family.

    For every budget ``b`` the topology is the data-derived capacitated one (clusters of at most
    ``b`` qubits, extracted from the *training* split only), plus -- optionally -- the same
    topology with one bridge edge per cluster boundary, which is the cut-size relaxation.
    """
    splits = load_splits(n_qubits, limits=(n_train, n_eval, n_eval), dataset=dataset)
    data = {k: splits[k] for k in ("train", "val", "test")}
    X_tr = data["train"][0]
    corr = np.abs(pearson_matrix(X_tr))

    families: Dict[str, List[Edge]] = {}
    for b in budgets:
        if b <= 1:
            families["b=1 (product)"] = []
            continue
        edges, _, _ = capacitated_topology(corr, budget=b, percentile=0.0)
        families[f"b={b} (data-derived)"] = list(edges)
        if include_bridges:
            extra = [e for e in bridge_edges(n_qubits, edges) if tuple(e) not in set(map(tuple, edges))]
            if extra:
                families[f"b={b} + bridges"] = list(edges) + extra
    if include_dense:
        families["dense (HEA)"] = complete_edges(n_qubits)

    rows = []
    t0 = time.time()
    pool = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
    futures: Dict[str, List] = {}
    if pool is not None:
        for name, edges in families.items():
            futures[name] = [pool.submit(_train_job, (n_qubits, list(edges), n_layers, data,
                                                      epochs, stepsize, s)) for s in seeds]
    for name, edges in families.items():
        sizes = sorted(cluster_sizes(cluster_labels(n_qubits, edges)).values(), reverse=True)
        if pool is not None:
            runs = [f.result() for f in futures[name]]
        else:
            runs = [train_classifier(n_qubits, edges, n_layers, data, epochs=epochs,
                                     stepsize=stepsize, seed=s) for s in seeds]
        accs = np.array([r["test_accuracy"] for r in runs])
        var = gradient_variance(n_qubits, edges, n_layers, n_samples=variance_samples)
        rows.append({
            "name": name,
            "edges": [list(e) for e in edges],
            "n_edges": len(edges),
            "cluster_sizes": sizes,
            "max_cluster": max(sizes) if sizes else 1,
            "global_dla_dimension": global_dla_dimension(n_qubits, edges),
            "cone_dla_dimension": cone_dla_dimension(n_qubits, edges, [0], n_layers),
            "gradient_variance": var,
            "test_accuracy_mean": float(accs.mean()),
            "test_accuracy_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            "test_majority_rate": runs[0]["test_majority_rate"],
            "beats_majority": bool(accs.mean() > runs[0]["test_majority_rate"]),
            "runs": runs,
        })
        if verbose:
            print(f"[exp7] {name:22s} |E|={len(edges):3d} dim_g={rows[-1]['global_dla_dimension']:<8d} "
                  f"dim_cone={rows[-1]['cone_dla_dimension']:<8d} Var={var:.3e} "
                  f"test acc {accs.mean():.3f} +- {rows[-1]['test_accuracy_std']:.3f} "
                  f"(majority {runs[0]['test_majority_rate']:.3f})", flush=True)

    if pool is not None:
        pool.shutdown()
    out = {
        "experiment": "capacity_sweep",
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "epochs": epochs,
        "seeds": list(seeds),
        "dataset": splits.get("source", dataset),
        "rows": rows,
        "seconds": time.time() - t0,
    }
    try:
        out["logistic_baseline"] = logistic_baseline(data)
    except Exception as exc:  # scikit-learn is optional
        out["logistic_baseline"] = {"error": str(exc)}
    if verbose and isinstance(out["logistic_baseline"], dict):
        print(f"[exp7] logistic baseline: {out['logistic_baseline']}", flush=True)
    return out
