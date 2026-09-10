"""Experiment C4: convergence and accuracy under the causal-cone design.

Two protocols are run.

**Noiseless (exact).**  Statevector simulation with parameter broadcasting over the whole batch,
the full official BreastMNIST / PneumoniaMNIST splits, a trained linear head over the readout,
Adam, and identical initialisation, budget and data for every topology.  This is the setting in
which accuracy can be compared fairly.

**Noisy at p = 0.10 (quantum trajectories).**  A density-matrix simulation of eight qubits at
four layers costs minutes per gradient step, which is why the first draft's training comparison
was run at four qubits on sixty images.  The depolarizing channel is instead unravelled into
Pauli trajectories: ``DepolarizingChannel(p)`` maps ``rho`` to ``(1-p) rho + (p/3)(X rho X +
Y rho Y + Z rho Z)``, so inserting, independently at every noise location, the identity with
probability ``1-p`` and one of ``X, Y, Z`` with probability ``p/3`` each, and averaging over the
insertions, reproduces the channel exactly in expectation.  Each trajectory is a *statevector*
simulation, so a full training run becomes affordable, and because the insertions do not depend
on the parameters the average of the trajectory gradients is an unbiased estimate of the gradient
of the noisy expectation value.  :func:`validate_trajectories` checks the unravelling against the
density-matrix simulator on the very circuits used here.

Convergence is reported as **epochs to reach a target training loss**, the statistic a
"converges k times faster" claim has to be based on.  For comparison, the first draft's statistic
— the quotient of the two percentage loss reductions — is computed as well, since it is what the
"4.6x faster" figure of the submitted abstract actually is (Lean
``CMT.Impl.table3_speedup_is_ratio_of_reductions``); the two are different quantities and are kept
apart in the report.
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

__all__ = [
    "noise_sites",
    "sample_errors",
    "make_trajectory_node",
    "validate_trajectories",
    "train_run",
    "convergence_experiment",
]

Edge = Tuple[int, int]


# ----------------------------------------------------------------------------------------------
# Pauli-trajectory unravelling of the depolarizing channel
# ----------------------------------------------------------------------------------------------

def noise_sites(n_qubits: int, edges: Sequence[Edge], n_layers: int) -> int:
    """Number of depolarizing channels in the circuit: ``L * (2|E| + n)``."""
    return n_layers * (2 * len(edges) + n_qubits)


def sample_errors(rng: np.random.Generator, n_sites: int, p: float) -> np.ndarray:
    """One error realisation: ``0`` = identity, ``1, 2, 3`` = ``X, Y, Z``."""
    u = rng.random(n_sites)
    which = rng.integers(1, 4, size=n_sites)
    return np.where(u < p, which, 0).astype(int)


def _apply_error(code: int, wire: int) -> None:
    if code == 1:
        qml.PauliX(wires=wire)
    elif code == 2:
        qml.PauliY(wires=wire)
    elif code == 3:
        qml.PauliZ(wires=wire)


def make_trajectory_node(n_qubits: int, edges: Sequence[Edge], n_layers: int):
    """``[<Z_0>, ..., <Z_{n-1}>]`` on a statevector, with a supplied Pauli-error realisation.

    ``errs`` is an integer array of length :func:`noise_sites`; it is data, not a parameter, so
    differentiating through the node differentiates the trajectory at fixed noise.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="backprop")
    def circuit(theta, x, errs):
        for q in range(n_qubits):
            qml.RY(x[..., q], wires=q)
        k = 0
        for layer in range(n_layers):
            for (j, kk) in edges:
                qml.CZ(wires=[j, kk])
                _apply_error(int(errs[k]), j); k += 1
                _apply_error(int(errs[k]), kk); k += 1
            for q in range(n_qubits):
                qml.RX(theta[layer, q, 0], wires=q)
                qml.RY(theta[layer, q, 1], wires=q)
                qml.RZ(theta[layer, q, 2], wires=q)
                _apply_error(int(errs[k]), q); k += 1
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return circuit


def _mixed_node(n_qubits: int, edges: Sequence[Edge], n_layers: int, p: float):
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev, diff_method="backprop")
    def circuit(theta, x):
        for q in range(n_qubits):
            qml.RY(x[..., q], wires=q)
        for layer in range(n_layers):
            for (j, k) in edges:
                qml.CZ(wires=[j, k])
                qml.DepolarizingChannel(p, wires=j)
                qml.DepolarizingChannel(p, wires=k)
            for q in range(n_qubits):
                qml.RX(theta[layer, q, 0], wires=q)
                qml.RY(theta[layer, q, 1], wires=q)
                qml.RZ(theta[layer, q, 2], wires=q)
                qml.DepolarizingChannel(p, wires=q)
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return circuit


def validate_trajectories(
    n_qubits: int = 8,
    edges: Sequence[Edge] = ((0, 1), (2, 3), (4, 5), (6, 7)),
    n_layers: int = 4,
    p: float = 0.10,
    n_trajectories: int = 4000,
    seed: int = 0,
) -> Dict:
    """Check the unravelling against the density-matrix simulator on the same circuit."""
    edges = [tuple(e) for e in edges]
    rng = np.random.default_rng(seed)
    theta = pnp.array(rng.uniform(0, 2 * np.pi, size=(n_layers, n_qubits, 3)), requires_grad=False)
    x = pnp.array(rng.uniform(0, np.pi, size=n_qubits), requires_grad=False)

    exact = np.asarray(_mixed_node(n_qubits, edges, n_layers, p)(theta, x), dtype=float)

    node = make_trajectory_node(n_qubits, edges, n_layers)
    ns = noise_sites(n_qubits, edges, n_layers)
    acc = np.zeros(n_qubits)
    for _ in range(n_trajectories):
        errs = sample_errors(rng, ns, p)
        acc += np.asarray(node(theta, x, errs), dtype=float)
    est = acc / n_trajectories
    err = np.abs(est - exact)
    return {
        "exact": exact.tolist(),
        "trajectory_estimate": est.tolist(),
        "max_abs_error": float(err.max()),
        "mc_standard_error": float(1.0 / np.sqrt(n_trajectories)),
        "n_trajectories": n_trajectories,
        "within_3_sigma": bool(err.max() <= 3.0 / np.sqrt(n_trajectories)),
    }


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------

def _accuracy(pred, y) -> float:
    return float(np.mean(np.sign(np.asarray(pred)) == np.asarray(y)))


def train_run(
    n_qubits: int,
    edges: Sequence[Edge],
    n_layers: int,
    data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    noise: float = 0.0,
    epochs: int = 60,
    stepsize: float = 0.05,
    seed: int = 0,
    batch_size: Optional[int] = None,
    trajectories: int = 4,
    eval_trajectories: int = 8,
    eval_subsample: int = 256,
    readout: str = "all_z",
    scale: float = np.pi,
    target_loss: float = 0.5,
    verbose: bool = False,
) -> Dict:
    """One training run.  ``noise = 0`` is exact; ``noise > 0`` uses the Pauli unravelling.

    ``readout`` is ``"all_z"`` (a trained linear head over every ``<Z_j>``, the standard
    variational classifier), ``"z0"`` (the single cone-protected observable with a trained scale
    and bias) or ``"z0_nobias"`` (the same without the bias, so that no part of the loss
    reduction can come from fitting the class prior).  The training loss is recorded at every
    epoch so that epochs-to-target can be read off.
    """
    edges = [tuple(e) for e in edges]
    X_tr, Y_tr = data["train"]
    X_va, Y_va = data["val"]
    X_te, Y_te = data["test"]
    rng = np.random.default_rng(seed)

    theta = pnp.array(rng.uniform(0, 2 * np.pi, size=(n_layers, n_qubits, 3)), requires_grad=True)
    w = pnp.array(rng.normal(0, 0.1, size=n_qubits), requires_grad=True)
    c = pnp.array(0.0, requires_grad=True)

    if noise > 0:
        node = make_trajectory_node(n_qubits, edges, n_layers)
        ns = noise_sites(n_qubits, edges, n_layers)

        def zs(theta, x, k):
            out = None
            for _ in range(k):
                errs = sample_errors(rng, ns, noise)
                z = qml.math.stack(node(theta, x, errs))
                out = z if out is None else out + z
            return out / k
    else:
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, diff_method="backprop")
        def clean(theta, x):
            for q in range(n_qubits):
                qml.RY(x[..., q], wires=q)
            for layer in range(n_layers):
                for (j, k) in edges:
                    qml.CZ(wires=[j, k])
                for q in range(n_qubits):
                    qml.RX(theta[layer, q, 0], wires=q)
                    qml.RY(theta[layer, q, 1], wires=q)
                    qml.RZ(theta[layer, q, 2], wires=q)
            return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

        def zs(theta, x, k):
            return qml.math.stack(clean(theta, x))

    def predict(theta, w, c, x, k):
        z = zs(theta, x, k)                      # (n_qubits, batch)
        if readout == "all_z":
            return qml.math.sum(w[:, None] * z, axis=0) + c
        if readout == "z0_nobias":
            # No bias term: the model cannot lower the loss by fitting the class prior, so every
            # reduction has to come from the circuit.  This is the setting in which a claim about
            # the *circuit's* trainability under noise can be tested.
            return w[0] * z[0]
        return w[0] * z[0] + c

    def make_cost(xb, yb, k):
        def cost(theta, w, c):
            return qml.math.mean((predict(theta, w, c, xb, k) - yb) ** 2)
        return cost

    ev = np.arange(len(X_tr))
    if eval_subsample and len(X_tr) > eval_subsample:
        ev = np.random.default_rng(1234).choice(len(X_tr), size=eval_subsample, replace=False)
    xt_all = pnp.array(scale * X_tr[ev], requires_grad=False)
    yt_all = pnp.array(Y_tr[ev], requires_grad=False)
    xv = pnp.array(scale * X_va, requires_grad=False)
    xe = pnp.array(scale * X_te, requires_grad=False)

    opt = qml.AdamOptimizer(stepsize=stepsize)
    history: List[Dict] = []
    best = None
    t0 = time.time()
    m = len(X_tr)
    bs = batch_size or m
    keval = eval_trajectories if noise > 0 else 1
    ktrain = trajectories if noise > 0 else 1

    loss0 = float(make_cost(xt_all, yt_all, keval)(theta, w, c))
    for ep in range(epochs):
        order = rng.permutation(m)
        for s in range(0, m, bs):
            idx = order[s:s + bs]
            xb = pnp.array(scale * X_tr[idx], requires_grad=False)
            yb = pnp.array(Y_tr[idx], requires_grad=False)
            theta, w, c = opt.step(make_cost(xb, yb, ktrain), theta, w, c)
        loss = float(make_cost(xt_all, yt_all, keval)(theta, w, c))
        acc_va = _accuracy(np.asarray(predict(theta, w, c, xv, keval)), Y_va)
        history.append({"epoch": ep + 1, "train_loss": loss, "val_accuracy": acc_va})
        if best is None or acc_va > best["val_accuracy"]:
            best = {"epoch": ep + 1, "val_accuracy": acc_va,
                    "theta": np.array(theta), "w": np.array(w), "c": float(c)}
        if verbose:
            print(f"    ep {ep + 1:3d} loss {loss:.4f} val {acc_va:.3f}", flush=True)

    losses = np.array([h["train_loss"] for h in history])
    reach = np.where(losses <= target_loss)[0]
    epochs_to_target = int(reach[0] + 1) if reach.size else None

    def first_epoch_below(level: float):
        w = np.where(losses <= level)[0]
        return int(w[0] + 1) if w.size else None

    fractions = (0.9, 0.8, 0.7, 0.6, 0.5)
    epochs_to_fraction = {str(f): first_epoch_below(f * loss0) for f in fractions}

    th = pnp.array(best["theta"], requires_grad=False)
    wb = pnp.array(best["w"], requires_grad=False)
    cb = pnp.array(best["c"], requires_grad=False)
    acc_te = _accuracy(np.asarray(predict(th, wb, cb, xe, keval)), Y_te)
    maj = float(max(np.mean(Y_te == 1), np.mean(Y_te == -1)))
    final = float(losses[-1])
    return {
        "n_edges": len(edges),
        "edges": [list(e) for e in edges],
        "seed": seed,
        "noise": noise,
        "readout": readout,
        "epochs": epochs,
        "initial_loss": loss0,
        "final_train_loss": final,
        "loss_reduction_percent": 100.0 * (loss0 - final) / loss0 if loss0 > 0 else 0.0,
        "epochs_to_target": epochs_to_target,
        "target_loss": target_loss,
        "epochs_to_fraction_of_initial": epochs_to_fraction,
        "best_val_accuracy": float(best["val_accuracy"]),
        "test_accuracy": acc_te,
        "test_majority_rate": maj,
        "history": history,
        "seconds": time.time() - t0,
    }


def _job(a):
    (n, edges, L, data, noise, epochs, step, seed, bs, ktr, kev, readout, target) = a
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return train_run(n, edges, L, data, noise=noise, epochs=epochs, stepsize=step, seed=seed,
                     batch_size=bs, trajectories=ktr, eval_trajectories=kev, readout=readout,
                     target_loss=target)


def convergence_experiment(
    family: Dict[str, Dict],
    n_qubits: int = 8,
    n_layers: int = 4,
    dataset: str = "breastmnist",
    noise: float = 0.0,
    epochs: int = 60,
    seeds: Sequence[int] = (0, 1, 2),
    stepsize: float = 0.05,
    batch_size: Optional[int] = None,
    trajectories: int = 4,
    eval_trajectories: int = 8,
    readout: str = "all_z",
    target_loss: float = 0.5,
    jobs: int = 4,
) -> Dict:
    """Experiment C4 over a whole family of topologies, seeds run in parallel."""
    splits = load_splits(n_qubits, dataset=dataset)
    data = {k: splits[k] for k in ("train", "val", "test")}

    jobs_list = []
    keys = []
    for name, d in family.items():
        for s in seeds:
            jobs_list.append((n_qubits, [tuple(e) for e in d["edges"]], n_layers, data, noise,
                              epochs, stepsize, s, batch_size, trajectories, eval_trajectories,
                              readout, target_loss))
            keys.append(name)

    t0 = time.time()
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_job, jobs_list))
    else:
        results = [_job(a) for a in jobs_list]

    rows = []
    for name, d in family.items():
        runs = [r for k, r in zip(keys, results) if k == name]
        acc = np.array([r["test_accuracy"] for r in runs])
        red = np.array([r["loss_reduction_percent"] for r in runs])
        ett = [r["epochs_to_target"] for r in runs]
        finished = [e for e in ett if e is not None]
        frac_keys = list(runs[0]["epochs_to_fraction_of_initial"].keys())
        mean_frac = {}
        for f in frac_keys:
            vals = [r["epochs_to_fraction_of_initial"][f] for r in runs]
            ok = [v for v in vals if v is not None]
            mean_frac[f] = {"epochs": vals,
                            "mean": float(np.mean(ok)) if len(ok) == len(vals) else None,
                            "runs_reaching": len(ok)}
        rows.append({
            "topology": name,
            "n_edges": len(d["edges"]),
            "cone_size": d.get("cone_size"),
            "cone_dla_dimension": cone_dla_dimension(n_qubits, [tuple(e) for e in d["edges"]],
                                                     [0], n_layers),
            "global_dla_dimension": global_dla_dimension(n_qubits,
                                                         [tuple(e) for e in d["edges"]]),
            "mean_initial_loss": float(np.mean([r["initial_loss"] for r in runs])),
            "mean_final_loss": float(np.mean([r["final_train_loss"] for r in runs])),
            "mean_loss_reduction_percent": float(red.mean()),
            "epochs_to_target": ett,
            "mean_epochs_to_target": float(np.mean(finished)) if finished else None,
            "runs_reaching_target": len(finished),
            "epochs_to_fraction_of_initial": mean_frac,
            "test_accuracy_mean": float(acc.mean()),
            "test_accuracy_std": float(acc.std(ddof=1)) if acc.size > 1 else 0.0,
            "test_majority_rate": runs[0]["test_majority_rate"],
            "runs": runs,
        })
        print(f"[C4] {name:34s} loss {rows[-1]['mean_final_loss']:.4f} "
              f"epochs->{target_loss}: {rows[-1]['mean_epochs_to_target']} "
              f"acc {acc.mean():.3f}", flush=True)

    dense = next((r for r in rows if r["topology"].startswith("HEA")), None)
    for r in rows:
        if dense and r["mean_epochs_to_target"] and dense["mean_epochs_to_target"]:
            r["speedup_epochs_vs_hea"] = (dense["mean_epochs_to_target"]
                                          / r["mean_epochs_to_target"])
        else:
            r["speedup_epochs_vs_hea"] = None
        if dense and dense["mean_loss_reduction_percent"] > 0:
            r["first_draft_reduction_ratio"] = (r["mean_loss_reduction_percent"]
                                                / dense["mean_loss_reduction_percent"])
        speed = {}
        for f, v in r["epochs_to_fraction_of_initial"].items():
            dv = dense["epochs_to_fraction_of_initial"][f] if dense else None
            speed[f] = (dv["mean"] / v["mean"]
                        if dv and dv["mean"] and v["mean"] else None)
        r["speedup_by_fraction_vs_hea"] = speed
    return {
        "experiment": "C4_convergence",
        "dataset": splits.get("source", dataset),
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "noise": noise,
        "readout": readout,
        "epochs": epochs,
        "seeds": list(seeds),
        "batch_size": batch_size,
        "trajectories": trajectories if noise > 0 else None,
        "target_loss": target_loss,
        "rows": rows,
        "seconds": time.time() - t0,
    }
