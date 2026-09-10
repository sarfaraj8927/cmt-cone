"""The four experiments, run under the corrected protocol.

Every function here is a drop-in replacement for one of the notebook cells; the docstring of each
lists what changed and which Lean theorem the change is needed for.

  * :func:`experiment_topology`           <-> `cmt-barren` cell 1 (Table 1)
  * :func:`experiment_gradient_variance`  <-> `cmt-barren` cell 2, `asymptotic-variance-8-qubits`
  * :func:`experiment_noise_sweep`        <-> `critical-noise-threshold-8-qubits`
  * :func:`experiment_training`           <-> `cmt-barren` cells 3-4, `sota-1-1`, `sota-2` (Table 3)
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .ansatz import make_qnode, matched_noise_plan, param_shape
from .data import load_splits
from .dla import dla_report
from .stats import (
    componentwise_variance,
    mean_confidence_interval,
    paired_variance_ratio_bootstrap,
    sample_variance,
    samples_for_confidence,
)
from .topology import (
    capacitated_topology,
    complete_edges,
    cooccurrence_matrix_notebook,
    describe_topology,
    pearson_matrix,
    threshold_topology,
)

__all__ = [
    "experiment_topology",
    "experiment_gradient_variance",
    "experiment_noise_sweep",
    "experiment_training",
]

Edge = Tuple[int, int]


# ----------------------------------------------------------------------------------------------
# Experiment 1: topology extraction
# ----------------------------------------------------------------------------------------------

def experiment_topology(
    n_qubits: int = 16,
    budget: int = 2,
    percentile: float = 80.0,
    n_train: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """Extract the entangling topology from the *training* split only.

    Changes from the notebook cell:

    * the statistic is the centred Pearson correlation, not the uncentred sigmoid co-occurrence
      (Lean: ``CMT.Corr.cov_eq_cooc_sub_mean_mul``, ``CMT.Corr.cooc_ranking_differs_from_cov``);
    * the extractor is capacity-constrained, so the cluster size is bounded by construction
      (Lean: ``CMT.Cluster.cluster_sizeLe``) and the DLA bound of the corrected Lemma 4.1 applies;
    * the report includes the cluster sizes, the proved DLA bound ``2*K*4**b`` and the *exact* DLA
      dimension -- the numbers that decide whether the sparsification removes algebra dimension at
      all.  The notebooks report only the edge count and the sparsity percentage, which say
      nothing about the DLA;
    * the plain thresholded topology is extracted from the same matrix and reported alongside, so
      the difference is visible.
    """
    splits = load_splits(n_qubits, limits=(n_train, None, None))
    X_train, _ = splits["train"]  # type: ignore[misc]

    corr = pearson_matrix(X_train)
    cooc = cooccurrence_matrix_notebook(X_train)

    edges_cap, labels, tau = capacitated_topology(corr, budget=budget, percentile=percentile)
    edges_thr = threshold_topology(corr, percentile=percentile)
    edges_nb = threshold_topology(cooc, percentile=percentile)

    out = {
        "data_source": splits["source"],
        "n_train": int(X_train.shape[0]),
        "threshold_tau": tau,
        "capacitated": describe_topology(n_qubits, edges_cap, labels),
        "thresholded_pearson": describe_topology(n_qubits, edges_thr),
        "thresholded_cooccurrence_notebook": describe_topology(n_qubits, edges_nb),
    }
    # The exact DLA dimension is only computable for moderate n; the bound always is.
    if n_qubits <= 12:
        out["capacitated"]["dla_exact"] = dla_report(n_qubits, edges_cap)
        out["thresholded_pearson"]["dla_exact"] = dla_report(n_qubits, edges_thr)

    if verbose:
        print(f"[exp1] data source: {out['data_source']}, {out['n_train']} training samples")
        print(f"[exp1] threshold tau = {tau:.4f}")
        for key in ("capacitated", "thresholded_pearson", "thresholded_cooccurrence_notebook"):
            d = out[key]
            print(
                f"[exp1] {key:36s} edges {d['n_edges']:3d}/{d['n_edges_dense']:3d}"
                f"  sparsity {100 * d['sparsity']:5.1f}%"
                f"  clusters {d['n_clusters']:2d} (max {d['max_cluster_size']})"
                f"  DLA <= {d['dla_upper_bound']}  vs dense >= {d['dense_dla_lower_bound']}"
            )
            if "dla_exact" in d:
                e = d["dla_exact"]
                print(
                    f"[exp1] {'':36s} exact DLA dim = {e['dla_dimension_exact']}"
                    f"  (bound holds: {e['bound_holds']}, reduces DLA: {e['reduces_dla']})"
                )
    return out


# ----------------------------------------------------------------------------------------------
# Experiment 2: gradient variance
# ----------------------------------------------------------------------------------------------

def experiment_gradient_variance(
    n_qubits: int,
    edges_sparse: Sequence[Edge],
    n_layers: int = 2,
    noise: float = 0.05,
    n_samples: int = 200,
    seed: int = 0,
    match_noise: bool = True,
    diff_method: str = "backprop",
    encoding_angle: float = np.pi / 4,
    verbose: bool = True,
) -> Dict:
    """Paired gradient-variance benchmark.

    Changes from the notebook cells:

    * the sparse topology is supplied by the caller and should be the *data-derived* one; the
      notebooks use ``hea_edges[:20%]``, the lexicographic prefix of the complete edge list, which
      is a star at qubit 0 and, from n = 10 on, connected -- so its DLA is the full ``4**n - 1``
      (Lean: ``CMT.Impl.cmtEdges_eight``, ``CMT.Impl.finrank_cmtDLA_impl_ge``);
    * the two circuits get the same number of depolarizing channels per layer when
      ``match_noise=True`` (Lean: ``CMT.NoiseBudget.padded_channels_eq``); with
      ``match_noise=False`` the notebook comparison is reproduced, which measures the gate-count
      mechanism of the corrected Lemma 4.2 instead;
    * the variance is averaged over **all** ``3*n*L`` parameters, not taken at ``grad[0,0,0]``;
    * both circuits are evaluated at the **same** parameter draws (paired design) from a fixed
      seed, and the ratio carries a bootstrap 95 % confidence interval
      (Lean: ``CMT.Estimator.sampleVar_unbiased``, ``CMT.Estimator.mean_concentration``);
    * ``ddof=1`` throughout (``np.var`` is biased low by ``(N-1)/N``).
    """
    edges_dense = complete_edges(n_qubits)
    plan = matched_noise_plan(n_qubits, len(edges_dense), len(edges_sparse))
    pad = plan["padding_channels_per_layer"] if match_noise else 0

    dense = make_qnode(n_qubits, edges_dense, n_layers, noise, 0, "z0", diff_method)
    sparse = make_qnode(n_qubits, edges_sparse, n_layers, noise, pad, "z0", diff_method)

    rng = np.random.default_rng(seed)
    x = pnp.array(np.full(n_qubits, encoding_angle), requires_grad=False)

    shape = param_shape(n_qubits, n_layers)
    gd, gs = [], []
    t0 = time.time()
    for i in range(n_samples):
        theta = pnp.array(rng.uniform(0, 2 * np.pi, size=shape), requires_grad=True)
        gd.append(np.asarray(qml.grad(dense)(theta, x)))
        gs.append(np.asarray(qml.grad(sparse)(theta, x)))
        if verbose and (i + 1) % max(1, n_samples // 5) == 0:
            print(f"[exp2] n={n_qubits} sample {i + 1}/{n_samples} ({time.time() - t0:.1f}s)")

    gd_arr, gs_arr = np.array(gd), np.array(gs)
    var_dense, per_dense = componentwise_variance(gd_arr)
    var_sparse, per_sparse = componentwise_variance(gs_arr)
    boot = paired_variance_ratio_bootstrap(gs_arr, gd_arr, seed=seed)

    # How many draws would be needed to resolve a 20 % effect on the ratio, at 95 % confidence.
    rel = sample_variance(per_sparse / max(per_dense.mean(), 1e-300)) if per_dense.size > 1 else 0.0

    out = {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "noise": noise,
        "n_samples": n_samples,
        "seed": seed,
        "match_noise": match_noise,
        "noise_plan": plan,
        "n_edges_sparse": len(edges_sparse),
        "n_edges_dense": len(edges_dense),
        "variance_dense": var_dense,
        "variance_sparse": var_sparse,
        "ratio": boot["ratio"],
        "ratio_ci95": (boot["ci_low"], boot["ci_high"]),
        "ratio_significant": boot["significant"],
        "mean_gradient_dense": float(gd_arr.mean()),
        "mean_gradient_sparse": float(gs_arr.mean()),
        "single_component_variance_dense": sample_variance(gd_arr[:, 0, 0, 0]),
        "single_component_variance_sparse": sample_variance(gs_arr[:, 0, 0, 0]),
        "component_spread_relative_variance": float(rel),
        "seconds": time.time() - t0,
    }
    if verbose:
        print(
            f"[exp2] n={n_qubits} p={noise} matched={match_noise}: "
            f"Var_dense={var_dense:.3e}  Var_sparse={var_sparse:.3e}  "
            f"ratio={out['ratio']:.3f} CI95=({boot['ci_low']:.3f}, {boot['ci_high']:.3f})"
            f"  {'significant' if boot['significant'] else 'NOT significant'}"
        )
    return out


# ----------------------------------------------------------------------------------------------
# Experiment 3: noise sweep
# ----------------------------------------------------------------------------------------------

def experiment_noise_sweep(
    n_qubits: int,
    edges_sparse: Sequence[Edge],
    noise_values: Sequence[float] = (0.0, 0.01, 0.03, 0.05, 0.07, 0.10),
    n_layers: int = 2,
    n_samples: int = 100,
    seed: int = 0,
    match_noise: bool = True,
    diff_method: str = "backprop",
    verbose: bool = True,
) -> List[Dict]:
    """The noise sweep, with a confidence interval at every noise level.

    The archived sweep is non-monotone in ``p`` and has no error bars, so the ordering of its
    points is not resolvable; with intervals, a claim of a critical threshold can be checked
    (it requires the interval to leave 1 at some ``p`` and not before).  A separate seed per noise
    level keeps the levels independent while the two circuits stay paired within a level.
    """
    rows = []
    for i, p in enumerate(noise_values):
        rows.append(
            experiment_gradient_variance(
                n_qubits,
                edges_sparse,
                n_layers=n_layers,
                noise=p,
                n_samples=n_samples,
                seed=seed + i,
                match_noise=match_noise,
                diff_method=diff_method,
                verbose=False,
            )
        )
        if verbose:
            r = rows[-1]
            print(
                f"[exp3] p={p:5.3f}  Var_dense={r['variance_dense']:.3e}  "
                f"Var_sparse={r['variance_sparse']:.3e}  ratio={r['ratio']:.3f} "
                f"CI95=({r['ratio_ci95'][0]:.3f}, {r['ratio_ci95'][1]:.3f})"
            )
    return rows


# ----------------------------------------------------------------------------------------------
# Experiment 4: training
# ----------------------------------------------------------------------------------------------

def _make_loss(circuit, X, Y):
    Xc = pnp.array(X, requires_grad=False)
    Yc = pnp.array(Y, requires_grad=False)

    def loss(theta):
        preds = pnp.stack([pnp.mean(pnp.stack(circuit(theta, x))) for x in Xc])
        return pnp.mean((preds - Yc) ** 2)

    def predict(theta):
        return np.array([float(np.mean(np.asarray(circuit(theta, x)))) for x in Xc])

    return loss, predict


def _init_params(shape, rng, scheme: str):
    if scheme == "uniform":
        return pnp.array(rng.uniform(-np.pi, np.pi, size=shape), requires_grad=True)
    if scheme == "beta":  # the BEINIT-style initialisation of the notebooks
        return pnp.array(rng.beta(2.0, 2.0, size=shape) * 2 * np.pi - np.pi, requires_grad=True)
    raise ValueError("scheme must be 'uniform' or 'beta'")


def experiment_training(
    n_qubits: int,
    topologies: Dict[str, Sequence[Edge]],
    n_layers: int = 2,
    noise: float = 0.05,
    epochs: int = 30,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    optimizer: str = "adam",
    stepsize: float = 0.05,
    init_scheme: str = "uniform",
    match_noise: bool = True,
    n_train: Optional[int] = 60,
    n_eval: Optional[int] = 60,
    diff_method: str = "backprop",
    target_loss: float = 0.5,
    verbose: bool = True,
) -> Dict:
    """Controlled convergence comparison.

    Changes from the notebook cells:

    * one **shared** train / validation / test split, from the official MedMNIST splits, and the
      topology is extracted from the training split only -- the notebooks train on 10-20 images
      with no held-out data and fit the topology on them;
    * **one** optimizer, learning rate, epoch budget and initialisation scheme for every method;
      the manuscript's Table 3 states Adam at eta = 0.05 on 16 qubits while the code runs plain
      SGD at eta = 0.15 on 4 qubits;
    * ``len(seeds) >= 5`` seeds, and the *same* seed gives the *same* initial parameters to every
      method (paired design);
    * the reported quantity is the final **test** loss and accuracy, mean +/- std over seeds, plus
      epochs-to-target-loss if a speed claim is made -- not the quotient of two percentage loss
      reductions from different starting points
      (Lean: ``CMT.Impl.table3_speedup_is_ratio_of_reductions``);
    * the noise budget is matched across topologies exactly as in Experiment 2.
    """
    splits = load_splits(n_qubits, limits=(n_train, n_eval, n_eval))
    X_tr, Y_tr = splits["train"]  # type: ignore[misc]
    X_va, Y_va = splits["val"]  # type: ignore[misc]
    X_te, Y_te = splits["test"]  # type: ignore[misc]

    max_edges = max(len(e) for e in topologies.values())
    shape = param_shape(n_qubits, n_layers)
    results: Dict[str, Dict] = {}

    for name, edges in topologies.items():
        pad = 2 * (max_edges - len(edges)) if match_noise else 0
        circuit = make_qnode(n_qubits, edges, n_layers, noise, pad, "all_z", diff_method)
        loss_tr, _ = _make_loss(circuit, X_tr, Y_tr)
        loss_va, _ = _make_loss(circuit, X_va, Y_va)
        loss_te, pred_te = _make_loss(circuit, X_te, Y_te)

        per_seed = []
        for s in seeds:
            rng = np.random.default_rng(s)
            theta = _init_params(shape, rng, init_scheme)
            opt = (
                qml.AdamOptimizer(stepsize=stepsize)
                if optimizer == "adam"
                else qml.GradientDescentOptimizer(stepsize=stepsize)
            )
            history = []
            epochs_to_target = None
            for e in range(epochs):
                theta, cost = opt.step_and_cost(loss_tr, theta)
                history.append(float(cost))
                if epochs_to_target is None and float(cost) <= target_loss:
                    epochs_to_target = e + 1
            preds = pred_te(theta)
            acc = float(np.mean(np.sign(preds) == np.sign(Y_te)))
            per_seed.append(
                {
                    "seed": int(s),
                    "train_loss": history,
                    "final_train_loss": history[-1],
                    "final_val_loss": float(loss_va(theta)),
                    "final_test_loss": float(loss_te(theta)),
                    "test_accuracy": acc,
                    "epochs_to_target": epochs_to_target,
                }
            )
            if verbose:
                print(
                    f"[exp4] {name:10s} seed {s}: train {history[-1]:.4f} "
                    f"val {per_seed[-1]['final_val_loss']:.4f} "
                    f"test {per_seed[-1]['final_test_loss']:.4f} acc {acc:.3f}"
                )

        te = [r["final_test_loss"] for r in per_seed]
        ac = [r["test_accuracy"] for r in per_seed]
        results[name] = {
            "n_edges": len(edges),
            "pad_channels": pad,
            "per_seed": per_seed,
            "test_loss_mean": float(np.mean(te)),
            "test_loss_std": float(np.std(te, ddof=1)) if len(te) > 1 else 0.0,
            "test_loss_ci": mean_confidence_interval(te) if len(te) > 1 else None,
            "test_accuracy_mean": float(np.mean(ac)),
            "test_accuracy_std": float(np.std(ac, ddof=1)) if len(ac) > 1 else 0.0,
            "epochs_to_target": [r["epochs_to_target"] for r in per_seed],
        }

    if verbose:
        print("\n[exp4] final test loss (mean +/- std over seeds), lower is better:")
        for name, r in sorted(results.items(), key=lambda kv: kv[1]["test_loss_mean"]):
            print(
                f"[exp4]   {name:10s} {r['test_loss_mean']:.4f} +/- {r['test_loss_std']:.4f}"
                f"   acc {r['test_accuracy_mean']:.3f} +/- {r['test_accuracy_std']:.3f}"
                f"   ({r['n_edges']} edges)"
            )

    return {
        "data_source": splits["source"],
        "protocol": {
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "noise": noise,
            "epochs": epochs,
            "seeds": list(seeds),
            "optimizer": optimizer,
            "stepsize": stepsize,
            "init_scheme": init_scheme,
            "match_noise": match_noise,
            "n_train": int(X_tr.shape[0]),
            "n_val": int(X_va.shape[0]),
            "n_test": int(X_te.shape[0]),
        },
        "results": results,
    }


def sample_size_advice(pilot_variance: float, tolerance: float = 0.2, delta: float = 0.05) -> int:
    """Draws needed to resolve an effect of relative size ``tolerance`` at confidence ``1-delta``.

    Lean: ``CMT.Estimator.samples_for_confidence``.
    """
    return samples_for_confidence(pilot_variance, tolerance, delta)
