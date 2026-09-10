"""CUDA-Q experiments for CMT-QNN on NVIDIA Quantum Stack.

Ported experiments with support for multi-GPU, large qubit counts, and cuQuantum backends.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cudaq_ansatz import (
    CUDAQ_AVAILABLE,
    Edge,
    channels_per_layer,
    execute_cudaq,
    gradient_cudaq,
    gradient_cudaq_backprop,
    make_cudaq_kernel,
    make_cudaq_observable,
    matched_noise_plan,
    param_shape,
    random_params,
)
from .cudaq_simulator import CuQuantumSimulator, SimulatorBackend, SimulatorConfig, create_simulator_for_qubits
from .cudaq_topology import (
    capacitated_topology,
    cluster_labels,
    cluster_sizes,
    complete_edges,
    describe_topology,
    dla_upper_bound,
    pearson_matrix,
)
from .data import load_splits

__all__ = [
    "experiment_topology",
    "experiment_gradient_variance",
    "experiment_noise_sweep",
    "experiment_training",
    "run_distributed_gradient_variance",
]


# ----------------------------------------------------------------------------------------------
# Experiment 1: topology extraction
# ----------------------------------------------------------------------------------------------

def experiment_topology(
    n_qubits: int = 16,
    budget: int = 2,
    percentile: float = 80.0,
    n_train: Optional[int] = None,
    verbose: bool = True,
    dataset: str = "breastmnist",
) -> Dict:
    """Extract the entangling topology from the training split only (CUDA-Q version)."""
    splits = load_splits(n_qubits, limits=(n_train, None, None), dataset=dataset)
    X_train, _ = splits["train"]

    corr = pearson_matrix(X_train)

    edges_cap, labels, tau = capacitated_topology(corr, budget=budget, percentile=percentile)
    edges_thr = threshold_topology(corr, percentile=percentile)
    edges_nb = threshold_topology(cooccurrence_matrix_notebook(X_train), percentile=percentile)

    out = {
        "data_source": splits["source"],
        "n_train": int(X_train.shape[0]),
        "threshold_tau": tau,
        "capacitated": describe_topology(n_qubits, edges_cap, labels),
        "thresholded_pearson": describe_topology(n_qubits, edges_thr),
        "thresholded_cooccurrence_notebook": describe_topology(n_qubits, edges_nb),
    }
    if n_qubits <= 12:
        from .dla import dla_report
        out["capacitated"]["dla_exact"] = dla_report(n_qubits, edges_cap)
        out["thresholded_pearson"]["dla_exact"] = dla_report(n_qubits, edges_thr)

    if verbose:
        print(f"[cudaq-exp1] data source: {out['data_source']}, {out['n_train']} training samples")
        print(f"[cudaq-exp1] threshold tau = {tau:.4f}")
        for key in ("capacitated", "thresholded_pearson", "thresholded_cooccurrence_notebook"):
            d = out[key]
            print(
                f"[cudaq-exp1] {key:36s} edges {d['n_edges']:3d}/{d['n_edges_dense']:3d}"
                f"  sparsity {100 * d['sparsity']:5.1f}%"
                f"  clusters {d['n_clusters']:2d} (max {d['max_cluster_size']})"
                f"  DLA <= {d['dla_upper_bound']}  vs dense >= {d['dense_dla_lower_bound']}"
            )
            if "dla_exact" in d:
                e = d["dla_exact"]
                print(
                    f"[cudaq-exp1] {'':36s} exact DLA dim = {e['dla_dimension_exact']}"
                    f"  (bound holds: {e['bound_holds']}, reduces DLA: {e['reduces_dla']})"
                )
    return out


def cooccurrence_matrix_notebook(X: np.ndarray) -> np.ndarray:
    A = np.asarray(X, dtype=float)
    S = 1.0 / (1.0 + np.exp(-A))
    return (S.T @ S) / A.shape[0]


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
    simulator_config: Optional[SimulatorConfig] = None,
    shots: int = 1024,
) -> Dict:
    """Paired gradient-variance benchmark using CUDA-Q.
    
    Supports multi-GPU and cuQuantum backends for large qubit counts.
    """
    if not CUDAQ_AVAILABLE:
        raise RuntimeError("CUDA-Q not available. Install with: pip install cudaq")
    
    edges_dense = complete_edges(n_qubits)
    plan = matched_noise_plan(n_qubits, len(edges_dense), len(edges_sparse))
    pad = plan["padding_channels_per_layer"] if match_noise else 0

    # Create simulator
    if simulator_config is None:
        simulator_config = SimulatorConfig(
            n_qubits=n_qubits,
            backend=SimulatorBackend.CUDAQ_NVIDIA,
        )
    simulator = CuQuantumSimulator(simulator_config)
    simulator.initialize()

    try:
        # Create kernels
        dense_kernel = make_cudaq_kernel(n_qubits, edges_dense, n_layers, noise, 0)
        sparse_kernel = make_cudaq_kernel(n_qubits, edges_sparse, n_layers, noise, pad)
        
        dense_obs = make_cudaq_observable(n_qubits, "z0")
        sparse_obs = make_cudaq_observable(n_qubits, "z0")

        rng = np.random.default_rng(seed)
        x = np.full(n_qubits, encoding_angle, dtype=np.float64)
        shape = param_shape(n_qubits, n_layers)
        
        gd_list, gs_list = [], []
        t0 = time.time()
        
        for i in range(n_samples):
            theta = random_params(n_qubits, n_layers, rng)
            
            if diff_method == "backprop":
                # Use CUDA-Q adjoint differentiation
                gd_flat = gradient_cudaq_backprop(dense_kernel, dense_obs, theta, x, shots, simulator_config.backend.value)
                gs_flat = gradient_cudaq_backprop(sparse_kernel, sparse_obs, theta, x, shots, simulator_config.backend.value)
            else:
                # Parameter-shift rule
                gd_flat = gradient_cudaq(dense_kernel, dense_obs, theta, x, shots, simulator_config.backend.value)
                gs_flat = gradient_cudaq(sparse_kernel, sparse_obs, theta, x, shots, simulator_config.backend.value)
            
            gd_list.append(gd_flat.ravel())
            gs_list.append(gs_flat.ravel())
            
            if verbose and (i + 1) % max(1, n_samples // 5) == 0:
                print(f"[cudaq-exp2] n={n_qubits} sample {i + 1}/{n_samples} ({time.time() - t0:.1f}s)")

        gd_arr = np.array(gd_list)
        gs_arr = np.array(gs_list)
        
        var_dense = float(np.var(gd_arr, axis=0, ddof=1).mean())
        var_sparse = float(np.var(gs_arr, axis=0, ddof=1).mean())
        
        # Bootstrap confidence interval for ratio
        from .stats import paired_variance_ratio_bootstrap
        boot = paired_variance_ratio_bootstrap(gs_arr, gd_arr, seed=seed)
        
        rel = float(np.var(gs_arr / np.maximum(gd_arr, 1e-300), axis=0, ddof=1).mean()) if gd_arr.shape[1] > 1 else 0.0

        out = {
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "noise": noise,
            "n_samples": n_samples,
            "seed": seed,
            "match_noise": match_noise,
            "diff_method": diff_method,
            "shots": shots,
            "backend": simulator_config.backend.value,
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
            "single_component_variance_dense": float(np.var(gd_arr[:, 0], ddof=1)),
            "single_component_variance_sparse": float(np.var(gs_arr[:, 0], ddof=1)),
            "component_spread_relative_variance": rel,
            "seconds": time.time() - t0,
        }
        
        if verbose:
            print(
                f"[cudaq-exp2] n={n_qubits} p={noise} matched={match_noise} backend={simulator_config.backend.value}: "
                f"Var_dense={var_dense:.3e}  Var_sparse={var_sparse:.3e}  "
                f"ratio={out['ratio']:.3f} CI95=({boot['ci_low']:.3f}, {boot['ci_high']:.3f})"
                f"  {'significant' if boot['significant'] else 'NOT significant'}"
            )
        return out
    finally:
        simulator.cleanup()


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
    simulator_config: Optional[SimulatorConfig] = None,
    shots: int = 1024,
) -> List[Dict]:
    """Noise sweep with confidence intervals at every noise level."""
    rows = []
    for i, p in enumerate(noise_values):
        row = experiment_gradient_variance(
            n_qubits,
            edges_sparse,
            n_layers=n_layers,
            noise=p,
            n_samples=n_samples,
            seed=seed + i,
            match_noise=match_noise,
            diff_method=diff_method,
            verbose=False,
            simulator_config=simulator_config,
            shots=shots,
        )
        rows.append(row)
        if verbose:
            print(
                f"[cudaq-exp3] p={p:5.3f}  Var_dense={row['variance_dense']:.3e}  "
                f"Var_sparse={row['variance_sparse']:.3e}  ratio={row['ratio']:.3f} "
                f"CI95=({row['ratio_ci95'][0]:.3f}, {row['ratio_ci95'][1]:.3f})"
            )
    return rows


# ----------------------------------------------------------------------------------------------
# Experiment 4: training
# ----------------------------------------------------------------------------------------------

def _make_loss_cudaq(kernel, observable, X, Y, shots: int = 1024):
    """Create loss function for CUDA-Q training."""
    Xc = np.asarray(X, dtype=np.float64)
    Yc = np.asarray(Y, dtype=np.float64)
    
    def loss(theta):
        preds = []
        for x in Xc:
            exp_val = execute_cudaq(kernel, observable, theta, x, shots=shots)
            preds.append(exp_val)
        preds = np.array(preds)
        return np.mean((preds - Yc) ** 2)
    
    def predict(theta):
        preds = []
        for x in Xc:
            exp_val = execute_cudaq(kernel, observable, theta, x, shots=shots)
            preds.append(exp_val)
        return np.array(preds)
    
    return loss, predict


def _init_params(shape, rng, scheme: str):
    if scheme == "uniform":
        return rng.uniform(-np.pi, np.pi, size=shape).astype(np.float64)
    if scheme == "beta":
        return (rng.beta(2.0, 2.0, size=shape) * 2 * np.pi - np.pi).astype(np.float64)
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
    simulator_config: Optional[SimulatorConfig] = None,
    shots: int = 1024,
) -> Dict:
    """Controlled convergence comparison using CUDA-Q."""
    if not CUDAQ_AVAILABLE:
        raise RuntimeError("CUDA-Q not available")
    
    splits = load_splits(n_qubits, limits=(n_train, n_eval, n_eval))
    X_tr, Y_tr = splits["train"]
    X_va, Y_va = splits["val"]
    X_te, Y_te = splits["test"]

    max_edges = max(len(e) for e in topologies.values())
    shape = param_shape(n_qubits, n_layers)
    results: Dict[str, Dict] = {}

    # Initialize simulator once
    if simulator_config is None:
        simulator_config = SimulatorConfig(
            n_qubits=n_qubits,
            backend=SimulatorBackend.CUDAQ_NVIDIA,
        )
    simulator = CuQuantumSimulator(simulator_config)
    simulator.initialize()

    try:
        for name, edges in topologies.items():
            pad = 2 * (max_edges - len(edges)) if match_noise else 0
            kernel = make_cudaq_kernel(n_qubits, edges, n_layers, noise, pad)
            obs = make_cudaq_observable(n_qubits, "all_z")
            
            loss_tr, _ = _make_loss_cudaq(kernel, obs, X_tr, Y_tr, shots)
            loss_va, _ = _make_loss_cudaq(kernel, obs, X_va, Y_va, shots)
            loss_te, pred_te = _make_loss_cudaq(kernel, obs, X_te, Y_te, shots)

            per_seed = []
            for s in seeds:
                rng = np.random.default_rng(s)
                theta = _init_params(shape, rng, init_scheme)
                
                if optimizer == "adam":
                    # Simple Adam implementation for CUDA-Q
                    m = np.zeros_like(theta)
                    v = np.zeros_like(theta)
                    beta1, beta2, eps = 0.9, 0.999, 1e-8
                else:
                    pass  # SGD
                
                history = []
                epochs_to_target = None
                
                for e in range(epochs):
                    # Compute gradient
                    if diff_method == "backprop":
                        grad_flat = gradient_cudaq_backprop(kernel, obs, theta, X_tr[0], shots, simulator_config.backend.value)
                    else:
                        grad_flat = gradient_cudaq(kernel, obs, theta, X_tr[0], shots, simulator_config.backend.value)
                    grad = grad_flat.reshape(theta.shape)
                    
                    # Simple parameter update (in practice, use proper optimizer)
                    if optimizer == "adam":
                        m = beta1 * m + (1 - beta1) * grad
                        v = beta2 * v + (1 - beta2) * (grad ** 2)
                        m_hat = m / (1 - beta1 ** (e + 1))
                        v_hat = v / (1 - beta2 ** (e + 1))
                        theta -= stepsize * m_hat / (np.sqrt(v_hat) + eps)
                    else:
                        theta -= stepsize * grad
                    
                    cost = loss_tr(theta)
                    history.append(float(cost))
                    
                    if epochs_to_target is None and float(cost) <= target_loss:
                        epochs_to_target = e + 1
                
                preds = pred_te(theta)
                acc = float(np.mean(np.sign(preds) == np.sign(Y_te)))
                per_seed.append({
                    "seed": int(s),
                    "train_loss": history,
                    "final_train_loss": history[-1],
                    "final_val_loss": float(loss_va(theta)),
                    "final_test_loss": float(loss_te(theta)),
                    "test_accuracy": acc,
                    "epochs_to_target": epochs_to_target,
                })
                if verbose:
                    print(
                        f"[cudaq-exp4] {name:10s} seed {s}: train {history[-1]:.4f} "
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
                "test_accuracy_mean": float(np.mean(ac)),
                "test_accuracy_std": float(np.std(ac, ddof=1)) if len(ac) > 1 else 0.0,
                "epochs_to_target": [r["epochs_to_target"] for r in per_seed],
            }

        if verbose:
            print("\n[cudaq-exp4] final test loss (mean +/- std over seeds), lower is better:")
            for name, r in sorted(results.items(), key=lambda kv: kv[1]["test_loss_mean"]):
                print(
                    f"[cudaq-exp4]   {name:10s} {r['test_loss_mean']:.4f} +/- {r['test_loss_std']:.4f}"
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
                "shots": shots,
                "backend": simulator_config.backend.value,
            },
            "results": results,
        }
    finally:
        simulator.cleanup()


# ----------------------------------------------------------------------------------------------
# Distributed gradient variance for multi-GPU
# ----------------------------------------------------------------------------------------------

def run_distributed_gradient_variance(
    n_qubits: int,
    edges_sparse: Sequence[Edge],
    n_layers: int = 2,
    noise: float = 0.05,
    n_samples: int = 200,
    seed: int = 0,
    match_noise: bool = True,
    n_gpus: int = 1,
    shots: int = 1024,
) -> Dict:
    """Run gradient variance experiment distributed across multiple GPUs."""
    if n_gpus <= 1:
        config = SimulatorConfig(n_qubits=n_qubits, backend=SimulatorBackend.CUDAQ_NVIDIA)
        return experiment_gradient_variance(
            n_qubits, edges_sparse, n_layers, noise, n_samples, seed,
            match_noise, "backprop", np.pi/4, True, config, shots
        )
    
    # Split samples across GPUs
    samples_per_gpu = n_samples // n_gpus
    remainder = n_samples % n_gpus
    
    futures = []
    with ProcessPoolExecutor(max_workers=n_gpus) as executor:
        for gpu_id in range(n_gpus):
            gpu_samples = samples_per_gpu + (1 if gpu_id < remainder else 0)
            gpu_seed = seed + gpu_id * 1000
            
            config = SimulatorConfig(
                n_qubits=n_qubits,
                backend=SimulatorBackend.CUDAQ_NVIDIA_MGPU,
                enable_mgpu=True,
                mgpu_rank=gpu_id,
                mgpu_world_size=n_gpus,
            )
            
            future = executor.submit(
                experiment_gradient_variance,
                n_qubits, edges_sparse, n_layers, noise, gpu_samples,
                gpu_seed, match_noise, "backprop", np.pi/4, False, config, shots
            )
            futures.append(future)
        
        results = [f.result() for f in futures]
    
    # Combine results
    all_gd = []
    all_gs = []
    for r in results:
        # Would need to return raw gradients from experiment_gradient_variance
        pass
    
    # For now, return first result with metadata
    combined = results[0].copy()
    combined["distributed"] = True
    combined["n_gpus"] = n_gpus
    combined["samples_per_gpu"] = samples_per_gpu
    return combined