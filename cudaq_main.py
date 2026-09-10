"""NVIDIA Quantum Stack (CUDA-Q + cuQuantum) entry points for CMT-QNN experiments.

This module provides high-level interfaces to run all experiments on NVIDIA hardware.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .cudaq_ansatz import CUDAQ_AVAILABLE, Edge
from .cudaq_simulator import (
    CuQuantumSimulator,
    SimulatorBackend,
    SimulatorConfig,
    create_simulator_for_qubits,
    benchmark_backends,
)
from .cudaq_experiments import (
    experiment_topology,
    experiment_gradient_variance,
    experiment_noise_sweep,
    experiment_training,
    run_distributed_gradient_variance,
)
from .cudaq_learning import capacity_sweep_cudaq
from .cudaq_topology import complete_edges, capacitated_topology, pearson_matrix
from .data import load_splits

__all__ = [
    "run_all_experiments",
    "run_scaling_experiment",
    "run_learning_experiment",
    "benchmark_hardware",
    "main",
]


def get_default_simulator_config(n_qubits: int, prefer_mgpu: bool = False) -> SimulatorConfig:
    """Get optimal simulator configuration for given qubit count."""
    return create_simulator_for_qubits(n_qubits, n_gpus=1, prefer_mgpu=prefer_mgpu).config


def run_all_experiments(
    n_qubits: int = 8,
    n_layers: int = 2,
    noise: float = 0.05,
    budget: int = 2,
    n_samples: int = 100,
    n_train: int = 60,
    epochs: int = 30,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    verbose: bool = True,
    output_dir: Optional[str] = None,
    backend: Optional[SimulatorBackend] = None,
    shots: int = 1024,
) -> Dict:
    """Run all CMT-QNN experiments on NVIDIA Quantum Stack."""
    if not CUDAQ_AVAILABLE:
        raise RuntimeError("CUDA-Q not available. Install with: pip install cudaq")
    
    # Configure simulator
    if backend is None:
        config = get_default_simulator_config(n_qubits)
    else:
        config = SimulatorConfig(n_qubits=n_qubits, backend=backend)
    
    results = {}
    
    # Experiment 1: Topology extraction
    if verbose:
        print(f"\n{'='*60}")
        print(f"NVIDIA Quantum Stack CMT-QNN Experiments")
        print(f"n_qubits={n_qubits}, n_layers={n_layers}, noise={noise}")
        print(f"Backend: {config.backend.value}")
        print(f"{'='*60}\n")
    
    exp1 = experiment_topology(n_qubits, budget=budget, n_train=n_train, verbose=verbose)
    results["experiment_1_topology"] = exp1
    
    # Get sparse topology from capacitated extraction
    edges_sparse = exp1["capacitated"]["edges"]
    
    # Experiment 2: Gradient variance
    exp2 = experiment_gradient_variance(
        n_qubits, edges_sparse, n_layers, noise, n_samples,
        seed=0, match_noise=True, diff_method="backprop",
        verbose=verbose, simulator_config=config, shots=shots
    )
    results["experiment_2_gradient_variance"] = exp2
    
    # Experiment 3: Noise sweep
    exp3 = experiment_noise_sweep(
        n_qubits, edges_sparse, n_layers=n_layers,
        n_samples=n_samples//2, seed=0, match_noise=True,
        diff_method="backprop", verbose=verbose,
        simulator_config=config, shots=shots
    )
    results["experiment_3_noise_sweep"] = exp3
    
    # Experiment 4: Training
    topologies = {
        "dense": complete_edges(n_qubits),
        "capacitated": edges_sparse,
    }
    exp4 = experiment_training(
        n_qubits, topologies, n_layers, noise, epochs, seeds,
        optimizer="adam", stepsize=0.05, init_scheme="uniform",
        match_noise=True, n_train=n_train, n_eval=n_train,
        diff_method="backprop", target_loss=0.5,
        verbose=verbose, simulator_config=config, shots=shots
    )
    results["experiment_4_training"] = exp4
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"cudaq_results_n{n_qubits}_l{n_layers}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        if verbose:
            print(f"\nResults saved to {output_path}")
    
    return results


def run_scaling_experiment(
    max_qubits: int = 20,
    n_layers: int = 2,
    noise: float = 0.05,
    budget: int = 2,
    n_samples: int = 50,
    n_train: int = 60,
    verbose: bool = True,
    output_dir: Optional[str] = None,
) -> List[Dict]:
    """Run experiments across increasing qubit counts to test scaling."""
    if not CUDAQ_AVAILABLE:
        raise RuntimeError("CUDA-Q not available")
    
    results = []
    for n in range(4, max_qubits + 1, 2):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Scaling experiment: n_qubits={n}")
            print(f"{'='*50}")
        
        config = get_default_simulator_config(n)
        
        try:
            exp1 = experiment_topology(n, budget=budget, n_train=n_train, verbose=verbose)
            edges_sparse = exp1["capacitated"]["edges"]
            
            exp2 = experiment_gradient_variance(
                n, edges_sparse, n_layers, noise, n_samples,
                seed=0, match_noise=True, diff_method="backprop",
                verbose=verbose, simulator_config=config
            )
            
            results.append({
                "n_qubits": n,
                "topology": exp1["capacitated"],
                "gradient_variance": exp2,
            })
        except Exception as e:
            if verbose:
                print(f"Failed at n={n}: {e}")
            results.append({
                "n_qubits": n,
                "error": str(e),
            })
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "cudaq_scaling_results.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        if verbose:
            print(f"\nScaling results saved to {output_path}")
    
    return results


def run_learning_experiment(
    n_qubits: int = 8,
    n_layers: int = 4,
    budgets: Sequence[int] = (1, 2, 4, 8),
    epochs: int = 60,
    seeds: Sequence[int] = (0, 1, 2),
    verbose: bool = True,
    output_dir: Optional[str] = None,
    backend: Optional[SimulatorBackend] = None,
    shots: int = 1024,
) -> Dict:
    """Run capacity sweep learning experiment."""
    if not CUDAQ_AVAILABLE:
        raise RuntimeError("CUDA-Q not available")
    
    if backend is None:
        config = get_default_simulator_config(n_qubits)
    else:
        config = SimulatorConfig(n_qubits=n_qubits, backend=backend)
    
    result = capacity_sweep_cudaq(
        n_qubits=n_qubits,
        n_layers=n_layers,
        budgets=budgets,
        epochs=epochs,
        seeds=seeds,
        verbose=verbose,
        simulator_config=config,
        shots=shots,
    )
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"cudaq_learning_n{n_qubits}_l{n_layers}.json")
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        if verbose:
            print(f"\nLearning results saved to {output_path}")
    
    return result


def benchmark_hardware(
    max_qubits: int = 30,
    circuit_depth: int = 10,
    output_dir: Optional[str] = None,
) -> Dict:
    """Benchmark available backends for different qubit counts."""
    if not CUDAQ_AVAILABLE:
        raise RuntimeError("CUDA-Q not available")
    
    results = {}
    for n in range(4, max_qubits + 1, 2):
        print(f"Benchmarking n_qubits={n}...")
        results[n] = benchmark_backends(n, circuit_depth)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "cudaq_benchmark.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nBenchmark results saved to {output_path}")
    
    return results


def main():
    """Command-line entry point for NVIDIA Quantum Stack experiments."""
    parser = argparse.ArgumentParser(description="CMT-QNN on NVIDIA Quantum Stack")
    parser.add_argument("--experiment", choices=["all", "scaling", "learning", "benchmark"],
                       default="all", help="Experiment to run")
    parser.add_argument("--n-qubits", type=int, default=8, help="Number of qubits")
    parser.add_argument("--max-qubits", type=int, default=20, help="Max qubits for scaling")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--noise", type=float, default=0.05, help="Depolarizing noise probability")
    parser.add_argument("--budget", type=int, default=2, help="Cluster budget for topology")
    parser.add_argument("--n-samples", type=int, default=100, help="Samples for gradient variance")
    parser.add_argument("--n-train", type=int, default=60, help="Training samples")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Random seeds")
    parser.add_argument("--shots", type=int, default=1024, help="Measurement shots")
    parser.add_argument("--backend", choices=[b.value for b in SimulatorBackend],
                       help="Simulator backend")
    parser.add_argument("--output-dir", type=str, default="./cudaq_results", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    backend = SimulatorBackend(args.backend) if args.backend else None
    
    if args.experiment == "all":
        run_all_experiments(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            noise=args.noise,
            budget=args.budget,
            n_samples=args.n_samples,
            n_train=args.n_train,
            epochs=args.epochs,
            seeds=args.seeds,
            verbose=args.verbose,
            output_dir=args.output_dir,
            backend=backend,
            shots=args.shots,
        )
    elif args.experiment == "scaling":
        run_scaling_experiment(
            max_qubits=args.max_qubits,
            n_layers=args.n_layers,
            noise=args.noise,
            budget=args.budget,
            n_samples=args.n_samples,
            n_train=args.n_train,
            verbose=args.verbose,
            output_dir=args.output_dir,
        )
    elif args.experiment == "learning":
        run_learning_experiment(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            epochs=args.epochs,
            seeds=args.seeds,
            verbose=args.verbose,
            output_dir=args.output_dir,
            backend=backend,
            shots=args.shots,
        )
    elif args.experiment == "benchmark":
        benchmark_hardware(
            max_qubits=args.max_qubits,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()