"""NVIDIA Quantum Stack (CUDA-Q + cuQuantum) implementation of CMT-QNN experiments.

This package provides high-performance quantum circuit simulation and execution
using NVIDIA's quantum computing stack for deep and large-scale qubit experiments.

Modules:
    cudaq_ansatz: CUDA-Q kernel implementations of CMT-QNN ansatz
    cudaq_simulator: cuQuantum-backed simulator interface with multi-GPU support
    cudaq_topology: Data-derived topology extraction (classical, CUDA-Q compatible)
    cudaq_experiments: Experiments 1-4 (topology, gradient variance, noise sweep, training)
    cudaq_learning: Capacity sweep and variational classifier (Experiment 7)
    cudaq_cone: Causal cone analysis and cone DLA dimension
    cudaq_main: High-level entry points and CLI

Usage:
    python -m cudaq_main --experiment all --n-qubits 16 --n-layers 4
    python -m cudaq_main --experiment scaling --max-qubits 24
    python -m cudaq_main --experiment learning --n-qubits 12 --n-layers 6
    python -m cudaq_main --experiment benchmark --max-qubits 30

Requirements:
    - CUDA-Q (pip install cudaq)
    - cuQuantum (pip install cuquantum)
    - CuPy (pip install cupy)
    - PennyLane (for classical utilities)
    - NumPy, SciPy, scikit-learn
"""

from .cudaq_ansatz import (
    CUDAQ_AVAILABLE,
    Edge,
    channels_per_layer,
    matched_noise_plan,
    cone_matched_padding,
    param_shape,
    random_params,
    make_cudaq_kernel,
    make_cudaq_observable,
    execute_cudaq,
    gradient_cudaq,
    gradient_cudaq_backprop,
)

from .cudaq_simulator import (
    SimulatorBackend,
    SimulatorConfig,
    CuQuantumSimulator,
    MultiGPUManager,
    CUQUANTUM_AVAILABLE,
    create_simulator_for_qubits,
    benchmark_backends,
)

from .cudaq_topology import (
    covariance_matrix,
    pearson_matrix,
    cooccurrence_matrix_notebook,
    capacitated_topology,
    threshold_topology,
    cluster_labels,
    cluster_sizes,
    dla_upper_bound,
    dense_dla_lower_bound,
    describe_topology,
    complete_edges,
    extract_topology_from_cudaq_data,
)

from .cudaq_experiments import (
    experiment_topology,
    experiment_gradient_variance,
    experiment_noise_sweep,
    experiment_training,
    run_distributed_gradient_variance,
)

from .cudaq_learning import (
    make_batched_cudaq_kernel,
    train_classifier_cudaq,
    logistic_baseline,
    capacity_sweep_cudaq,
    bridge_edges,
    cluster_block_edges,
)

from .cudaq_cone import (
    adjacency,
    ball,
    components_in,
    cone_dla_dimension,
    global_dla_dimension,
    eccentricity,
    cone_report,
)

from .cudaq_main import (
    run_all_experiments,
    run_scaling_experiment,
    run_learning_experiment,
    benchmark_hardware,
    main,
)

__all__ = [
    # Ansatz
    "CUDAQ_AVAILABLE",
    "Edge",
    "channels_per_layer",
    "matched_noise_plan",
    "cone_matched_padding",
    "param_shape",
    "random_params",
    "make_cudaq_kernel",
    "make_cudaq_observable",
    "execute_cudaq",
    "gradient_cudaq",
    "gradient_cudaq_backprop",
    # Simulator
    "SimulatorBackend",
    "SimulatorConfig",
    "CuQuantumSimulator",
    "MultiGPUManager",
    "CUQUANTUM_AVAILABLE",
    "create_simulator_for_qubits",
    "benchmark_backends",
    # Topology
    "covariance_matrix",
    "pearson_matrix",
    "cooccurrence_matrix_notebook",
    "capacitated_topology",
    "threshold_topology",
    "cluster_labels",
    "cluster_sizes",
    "dla_upper_bound",
    "dense_dla_lower_bound",
    "describe_topology",
    "complete_edges",
    "extract_topology_from_cudaq_data",
    # Experiments
    "experiment_topology",
    "experiment_gradient_variance",
    "experiment_noise_sweep",
    "experiment_training",
    "run_distributed_gradient_variance",
    # Learning
    "make_batched_cudaq_kernel",
    "train_classifier_cudaq",
    "logistic_baseline",
    "capacity_sweep_cudaq",
    "bridge_edges",
    "cluster_block_edges",
    # Cone
    "adjacency",
    "ball",
    "components_in",
    "cone_dla_dimension",
    "global_dla_dimension",
    "eccentricity",
    "cone_report",
    # Main
    "run_all_experiments",
    "run_scaling_experiment",
    "run_learning_experiment",
    "benchmark_hardware",
    "main",
]

__version__ = "1.0.0"