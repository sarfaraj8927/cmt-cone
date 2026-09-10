# CMT-QNN on NVIDIA Quantum Stack

High-performance implementation of the **Cone Matching Theory Quantum Neural Network (CMT-QNN)** experiments using **NVIDIA CUDA-Q** and **cuQuantum** for deep circuits and large qubit counts.

## Overview

This implementation ports the PennyLane-based CMT-QNN experiments to NVIDIA's quantum computing stack, enabling:

- **Large-scale simulation**: Up to 40+ qubits with cuQuantum state vector / tensor network backends
- **Multi-GPU acceleration**: Distributed simulation across multiple GPUs
- **Hardware deployment**: Direct execution on NVIDIA GPUs and quantum hardware via CUDA-Q
- **High-performance training**: Batched execution and adjoint differentiation for gradient computation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NVIDIA Quantum Stack                      │
├─────────────────────────────────────────────────────────────┤
│  CUDA-Q (Quantum Kernel Language)                           │
│  ├── Kernel Definition (cudaq_ansatz.py)                    │
│  ├── Adjoint Differentiation                                │
│  └── Multi-GPU Execution (nvidia-mgpu target)               │
├─────────────────────────────────────────────────────────────┤
│  cuQuantum Simulators                                       │
│  ├── cuStateVec (State Vector, up to ~40 qubits)            │
│  ├── cuTensorNet (Tensor Network, 50+ qubits)               │
│  └── cuDensityMat (Noisy Simulation)                        │
├─────────────────────────────────────────────────────────────┤
│  Classical Pre/Post-processing (CUDA-Q compatible)          │
│  ├── Topology Extraction (cudaq_topology.py)                │
│  ├── Cone Analysis (cudaq_cone.py)                          │
│  └── Statistics (ported from original)                      │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Install CUDA-Q (requires NVIDIA GPU with CUDA 12+)
pip install cudaq

# Install cuQuantum
pip install cuquantum

# Install CuPy (match your CUDA version)
pip install cupy-cuda12x

# Install other dependencies
pip install -r requirements_cudaq.txt
```

### System Requirements

- NVIDIA GPU with Compute Capability 7.0+ (Volta or newer)
- CUDA Toolkit 12.0+
- cuQuantum SDK 23.10+
- 16GB+ GPU memory recommended for >25 qubits
- Multi-GPU: NVLink recommended for best performance

## Quick Start

```bash
# Run all experiments (8 qubits, 2 layers)
python -m cudaq_main --experiment all --n-qubits 8 --n-layers 2

# Scaling experiment (4 to 20 qubits)
python -m cudaq_main --experiment scaling --max-qubits 20

# Learning capacity sweep (12 qubits, 4 layers)
python -m cudaq_main --experiment learning --n-qubits 12 --n-layers 4

# Benchmark backends
python -m cudaq_main --experiment benchmark --max-qubits 30
```

## Key Modules

### `cudaq_ansatz.py`
CUDA-Q kernel implementations of the CMT-QNN ansatz:
- RY encoding layer
- CZ entangling layers with configurable topology
- RX/RY/RZ rotation blocks per layer
- Depolarizing noise channels with matched noise budget
- Adjoint differentiation support

### `cudaq_simulator.py`
Unified interface to cuQuantum simulators:
- `SimulatorBackend`: CUDA_Q, CUSTATEVEC, CUTENSORNET, MPS
- `SimulatorConfig`: Configure precision, workspace, multi-GPU
- `CuQuantumSimulator`: High-level execution interface
- `MultiGPUManager`: Distributed simulation across GPUs
- Automatic backend selection based on qubit count

### `cudaq_experiments.py`
Experiments 1-4 ported to CUDA-Q:
- **Exp 1**: Topology extraction from data (classical, unchanged)
- **Exp 2**: Paired gradient variance with bootstrap CI
- **Exp 3**: Noise sweep with confidence intervals
- **Exp 4**: Controlled training comparison
- **Distributed**: Multi-GPU gradient variance

### `cudaq_learning.py`
Experiment 7: Capacity sweep and variational classifier:
- Batched execution for training speed
- Linear readout head trained jointly
- Logistic regression baseline
- Bridge edges (cut relaxation) comparison

### `cudaq_cone.py`
Causal cone analysis (identical to PennyLane version):
- Graph ball computation
- Cone DLA dimension
- Eccentricity and cone bounds

## Running Large-Scale Experiments

### Single GPU (up to ~30 qubits)
```python
from cudaq import SimulatorConfig, SimulatorBackend

config = SimulatorConfig(
    n_qubits=28,
    backend=SimulatorBackend.CUDAQ_NVIDIA,
    precision="fp64",
)
```

### Multi-GPU (30-40 qubits)
```python
config = SimulatorConfig(
    n_qubits=36,
    backend=SimulatorBackend.CUDAQ_NVIDIA_MGPU,
    enable_mgpu=True,
    n_gpus=4,
    mgpu_world_size=4,
)
# Run with: mpirun -np 4 python script.py
```

### Tensor Network (50+ qubits)
```python
config = SimulatorConfig(
    n_qubits=50,
    backend=SimulatorBackend.CUTENSORNET,
    svd_cutoff=1e-12,
    mps_bond_dim=512,
)
```

## Key Features

### Matched Noise Budget
Preserves the corrected noise matching protocol from the paper:
```python
from cudaq_ansatz import matched_noise_plan, make_cudaq_kernel

plan = matched_noise_plan(n_qubits, n_edges_dense, n_edges_sparse)
pad = plan["padding_channels_per_layer"]
kernel = make_cudaq_kernel(n_qubits, edges_sparse, n_layers, noise, pad)
```

### Adjoint Differentiation
Exact gradients via CUDA-Q's adjoint differentiation (backprop):
```python
grad = gradient_cudaq_backprop(kernel, observable, theta, x, shots=1024)
```

### Parameter-Shift Rule
Hardware-compatible gradients:
```python
grad = gradient_cudaq(kernel, observable, theta, x, shots=1024)
```

## Performance Guidelines

| Qubits | Recommended Backend | Est. Memory | Est. Time (100 samples) |
|--------|---------------------|-------------|-------------------------|
| 4-20   | CUDAQ_NVIDIA        | < 2 GB      | ~10s                   |
| 20-28  | CUDAQ_NVIDIA        | 2-16 GB     | ~30s                   |
| 28-36  | CUDAQ_NVIDIA_MGPU   | 16-256 GB   | ~60s (4 GPUs)          |
| 36-50  | CUTENSORNET         | < 64 GB     | ~120s                  |
| 50+    | MPS                 | < 32 GB     | ~300s                  |

## Differences from PennyLane Version

| Feature | PennyLane | CUDA-Q |
|---------|-----------|--------|
| Diff Method | backprop, param-shift | adjoint, param-shift |
| Simulator | default.mixed, default.qubit | cuStateVec, cuTensorNet, MPS |
| Multi-GPU | Limited | Native (nvidia-mgpu) |
| Max Qubits (SV) | ~20 | ~40 |
| Max Qubits (TN) | N/A | 50+ |
| Hardware Deploy | Limited | Direct (NVIDIA QPUs) |
| Batched Execution | Native | Manual loop |

## Citation

If you use this implementation, please cite:
- NVIDIA CUDA-Q: https://github.com/NVIDIA/cuda-quantum
- cuQuantum: https://github.com/NVIDIA/cuQuantum
- Original CMT-QNN paper: [arXiv reference]

## License

MIT License - See LICENSE file for details.