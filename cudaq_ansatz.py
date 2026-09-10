"""CUDA-Q implementation of the CMT-QNN / HEA ansatz with matched noise budget.

This module replaces the PennyLane ansatz with CUDA-Q kernels for NVIDIA Quantum Stack.
The circuit structure (RY encoding, CZ on active edges, RX/RY/RZ per qubit per layer)
is preserved while using CUDA-Q's hardware-efficient execution model.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

try:
    import cudaq
    from cudaq import spin
    CUDAQ_AVAILABLE = True
except ImportError:
    CUDAQ_AVAILABLE = False
    cudaq = None
    spin = None

__all__ = [
    "channels_per_layer",
    "matched_noise_plan",
    "cone_matched_padding",
    "param_shape",
    "random_params",
    "make_cudaq_kernel",
    "make_cudaq_observable",
    "CUDAQ_AVAILABLE",
]

Edge = Tuple[int, int]


def channels_per_layer(n_qubits: int, n_edges: int) -> int:
    """``2 * |E| + n``: the per-layer depolarizing channel count of the ansatz."""
    return 2 * n_edges + n_qubits


def matched_noise_plan(n_qubits: int, n_edges_dense: int, n_edges_sparse: int) -> Dict[str, int]:
    """Padding that gives the sparse circuit the same per-layer noise budget as the dense one."""
    if n_edges_sparse > n_edges_dense:
        raise ValueError("the sparse topology must not have more edges than the dense one")
    pad = 2 * (n_edges_dense - n_edges_sparse)
    return {
        "dense_channels_per_layer": channels_per_layer(n_qubits, n_edges_dense),
        "sparse_channels_per_layer": channels_per_layer(n_qubits, n_edges_sparse),
        "padding_channels_per_layer": pad,
        "sparse_channels_after_padding": channels_per_layer(n_qubits, n_edges_sparse) + pad,
    }


def cone_matched_padding(n_qubits: int, edges: Sequence[Edge], cone: Sequence[int]) -> int:
    """Channels per layer that must be added on the cone to match the dense circuit there."""
    deg = {w: 0 for w in range(n_qubits)}
    for j, k in edges:
        deg[j] += 1
        deg[k] += 1
    own = sum(deg[w] + 1 for w in cone)
    dense = sum((n_qubits - 1) + 1 for _ in cone)
    return max(0, dense - own)


def param_shape(n_qubits: int, n_layers: int) -> Tuple[int, int, int]:
    return (n_layers, n_qubits, 3)


def random_params(n_qubits: int, n_layers: int, rng: np.random.Generator, requires_grad: bool = True) -> np.ndarray:
    """Uniform parameter draw of the right shape."""
    theta = rng.uniform(0.0, 2 * np.pi, size=param_shape(n_qubits, n_layers))
    return theta.astype(np.float64)


def make_cudaq_kernel(
    n_qubits: int,
    edges: Sequence[Edge],
    n_layers: int = 2,
    noise: float = 0.05,
    pad_channels: int = 0,
    pad_wires: Sequence[int] = (),
) -> "cudaq.Kernel":
    """Build the CUDA-Q kernel for the CMT-QNN ansatz.
    
    Args:
        n_qubits: Number of qubits
        edges: Active edges for CZ gates
        n_layers: Number of ansatz layers
        noise: Depolarizing noise probability
        pad_channels: Extra depolarizing channels per layer for noise matching
        pad_wires: Wires to apply padding channels to (all if empty)
    
    Returns:
        CUDA-Q kernel that can be executed on simulators or hardware
    """
    if not CUDAQ_AVAILABLE:
        raise RuntimeError("CUDA-Q not available. Install with: pip install cudaq")
    
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(n_qubits)
    theta = kernel.alloc_vector("theta", n_layers * n_qubits * 3)
    x = kernel.alloc_vector("x", n_qubits)
    
    pad_wire_list = list(pad_wires) if pad_wires else list(range(n_qubits))
    
    # RY encoding
    for q in range(n_qubits):
        kernel.ry(x[q], qubits[q])
    
    for layer in range(n_layers):
        # CZ entangling layer
        for (j, k) in edges:
            kernel.cz(qubits[j], qubits[k])
            if noise > 0:
                kernel.depolarize(noise, qubits[j])
                kernel.depolarize(noise, qubits[k])
        
        # Single-qubit rotation block
        for q in range(n_qubits):
            idx = layer * n_qubits * 3 + q * 3
            kernel.rx(theta[idx], qubits[q])
            kernel.ry(theta[idx + 1], qubits[q])
            kernel.rz(theta[idx + 2], qubits[q])
            if noise > 0:
                kernel.depolarize(noise, qubits[q])
        
        # Padding channels for noise matching
        if noise > 0 and pad_channels > 0:
            for i in range(pad_channels):
                kernel.depolarize(noise, qubits[pad_wire_list[i % len(pad_wire_list)]])
    
    return kernel


def make_cudaq_observable(n_qubits: int, observable: str = "z0") -> "cudaq.SpinOperator":
    """Create the CUDA-Q observable for measurement.
    
    Args:
        n_qubits: Number of qubits
        observable: "z0" for <Z_0>, "all_z" for list of <Z_j>
    
    Returns:
        CUDA-Q SpinOperator or list of SpinOperators
    """
    if not CUDAQ_AVAILABLE:
        raise RuntimeError("CUDA-Q not available")
    
    if observable == "z0":
        return spin.z(0)
    elif observable == "all_z":
        return [spin.z(q) for q in range(n_qubits)]
    else:
        raise ValueError("observable must be 'z0' or 'all_z'")


def execute_cudaq(
    kernel: "cudaq.Kernel",
    observable: "cudaq.SpinOperator",
    theta: np.ndarray,
    x: np.ndarray,
    shots: int = 1024,
    backend: str = "nvidia",
) -> float:
    """Execute CUDA-Q kernel and return expectation value.
    
    Args:
        kernel: CUDA-Q kernel
        observable: Observable to measure
        theta: Parameter values (n_layers, n_qubits, 3)
        x: Input encoding values (n_qubits,)
        shots: Number of shots
        backend: CUDA-Q backend ("nvidia", "nvidia-mgpu", "qpp-cpu", "density-matrix-cpu")
    
    Returns:
        Expectation value
    """
    if not CUDAQ_AVAILABLE:
        raise RuntimeError("CUDA-Q not available")
    
    cudaq.set_target(backend)
    
    theta_flat = theta.ravel()
    
    if isinstance(observable, list):
        results = []
        for obs in observable:
            exp_val = cudaq.observe(kernel, obs, theta_flat, x, shots_count=shots).expectation()
            results.append(exp_val)
        return results
    else:
        return cudaq.observe(kernel, observable, theta_flat, x, shots_count=shots).expectation()


def gradient_cudaq(
    kernel: "cudaq.Kernel",
    observable: "cudaq.SpinOperator",
    theta: np.ndarray,
    x: np.ndarray,
    shots: int = 1024,
    backend: str = "nvidia",
) -> np.ndarray:
    """Compute gradient using CUDA-Q's parameter-shift rule.
    
    Args:
        kernel: CUDA-Q kernel
        observable: Observable to measure
        theta: Parameter values
        x: Input encoding values
        shots: Number of shots
        backend: CUDA-Q backend
    
    Returns:
        Gradient array of same shape as theta
    """
    if not CUDAQ_AVAILABLE:
        raise RuntimeError("CUDA-Q not available")
    
    cudaq.set_target(backend)
    
    theta_flat = theta.ravel()
    n_params = len(theta_flat)
    grad = np.zeros(n_params, dtype=np.float64)
    
    for i in range(n_params):
        theta_plus = theta_flat.copy()
        theta_minus = theta_flat.copy()
        theta_plus[i] += np.pi / 2
        theta_minus[i] -= np.pi / 2
        
        exp_plus = cudaq.observe(kernel, observable, theta_plus, x, shots_count=shots).expectation()
        exp_minus = cudaq.observe(kernel, observable, theta_minus, x, shots_count=shots).expectation()
        
        grad[i] = 0.5 * (exp_plus - exp_minus)
    
    return grad.reshape(theta.shape)


def gradient_cudaq_backprop(
    kernel: "cudaq.Kernel",
    observable: "cudaq.SpinOperator",
    theta: np.ndarray,
    x: np.ndarray,
    shots: int = 1024,
    backend: str = "nvidia",
) -> np.ndarray:
    """Compute gradient using CUDA-Q's adjoint differentiation (backprop).
    
    Requires nvidia backend with adjoint differentiation support.
    """
    if not CUDAQ_AVAILABLE:
        raise RuntimeError("CUDA-Q not available")
    
    cudaq.set_target(backend)
    
    # CUDA-Q supports adjoint differentiation via the `gradient` function
    theta_flat = theta.ravel()
    
    if isinstance(observable, list):
        grads = []
        for obs in observable:
            grad = cudaq.gradient(kernel, obs, theta_flat, x)
            grads.append(np.array(grad))
        return np.array(grads)
    else:
        grad = cudaq.gradient(kernel, observable, theta_flat, x)
        return np.array(grad).reshape(theta.shape)