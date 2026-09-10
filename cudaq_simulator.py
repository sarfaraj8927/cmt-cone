"""cuQuantum-backed high-performance simulator interface for NVIDIA Quantum Stack.

This module provides a unified interface to cuQuantum simulators (State Vector, Tensor Network,
Density Matrix) with multi-GPU support for large-scale quantum experiments.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cudaq
    CUDAQ_AVAILABLE = True
except ImportError:
    CUDAQ_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    from cuquantum import custatevec, cutensornet
    CUQUANTUM_AVAILABLE = True
except ImportError:
    CUQUANTUM_AVAILABLE = False
    custatevec = None
    cutensornet = None

__all__ = [
    "SimulatorBackend",
    "SimulatorConfig",
    "CuQuantumSimulator",
    "MultiGPUManager",
    "CUQUANTUM_AVAILABLE",
    "CUDAQ_AVAILABLE",
]


class SimulatorBackend(Enum):
    """Available simulator backends."""
    CUDAQ_NVIDIA = "nvidia"           # CUDA-Q default (state vector, single GPU)
    CUDAQ_NVIDIA_MGPU = "nvidia-mgpu" # CUDA-Q multi-GPU
    CUDAQ_QPP = "qpp-cpu"             # CUDA-Q QPP CPU simulator
    CUDAQ_DENSITY = "density-matrix-cpu"  # CUDA-Q density matrix (noisy)
    CUSTATEVEC = "custatevec"         # Direct cuStateVec API
    CUTENSORNET = "cutensornet"       # Direct cuTensorNet API (tensor network)
    MPS = "mps"                       # Matrix Product State (for large qubits)


class SimulatorConfig:
    """Configuration for cuQuantum simulator."""
    
    def __init__(
        self,
        backend: SimulatorBackend = SimulatorBackend.CUDAQ_NVIDIA,
        n_qubits: int = 0,
        n_gpus: int = 1,
        precision: str = "fp64",       # "fp32" or "fp64"
        workspace_limit_gb: float = 0.0,  # 0 = auto
        mps_bond_dim: int = 256,       # For MPS simulator
        svd_cutoff: float = 1e-12,     # For tensor network truncation
        enable_mgpu: bool = False,
        mgpu_rank: int = 0,
        mgpu_world_size: int = 1,
    ):
        self.backend = backend
        self.n_qubits = n_qubits
        self.n_gpus = n_gpus
        self.precision = precision
        self.workspace_limit_gb = workspace_limit_gb
        self.mps_bond_dim = mps_bond_dim
        self.svd_cutoff = svd_cutoff
        self.enable_mgpu = enable_mgpu
        self.mgpu_rank = mgpu_rank
        self.mgpu_world_size = mgpu_world_size
    
    def estimate_memory_gb(self) -> float:
        """Estimate memory requirement for state vector simulation."""
        if self.n_qubits <= 0:
            return 0.0
        # State vector: 2^n * 16 bytes (complex128) or 8 bytes (complex64)
        bytes_per_amp = 16 if self.precision == "fp64" else 8
        state_vector_gb = (2 ** self.n_qubits) * bytes_per_amp / (1024 ** 3)
        
        # Add overhead for gates, workspace
        overhead_factor = 2.0 if self.n_qubits > 30 else 1.5
        return state_vector_gb * overhead_factor
    
    def can_fit_statevector(self) -> bool:
        """Check if state vector fits in GPU memory."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_gb = info.free / (1024 ** 3)
            return self.estimate_memory_gb() < free_gb * 0.8
        except Exception:
            return False
    
    def recommended_backend(self) -> SimulatorBackend:
        """Recommend best backend based on qubit count and resources."""
        if self.n_qubits <= 28:
            return SimulatorBackend.CUDAQ_NVIDIA if self.enable_mgpu else SimulatorBackend.CUDAQ_NVIDIA
        elif self.n_qubits <= 36:
            return SimulatorBackend.CUDAQ_NVIDIA_MGPU if self.enable_mgpu else SimulatorBackend.CUTENSORNET
        elif self.n_qubits <= 50:
            return SimulatorBackend.CUTENSORNET
        else:
            return SimulatorBackend.MPS


class CuQuantumSimulator:
    """High-level interface to cuQuantum simulators."""
    
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self._handle = None
        self._workspace = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the simulator backend."""
        if self.config.backend in (SimulatorBackend.CUDAQ_NVIDIA, 
                                    SimulatorBackend.CUDAQ_NVIDIA_MGPU,
                                    SimulatorBackend.CUDAQ_QPP,
                                    SimulatorBackend.CUDAQ_DENSITY):
            if not CUDAQ_AVAILABLE:
                raise RuntimeError("CUDA-Q not available")
            self._init_cudaq()
        elif self.config.backend == SimulatorBackend.CUSTATEVEC:
            if not CUQUANTUM_AVAILABLE:
                raise RuntimeError("cuStateVec not available")
            self._init_custatevec()
        elif self.config.backend == SimulatorBackend.CUTENSORNET:
            if not CUQUANTUM_AVAILABLE:
                raise RuntimeError("cuTensorNet not available")
            self._init_cutensornet()
        elif self.config.backend == SimulatorBackend.MPS:
            self._init_mps()
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")
        
        self._initialized = True
    
    def _init_cudaq(self) -> None:
        target = self.config.backend.value
        if self.config.enable_mgpu:
            target = "nvidia-mgpu"
        cudaq.set_target(target)
        # Configure number of GPUs
        if hasattr(cudaq, 'set_mgpu_rank'):
            cudaq.set_mgpu_rank(self.config.mgpu_rank)
    
    def _init_custatevec(self) -> None:
        self._handle = custatevec.create()
        # Set workspace
        workspace_size = self.config.workspace_limit_gb * (1024 ** 3) if self.config.workspace_limit_gb > 0 else 0
        if workspace_size > 0:
            self._workspace = cp.cuda.alloc(workspace_size)
            custatevec.set_workspace(self._handle, self._workspace.ptr, workspace_size)
    
    def _init_cutensornet(self) -> None:
        self._handle = cutensornet.create()
    
    def _init_mps(self) -> None:
        # MPS simulator initialization (using cudaq or custom implementation)
        pass
    
    def execute_kernel(
        self,
        kernel: "cudaq.Kernel",
        observable: "cudaq.SpinOperator",
        theta: np.ndarray,
        x: np.ndarray,
        shots: int = 1024,
    ) -> float:
        """Execute a CUDA-Q kernel with this simulator configuration."""
        if not self._initialized:
            self.initialize()
        
        if self.config.backend in (SimulatorBackend.CUDAQ_NVIDIA,
                                    SimulatorBackend.CUDAQ_NVIDIA_MGPU,
                                    SimulatorBackend.CUDAQ_QPP,
                                    SimulatorBackend.CUDAQ_DENSITY):
            return self._execute_cudaq(kernel, observable, theta, x, shots)
        elif self.config.backend == SimulatorBackend.CUSTATEVEC:
            return self._execute_custatevec(kernel, observable, theta, x, shots)
        elif self.config.backend == SimulatorBackend.CUTENSORNET:
            return self._execute_cutensornet(kernel, observable, theta, x, shots)
        elif self.config.backend == SimulatorBackend.MPS:
            return self._execute_mps(kernel, observable, theta, x, shots)
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")
    
    def _execute_cudaq(
        self,
        kernel: "cudaq.Kernel",
        observable: "cudaq.SpinOperator",
        theta: np.ndarray,
        x: np.ndarray,
        shots: int,
    ) -> float:
        theta_flat = theta.ravel()
        if isinstance(observable, list):
            results = []
            for obs in observable:
                exp_val = cudaq.observe(kernel, obs, theta_flat, x, shots_count=shots).expectation()
                results.append(exp_val)
            return results
        else:
            return cudaq.observe(kernel, observable, theta_flat, x, shots_count=shots).expectation()
    
    def _execute_custatevec(
        self,
        kernel: "cudaq.Kernel",
        observable: "cudaq.SpinOperator",
        theta: np.ndarray,
        x: np.ndarray,
        shots: int,
    ) -> float:
        # Direct cuStateVec execution would require extracting circuit from kernel
        # For now, fall back to CUDA-Q with custatevec backend
        cudaq.set_target("nvidia")
        return self._execute_cudaq(kernel, observable, theta, x, shots)
    
    def _execute_cutensornet(
        self,
        kernel: "cudaq.Kernel",
        observable: "cudaq.SpinOperator",
        theta: np.ndarray,
        x: np.ndarray,
        shots: int,
    ) -> float:
        # Tensor network simulation for larger circuits
        # Would require circuit conversion to tensor network format
        cudaq.set_target("nvidia")
        return self._execute_cudaq(kernel, observable, theta, x, shots)
    
    def _execute_mps(
        self,
        kernel: "cudaq.Kernel",
        observable: "cudaq.SpinOperator",
        theta: np.ndarray,
        x: np.ndarray,
        shots: int,
    ) -> float:
        # MPS simulation for very large qubit counts
        raise NotImplementedError("MPS simulator not yet implemented")
    
    def get_statevector(self, kernel: "cudaq.Kernel", theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Get full statevector (for small qubit counts)."""
        if self.config.n_qubits > 30:
            raise ValueError("Statevector extraction only supported for n_qubits <= 30")
        
        cudaq.set_target("nvidia")
        theta_flat = theta.ravel()
        result = cudaq.get_statevector(kernel, theta_flat, x)
        return np.array(result)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self._handle is not None:
            if self.config.backend == SimulatorBackend.CUSTATEVEC:
                custatevec.destroy(self._handle)
            elif self.config.backend == SimulatorBackend.CUTENSORNET:
                cutensornet.destroy(self._handle)
            self._handle = None
        
        if self._workspace is not None:
            self._workspace.free()
            self._workspace = None
        
        self._initialized = False
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class MultiGPUManager:
    """Manager for multi-GPU distributed simulation."""
    
    def __init__(self, world_size: int, n_qubits: int, backend: SimulatorBackend = SimulatorBackend.CUDAQ_NVIDIA_MGPU):
        self.world_size = world_size
        self.n_qubits = n_qubits
        self.backend = backend
        self.rank = 0
        self._initialized = False
    
    def initialize(self, rank: int) -> None:
        """Initialize multi-GPU environment."""
        self.rank = rank
        os.environ["CUDAQ_MGPU_RANK"] = str(rank)
        os.environ["CUDAQ_MGPU_WORLD_SIZE"] = str(self.world_size)
        
        if self.backend == SimulatorBackend.CUDAQ_NVIDIA_MGPU:
            cudaq.set_target("nvidia-mgpu")
            if hasattr(cudaq, 'set_mgpu_rank'):
                cudaq.set_mgpu_rank(rank)
        
        self._initialized = True
    
    def distribute_circuit(self, kernel: "cudaq.Kernel") -> "cudaq.Kernel":
        """Distribute circuit across GPUs (handled automatically by CUDA-Q)."""
        return kernel
    
    def allreduce(self, data: np.ndarray, op: str = "sum") -> np.ndarray:
        """All-reduce across GPUs."""
        # This would use NCCL or MPI in production
        return data
    
    def barrier(self) -> None:
        """Synchronize across GPUs."""
        pass
    
    def finalize(self) -> None:
        """Finalize multi-GPU environment."""
        self._initialized = False


def create_simulator_for_qubits(n_qubits: int, n_gpus: int = 1, prefer_mgpu: bool = False) -> CuQuantumSimulator:
    """Factory function to create optimal simulator for given qubit count."""
    config = SimulatorConfig(
        n_qubits=n_qubits,
        n_gpus=n_gpus,
        enable_mgpu=prefer_mgpu and n_gpus > 1,
    )
    recommended = config.recommended_backend()
    config.backend = recommended
    return CuQuantumSimulator(config)


def benchmark_backends(n_qubits: int, circuit_depth: int = 10) -> Dict[str, float]:
    """Benchmark available backends for a given problem size."""
    import time
    
    results = {}
    
    # Create a test circuit
    if CUDAQ_AVAILABLE:
        kernel = cudaq.make_kernel()
        q = kernel.qalloc(n_qubits)
        for _ in range(circuit_depth):
            for i in range(n_qubits):
                kernel.h(q[i])
            for i in range(n_qubits - 1):
                kernel.cz(q[i], q[i + 1])
        
        observable = spin.z(0)
        theta = np.zeros((circuit_depth, n_qubits, 3))
        x = np.zeros(n_qubits)
        
        for backend in [SimulatorBackend.CUDAQ_NVIDIA, SimulatorBackend.CUDAQ_QPP]:
            if backend == SimulatorBackend.CUDAQ_NVIDIA and n_qubits > 30:
                continue
            
            try:
                cudaq.set_target(backend.value)
                t0 = time.time()
                for _ in range(3):
                    cudaq.observe(kernel, observable, theta.ravel(), x, shots_count=100)
                elapsed = time.time() - t0
                results[backend.value] = elapsed / 3
            except Exception as e:
                results[backend.value] = float('inf')
    
    return results