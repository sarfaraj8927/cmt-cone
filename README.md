# CMT-QNN on NVIDIA Quantum Stack

CUDA-Q and cuQuantum implementation of the Cone Matching Theory Quantum
Neural Network (CMT-QNN) experiments. It supports GPU-accelerated simulation,
adjoint differentiation, multiple simulator backends, and multi-GPU execution.

## Install

CUDA 12+ and a compatible NVIDIA GPU are required for the GPU backends.

```bash
pip install cudaq cuquantum cupy-cuda12x
pip install -r requirements_cudaq.txt
```

## Run

```bash
python -m cudaq_main --experiment all --n-qubits 8 --n-layers 2
python -m cudaq_main --experiment scaling --max-qubits 20
```

See [README_CUDAQ.md](README_CUDAQ.md) for architecture, backend selection,
large-scale execution guidance, and the complete experiment reference.

## License

MIT. See [LICENSE](LICENSE).
