"""The CMT-QNN / HEA ansatz in PennyLane, with a *matched* noise budget.

The circuit structure of the notebooks (RY encoding, CZ on the active edges, RX/RY/RZ per qubit
per layer, ``default.mixed``) is implemented correctly there and is kept unchanged.  What is
corrected here is the noise accounting.

A layer of the notebooks' ansatz applies ``2 * |E| + n`` depolarizing channels: two per CZ and one
per single-qubit rotation block.  At n = 8 that is 64 channels for the dense circuit against 18
for the 5-edge sparse one, so the two circuits are not run under the same noise model.  The gap
between the two contraction bounds is then a function of the edge counts alone (Lean:
``CMT.NoiseBudget.survival_factor_split``) -- a gate-count effect, which is the corrected
Lemma 4.2, and not the algebraic mechanism of Theorem 4.1.  :func:`matched_noise_plan` computes
the padding that equalises the budgets (Lean: ``CMT.NoiseBudget.padded_channels_eq``) and
:func:`make_qnode` applies it.

Run the benchmark **both** ways: ``match_noise=True`` isolates the algebraic effect, and
``match_noise=False`` reproduces the notebooks and measures the gate-count effect.  They are
different claims and the paper should report them separately.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import pennylane as qml
from pennylane import numpy as pnp

__all__ = [
    "channels_per_layer",
    "matched_noise_plan",
    "cone_matched_padding",
    "param_shape",
    "random_params",
    "make_qnode",
]

Edge = Tuple[int, int]


def channels_per_layer(n_qubits: int, n_edges: int) -> int:
    """``2 * |E| + n``: the per-layer depolarizing channel count of the notebooks' ansatz."""
    return 2 * n_edges + n_qubits


def matched_noise_plan(n_qubits: int, n_edges_dense: int, n_edges_sparse: int) -> Dict[str, int]:
    """Padding that gives the sparse circuit the same per-layer noise budget as the dense one.

    Lean: ``CMT.NoiseBudget.padded_channels_eq``.
    """
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
    """Channels per layer that must be added *on the cone* to match the dense circuit there.

    A wire ``w`` carries ``deg(w) + 1`` depolarizing channels per layer (one per entangling gate
    it takes part in, one for its rotation block).  Padding a topology up to the dense count on
    the qubits of its own causal cone gives the two circuits the same local decoherence exactly
    where the gradient of the readout can see it, so what is left is the algebraic difference.
    """
    deg = {w: 0 for w in range(n_qubits)}
    for j, k in edges:
        deg[j] += 1
        deg[k] += 1
    own = sum(deg[w] + 1 for w in cone)
    dense = sum((n_qubits - 1) + 1 for _ in cone)
    return max(0, dense - own)


def param_shape(n_qubits: int, n_layers: int) -> Tuple[int, int, int]:
    return (n_layers, n_qubits, 3)


def random_params(n_qubits: int, n_layers: int, rng, requires_grad: bool = True):
    """Uniform parameter draw of the right shape, as a differentiable PennyLane array."""
    theta = rng.uniform(0.0, 2 * 3.141592653589793, size=param_shape(n_qubits, n_layers))
    return pnp.array(theta, requires_grad=requires_grad)


def _ansatz(theta, x, n_qubits: int, edges: Sequence[Edge], n_layers: int,
            noise: float, pad_channels: int, pad_wires: Sequence[int] = ()) -> None:
    for q in range(n_qubits):
        qml.RY(x[q], wires=q)
    for layer in range(n_layers):
        for (j, k) in edges:
            qml.CZ(wires=[j, k])
            if noise > 0:
                qml.DepolarizingChannel(noise, wires=j)
                qml.DepolarizingChannel(noise, wires=k)
        for q in range(n_qubits):
            qml.RX(theta[layer, q, 0], wires=q)
            qml.RY(theta[layer, q, 1], wires=q)
            qml.RZ(theta[layer, q, 2], wires=q)
            if noise > 0:
                qml.DepolarizingChannel(noise, wires=q)
        if noise > 0:
            wires = list(pad_wires) if len(pad_wires) else list(range(n_qubits))
            for i in range(pad_channels):
                qml.DepolarizingChannel(noise, wires=wires[i % len(wires)])


def make_qnode(
    n_qubits: int,
    edges: Sequence[Edge],
    n_layers: int = 2,
    noise: float = 0.05,
    pad_channels: int = 0,
    observable: str = "z0",
    diff_method: str = "backprop",
    device_name: str = "default.mixed",
    pad_wires: Sequence[int] = (),
):
    """Build the QNode.

    ``observable``
        ``"z0"`` returns ``<Z_0>`` (the quantity the variance experiments differentiate);
        ``"all_z"`` returns the list ``[<Z_j>]``, used by the training experiment.
    ``pad_channels``
        extra depolarizing channels per layer, spread round-robin over ``pad_wires`` (all wires
        when that is empty).  With ``pad_wires`` set to the causal cone of the observable and the
        count taken from :func:`cone_matched_padding`, the padding equalises the noise *where the
        gradient can see it*, which is the control the cone designs need; with the count from
        :func:`matched_noise_plan` and no ``pad_wires`` it equalises the total channel budget.
    ``diff_method``
        ``"backprop"`` is exact and much faster on a simulator; ``"parameter-shift"`` is the
        hardware-realistic rule the notebooks use.  Both give the same gradients here (the
        self-test checks this), so ``backprop`` is the default for the larger sweeps.
    """
    dev = qml.device(device_name, wires=n_qubits)

    if observable == "z0":
        @qml.qnode(dev, diff_method=diff_method)
        def circuit(theta, x):
            _ansatz(theta, x, n_qubits, edges, n_layers, noise, pad_channels, pad_wires)
            return qml.expval(qml.PauliZ(0))
    elif observable == "all_z":
        @qml.qnode(dev, diff_method=diff_method)
        def circuit(theta, x):
            _ansatz(theta, x, n_qubits, edges, n_layers, noise, pad_channels, pad_wires)
            return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]
    else:
        raise ValueError("observable must be 'z0' or 'all_z'")

    return circuit
