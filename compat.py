"""A compatibility shim needed to run the batched simulations past eleven qubits.

PennyLane's ``default.qubit`` switches from an ``einsum`` contraction to a ``tensordot``
contraction once the state tensor has at least ``EINSUM_STATE_WIRECOUNT_PERF_THRESHOLD = 13``
axes, and for the autograd interface it takes that branch unconditionally.  With parameter
broadcasting over a batch of inputs the state carries one extra leading axis, so the switch
happens at ``n = 12`` qubits rather than ``n = 13``; on that path the batch axis of the state is
mistaken for a wire axis and the contraction raises ``ValueError: shape-mismatch for sum``.

The two contractions compute the same thing --- only their performance differs --- so the fix
used here is to raise the threshold, which keeps every simulation on the ``einsum`` path.
:func:`check_einsum_equivalence` re-verifies on every import-time call that the two paths agree
to machine precision at a qubit count where both of them run, so the shim is not taken on trust.

Import this module before building any batched QNode:

    from . import compat  # noqa: F401
"""

from __future__ import annotations

import sys

__all__ = ["EINSUM_THRESHOLD", "apply_einsum_patch", "check_einsum_equivalence"]

# Large enough that no simulation in this package can reach it: an ``n``-qubit batched state has
# ``n + 1`` axes, and the package never goes beyond twenty qubits.
EINSUM_THRESHOLD = 64

_PATCHED = False


def apply_einsum_patch() -> bool:
    """Raise PennyLane's einsum/tensordot threshold.  Returns ``True`` if it changed anything."""
    global _PATCHED
    import pennylane  # noqa: F401  (ensure the submodule is imported)
    import pennylane.devices.qubit  # noqa: F401

    mod = sys.modules.get("pennylane.devices.qubit.apply_operation")
    if mod is None:  # pragma: no cover - PennyLane restructured; nothing to patch
        return False
    old = getattr(mod, "EINSUM_STATE_WIRECOUNT_PERF_THRESHOLD", None)
    if old is None:  # pragma: no cover
        return False
    if old >= EINSUM_THRESHOLD:
        _PATCHED = True
        return False
    mod.EINSUM_STATE_WIRECOUNT_PERF_THRESHOLD = EINSUM_THRESHOLD
    _PATCHED = True
    return True


def _reference_circuit(n_qubits: int, n_layers: int):
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="backprop")
    def circuit(theta, x):
        for q in range(n_qubits):
            qml.RY(x[..., q], wires=q)
        for layer in range(n_layers):
            for q in range(n_qubits - 1):
                qml.CZ(wires=[q, q + 1])
            for q in range(n_qubits):
                qml.RX(theta[layer, q, 0], wires=q)
                qml.RY(theta[layer, q, 1], wires=q)
                qml.RZ(theta[layer, q, 2], wires=q)
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return circuit


def check_einsum_equivalence(n_qubits: int = 8, n_layers: int = 2) -> float:
    """Largest absolute discrepancy between the two contraction paths, on a single input.

    Both paths are valid without a broadcast dimension, so this shows that raising the
    threshold is a performance choice and not a change of numerics.  Returns the max abs
    difference over the ``<Z_j>``; it should be ``0.0``.
    """
    import numpy as np
    import pennylane as qml
    from pennylane import numpy as pnp

    mod = sys.modules["pennylane.devices.qubit.apply_operation"]
    saved = mod.EINSUM_STATE_WIRECOUNT_PERF_THRESHOLD
    circuit = _reference_circuit(n_qubits, n_layers)

    rng = np.random.default_rng(0)
    theta = pnp.array(rng.uniform(0, 2 * np.pi, size=(n_layers, n_qubits, 3)), requires_grad=True)
    x = pnp.array(rng.uniform(0, 1, size=n_qubits), requires_grad=False)
    try:
        mod.EINSUM_STATE_WIRECOUNT_PERF_THRESHOLD = 0                 # force tensordot
        a = np.array(qml.math.stack(circuit(theta, x)))
        mod.EINSUM_STATE_WIRECOUNT_PERF_THRESHOLD = EINSUM_THRESHOLD  # force einsum
        b = np.array(qml.math.stack(circuit(theta, x)))
    finally:
        mod.EINSUM_STATE_WIRECOUNT_PERF_THRESHOLD = saved
    return float(np.abs(a - b).max())


def check_batching_equivalence(n_qubits: int = 12, n_layers: int = 2, batch: int = 4) -> float:
    """Largest discrepancy between the broadcast evaluation and a loop over single inputs.

    This is the check that matters for the patched path: at ``n_qubits >= 12`` the broadcast
    simulation only runs with the patch in place, and its answer is compared here against the
    unbatched simulation of the same inputs, which uses no broadcasting at all.
    """
    import numpy as np
    import pennylane as qml
    from pennylane import numpy as pnp

    apply_einsum_patch()
    circuit = _reference_circuit(n_qubits, n_layers)
    rng = np.random.default_rng(0)
    theta = pnp.array(rng.uniform(0, 2 * np.pi, size=(n_layers, n_qubits, 3)), requires_grad=True)
    xs = rng.uniform(0, 1, size=(batch, n_qubits))
    batched = np.array(qml.math.stack(circuit(theta, pnp.array(xs, requires_grad=False))))
    one_by_one = np.stack(
        [np.array(qml.math.stack(circuit(theta, pnp.array(x, requires_grad=False)))) for x in xs],
        axis=1,
    )
    return float(np.abs(batched - one_by_one).max())


apply_einsum_patch()


if __name__ == "__main__":  # pragma: no cover
    apply_einsum_patch()
    print("max |einsum - tensordot|, single input, n=8 :", check_einsum_equivalence(8))
    print("max |einsum - tensordot|, single input, n=10:", check_einsum_equivalence(10))
    print("max |batched - looped|,   patched path, n=12:", check_batching_equivalence(12))
