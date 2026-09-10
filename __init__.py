"""Corrected PennyLane implementation of the CMT-QNN experiments.

Each module states which defect of the shipped notebooks it repairs and which machine-checked
statement in ``RequestProject/*.lean`` the repair is needed for.  See ``pennylane_cmt/README.md``
for the mapping from notebook cells to functions here, and run

    python -m pennylane_cmt.run_all --quick

for an end-to-end smoke run, or

    python -m pennylane_cmt.self_test

for the checks that need no simulation.
"""

from .ansatz import channels_per_layer, make_qnode, matched_noise_plan, random_params
from .data import load_splits, synthetic_splits
from .dla import dla_report, lie_closure_dimension
from .isoresource import experiment_isoresource
from .experiments import (
    experiment_gradient_variance,
    experiment_noise_sweep,
    experiment_topology,
    experiment_training,
)
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
    dla_upper_bound,
    pearson_matrix,
    threshold_topology,
)

__all__ = [
    "capacitated_topology",
    "channels_per_layer",
    "complete_edges",
    "componentwise_variance",
    "cooccurrence_matrix_notebook",
    "describe_topology",
    "dla_report",
    "dla_upper_bound",
    "experiment_gradient_variance",
    "experiment_noise_sweep",
    "experiment_topology",
    "experiment_training",
    "lie_closure_dimension",
    "load_splits",
    "make_qnode",
    "matched_noise_plan",
    "mean_confidence_interval",
    "paired_variance_ratio_bootstrap",
    "pearson_matrix",
    "random_params",
    "sample_variance",
    "samples_for_confidence",
    "synthetic_splits",
    "threshold_topology",
]
