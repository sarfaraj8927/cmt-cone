"""Checks of the corrected implementation, run with ``python -m pennylane_cmt.self_test``.

They reproduce numerically the statements that are proved in Lean in this repository, and they
check the PennyLane pieces on circuits small enough to verify by other means.  Nothing here needs
network access or a dataset.
"""

from __future__ import annotations

import numpy as np

from .ansatz import channels_per_layer, make_qnode, matched_noise_plan
from .dla import dla_report, lie_closure_dimension
from .stats import (
    componentwise_variance,
    naive_variance,
    paired_variance_ratio_bootstrap,
    sample_variance,
    samples_for_confidence,
)
from .topology import (
    capacitated_topology,
    cluster_sizes,
    complete_edges,
    cooccurrence_matrix_notebook,
    covariance_matrix,
    describe_topology,
    dla_upper_bound,
    pearson_matrix,
)


def check_centring_identity() -> None:
    """``cov = E[x x^T] - mu mu^T``: centring removes exactly the rank-one mean term.

    Lean: ``CMT.Corr.cov_eq_cooc_sub_mean_mul``.
    """
    rng = np.random.default_rng(0)
    X = rng.random((50, 6))
    second = (X.T @ X) / X.shape[0]
    mu = X.mean(axis=0)
    assert np.allclose(covariance_matrix(X), second - np.outer(mu, mu))
    print("[ok] centring removes exactly the rank-one mean term")


def check_ranking_counterexample() -> None:
    """The uncentred statistic ranks constant pixels above perfectly correlated ones.

    Lean: ``CMT.Corr.cooc_ranking_differs_from_cov``.
    """
    X = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    unc = (X.T @ X) / X.shape[0]
    cov = covariance_matrix(X)
    assert unc[0, 1] > unc[2, 3]
    assert cov[0, 1] < cov[2, 3]
    print(
        f"[ok] uncentred ranks (0,1) over (2,3): {unc[0,1]:.3f} > {unc[2,3]:.3f}; "
        f"covariance ranks the other way: {cov[2,3]:.3f} > {cov[0,1]:.3f}"
    )


def check_scale_invariance() -> None:
    """Pearson is invariant under an affine renormalisation of the pixel values; sigmoid is not."""
    rng = np.random.default_rng(1)
    X = rng.random((40, 5))
    Y = 3.0 * X + 0.7
    assert np.allclose(pearson_matrix(X), pearson_matrix(Y))
    assert not np.allclose(
        cooccurrence_matrix_notebook(X), cooccurrence_matrix_notebook(Y)
    )
    print("[ok] Pearson is affine-invariant, the sigmoid co-occurrence is not")


def check_worked_example() -> None:
    """The example proved in Lean: two independent correlated pairs give two clusters of two.

    Lean: ``CMT.Corr.finrank_cmtDLA_blockData_le`` (dim <= 64 against a dense >= 255).
    """
    data = np.array(
        [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    corr = pearson_matrix(data)
    edges, labels, _tau = capacitated_topology(corr, budget=2, percentile=None, threshold=0.1)
    assert sorted(edges) == [(0, 1), (2, 3)], edges
    assert sorted(cluster_sizes(labels).values()) == [2, 2]
    assert dla_upper_bound(labels) == 64
    rep = dla_report(4, edges)
    assert rep["dla_dimension_exact"] == 2 * (4 ** 2 - 1) == 30
    assert rep["bound_holds"] and rep["reduces_dla"]
    print(
        f"[ok] worked example: edges {edges}, exact DLA dim "
        f"{rep['dla_dimension_exact']} <= bound {rep['dla_upper_bound_proved']} "
        f"<< dense {rep['dense_dla']}"
    )


def check_capacity_bound_random() -> None:
    """On random data, every cluster respects the budget and the proved bound holds exactly."""
    rng = np.random.default_rng(2)
    for n, b in ((6, 2), (6, 3), (8, 2), (8, 4)):
        X = rng.random((80, n))
        edges, labels, _ = capacitated_topology(pearson_matrix(X), budget=b, percentile=50.0)
        sizes = cluster_sizes(labels)
        assert max(sizes.values()) <= b, (n, b, sizes)
        rep = dla_report(n, edges)
        assert rep["bound_holds"], rep
        assert rep["dla_dimension_exact"] == rep["dla_dimension_predicted"]
    print("[ok] capacity constraint and DLA bound hold on random data (n = 6, 8; b = 2, 3, 4)")


def check_notebook_topology_is_a_star() -> None:
    """``hea_edges[:20%]`` is a star at qubit 0, and connected from n = 10 on.

    Lean: ``CMT.Impl.cmtEdges_eight``, ``CMT.Impl.qubits_six_seven_isolated``,
    ``CMT.Impl.finrank_cmtDLA_impl_ge``.
    """
    for n in (4, 6, 8, 10):
        dense = complete_edges(n)
        nb = dense[: max(1, int(len(dense) * 0.20))]
        assert all(e[0] == 0 for e in nb), (n, nb)
        d = describe_topology(n, nb)
        if n == 8:
            assert nb == [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
            assert d["cluster_sizes"] == [6, 1, 1]
        if n == 10:
            # connected: the DLA is the full 4**n - 1, so nothing is removed
            assert d["n_clusters"] == 1
        print(
            f"[ok] notebook topology n={n:2d}: {len(nb)} edges, clusters {d['cluster_sizes']}, "
            f"DLA {'= dense' if d['n_clusters'] == 1 else '< dense'}"
        )


def check_dla_connected_is_full() -> None:
    """A connected topology, however sparse, gives the full ``4**n - 1``.

    Lean: ``CMT.finrank_cmtDLA_ge_of_backwardConnected``, ``CMT.finrank_cmtDLA_path_ge``.
    """
    for n in (3, 4, 5, 6):
        chain = [(i, i + 1) for i in range(n - 1)]
        assert lie_closure_dimension(n, chain) == 4 ** n - 1
    print("[ok] the degree-2 chain already has the full DLA at n = 3..6")


def check_noise_budget() -> None:
    """At n = 8 the notebooks give the dense circuit 64 channels per layer against 18.

    Lean: ``CMT.NoiseBudget.channels_eight``, ``CMT.NoiseBudget.padded_channels_eq``.
    """
    assert channels_per_layer(8, 28) == 64
    assert channels_per_layer(8, 5) == 18
    plan = matched_noise_plan(8, 28, 5)
    assert plan["sparse_channels_after_padding"] == 64
    print(f"[ok] matched noise plan at n = 8: {plan}")


def check_estimators() -> None:
    """ddof, Chebyshev sample size, and a bootstrap interval that correctly contains 1."""
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isclose(sample_variance(xs), 2.5)
    assert np.isclose(naive_variance(xs), 2.0)
    assert np.isclose(naive_variance(xs), sample_variance(xs) * (len(xs) - 1) / len(xs))
    assert samples_for_confidence(1.0, 0.2, 0.05) == 500
    rng = np.random.default_rng(3)
    a = rng.normal(size=(200, 4))
    b = rng.normal(size=(200, 4))
    boot = paired_variance_ratio_bootstrap(a, b, resamples=500, seed=0)
    assert boot["ci_low"] < 1.0 < boot["ci_high"], boot
    print(
        f"[ok] estimators: ddof correction, N = 500 for a 20 % effect, "
        f"null ratio CI = ({boot['ci_low']:.3f}, {boot['ci_high']:.3f}) contains 1"
    )


def check_gradient_methods_agree() -> None:
    """Backprop and parameter-shift give the same gradients on the noisy ansatz."""
    from pennylane import numpy as pnp
    import pennylane as qml

    n, layers = 3, 2
    edges = [(0, 1)]
    rng = np.random.default_rng(4)
    theta = pnp.array(rng.uniform(0, 2 * np.pi, size=(layers, n, 3)), requires_grad=True)
    x = pnp.array(np.full(n, np.pi / 4), requires_grad=False)
    g1 = qml.grad(make_qnode(n, edges, layers, 0.05, 0, "z0", "backprop"))(theta, x)
    g2 = qml.grad(make_qnode(n, edges, layers, 0.05, 0, "z0", "parameter-shift"))(theta, x)
    assert np.allclose(np.asarray(g1), np.asarray(g2), atol=1e-7)
    print("[ok] backprop and parameter-shift gradients agree on the noisy ansatz")


def check_identical_circuits_give_ratio_one() -> None:
    """At n = 2 the two circuits are literally identical, so the paired ratio must be exactly 1.

    Lean: ``CMT.Impl.cmt_eq_hea_two``.  The manuscript tabulates a 1.6x improvement at n = 2.
    """
    from .experiments import experiment_gradient_variance

    res = experiment_gradient_variance(
        2, [(0, 1)], n_layers=2, noise=0.05, n_samples=8, seed=0, verbose=False
    )
    assert abs(res["ratio"] - 1.0) < 1e-9, res["ratio"]
    print("[ok] n = 2: identical circuits give a paired ratio of exactly 1.000")


def check_variance_averaging() -> None:
    """Averaging over all parameters is not the same as looking at one component."""
    rng = np.random.default_rng(5)
    g = rng.normal(scale=np.array([1.0, 0.1, 5.0]), size=(400, 3))
    mean_var, per = componentwise_variance(g)
    assert per.shape == (3,)
    assert abs(mean_var - per.mean()) < 1e-12
    assert per.max() / per.min() > 100
    print(
        "[ok] parameter-averaged variance differs from a single component by "
        f"a factor {per.max() / mean_var:.1f} here"
    )


def main() -> None:
    check_centring_identity()
    check_ranking_counterexample()
    check_scale_invariance()
    check_worked_example()
    check_capacity_bound_random()
    check_notebook_topology_is_a_star()
    check_dla_connected_is_full()
    check_noise_budget()
    check_estimators()
    check_variance_averaging()
    check_gradient_methods_agree()
    check_identical_circuits_give_ratio_one()
    print("\nall checks passed")


if __name__ == "__main__":
    main()
