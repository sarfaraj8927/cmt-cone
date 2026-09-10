"""Command-line driver for the corrected experiments.

Examples
--------
    python -m pennylane_cmt.run_all --quick                 # small smoke run, a few minutes
    python -m pennylane_cmt.run_all --exp 1                 # Table 1, corrected
    python -m pennylane_cmt.run_all --exp 2 --qubits 4 6 8 --samples 500
    python -m pennylane_cmt.run_all --exp 3 --qubits 6 --samples 200
    python -m pennylane_cmt.run_all --exp 4 --qubits 4 --epochs 40 --seeds 5

Every run writes a JSON file into ``--outdir`` (default ``results/``) containing the full protocol
and the raw numbers, so that a table in the paper can be regenerated from it.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .data import load_splits
from .experiments import (
    experiment_gradient_variance,
    experiment_noise_sweep,
    experiment_topology,
    experiment_training,
)
from .isoresource import experiment_isoresource
from .topology import capacitated_topology, complete_edges, describe_topology, pearson_matrix

Edge = Tuple[int, int]


def data_derived_edges(n_qubits: int, budget: int, percentile: float = 80.0) -> Tuple[List[Edge], Dict]:
    """The topology the *corrected* pipeline uses everywhere: capacity-constrained, from data."""
    splits = load_splits(n_qubits)
    X_train, _ = splits["train"]  # type: ignore[misc]
    edges, labels, _tau = capacitated_topology(pearson_matrix(X_train), budget=budget,
                                               percentile=percentile)
    return edges, describe_topology(n_qubits, edges, labels)


def _save(obj, outdir: str, name: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)
    print(f"[run] wrote {path}")
    return path


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=int, nargs="*", default=[1, 2, 3, 4],
                    help="which experiments to run (1 topology, 2 variance, 3 noise sweep, "
                         "4 training, 5 iso-resource DLA test)")
    ap.add_argument("--qubits", type=int, nargs="*", default=[4, 6],
                    help="system sizes for experiments 2-4")
    ap.add_argument("--topology-qubits", type=int, default=16,
                    help="feature/qubit count for experiment 1")
    ap.add_argument("--budget", type=int, default=2, help="maximum cluster size b")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--samples", type=int, default=200,
                    help="parameter draws per configuration (>= 500 to resolve a 20 %% effect)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--stepsize", type=float, default=0.05)
    ap.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--diff-method", choices=["backprop", "parameter-shift"], default="backprop")
    ap.add_argument("--unmatched-noise", action="store_true",
                    help="reproduce the notebooks' unmatched noise budget as well")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--quick", action="store_true", help="tiny sizes, for a smoke run")
    args = ap.parse_args(argv)

    if args.quick:
        args.qubits = [4]
        args.topology_qubits = 8
        args.samples = 20
        args.epochs = 4
        args.seeds = 2

    if 1 in args.exp:
        print("\n=== Experiment 1: data-derived topology (corrected) ===")
        res = experiment_topology(args.topology_qubits, budget=args.budget)
        _save(res, args.outdir, f"exp1_topology_n{args.topology_qubits}.json")

    if 2 in args.exp:
        print("\n=== Experiment 2: gradient variance (paired, matched noise) ===")
        rows = []
        for n in args.qubits:
            edges, desc = data_derived_edges(n, args.budget)
            print(f"[run] n={n}: data-derived topology {desc['edges']} "
                  f"clusters {desc['cluster_sizes']} DLA <= {desc['dla_upper_bound']} "
                  f"(dense >= {desc['dense_dla_lower_bound']})")
            r = experiment_gradient_variance(
                n, edges, n_layers=args.layers, noise=args.noise, n_samples=args.samples,
                seed=0, match_noise=True, diff_method=args.diff_method)
            r["topology"] = desc
            rows.append(r)
            if args.unmatched_noise:
                r2 = experiment_gradient_variance(
                    n, edges, n_layers=args.layers, noise=args.noise, n_samples=args.samples,
                    seed=0, match_noise=False, diff_method=args.diff_method)
                r2["topology"] = desc
                rows.append(r2)
        _save(rows, args.outdir, "exp2_gradient_variance.json")

    if 3 in args.exp:
        print("\n=== Experiment 3: noise sweep with confidence intervals ===")
        out = {}
        for n in args.qubits:
            edges, desc = data_derived_edges(n, args.budget)
            out[str(n)] = {
                "topology": desc,
                "sweep": experiment_noise_sweep(
                    n, edges, n_layers=args.layers, n_samples=args.samples,
                    diff_method=args.diff_method),
            }
        _save(out, args.outdir, "exp3_noise_sweep.json")

    if 4 in args.exp:
        print("\n=== Experiment 4: controlled training comparison ===")
        out = {}
        for n in args.qubits:
            edges, desc = data_derived_edges(n, args.budget)
            topologies = {"HEA": complete_edges(n), "CMT-QNN": edges}
            out[str(n)] = {
                "topology": desc,
                "training": experiment_training(
                    n, topologies, n_layers=args.layers, noise=args.noise, epochs=args.epochs,
                    seeds=tuple(range(args.seeds)), optimizer=args.optimizer,
                    stepsize=args.stepsize, diff_method=args.diff_method),
            }
        _save(out, args.outdir, "exp4_training.json")

    if 5 in args.exp:
        print("\n=== Experiment 5: iso-resource test of the algebraic mechanism ===")
        for n in args.qubits:
            out = {str(n): experiment_isoresource(
                n, n_layers=args.layers, noise=args.noise, n_samples=args.samples,
                diff_method=args.diff_method)}
            _save(out, args.outdir, f"exp5_isoresource_n{n}_L{args.layers}.json")

    print("\n[run] done.  Reminder: the ratio is a claim only when its confidence interval "
          "excludes 1, and a DLA claim is a claim only when the cluster sizes are reported.")


if __name__ == "__main__":
    main()
