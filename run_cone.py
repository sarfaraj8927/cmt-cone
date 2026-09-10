"""Runner for the causal-cone experiment suite.  Writes one JSON file per experiment.

Examples
--------
    python -m pennylane_cmt.run_cone --exp 1
    python -m pennylane_cmt.run_cone --exp 2 --samples 150 --jobs 8
    python -m pennylane_cmt.run_cone --exp 2 --samples 150 --jobs 8 --matched
    python -m pennylane_cmt.run_cone --exp 3 --samples 60 --jobs 8
    python -m pennylane_cmt.run_cone --exp 5
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

import numpy as np

from .coneexp import (
    build_family,
    cone_locality_check,
    cone_scaling_table,
    design_table,
    variance_scaling,
    variance_sweep,
)


def _dump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=default)
    print(f"[run_cone] wrote {path}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", type=int, nargs="+", default=[1])
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dataset", type=str, default="breastmnist")
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--matched", action="store_true",
                    help="pad the sparse circuits to the dense depolarizing-channel budget")
    ap.add_argument("--cone-matched", action="store_true",
                    help="pad each circuit on its own causal cone up to the dense count there")
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.05, 0.10])
    ap.add_argument("--outdir", type=str, default="results/cone")
    args = ap.parse_args(argv)

    os.environ.setdefault("OMP_NUM_THREADS", "1")

    if 1 in args.exp:
        _dump(design_table(args.qubits, args.layers), f"{args.outdir}/c1_design.json")

    if 2 in args.exp or 3 in args.exp:
        fam = build_family(args.qubits, args.layers, (0,), args.dataset)

    if 2 in args.exp:
        protocol = "cone" if args.cone_matched else bool(args.matched)
        res = variance_sweep(fam, args.qubits, args.layers, tuple(args.noise),
                             n_samples=args.samples, match_noise=protocol, jobs=args.jobs)
        tag = "conematched" if args.cone_matched else ("matched" if args.matched else "unmatched")
        _dump(res, f"{args.outdir}/c2_variance_{tag}_{args.dataset}.json")

    if 3 in args.exp:
        res = cone_locality_check(fam, args.qubits, args.layers, noise=max(args.noise),
                                  n_samples=args.samples, jobs=args.jobs)
        _dump(res, f"{args.outdir}/c3_locality_{args.dataset}.json")

    if 5 in args.exp:
        _dump(cone_scaling_table(n_layers=args.layers), f"{args.outdir}/c5_scaling.json")

    if 6 in args.exp:
        _dump(variance_scaling(n_layers=args.layers, n_samples=args.samples, jobs=args.jobs,
                               dataset=args.dataset),
              f"{args.outdir}/c6_variance_scaling_{args.dataset}.json")


if __name__ == "__main__":
    main()
