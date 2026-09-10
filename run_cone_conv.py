"""Runner for Experiment C4 (convergence and accuracy under the causal-cone design).

    python -m pennylane_cmt.run_cone_conv --noise 0.0  --epochs 60 --jobs 8
    python -m pennylane_cmt.run_cone_conv --noise 0.10 --epochs 40 --batch 32 --jobs 8
    python -m pennylane_cmt.run_cone_conv --validate
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

import numpy as np

from .coneconv import convergence_experiment, validate_trajectories
from .coneexp import build_family


def _dump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=default)
    print(f"[run_cone_conv] wrote {path}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dataset", type=str, default="breastmnist")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--stepsize", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=0, help="0 = full batch")
    ap.add_argument("--trajectories", type=int, default=4)
    ap.add_argument("--eval-trajectories", type=int, default=8)
    ap.add_argument("--readout", type=str, default="all_z", choices=("all_z", "z0", "z0_nobias"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--outdir", type=str, default="results/cone")
    ap.add_argument("--validate", action="store_true",
                    help="check the Pauli unravelling against the density-matrix simulator")
    args = ap.parse_args(argv)

    os.environ.setdefault("OMP_NUM_THREADS", "1")

    if args.validate:
        _dump(validate_trajectories(n_trajectories=3000),
              f"{args.outdir}/c4_trajectory_validation.json")
        return

    fam = build_family(args.qubits, args.layers, (0,), args.dataset)
    res = convergence_experiment(
        fam, args.qubits, args.layers, dataset=args.dataset, noise=args.noise,
        epochs=args.epochs, seeds=tuple(args.seeds), stepsize=args.stepsize,
        batch_size=args.batch or None, trajectories=args.trajectories,
        eval_trajectories=args.eval_trajectories, readout=args.readout, jobs=args.jobs)
    tag = f"p{args.noise:g}_{args.readout}_{args.dataset}"
    _dump(res, f"{args.outdir}/c4_convergence_{tag}.json")


if __name__ == "__main__":
    main()
