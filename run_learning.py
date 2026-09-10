"""CLI for Experiment 7: the capacity sweep, on either binary MedMNIST task.

    python -m pennylane_cmt.run_learning --out results/exp7 --epochs 150 --jobs 7
    python -m pennylane_cmt.run_learning --out results/exp7 --dataset pneumoniamnist
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/exp7")
    ap.add_argument("--qubits", type=int, nargs="+", default=[8, 12])
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--dataset", default="breastmnist")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=None)
    ap.add_argument("--no-dense", action="store_true",
                    help="skip the complete-graph reference (it dominates the runtime at n >= 12)")
    args = ap.parse_args()

    from .learning import capacity_sweep

    os.makedirs(args.out, exist_ok=True)
    tag = "" if args.dataset == "breastmnist" else f"_{args.dataset}"
    for n in args.qubits:
        path = os.path.join(args.out, f"exp7_capacity_n{n}_L{args.layers}{tag}.json")
        if os.path.exists(path):
            print(f"{path} (cached)", flush=True)
            continue
        res = capacity_sweep(n_qubits=n, n_layers=args.layers, budgets=(1, 2, 4),
                             epochs=args.epochs, seeds=tuple(args.seeds), jobs=args.jobs,
                             include_dense=not args.no_dense, dataset=args.dataset,
                             n_train=args.n_train, n_eval=args.n_eval)
        with open(path, "w") as fh:
            json.dump(res, fh, indent=1)
        print(path, flush=True)


if __name__ == "__main__":
    main()
