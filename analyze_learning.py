"""Summarise Experiment 7 (learning and the capacity sweep).

    python -m pennylane_cmt.analyze_learning --results results/exp7

Reports, for every configuration, whether anything actually learns (against the majority-class
rate and against logistic regression on the same features), and the trainability-versus-accuracy
trade-off across the cluster-budget family: the gradient variance, the exact global algebra
dimension, the cone dimension of the observable, and the held-out accuracy.

Writes ``results/exp7_summary.json``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np


def _spearman(a: List[float], b: List[float]) -> float:
    """Rank correlation without a SciPy dependency."""
    def ranks(v):
        order = np.argsort(np.asarray(v, dtype=float))
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        return r
    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(ra @ rb / denom) if denom > 0 else float("nan")


def summarise(results: str) -> Dict:
    paths = sorted(glob.glob(os.path.join(results, "exp7_capacity_n*.json")))
    out: Dict[str, object] = {"experiment": "capacity_sweep_summary", "configurations": []}
    for path in paths:
        with open(path) as fh:
            data = json.load(fh)
        rows = data["rows"]
        maj = rows[0]["test_majority_rate"]
        base = data.get("logistic_baseline", {})
        base_acc = base.get("test_accuracy") if isinstance(base, dict) else None
        accs = [r["test_accuracy_mean"] for r in rows]
        vars_ = [r["gradient_variance"] for r in rows]
        cones = [r["cone_dla_dimension"] for r in rows]
        globals_ = [r["global_dla_dimension"] for r in rows]
        clustered = [r for r in rows if r["max_cluster"] < data["n_qubits"]]
        dense = [r for r in rows if r["name"].startswith("dense")]
        entry = {
            "file": os.path.basename(path),
            "dataset": data.get("dataset"),
            "n_qubits": data["n_qubits"],
            "n_layers": data["n_layers"],
            "epochs": data["epochs"],
            "seeds": data["seeds"],
            "majority_rate": maj,
            "logistic_accuracy": base_acc,
            "best_accuracy": max(accs),
            "best_accuracy_model": rows[int(np.argmax(accs))]["name"],
            "all_beat_majority": bool(min(accs) > maj),
            "any_beats_logistic": bool(base_acc is not None and max(accs) > base_acc),
            "accuracy_spread": float(max(accs) - min(accs)),
            "variance_ratio_best_over_dense":
                (max(vars_) / dense[0]["gradient_variance"]) if dense else None,
            "clustered_variance_over_dense":
                (max(r["gradient_variance"] for r in clustered) / dense[0]["gradient_variance"])
                if (clustered and dense) else None,
            "spearman_variance_vs_cone": _spearman(vars_, cones),
            "spearman_variance_vs_global": _spearman(vars_, globals_),
            "spearman_accuracy_vs_cone": _spearman(accs, cones),
            "rows": [
                {
                    "name": r["name"],
                    "n_edges": r["n_edges"],
                    "cluster_sizes": r["cluster_sizes"],
                    "global_dla_dimension": r["global_dla_dimension"],
                    "cone_dla_dimension": r["cone_dla_dimension"],
                    "gradient_variance": r["gradient_variance"],
                    "test_accuracy_mean": r["test_accuracy_mean"],
                    "test_accuracy_std": r["test_accuracy_std"],
                }
                for r in rows
            ],
        }
        out["configurations"].append(entry)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/exp7")
    ap.add_argument("--out", default="results/exp7_summary.json")
    args = ap.parse_args(argv)

    summary = summarise(args.results)
    for cfg in summary["configurations"]:
        print(f"\n=== {cfg['dataset']}  n={cfg['n_qubits']} L={cfg['n_layers']} "
              f"({cfg['epochs']} epochs, seeds {cfg['seeds']}) ===")
        print(f"{'topology':24s} {'|E|':>4s} {'dim g':>8s} {'dim cone':>9s} "
              f"{'Var':>10s} {'test acc':>16s}")
        for r in cfg["rows"]:
            print(f"{r['name']:24s} {r['n_edges']:4d} {r['global_dla_dimension']:8d} "
                  f"{r['cone_dla_dimension']:9d} {r['gradient_variance']:10.3e} "
                  f"{r['test_accuracy_mean']:.3f} +- {r['test_accuracy_std']:.3f}")
        print(f"majority {cfg['majority_rate']:.3f}   logistic {cfg['logistic_accuracy']}   "
              f"all beat majority: {cfg['all_beat_majority']}   "
              f"best {cfg['best_accuracy']:.3f} ({cfg['best_accuracy_model']})")
        print(f"accuracy spread across the family {cfg['accuracy_spread']:.3f};  "
              f"clustered/dense variance {cfg['clustered_variance_over_dense']}")
        print(f"Spearman(Var, dim cone) {cfg['spearman_variance_vs_cone']:+.3f}   "
              f"Spearman(Var, dim g) {cfg['spearman_variance_vs_global']:+.3f}   "
              f"Spearman(acc, dim cone) {cfg['spearman_accuracy_vs_cone']:+.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
