"""Analysis of Experiment 6: which predictor explains the measured gradient variance?

Two candidate predictors are scored on the whole sweep.

* ``dim g`` -- the dimension of the dynamical Lie algebra of the topology, the quantity in the
  proved bound of the corrected Theorem 4.1;
* ``dim g_cone`` -- the dimension of the algebra generated inside the depth-``L`` causal cone of
  the observable, the quantity in the observable-aware bound
  (Lean ``CMT.Cone.gradient_variance_cone_lower_bound``).

For each configuration (qubit number, depth, observable, edge count) the family of topologies has
identical resources, so the variance ordering inside a configuration is exactly the effect the
theory is supposed to predict.  The script reports, per configuration and pooled:

* Kendall-style concordance of each predictor with the measured ordering;
* the log-log slope of variance against each predictor;
* the largest gradient variance of a parameter that lies *outside* the cone, which the
  depth-conditionality lemma says must be exactly zero;
* how the matched-resource ratio (the smallest-cone topology over the largest-cone one) behaves
  as the qubit number grows -- the scaling claim itself.
"""

from __future__ import annotations

import glob
import json
import math
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = ["load_runs", "concordance", "analyse", "main"]


def load_runs(pattern: str = "results/exp6/exp6_n*.json") -> List[Dict]:
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


def concordance(pred: Sequence[float], var: Sequence[float]) -> Dict[str, float]:
    """Fraction of comparable pairs whose ordering the predictor gets right.

    A pair is *comparable* when the predictor separates it (``pred_i != pred_j``); pairs on which
    the predictor is silent are counted separately as ``ties``, and a tie is scored as correct
    only if the two variances agree to within 1 %.
    """
    p = np.asarray(pred, dtype=float)
    v = np.asarray(var, dtype=float)
    good = bad = ties = ties_ok = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] == p[j]:
                ties += 1
                if abs(v[i] - v[j]) <= 0.01 * max(v[i], v[j]):
                    ties_ok += 1
            else:
                # larger predictor dimension should mean smaller variance
                if (p[i] - p[j]) * (v[i] - v[j]) < 0:
                    good += 1
                else:
                    bad += 1
    total = good + bad
    return {
        "concordant": good,
        "discordant": bad,
        "concordance": (good / total) if total else float("nan"),
        "ties": ties,
        "ties_consistent": ties_ok,
    }


def _slope(x: Sequence[float], y: Sequence[float]) -> float:
    if len(set(x)) < 2:
        return float("nan")
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def analyse(runs: Sequence[Dict]) -> Dict:
    per_config = []
    pooled_cone: List[Tuple[float, float]] = []
    pooled_global: List[Tuple[float, float]] = []
    max_out_of_cone = 0.0
    ratios: Dict[Tuple[int, str, str], List[Tuple[int, float]]] = {}
    param_level: List[Tuple[float, float]] = []

    for r in runs:
        rows = r["rows"]
        dims_cone = [row["cone_dla_dimension"] for row in rows]
        dims_glob = [row["global_dla_dimension"] for row in rows]
        vars_ = [row["variance_mean"] for row in rows]
        cfg = {
            "n_qubits": r["n_qubits"],
            "n_layers": r["n_layers"],
            "observable": r["observable"],
            "n_edges": r["n_edges_common"],
            "topologies": [row["name"] for row in rows],
            "cone_dims": dims_cone,
            "global_dims": dims_glob,
            "variances": vars_,
            "cone": concordance(dims_cone, vars_),
            "global": concordance(dims_glob, vars_),
            "slope_cone": _slope(dims_cone, vars_),
            "slope_global": _slope(dims_glob, vars_),
            "max_variance_out_of_cone": max(row["max_variance_out_of_cone"] for row in rows),
        }
        per_config.append(cfg)
        pooled_cone += list(zip(dims_cone, vars_))
        pooled_global += list(zip(dims_glob, vars_))
        max_out_of_cone = max(max_out_of_cone, cfg["max_variance_out_of_cone"])

        # per-parameter data: variance against the cone of that parameter's own layer
        for row in rows:
            for prec in row["parameters"]:
                if prec["in_cone"] and prec["variance"] > 0:
                    param_level.append((prec["cone_dla_dimension"], prec["variance"]))

        rule = "n/2" if r["n_edges_common"] == r["n_qubits"] // 2 else (
            "n-1" if r["n_edges_common"] == r["n_qubits"] - 1 else str(r["n_edges_common"]))
        key = (r["n_layers"], r["observable"], rule)
        by_name = {row["name"]: row for row in rows}
        if "matching" in by_name and "star" in by_name:
            ratios.setdefault(key, []).append(
                (r["n_qubits"], by_name["matching"]["variance_mean"] /
                 by_name["star"]["variance_mean"]))

    def _pool(pairs: Sequence[Tuple[float, float]]) -> Dict:
        xs = [p for p, _ in pairs]
        ys = [v for _, v in pairs]
        return {"n_points": len(pairs), "slope": _slope(xs, ys)}

    def _within(key: str) -> Dict:
        """Fixed-effects slope: each configuration is centred before pooling.

        Every configuration has its own overall scale (set by n, the depth and the observable),
        so the pooled regression mixes that scale with the effect of interest.  Centring each
        configuration removes it and leaves the within-configuration dependence, which is the
        quantity the bound predicts.
        """
        xs, ys = [], []
        for c in per_config:
            d = np.log(np.asarray(c[key], dtype=float))
            v = np.log(np.asarray(c["variances"], dtype=float))
            if len(set(d.tolist())) < 2:
                continue
            xs += (d - d.mean()).tolist()
            ys += (v - v.mean()).tolist()
        if len(xs) < 2:
            return {"n_points": len(xs), "slope": float("nan"), "r2": float("nan")}
        a, b = np.polyfit(xs, ys, 1)
        resid = np.asarray(ys) - (a * np.asarray(xs) + b)
        ss_tot = float(np.sum((np.asarray(ys) - np.mean(ys)) ** 2))
        r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
        return {"n_points": len(xs), "slope": float(a), "r2": r2}

    out = {
        "n_configurations": len(per_config),
        "pooled_cone_fit": _pool(pooled_cone),
        "pooled_global_fit": _pool(pooled_global),
        "per_parameter_cone_fit": _pool(param_level),
        "concordance_cone": {
            "concordant": sum(c["cone"]["concordant"] for c in per_config),
            "discordant": sum(c["cone"]["discordant"] for c in per_config),
            "ties": sum(c["cone"]["ties"] for c in per_config),
            "ties_consistent": sum(c["cone"]["ties_consistent"] for c in per_config),
        },
        "concordance_global": {
            "concordant": sum(c["global"]["concordant"] for c in per_config),
            "discordant": sum(c["global"]["discordant"] for c in per_config),
            "ties": sum(c["global"]["ties"] for c in per_config),
            "ties_consistent": sum(c["global"]["ties_consistent"] for c in per_config),
        },
        "within_configuration_cone_fit": _within("cone_dims"),
        "within_configuration_global_fit": _within("global_dims"),
        "max_variance_out_of_cone": max_out_of_cone,
        "matching_over_star_ratio": {
            f"L{k[0]}_{k[1]}_edges-{k[2]}": sorted(v) for k, v in ratios.items()
        },
        "configurations": per_config,
    }
    growth = {}
    for k, v in out["matching_over_star_ratio"].items():
        ns = np.array([n for n, _ in v], dtype=float)
        rs = np.array([r for _, r in v], dtype=float)
        if len(ns) >= 3:
            a = float(np.polyfit(ns, np.log(rs), 1)[0])
            growth[k] = {"log_ratio_slope_per_qubit": a,
                         "factor_per_two_qubits": float(math.exp(2 * a)),
                         "n_points": len(ns)}
    out["ratio_growth"] = growth
    for key in ("concordance_cone", "concordance_global"):
        c = out[key]
        tot = c["concordant"] + c["discordant"]
        c["fraction"] = (c["concordant"] / tot) if tot else float("nan")
    return out


def load_noise_runs(pattern: str = "results/exp6/exp6noise_*.json") -> List[Dict]:
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


def analyse_noise(runs: Sequence[Dict]) -> List[Dict]:
    out = []
    for r in runs:
        dims = [row["cone_dla_dimension"] for row in r["rows"]]
        vars_ = [row["variance_mean"] for row in r["rows"]]
        out.append({
            "n_qubits": r["n_qubits"],
            "n_layers": r["n_layers"],
            "noise_model": r["noise_model"],
            "noise": r["noise"],
            "cone_order_monotone": r["cone_order_monotone"],
            "concordance_cone": concordance(dims, vars_),
            "concordance_global": concordance(
                [row["global_dla_dimension"] for row in r["rows"]], vars_),
            "rows": [{k: row[k] for k in ("name", "global_dla_dimension",
                                          "cone_dla_dimension", "variance_mean")}
                     for row in r["rows"]],
        })
    return out


def main() -> None:
    runs = load_runs()
    if not runs:
        raise SystemExit("no results found; run `python -m pennylane_cmt.run_scaling` first")
    res = analyse(runs)
    res["noise_checks"] = analyse_noise(load_noise_runs())
    os.makedirs("results", exist_ok=True)
    with open("results/exp6_summary.json", "w") as fh:
        json.dump(res, fh, indent=1)
    c, g = res["concordance_cone"], res["concordance_global"]
    print(f"configurations: {res['n_configurations']}")
    print(f"cone predictor   : {c['concordant']}/{c['concordant'] + c['discordant']} pairs correct "
          f"({c['fraction']:.3f}), ties {c['ties']} of which {c['ties_consistent']} consistent")
    print(f"global predictor : {g['concordant']}/{g['concordant'] + g['discordant']} pairs correct "
          f"({g['fraction']:.3f}), ties {g['ties']} of which {g['ties_consistent']} consistent")
    print(f"pooled slope  log Var vs log dim_cone : {res['pooled_cone_fit']['slope']:.3f}")
    print(f"pooled slope  log Var vs log dim_g    : {res['pooled_global_fit']['slope']:.3f}")
    print(f"per-parameter slope vs its own cone   : {res['per_parameter_cone_fit']['slope']:.3f} "
          f"({res['per_parameter_cone_fit']['n_points']} parameters)")
    print(f"within-configuration slope, cone      : {res['within_configuration_cone_fit']['slope']:.3f} "
          f"(R2 = {res['within_configuration_cone_fit']['r2']:.3f})")
    print(f"within-configuration slope, global    : {res['within_configuration_global_fit']['slope']:.3f} "
          f"(R2 = {res['within_configuration_global_fit']['r2']:.3f})")
    print(f"largest out-of-cone variance          : {res['max_variance_out_of_cone']:.2e}")
    for k, v in res["ratio_growth"].items():
        print(f"ratio growth {k}: x{v['factor_per_two_qubits']:.2f} per two qubits")
    for k, v in res["matching_over_star_ratio"].items():
        print(f"matching/star ratio {k}: " + ", ".join(f"n={n}: {r:.2f}" for n, r in v))


if __name__ == "__main__":
    main()
