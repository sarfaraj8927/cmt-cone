"""Figures for the revised manuscript, drawn from the JSON produced by ``run_all``.

    python -m pennylane_cmt.make_figures --results results --outdir manuscript/figs

Every figure is generated from a results file; nothing is drawn by hand.  Missing input files are
skipped with a message, so the script can be run after a partial set of experiments.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 9, "figure.dpi": 200, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})


def _load(path: str):
    if not os.path.exists(path):
        print(f"[fig] missing {path}, skipped")
        return None
    with open(path) as f:
        return json.load(f)


def fig_noise_sweep(results: str, outdir: str) -> None:
    data = _load(os.path.join(results, "exp3_noise_sweep.json"))
    if data is None:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for n, block in data.items():
        rows = block["sweep"]
        p = [r["noise"] for r in rows]
        ax1.semilogy(p, [r["variance_dense"] for r in rows], "o-", label=f"dense HEA (n={n})")
        ax1.semilogy(p, [r["variance_sparse"] for r in rows], "s-",
                     label=f"cluster CMT (n={n})")
        ratio = np.array([r["ratio"] for r in rows])
        lo = np.array([r["ratio_ci95"][0] for r in rows])
        hi = np.array([r["ratio_ci95"][1] for r in rows])
        ax2.errorbar(p, ratio, yerr=[ratio - lo, hi - ratio], fmt="o-", capsize=2,
                     label=f"n={n}, matched budget")
    ax1.set_xlabel("depolarizing probability $p$")
    ax1.set_ylabel(r"$\delta[\partial_\theta L]$")
    ax1.legend(fontsize=7)
    ax2.axhline(1.0, color="k", lw=0.8, ls="--")
    ax2.set_xlabel("depolarizing probability $p$")
    ax2.set_ylabel("variance ratio CMT / HEA")
    ax2.legend(fontsize=7)
    path = os.path.join(outdir, "fig_noise_sweep.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] wrote {path}")


def fig_isoresource(results: str, outdir: str) -> None:
    blocks: List[Dict] = []
    for name in sorted(os.listdir(results)):
        if name.startswith("exp5_isoresource") and name.endswith(".json"):
            d = _load(os.path.join(results, name))
            if d:
                for _n, block in d.items():
                    blocks.append(block)
    if not blocks:
        print("[fig] no experiment-5 results, skipped")
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    markers = "osd^v"
    for i, block in enumerate(blocks):
        rows = sorted(block["rows"], key=lambda r: r["dla_dimension_exact"])
        dims = np.array([r["dla_dimension_exact"] for r in rows], dtype=float)
        var = np.array([r["variance"] for r in rows], dtype=float)
        lbl = f"n={block['n_qubits']}, L={block['n_layers']}"
        ax.loglog(dims, var, markers[i % len(markers)], ls="none", label=lbl)
        for r, d, v in zip(rows, dims, var):
            ax.annotate(r["name"], (d, v), textcoords="offset points", xytext=(4, 2),
                        fontsize=6)
        c = var[0] * dims[0]
        xs = np.linspace(dims.min(), dims.max(), 50)
        ax.loglog(xs, c / xs, lw=0.8, ls="--", color="grey")
    ax.set_xlabel(r"$\dim \mathfrak{g}$ (exact Lie closure)")
    ax.set_ylabel(r"$\delta[\partial_\theta L]$")
    ax.set_title("equal $n$, $|E|$, depth and noise channels", fontsize=8)
    ax.legend(fontsize=7)
    path = os.path.join(outdir, "fig_isoresource.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] wrote {path}")


def fig_variance_scaling(results: str, outdir: str) -> None:
    rows = _load(os.path.join(results, "exp2_gradient_variance.json")) or []
    extra = _load(os.path.join(results, "n8", "exp2_gradient_variance.json"))
    if extra:
        rows = list(rows) + list(extra)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for matched, marker, lbl in ((True, "o", "matched noise budget"),
                                 (False, "s", "unmatched (notebook protocol)")):
        sel = sorted([r for r in rows if r["match_noise"] is matched],
                     key=lambda r: r["n_qubits"])
        if not sel:
            continue
        n = [r["n_qubits"] for r in sel]
        ratio = np.array([r["ratio"] for r in sel])
        lo = np.array([r["ratio_ci95"][0] for r in sel])
        hi = np.array([r["ratio_ci95"][1] for r in sel])
        ax.errorbar(n, ratio, yerr=[ratio - lo, hi - ratio], fmt=marker + "-", capsize=2,
                    label=lbl)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel("variance ratio CMT / HEA")
    ax.legend(fontsize=7)
    path = os.path.join(outdir, "fig_variance_scaling.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] wrote {path}")




def fig_cone_predictor(results: str, outdir: str) -> None:
    """Measured variance against the two predictors, over the whole Experiment 6 sweep."""
    data = _load(os.path.join(results, "exp6_summary.json"))
    if data is None:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    for ax, key, lbl in ((ax1, "global_dims", r"$\dim\,\mathfrak{g}$"),
                         (ax2, "cone_dims", r"$\dim\,\mathfrak{g}_{\rm cone}$")):
        for c in data["configurations"]:
            d = np.asarray(c[key], dtype=float)
            v = np.asarray(c["variances"], dtype=float)
            order = np.argsort(d)
            ax.plot(d[order], v[order], "o-", ms=2.5, lw=0.6, alpha=0.5,
                    color="C0" if c["observable"] == "z0" else "C1")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(lbl)
    ax1.set_ylabel("gradient variance")
    ax1.set_title(f"global algebra: {data['concordance_global']['fraction']:.0%} of pairs ordered "
                  f"correctly", fontsize=8)
    ax2.set_title(f"cone algebra: {data['concordance_cone']['fraction']:.0%} of pairs ordered "
                  f"correctly", fontsize=8)
    path = os.path.join(outdir, "fig_cone_predictor.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] wrote {path}")


def fig_scaling_ratio(results: str, outdir: str) -> None:
    """The matched-resource advantage as a function of the qubit number, up to n = 14."""
    data = _load(os.path.join(results, "exp6_summary.json"))
    if data is None:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for k, v in sorted(data["matching_over_star_ratio"].items()):
        ns = [n for n, _ in v]
        rs = [r for _, r in v]
        style = "o-" if "_z0_" in k else "s--"
        ax.plot(ns, rs, style, ms=3, lw=1, label=k.replace("_edges-n/2", "").replace("_", " "))
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel("variance ratio, matching / star")
    ax.legend(fontsize=6)
    path = os.path.join(outdir, "fig_scaling_ratio.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] wrote {path}")


def fig_capacity_tradeoff(results: str, outdir: str) -> None:
    """Trainability against task performance across the cluster-budget family."""
    paths = sorted(glob.glob(os.path.join(results, "exp7", "exp7_capacity_n*.json")))
    if not paths:
        print("[fig] missing exp7 capacity sweep, skipped")
        return
    fig, axes = plt.subplots(1, len(paths), figsize=(3.4 * len(paths), 2.7), squeeze=False)
    for ax, path in zip(axes[0], paths):
        with open(path) as fh:
            data = json.load(fh)
        for row in data["rows"]:
            ax.errorbar(row["gradient_variance"], row["test_accuracy_mean"],
                        yerr=row["test_accuracy_std"], fmt="o", ms=4, capsize=2)
            ax.annotate(row["name"].split(" ")[0], (row["gradient_variance"],
                                                    row["test_accuracy_mean"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.axhline(data["rows"][0]["test_majority_rate"], color="k", lw=0.8, ls="--",
                   label="majority class")
        base = data.get("logistic_baseline", {})
        if isinstance(base, dict) and "test_accuracy" in base:
            ax.axhline(base["test_accuracy"], color="C3", lw=0.8, ls=":",
                       label="logistic regression")
        ax.set_xscale("log")
        ax.set_xlabel("gradient variance (trainability)")
        ax.set_ylabel("test accuracy")
        ax.set_title(f"n = {data['n_qubits']}, L = {data['n_layers']}", fontsize=8)
        ax.legend(fontsize=6)
    path = os.path.join(outdir, "fig_capacity_tradeoff.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] wrote {path}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="manuscript/figs")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    fig_noise_sweep(args.results, args.outdir)
    fig_isoresource(args.results, args.outdir)
    fig_variance_scaling(args.results, args.outdir)
    fig_cone_predictor(args.results, args.outdir)
    fig_scaling_ratio(args.results, args.outdir)
    fig_capacity_tradeoff(args.results, args.outdir)


if __name__ == "__main__":
    main()
