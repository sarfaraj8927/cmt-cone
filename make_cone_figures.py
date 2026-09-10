"""Figures for the causal-cone results document.

    python -m pennylane_cmt.make_cone_figures

Reads the JSON produced by ``run_cone`` / ``run_cone_conv`` from ``results/cone`` and writes PNG
figures to ``results/cone/figs``.  Every figure is generated from the raw run files, so it cannot
drift away from the tables.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = "results/cone"
FIGS = f"{RES}/figs"

SHORT = {
    "HEA (dense)": "HEA (28 gates)",
    "sparsity-only (80th pct)": "sparsity-only (6)",
    "CMT b=2 (cluster rule)": "CMT cluster (4)",
    "CC-sparse (cone rule, k=2)": "CC-sparse (5)",
    "CC-expressive (cone rule, k=2)": "CC-expressive (16)",
    "CC-expressive (cone rule, k=3)": "CC k=3 (13)",
}
#: distinct markers and dashes so that exactly coincident curves stay visible
STYLE = {
    "HEA (dense)": {"marker": "o", "ls": "-", "ms": 5},
    "sparsity-only (80th pct)": {"marker": "v", "ls": "-", "ms": 5},
    "CMT b=2 (cluster rule)": {"marker": "o", "ls": "-", "ms": 11},
    "CC-sparse (cone rule, k=2)": {"marker": "s", "ls": "--", "ms": 7},
    "CC-expressive (cone rule, k=2)": {"marker": "^", "ls": ":", "ms": 4},
    "CC-expressive (cone rule, k=3)": {"marker": "D", "ls": "-", "ms": 5},
}
COLOUR = {
    "HEA (dense)": "#444444",
    "sparsity-only (80th pct)": "#c05a12",
    "CMT b=2 (cluster rule)": "#1f77b4",
    "CC-sparse (cone rule, k=2)": "#2ca02c",
    "CC-expressive (cone rule, k=2)": "#d62728",
    "CC-expressive (cone rule, k=3)": "#9467bd",
}


def _load(name: str) -> Optional[Dict]:
    path = f"{RES}/{name}"
    if not os.path.exists(path):
        print(f"[figs] missing {path}")
        return None
    with open(path) as fh:
        return json.load(fh)


def fig_variance_vs_noise(tag: str = "unmatched") -> None:
    d = _load(f"c2_variance_{tag}_breastmnist.json")
    if d is None:
        return
    rows = d["rows"]
    tops = list(dict.fromkeys(r["topology"] for r in rows))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for t in tops:
        rs = sorted([r for r in rows if r["topology"] == t], key=lambda r: r["noise"])
        p = [r["noise"] for r in rs]
        st = STYLE.get(t, {"marker": "o", "ls": "-", "ms": 5})
        ax[0].plot(p, [r["variance"] for r in rs], marker=st["marker"], ls=st["ls"],
                   ms=st["ms"], mfc="none", label=SHORT.get(t, t), color=COLOUR.get(t))
        if t != "HEA (dense)":
            y = [r["ratio_to_dense"] for r in rs]
            lo = [r["ratio_ci95"][0] for r in rs]
            hi = [r["ratio_ci95"][1] for r in rs]
            ax[1].errorbar(p, y, yerr=[np.array(y) - np.array(lo), np.array(hi) - np.array(y)],
                           marker=st["marker"], ls=st["ls"], ms=st["ms"], mfc="none",
                           capsize=3, label=SHORT.get(t, t), color=COLOUR.get(t))
    ax[0].set_yscale("log"); ax[0].set_xlabel("depolarizing probability $p$")
    ax[0].set_ylabel(r"gradient variance of $\langle Z_0\rangle$")
    ax[0].set_title(f"Gradient variance, 8 qubits, 4 layers ({tag} budget)")
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[1].set_yscale("log"); ax[1].axhline(35, ls="--", c="k", lw=1)
    ax[1].text(0.001, 38, "35x (submitted abstract)", fontsize=8)
    ax[1].set_xlabel("depolarizing probability $p$")
    ax[1].set_ylabel("variance ratio to HEA")
    ax[1].set_title("Advantage over the dense ansatz (95 % bootstrap)")
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/c2_variance_vs_noise_{tag}.png", dpi=160)
    plt.close(fig)


def fig_gate_count_vs_cone() -> None:
    d = _load("c2_variance_unmatched_breastmnist.json")
    if d is None:
        return
    rows = [r for r in d["rows"] if abs(r["noise"] - 0.10) < 1e-12]
    if not rows:
        return
    names = [r["topology"] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - 0.2, [r["n_edges"] for r in rows], width=0.4, label="entangling gates per layer",
           color="#8899aa")
    ax.set_ylabel("entangling gates per layer")
    ax.set_xticks(x); ax.set_xticklabels([SHORT.get(n, n) for n in names], rotation=20,
                                         ha="right", fontsize=8)
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, [r["variance"] for r in rows], width=0.4, color="#2ca02c",
            label="gradient variance at $p=0.10$")
    ax2.set_yscale("log"); ax2.set_ylabel("gradient variance (log)")
    ax.set_title("Gate count does not predict trainability; the cone does")
    for i, r in enumerate(rows):
        ax2.text(i + 0.2, r["variance"] * 1.4, f"cone dim {r['cone_dla_dimension']}",
                 ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/c2_gate_count_vs_cone.png", dpi=160)
    plt.close(fig)


def fig_cone_vs_global() -> None:
    d = _load("c2_variance_unmatched_breastmnist.json")
    if d is None:
        return
    rows = [r for r in d["rows"] if abs(r["noise"] - 0.10) < 1e-12]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for r in rows:
        ax[0].scatter(r["cone_dla_dimension"], r["variance"], s=60,
                      color=COLOUR.get(r["topology"]), label=SHORT.get(r["topology"]))
        ax[1].scatter(r["global_dla_dimension"], r["variance"], s=60,
                      color=COLOUR.get(r["topology"]))
    for a, name in zip(ax, ("cone algebra dimension", "global DLA dimension")):
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(name); a.grid(alpha=.3)
    ax[0].set_ylabel("gradient variance at $p=0.10$")
    ax[0].legend(fontsize=7)
    ax[0].set_title("The cone dimension orders the data")
    ax[1].set_title("The global dimension does not")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/c3_cone_vs_global.png", dpi=160)
    plt.close(fig)


def fig_offcone() -> None:
    d = _load("c3_locality_breastmnist.json")
    if d is None:
        return
    rows = d["rows"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [max(r["mean_variance_in_cone"], 1e-40) for r in rows], width=0.4,
           label="inside the cone", color="#2ca02c")
    ax.bar(x + 0.2, [max(r["max_variance_out_of_cone"], 1e-40) for r in rows], width=0.4,
           label="outside the cone (max)", color="#d62728")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(r["topology"], r["topology"]) for r in rows], rotation=20,
                       ha="right", fontsize=8)
    ax.set_ylabel("gradient variance per parameter")
    ax.set_title(f"Parameters outside the causal cone have zero gradient (p = {d['noise']})")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/c3_off_cone_zero.png", dpi=160)
    plt.close(fig)


def fig_training(tag: str, title: str) -> None:
    d = _load(f"c4_convergence_{tag}.json")
    if d is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for row in d["rows"]:
        hs = [r["history"] for r in row["runs"]]
        L = min(len(h) for h in hs)
        loss = np.mean([[e["train_loss"] for e in h[:L]] for h in hs], axis=0)
        accv = np.mean([[e["val_accuracy"] for e in h[:L]] for h in hs], axis=0)
        ep = np.arange(1, L + 1)
        c = COLOUR.get(row["topology"])
        st = STYLE.get(row["topology"], {"ls": "-"})
        ax[0].plot(ep, loss, ls=st["ls"], label=SHORT.get(row["topology"], row["topology"]),
                   color=c)
        ax[1].plot(ep, accv, ls=st["ls"], color=c)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss (MSE)")
    ax[0].set_title(title); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation accuracy")
    ax[1].set_title("Validation accuracy"); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/c4_training_{tag}.png", dpi=160)
    plt.close(fig)


def fig_scaling() -> None:
    d = _load("c5_scaling.json")
    if d is None:
        return
    rows = d["rows"]
    n = [r["n_qubits"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(n, [r["dense_dla_dimension"] for r in rows], "o-", label=r"dense algebra $4^n-1$",
            color="#444444")
    ax.plot(n, [r["global_dla_dimension"] for r in rows], "s-",
            label="global algebra of the design", color="#1f77b4")
    ax.plot(n, [r["cone_dla_dimension"] for r in rows], "^-",
            label="cone algebra of the readout", color="#2ca02c")
    ax.set_yscale("log"); ax.set_xlabel("number of qubits $n$")
    ax.set_ylabel("algebra dimension (log)")
    ax.set_title("The certified cone stays constant as the circuit grows (depth 4)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/c5_cone_scaling.png", dpi=160)
    plt.close(fig)


def fig_variance_scaling() -> None:
    d = _load("c6_variance_scaling_breastmnist.json")
    if d is None:
        return
    rows = d["rows"]
    n = [r["n_qubits"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(n, [r["variance_dense"] for r in rows], "o-", label="HEA (dense)",
               color="#444444")
    ax[0].plot(n, [r["variance_cone_design"] for r in rows], "s--", label="cone design",
               color="#2ca02c")
    ax[0].set_yscale("log"); ax[0].set_xlabel("number of qubits $n$")
    ax[0].set_ylabel(r"gradient variance of $\langle Z_0\rangle$ (noiseless)")
    ax[0].set_title("Barren plateau of the dense ansatz, absent for the cone design")
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    y = [r["ratio"] for r in rows]
    lo = [r["ratio_ci95"][0] for r in rows]
    hi = [r["ratio_ci95"][1] for r in rows]
    ax[1].errorbar(n, y, yerr=[np.array(y) - np.array(lo), np.array(hi) - np.array(y)],
                   fmt="s-", capsize=3, color="#2ca02c")
    ax[1].set_yscale("log"); ax[1].set_xlabel("number of qubits $n$")
    ax[1].set_ylabel("variance ratio, cone design / HEA")
    ax[1].set_title("The advantage grows with the register (95 % bootstrap)")
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/c6_variance_scaling.png", dpi=160)
    plt.close(fig)


def main() -> None:
    os.makedirs(FIGS, exist_ok=True)
    fig_variance_vs_noise("unmatched")
    fig_variance_vs_noise("matched")
    fig_variance_vs_noise("conematched")
    fig_gate_count_vs_cone()
    fig_cone_vs_global()
    fig_offcone()
    fig_training("p0_all_z_breastmnist", "Training loss, noiseless, BreastMNIST")
    fig_training("p0.1_all_z_breastmnist", "Training loss, p = 0.10 (trajectories), BreastMNIST")
    fig_training("p0_all_z_pneumoniamnist", "Training loss, noiseless, PneumoniaMNIST")
    fig_training("p0.1_z0_nobias_breastmnist",
                 "Circuit-only training at p = 0.10, BreastMNIST")
    fig_training("p0.05_z0_nobias_breastmnist",
                 "Circuit-only training at p = 0.05, BreastMNIST")
    fig_training("p0_z0_nobias_breastmnist",
                 "Circuit-only training, noiseless, BreastMNIST")
    fig_scaling()
    fig_variance_scaling()
    print(f"[figs] wrote figures to {FIGS}")


if __name__ == "__main__":
    main()
