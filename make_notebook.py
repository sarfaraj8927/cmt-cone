"""Generate a self-contained corrected notebook from the package sources.

    python -m pennylane_cmt.make_notebook

writes ``corrected-cmt-experiments.ipynb`` in the project root.  The notebook inlines the code of
this package (so it can be uploaded to Kaggle/Colab on its own) and runs the four corrected
experiments.  Regenerate it after editing any module, so that the two never drift apart.
"""

from __future__ import annotations

import json
import os
import re

MODULES = ["stats", "topology", "dla", "ansatz", "data", "experiments", "isoresource",
           "self_test"]

HEADER = """# Corrected CMT-QNN experiments

Drop-in replacement for the cells of `cmt-barren.ipynb`, `asymptotic-variance-8-qubits.ipynb`,
`critical-noise-threshold-8-qubits.ipynb`, `sota-1-1.ipynb` and `sota-2.ipynb`.

Five corrections, each needed for one of the machine-checked statements in `RequestProject/*.lean`:

1. **centred** Pearson correlation instead of the uncentred sigmoid co-occurrence matrix;
2. a **capacity-constrained** topology extractor with a proved cluster-size bound, plus the
   **exact dynamical Lie algebra dimension** of the extracted topology (the number that decides
   whether the sparsification removes anything at all);
3. a **matched per-layer noise budget** for the two ansätze (64 vs 18 channels at n = 8 otherwise);
4. an **unbiased, parameter-averaged, paired** gradient-variance estimator with a bootstrap
   confidence interval, and the sample size the target effect needs (~500, not 15–50);
5. a **controlled training protocol**: official train/val/test splits, topology fitted on the
   training split only, one optimizer and epoch budget for every method, ≥ 5 seeds, final **test**
   loss reported (not a ratio of percentage reductions).

The code below is the `pennylane_cmt` package inlined verbatim; see `pennylane_cmt/README.md` for
the mapping from each correction to the theorem that requires it.
"""


def _strip(src: str) -> str:
    """Remove the relative imports and the ``__future__`` import: the notebook is one module."""
    src = re.sub(r"^[ \t]*from \.[\w.]* import \([^)]*\)[ \t]*\n", "", src, flags=re.M)
    src = re.sub(r"^[ \t]*from \.[\w.]* import [^(\n]*\n", "", src, flags=re.M)
    src = re.sub(r"^[ \t]*from __future__ import .*\n", "", src, flags=re.M)
    # drop the ``if __name__ == "__main__"`` blocks: in a notebook every cell is __main__
    src = re.sub(r"\nif __name__ == \"__main__\":\n(?:[ \t]+.*\n|\n)*", "\n", src)
    return src.strip() + "\n"


def build() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    code = ["from __future__ import annotations"]
    for m in MODULES:
        with open(os.path.join(here, f"{m}.py")) as f:
            code.append(f"# {'=' * 90}\n# {m}.py\n# {'=' * 90}\n" + _strip(f.read()))
    body = "\n\n".join(code)

    def md(text: str) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}

    def cd(text: str) -> dict:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.splitlines(True),
        }

    cells = [
        md(HEADER),
        cd("!pip install -q pennylane medmnist scikit-image"),
        md("## The corrected implementation (inlined)"),
        cd(body),
        md(
            "## Checks\n\n"
            "These reproduce numerically the statements proved in Lean, and check the PennyLane "
            "pieces on circuits small enough to verify independently. They need no data."
        ),
        cd("main()  # the self-test"),
        md(
            "## Experiment 1 — data-derived topology (replaces `cmt-barren` cell 1 / Table 1)\n\n"
            "Reports what Table 1 should report: the cluster sizes, the proved DLA bound "
            "`2*K*4**b` and the exact DLA dimension, next to the sparsity. A percentile-thresholded "
            "graph is typically connected, and a connected topology has the full `4**n - 1`, i.e. "
            "no algebra dimension is removed."
        ),
        cd("topology_report = experiment_topology(n_qubits=16, budget=2)"),
        md(
            "## Experiment 2 — gradient variance (replaces `cmt-barren` cell 2 and "
            "`asymptotic-variance-8-qubits`)\n\n"
            "Paired draws, all parameters, `ddof=1`, matched noise budget, bootstrap 95 % interval. "
            "Set `match_noise=False` to reproduce the notebooks' unmatched comparison, which "
            "measures the gate-count effect instead."
        ),
        cd(
            "results_exp2 = {}\n"
            "for n in (4, 6):\n"
            "    splits = load_splits(n)\n"
            "    edges, labels, tau = capacitated_topology(pearson_matrix(splits['train'][0]),\n"
            "                                              budget=2, percentile=80.0)\n"
            "    print(n, 'qubits:', describe_topology(n, edges, labels))\n"
            "    results_exp2[n] = experiment_gradient_variance(n, edges, n_samples=500,\n"
            "                                                   match_noise=True)"
        ),
        md(
            "## Experiment 3 — noise sweep (replaces `critical-noise-threshold-8-qubits`)\n\n"
            "A confidence interval at every noise level: a critical-threshold claim needs the "
            "interval to exclude 1 above some `p` and not below it."
        ),
        cd(
            "splits = load_splits(6)\n"
            "edges6, labels6, _ = capacitated_topology(pearson_matrix(splits['train'][0]),\n"
            "                                          budget=2, percentile=80.0)\n"
            "sweep = experiment_noise_sweep(6, edges6, n_samples=300)"
        ),
        md(
            "## Experiment 4 — training (replaces `cmt-barren` cells 3–4, `sota-1-1`, `sota-2` / "
            "Table 3)\n\n"
            "Shared official splits, identical optimizer/budget/initialisation, 5 seeds, final test "
            "loss and accuracy."
        ),
        cd(
            "n = 4\n"
            "splits = load_splits(n, limits=(60, 60, 60))\n"
            "edges4, labels4, _ = capacitated_topology(pearson_matrix(splits['train'][0]),\n"
            "                                          budget=2, percentile=80.0)\n"
            "training = experiment_training(n, {'HEA': complete_edges(n), 'CMT-QNN': edges4},\n"
            "                               epochs=30, seeds=(0, 1, 2, 3, 4), optimizer='adam',\n"
            "                               stepsize=0.05)"
        ),
        md(
            "## Experiment 5 — iso-resource test of the algebraic mechanism (new)\n\n"
            "Experiments 2-4 change the cluster structure and the gate count at the same time. "
            "Here every topology has the same qubit number, edge count, depth, parameter count and "
            "depolarizing-channel count, so the only difference is the dynamical Lie algebra. If "
            "the variance tracks `dim g` here, the algebraic mechanism is doing the work; the "
            "`path` and `star` rows have the *same* `dim g`, so their difference measures what the "
            "dimension does not capture."
        ),
        cd(
            "iso = experiment_isoresource(6, n_layers=4, n_samples=300)"
        ),
        md(
            "## How to report\n\n"
            "1. A variance ratio is a claim only if its bootstrap interval excludes 1.\n"
            "2. An algebraic (Lemma 4.1) claim needs the cluster sizes of the topology actually "
            "simulated, with `2*K*4**b < 4**n - 1`; otherwise the effect measured is the gate-count "
            "mechanism of the corrected Lemma 4.2.\n"
            "3. Say whether the noise budget was matched.\n"
            "4. Report final test loss, never a ratio of percentage reductions.\n"
            "5. At n = 2 with one edge the two circuits are identical; the honest ratio is 1.000.\n"
            "6. A DLA-based claim should be checked at fixed resources (Experiment 5), and is "
            "depth-conditional: at small depth the causal cone of the observable can hide the "
            "algebra entirely."
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    nb = build()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "corrected-cmt-experiments.ipynb")
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
