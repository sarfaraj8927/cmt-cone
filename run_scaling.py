"""CLI for Experiment 6 (the scaling iso-resource sweep) and the noise-model checks.

    python -m pennylane_cmt.run_scaling --out results/exp6 --samples 200

Each configuration is written to its own JSON file as soon as it finishes, so the sweep can be
interrupted and resumed.  ``--jobs`` processes run in parallel.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple


def _config_name(n: int, L: int, obs: str, m: int) -> str:
    return f"exp6_n{n}_L{L}_{obs}_m{m}.json"


def _run_one(args: Tuple[int, int, str, int, int, int, str]) -> str:
    n, L, obs, m, samples, seed, outdir = args
    from .scaling import experiment_scaling

    path = os.path.join(outdir, _config_name(n, L, obs, m))
    if os.path.exists(path):
        return path + " (cached)"
    res = experiment_scaling(n, L, obs, n_edges=m, n_samples=samples, seed=seed, verbose=False)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    return path


def _run_noise(args: Tuple[int, int, str, float, str, int, int, str]) -> str:
    n, L, obs, p, model, samples, seed, outdir = args
    from .scaling import noise_ordering_check

    path = os.path.join(outdir, f"exp6noise_n{n}_L{L}_{obs}_{model}_p{p}.json")
    if os.path.exists(path):
        return path + " (cached)"
    res = noise_ordering_check(n, L, obs, noise=p, noise_model=model,
                               n_samples=samples, seed=seed, verbose=False)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/exp6")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--noise-samples", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--qubits", type=int, nargs="+", default=[4, 6, 8, 10, 12, 14])
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--observables", nargs="+", default=["z0", "zlast"])
    ap.add_argument("--skip-noise", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    jobs: List[Tuple] = []
    for n, L, obs in itertools.product(args.qubits, args.layers, args.observables):
        for m in sorted({n // 2, n - 1}):
            jobs.append((n, L, obs, m, args.samples, args.seed, args.out))
    # cheap configurations first so that partial results are useful early
    jobs.sort(key=lambda j: (j[0], j[1]))

    noise_jobs: List[Tuple] = []
    if not args.skip_noise:
        for n, L in [(6, 4), (8, 4), (6, 6)]:
            for model, p in [("depolarizing", 0.02), ("depolarizing", 0.05),
                             ("amplitude_damping", 0.02), ("amplitude_damping", 0.05)]:
                noise_jobs.append((n, L, "z0", p, model, args.noise_samples, args.seed, args.out))

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(_run_one, j) for j in jobs]
        futs += [ex.submit(_run_noise, j) for j in noise_jobs]
        for f in as_completed(futs):
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
