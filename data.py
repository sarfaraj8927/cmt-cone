"""BreastMNIST loading, with the train / validation / test discipline the experiments need.

The notebooks take the first 10-20 images of the training split, extract the topology from those
same images, train on them and report the training loss: there is no held-out data anywhere, and
the topology is fitted on the evaluation set.  This module returns the three official MedMNIST
splits separately; :func:`load_splits` is the only entry point the experiments use, and the
topology is always extracted from the *training* split alone.

If ``medmnist`` is not installed or the download fails, :func:`load_splits` falls back to a
deterministic synthetic dataset with the same interface and says so in the returned metadata, so
that the pipeline is runnable offline.  Results obtained on the fallback are clearly not results
about BreastMNIST.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = ["load_splits", "synthetic_splits", "image_to_features", "DATASETS"]

Split = Tuple[np.ndarray, np.ndarray]

#: The binary MedMNIST tasks the experiments may be run on.  Both are 28x28 greyscale with
#: official train/val/test splits, so the whole protocol transfers unchanged; the second one is
#: there so that no claim in the paper rests on a single dataset.
DATASETS = ("breastmnist", "pneumoniamnist")


def image_to_features(img: np.ndarray, n_features: int) -> np.ndarray:
    """Downsample a 28x28 image to ``n_features`` values in [0, 1].

    A square feature count whose side divides 28 (n = 4, 16, 49, ...) is obtained by mean pooling,
    which is what the notebooks do (``block_reduce`` with block sizes 14x14 and 7x7); any other
    count is obtained by anti-aliased resizing.
    """
    a = np.asarray(img, dtype=float)
    s = int(round(math.sqrt(n_features)))
    if s * s == n_features and a.shape[0] % s == 0 and a.shape[1] % s == 0:
        bs = (a.shape[0] // s, a.shape[1] // s)
        pooled = a.reshape(s, bs[0], s, bs[1]).mean(axis=(1, 3))
        return (pooled / 255.0).ravel()
    from skimage.transform import resize  # imported lazily

    if s * s == n_features:
        return (resize(a, (s, s), anti_aliasing=True) / 255.0).ravel()
    return (resize(a, (1, n_features), anti_aliasing=True) / 255.0).ravel()


def _load_medmnist_split(split: str, n_features: int, limit: Optional[int], root: str,
                         dataset: str = "breastmnist") -> Split:
    import medmnist

    key = dataset.lower()
    if key not in DATASETS:
        raise ValueError(f"unsupported dataset {dataset!r}; expected one of {DATASETS}")
    cls = {"breastmnist": "BreastMNIST", "pneumoniamnist": "PneumoniaMNIST"}[key]
    ds = getattr(medmnist, cls)(split=split, download=True, root=root)
    X, y = [], []
    for img, label in ds:
        X.append(image_to_features(np.array(img), n_features))
        y.append(1.0 if int(np.asarray(label).ravel()[0]) == 1 else -1.0)
        if limit is not None and len(X) >= limit:
            break
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def synthetic_splits(
    n_features: int, sizes: Tuple[int, int, int] = (200, 60, 60), seed: int = 0
) -> Dict[str, object]:
    """Deterministic stand-in dataset: two blocks of mutually correlated features plus noise.

    Built so that the corrected extractor finds small clusters, i.e. so that the offline run
    exercises the same code path as the real one.
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, object] = {"source": "synthetic", "n_features": n_features}
    half = max(1, n_features // 2)
    for name, m in zip(("train", "val", "test"), sizes):
        latent = rng.normal(size=(m, 2))
        base = np.empty((m, n_features))
        for j in range(n_features):
            base[:, j] = latent[:, 0 if j < half else 1]
        X = 1.0 / (1.0 + np.exp(-(base + 0.35 * rng.normal(size=(m, n_features)))))
        y = np.where(latent[:, 0] > 0, 1.0, -1.0)
        out[name] = (X, y)
    return out


def load_splits(
    n_features: int,
    limits: Tuple[Optional[int], Optional[int], Optional[int]] = (None, None, None),
    root: Optional[str] = None,
    allow_fallback: bool = True,
    dataset: str = "breastmnist",
) -> Dict[str, object]:
    """The three official splits of ``dataset``, downsampled to ``n_features`` features.

    ``dataset`` is one of :data:`DATASETS`.  Returns
    ``{"source": ..., "train": (X, y), "val": (X, y), "test": (X, y)}``.
    """
    root = root or os.path.expanduser("~/.medmnist")
    try:
        os.makedirs(root, exist_ok=True)
        out: Dict[str, object] = {"source": dataset.lower(), "n_features": n_features}
        for name, lim in zip(("train", "val", "test"), limits):
            out[name] = _load_medmnist_split(name, n_features, lim, root, dataset=dataset)
        return out
    except Exception as exc:  # pragma: no cover - depends on the environment
        if not allow_fallback:
            raise
        print(f"[data] {dataset} unavailable ({exc!r}); using the synthetic fallback dataset.")
        return synthetic_splits(n_features)
