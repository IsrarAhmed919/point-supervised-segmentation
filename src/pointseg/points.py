"""Simulating point labels from dense masks.

The premise of this work is that dense masks are expensive and point labels are cheap. A LoveDA
mask has 1,048,576 labelled pixels. An annotator clicking twenty points produces 20. This module
simulates that annotator, and the choice of *which* twenty pixels turns out to matter more than
how many.

Two strategies, meant to model two different annotators:

``uniform``
    N pixels drawn uniformly at random from the valid region. Each class therefore receives
    points roughly in proportion to its **area**. This is the honest model of "click anywhere",
    and on a dataset as imbalanced as LoveDA it means rare classes are frequently never labelled
    at all.

``class_balanced``
    N split evenly across the classes **present in that image**. This models an annotator told
    to cover every class they can see. The budget is identical; only its allocation changes.

That distinction is the single highest-leverage variable measured in this project.
"""
from __future__ import annotations

import numpy as np

from pointseg.config import IGNORE

# tqdm and pointseg.data (cv2, torch) are imported lazily inside build_point_labels. The
# sampling logic itself is pure numpy, and keeping it importable without the deep learning
# stack is what lets the sampling tests run anywhere.

Points = tuple[np.ndarray, np.ndarray, np.ndarray]  # (ys, xs, labels)


def sample_points(mask: np.ndarray, n: int, strategy: str, rng: np.random.Generator) -> Points:
    """Draw ``n`` labelled pixels from one dense mask.

    Returns ``(ys, xs, labels)``. No-data pixels are never sampled. Where a class has fewer
    pixels than its share of the budget, it contributes everything it has and the total comes
    in under ``n`` rather than sampling with replacement, which would duplicate supervision
    and silently reweight the loss.
    """
    flat = mask.ravel()
    valid = np.flatnonzero(flat != IGNORE)
    if valid.size == 0:
        empty = np.empty(0, int)
        return empty, empty, empty

    if strategy == "uniform":
        idx = rng.choice(valid, size=min(n, valid.size), replace=False)

    elif strategy == "class_balanced":
        classes = np.unique(flat[valid])
        # Split n as evenly as possible, giving the remainder to the first few classes, and
        # guarantee every present class at least one point.
        k = np.full(len(classes), n // len(classes))
        k[: n % len(classes)] += 1
        k = np.maximum(k, 1)
        idx = np.concatenate([
            rng.choice(np.flatnonzero(flat == c), size=min(kk, int((flat == c).sum())),
                       replace=False)
            for c, kk in zip(classes, k)
        ])
    else:
        raise ValueError(f"unknown strategy {strategy!r}, expected 'uniform' or 'class_balanced'")

    ys, xs = np.unravel_index(idx, mask.shape)
    return ys, xs, flat[idx].astype(int)


def build_point_labels(items: list[dict], n: int, strategy: str, seed: int = 0,
                       show_progress: bool = True) -> dict[str, Points]:
    """Point labels for a whole split: ``{item_id: (ys, xs, labels)}``.

    Generated once from a seeded generator and reused for every epoch, so the simulated
    annotator does not get a fresh set of clicks each time the image is seen. That would be
    a slow leak of the dense mask back into training and would overstate point supervision.
    """
    from tqdm.auto import tqdm

    from pointseg.data import read_mask

    rng = np.random.default_rng(seed)
    it = tqdm(items, desc=f"points {strategy} N={n}", leave=False, disable=not show_progress)
    return {item["id"]: sample_points(read_mask(item["mask"]), n, strategy, rng) for item in it}


def label_distribution(points: dict[str, Points], num_classes: int) -> np.ndarray:
    """How many points each class received across a split. Used by the EDA notebook to show,
    concretely, which classes a uniform annotator simply never labels."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, _, labels in points.values():
        if len(labels):
            counts += np.bincount(labels, minlength=num_classes)
    return counts
