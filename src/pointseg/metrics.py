"""Segmentation metrics, all derived from one confusion matrix.

Everything reported in this project is a function of the confusion matrix, which is why the raw
matrices are stored in ``results/results.json``: any metric can be recomputed later without
re-running inference, and the reported numbers can be independently verified rather than taken
on trust. ``pointseg.report`` does exactly that check on every load.

On the choice of headline metric. **Overall accuracy is the wrong one for LoveDA** and quoting it
alone would be misleading. The dataset is badly imbalanced, background alone is over half of all
pixels, so a model can score well on OA while failing completely on the classes anyone cares
about. In these results the run with the *highest* OA has nearly the *lowest* mIoU. mIoU averages
over classes rather than pixels and so refuses to let a large class hide a small one, which is
why it is the number this work is judged on. Both are reported, side by side, always.
"""
from __future__ import annotations

import numpy as np


def confusion_from_pair(target: np.ndarray, pred: np.ndarray, num_classes: int,
                        ignore_index: int = 255) -> np.ndarray:
    """Confusion matrix for one image pair. Rows are ground truth, columns are prediction."""
    k = target != ignore_index
    return np.bincount(
        target[k].astype(np.int64) * num_classes + pred[k].astype(np.int64),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)


def iou_per_class(cm: np.ndarray) -> np.ndarray:
    """Intersection over union per class. Classes absent from both truth and prediction
    give NaN rather than a misleading zero, and are excluded from the mean."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - tp
    return np.where(union > 0, tp / np.maximum(union, 1.0), np.nan)


def scores(cm: np.ndarray) -> dict:
    """Every headline metric for one confusion matrix.

    Returns mIoU, overall accuracy, frequency-weighted IoU, macro precision/recall/F1, plus the
    per-class IoU, precision and recall vectors.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    gt = cm.sum(axis=1)       # ground-truth pixels per class
    pr = cm.sum(axis=0)       # predicted pixels per class
    total = cm.sum()

    iou = iou_per_class(cm)
    precision = np.where(pr > 0, tp / np.maximum(pr, 1.0), np.nan)
    recall = np.where(gt > 0, tp / np.maximum(gt, 1.0), np.nan)
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), np.nan)

    freq = gt / max(total, 1.0)
    fw_iou = float(np.nansum(freq * np.nan_to_num(iou)))

    return dict(
        mIoU=float(np.nanmean(iou)),
        OA=float(tp.sum() / max(total, 1.0)),
        fwIoU=fw_iou,
        mPrecision=float(np.nanmean(precision)),
        mRecall=float(np.nanmean(recall)),
        mF1=float(np.nanmean(f1)),
        IoU=iou.tolist(),
        precision=precision.tolist(),
        recall=recall.tolist(),
    )


def normalize_confusion(cm: np.ndarray) -> np.ndarray:
    """Row-normalised confusion matrix: of the pixels truly in class i, the fraction predicted
    as class j. Row-normalising rather than column-normalising is what makes the characteristic
    failure visible, namely a rare class whose row drains almost entirely into background."""
    cm = np.asarray(cm, dtype=np.float64)
    return cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
