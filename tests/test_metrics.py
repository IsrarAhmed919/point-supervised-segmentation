"""Metrics, and the integrity check that makes the published scores verifiable."""
import numpy as np
import pytest

from pointseg.config import RESULTS_JSON
from pointseg.metrics import iou_per_class, normalize_confusion, scores


def test_perfect_prediction_scores_one():
    cm = np.diag([10, 20, 30]).astype(np.int64)
    s = scores(cm)
    assert s["mIoU"] == pytest.approx(1.0)
    assert s["OA"] == pytest.approx(1.0)


def test_absent_class_is_nan_not_zero():
    """A class in neither truth nor prediction must not drag the mean down."""
    cm = np.zeros((3, 3), dtype=np.int64)
    cm[0, 0] = 10
    cm[1, 1] = 10
    assert np.isnan(iou_per_class(cm)[2])
    assert scores(cm)["mIoU"] == pytest.approx(1.0)


def test_oa_can_be_high_while_miou_is_low():
    """The reason mIoU is the headline metric: a model that predicts only the dominant class
    scores well on accuracy and badly on mIoU."""
    cm = np.array([[990, 0], [10, 0]], dtype=np.int64)
    s = scores(cm)
    assert s["OA"] > 0.98
    assert s["mIoU"] < 0.55


def test_row_normalised_confusion_rows_sum_to_one():
    cm = np.array([[5, 5], [2, 8]], dtype=np.int64)
    assert np.allclose(normalize_confusion(cm).sum(axis=1), 1.0)


@pytest.mark.skipif(not RESULTS_JSON.exists(), reason="results.json not present")
def test_published_scores_match_their_confusion_matrices():
    """Guards the published numbers: every stored mIoU and OA must be reproducible from the
    stored confusion matrix."""
    from pointseg.report import load_results
    results = load_results(RESULTS_JSON, verify=True)
    assert len(results) >= 1
