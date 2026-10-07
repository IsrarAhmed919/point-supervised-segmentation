"""Point sampling must respect the budget and the strategy."""
import numpy as np
import pytest

from pointseg.config import IGNORE
from pointseg.points import label_distribution, sample_points


@pytest.fixture
def mask():
    """64x64 mask: class 0 dominates, class 3 is rare, a corner is no-data."""
    m = np.zeros((64, 64), dtype=np.uint8)
    m[:8, :8] = 1
    m[:4, 20:24] = 2
    m[0, 40] = 3                 # a single pixel of a rare class
    m[60:, 60:] = IGNORE
    return m


def test_uniform_respects_budget(mask):
    ys, xs, labels = sample_points(mask, 20, "uniform", np.random.default_rng(0))
    assert len(ys) == len(xs) == len(labels) == 20


def test_never_samples_ignore(mask):
    for strategy in ("uniform", "class_balanced"):
        ys, xs, _ = sample_points(mask, 100, strategy, np.random.default_rng(0))
        assert not (mask[ys, xs] == IGNORE).any()


def test_labels_match_the_mask(mask):
    ys, xs, labels = sample_points(mask, 30, "uniform", np.random.default_rng(0))
    assert (mask[ys, xs] == labels).all()


def test_class_balanced_covers_every_present_class(mask):
    """The whole point of the strategy: a class present in the image gets at least one point,
    including the one that occupies a single pixel."""
    present = set(np.unique(mask[mask != IGNORE]).tolist())
    _, _, labels = sample_points(mask, 20, "class_balanced", np.random.default_rng(0))
    assert set(labels.tolist()) == present


def test_uniform_usually_misses_the_rare_class(mask):
    """The failure that class-balanced sampling exists to fix. One pixel in ~4000 is almost
    never drawn in a 20-point budget."""
    hits = sum(3 in sample_points(mask, 20, "uniform", np.random.default_rng(s))[2]
               for s in range(50))
    assert hits <= 2


def test_no_duplicate_pixels(mask):
    ys, xs, _ = sample_points(mask, 50, "uniform", np.random.default_rng(0))
    assert len({(int(a), int(b)) for a, b in zip(ys, xs)}) == len(ys)


def test_budget_caps_at_available_pixels():
    tiny = np.zeros((3, 3), dtype=np.uint8)
    ys, _, _ = sample_points(tiny, 100, "uniform", np.random.default_rng(0))
    assert len(ys) == 9


def test_all_ignore_returns_empty():
    ys, xs, labels = sample_points(np.full((8, 8), IGNORE, np.uint8), 10, "uniform",
                                   np.random.default_rng(0))
    assert len(ys) == len(xs) == len(labels) == 0


def test_unknown_strategy_raises(mask):
    with pytest.raises(ValueError):
        sample_points(mask, 10, "nonsense", np.random.default_rng(0))


def test_label_distribution_counts(mask):
    pts = {"a": sample_points(mask, 20, "class_balanced", np.random.default_rng(0))}
    counts = label_distribution(pts, 7)
    assert counts.sum() == len(pts["a"][2])
