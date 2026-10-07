"""Partial cross-entropy must behave exactly as the formula says.

These are the correctness guarantees the whole project rests on. If pCE does not reduce to
standard cross-entropy when every pixel is labelled, or if unlabelled pixels leak gradient,
then every number in the README is measuring something other than point supervision.
"""
import pytest
import torch
import torch.nn.functional as F

from pointseg.config import IGNORE, NC
from pointseg.losses import PartialCrossEntropy


@pytest.fixture
def batch():
    torch.manual_seed(0)
    logits = torch.randn(2, NC, 32, 32)
    target = torch.randint(0, NC, (2, 32, 32))
    return logits, target


@pytest.fixture
def sparse(batch):
    """About 3% of pixels labelled, the rest IGNORE: roughly the 20-point regime."""
    logits, target = batch
    torch.manual_seed(1)
    keep = torch.rand(target.shape) < 0.03
    sp = torch.full_like(target, IGNORE)
    sp[keep] = target[keep]
    return logits, target, sp, keep


def test_reduces_to_cross_entropy_when_fully_labelled(batch):
    logits, target = batch
    assert torch.allclose(PartialCrossEntropy()(logits, target),
                          F.cross_entropy(logits, target), atol=1e-6)


def test_equals_cross_entropy_over_labelled_pixels_only(sparse):
    logits, target, sp, keep = sparse
    expected = F.cross_entropy(logits.permute(0, 2, 3, 1)[keep], target[keep])
    assert torch.allclose(PartialCrossEntropy()(logits, sp), expected, atol=1e-6)


def test_unlabelled_pixels_get_exactly_zero_gradient(sparse):
    logits, _, sp, keep = sparse
    x = logits.clone().requires_grad_(True)
    PartialCrossEntropy()(x, sp).backward()
    assert x.grad.abs().sum(1)[~keep].max() == 0


def test_batch_with_no_labels_is_zero_and_finite(batch):
    logits, target = batch
    x = logits.clone().requires_grad_(True)
    value = PartialCrossEntropy()(x, torch.full_like(target, IGNORE))
    value.backward()
    assert value.item() == 0
    assert torch.isfinite(x.grad).all()


def test_focal_with_gamma_zero_equals_plain_pce(sparse):
    logits, _, sp, _ = sparse
    assert torch.allclose(PartialCrossEntropy(gamma=0.0)(logits, sp),
                          PartialCrossEntropy()(logits, sp))


def test_focal_downweights_confident_pixels():
    """gamma>0 must reduce the loss on an already-confident correct pixel."""
    logits = torch.tensor([[[[6.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]]]])
    target = torch.zeros(1, 1, 1, dtype=torch.long)
    plain = PartialCrossEntropy(gamma=0.0)(logits, target)
    focal = PartialCrossEntropy(gamma=2.0)(logits, target)
    assert focal < plain


def test_denominator_is_labelled_count_not_pixel_count(batch):
    """Halving the number of labelled pixels while keeping them identical in value must not
    change the loss. If the denominator were the pixel count, it would."""
    logits, target = batch
    a = torch.full_like(target, IGNORE); a[0, :4, :4] = target[0, :4, :4]
    b = torch.full_like(target, IGNORE); b[0, :4, :4] = target[0, :4, :4]
    b[1, :4, :4] = target[1, :4, :4]
    loss_a = PartialCrossEntropy()(logits, a)
    manual_a = F.cross_entropy(logits[0:1, :, :4, :4], target[0:1, :4, :4])
    assert torch.allclose(loss_a, manual_a, atol=1e-6)
