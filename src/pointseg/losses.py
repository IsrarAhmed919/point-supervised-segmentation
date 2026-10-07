r"""Partial cross-entropy, the loss that makes point supervision work.

Standard cross-entropy assumes every pixel carries a label. Under point supervision almost none
do, so the question is what to do with the unlabelled ones. Partial cross-entropy answers it in
the simplest possible way: **average the loss over the labelled pixels and ignore the rest.**

.. math::

    \mathrm{pCE} = \frac{\sum_i \ell_i M_i}{\sum_i M_i},
    \qquad M_i = \begin{cases} 1 & \text{pixel } i \text{ is labelled} \\
                               0 & \text{otherwise} \end{cases}

with :math:`\ell_i = -\log p_{i,y_i}`, or the focal variant
:math:`\ell_i = -(1 - p_{i,y_i})^{\gamma}\log p_{i,y_i}`.

Two consequences worth being explicit about, because they are the whole point:

* **Unlabelled pixels receive exactly zero gradient.** They are not pushed toward a background
  class, not pseudo-labelled, not penalised. The network is simply told nothing about them, and
  everything it predicts there comes from what it generalises off the labelled pixels.
* **The denominator is the labelled count, not the pixel count.** Dividing by the full pixel
  count would scale the loss by the labelling density and make the learning rate implicitly
  depend on the point budget, so a 5-point run and a 500-point run would not be comparable.

Why the focal option is here: under ``class_balanced`` sampling a rare class gets as many points
as a common one, but those points are often harder. :math:`\gamma > 0` down-weights pixels the
network already predicts confidently, which concentrates the remaining budget on the ones it
does not. The experiments measure whether that helps, and it interacts with sampling strategy
rather than acting independently.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointseg.config import IGNORE


class PartialCrossEntropy(nn.Module):
    """Cross-entropy restricted to labelled pixels, with an optional focal term.

    Parameters
    ----------
    ignore_index:
        Label marking an unlabelled pixel. Defaults to :data:`pointseg.config.IGNORE`.
    gamma:
        Focal exponent. ``0.0`` is plain partial cross-entropy; ``2.0`` is the value used in
        the focal experiments.

    Notes
    -----
    With every pixel labelled and ``gamma=0`` this is numerically identical to
    :func:`torch.nn.functional.cross_entropy`, which :mod:`tests.test_losses` asserts.
    """

    def __init__(self, ignore_index: int = IGNORE, gamma: float = 0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target != self.ignore_index
        if not mask.any():
            # No labelled pixel anywhere in the batch. Return a real zero that is still part of
            # the autograd graph, so the step is a no-op instead of a crash or a NaN.
            return logits.sum() * 0.0

        # Gather needs a valid class index everywhere, including at pixels about to be masked
        # out. The value is irrelevant because the mask zeroes their contribution.
        t = target.masked_fill(~mask, 0)

        # log_softmax in float32 even under autocast: computing this in fp16 is the usual
        # source of NaNs in a mixed-precision segmentation loss.
        logp = F.log_softmax(logits.float(), dim=1).gather(1, t.unsqueeze(1)).squeeze(1)
        loss = -logp

        if self.gamma > 0:
            # (1 - p)^gamma, with p recovered as exp(logp). clamp guards against a tiny
            # negative from floating point before the fractional power.
            loss = loss * (1 - logp.exp()).clamp(min=0).pow(self.gamma)

        m = mask.float()
        return (loss * m).sum() / m.sum()

    def extra_repr(self) -> str:
        return f"ignore_index={self.ignore_index}, gamma={self.gamma}"
