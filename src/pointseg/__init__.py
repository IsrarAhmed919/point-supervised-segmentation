"""Point-supervised semantic segmentation with partial cross-entropy.

Deliberately torch-free at import time. Scoring and reporting read stored confusion matrices
and need nothing but numpy, so ``pointseg-report`` runs on a laptop with no GPU and no deep
learning stack installed. Import the loss or the model directly when you need them:

    from pointseg.losses import PartialCrossEntropy
    from pointseg.model import build_model
"""
from pointseg.config import CFG, CLASSES, EXPERIMENTS, IGNORE, NC, PALETTE

__version__ = "1.0.0"
__all__ = ["CLASSES", "IGNORE", "NC", "PALETTE", "CFG", "EXPERIMENTS"]
