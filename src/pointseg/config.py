"""Dataset constants and training configuration.

LoveDA stores masks with 0 = no-data and 1..7 for the seven land-cover classes. Everything
downstream expects 0..6 for the classes and 255 for "ignore", so the mapping happens once, in
``data.read_mask``, and nothing else in the codebase has to remember it.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- dataset
CLASSES: list[str] = [
    "background", "building", "road", "water", "barren", "forest", "agriculture",
]
NC: int = len(CLASSES)

#: Label used for every pixel the loss must ignore: LoveDA no-data, and, under point
#: supervision, every pixel that was not sampled.
IGNORE: int = 255

#: Official LoveDA colours, so figures here match the ones in the dataset papers.
PALETTE = np.array(
    [[255, 255, 255],   # background
     [255, 0, 0],       # building
     [255, 255, 0],     # road
     [0, 0, 255],       # water
     [159, 129, 183],   # barren
     [0, 255, 0],       # forest
     [255, 195, 128]],  # agriculture
    dtype=np.uint8,
)

# ImageNet statistics, because the encoder is ImageNet-pretrained.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --------------------------------------------------------------------------- paths
#: Repository root. Override with POINTSEG_ROOT when the data lives elsewhere.
ROOT = Path(os.environ.get("POINTSEG_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.environ.get("POINTSEG_DATA", ROOT / "data"))
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CHECKPOINT_DIR = ROOT / "checkpoints"

RESULTS_JSON = RESULTS_DIR / "results.json"
PREDS_NPZ = RESULTS_DIR / "results.preds.npz"

# --------------------------------------------------------------------------- training
#: Shared by every experiment. Holding these fixed is what makes the runs comparable:
#: the only thing that varies between experiments is supervision, point budget, sampling
#: strategy and the focal gamma.
CFG: dict = dict(
    size=512,            # training crop; evaluation runs at native 1024
    batch_size=8,
    epochs=15,
    lr=5e-4,
    weight_decay=1e-4,
    num_workers=6,
    encoder="resnet50",
    val_every=5,         # mid-training validation cadence, in epochs
    val_subset=200,      # images used for that cheap mid-training check
    seed=42,
)

#: The eight runs reported in the README. ``supervision="full"`` is the dense-mask upper
#: bound; everything else is point-supervised.
EXPERIMENTS: list[dict] = [
    dict(name="full_mask", supervision="full"),
    dict(name="pts5", n=5),
    dict(name="pts20", n=20),
    dict(name="pts100", n=100),
    dict(name="pts500", n=500),
    dict(name="pts20_focal", n=20, gamma=2.0),
    dict(name="pts20_balanced", n=20, strategy="class_balanced"),
    dict(name="pts20_balanced_focal", n=20, strategy="class_balanced", gamma=2.0),
]
