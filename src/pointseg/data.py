"""Reading LoveDA, and the Dataset that turns it into training tensors.

Images are read **straight out of the two zip archives**. LoveDA unpacks to about 6.4 GB and
nothing here needs the files on disk, so unzipping only costs space and time. ``zipfile`` handles
are per-process rather than global, because each DataLoader worker is a separate process and a
shared handle is not safe across them.
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pointseg.config import DATA_DIR, IGNORE, MEAN, STD, PALETTE

_ZIP_HANDLES: dict[tuple[str, int], zipfile.ZipFile] = {}


def zip_of(split: str) -> zipfile.ZipFile:
    """One open archive handle per (split, process)."""
    key = (split, os.getpid())
    if key not in _ZIP_HANDLES:
        path = DATA_DIR / f"{split}.zip"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. See the README: the two LoveDA archives go in {DATA_DIR}/"
            )
        _ZIP_HANDLES[key] = zipfile.ZipFile(path)
    return _ZIP_HANDLES[key]


def read_png(member: str, flag: int) -> np.ndarray | None:
    buf = np.frombuffer(zip_of(member.split("/")[0]).read(member), np.uint8)
    return cv2.imdecode(buf, flag)


def list_split(split: str) -> list[dict]:
    """Every image in a split, with its mask path, sorted and tagged by domain.

    The Rural/Urban split is kept on each item because the domain gap is one of the things
    that makes LoveDA hard, and the EDA notebook reports per-domain statistics.
    """
    names = set(zip_of(split).namelist())
    items: list[dict] = []
    for domain in ("Rural", "Urban"):
        prefix = f"{split}/{domain}/images_png/"
        imgs = sorted(
            (n for n in names if n.startswith(prefix) and n.endswith(".png")),
            key=lambda n: int(Path(n).stem),
        )
        for n in imgs:
            items.append(dict(
                id=f"{domain}_{Path(n).stem}",
                domain=domain,
                image=n,
                mask=n.replace("/images_png/", "/masks_png/"),
            ))
    return items


def read_image(path: str) -> np.ndarray:
    return cv2.cvtColor(read_png(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def read_mask(path: str) -> np.ndarray:
    """LoveDA mask to training labels: 0 = no-data becomes IGNORE, 1..7 become 0..6."""
    raw = read_png(path, cv2.IMREAD_GRAYSCALE).astype(np.int16)
    m = raw - 1
    m[raw == 0] = IGNORE
    return m.astype(np.uint8)


def colorize(mask: np.ndarray) -> np.ndarray:
    """Label map to RGB using the official palette. IGNORE renders black."""
    out = np.zeros((*mask.shape, 3), np.uint8)
    valid = mask != IGNORE
    out[valid] = PALETTE[mask[valid]]
    return out


def normalize(im: np.ndarray) -> torch.Tensor:
    """HWC uint8 RGB to normalised CHW float tensor."""
    arr = ((im.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(arr))


class LoveDA(Dataset):
    """LoveDA for training or evaluation.

    Training returns a 512x512 crop with augmentation. Evaluation returns the **native 1024x1024
    image and its full dense mask**, because the point labels are a property of training only.
    Every run, point-supervised or not, is scored against complete ground truth.

    Parameters
    ----------
    items:
        Output of :func:`list_split`.
    points:
        ``{item_id: (ys, xs, labels)}`` from :mod:`pointseg.points`. ``None`` means full-mask
        supervision, which is the dense upper bound this work is measured against.
    """

    def __init__(self, items: list[dict], points: dict | None = None,
                 train: bool = True, size: int = 512):
        self.items, self.points, self.train, self.size = items, points, train, size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        im = read_image(it["image"])

        if not self.train:
            return normalize(im), torch.from_numpy(read_mask(it["mask"]).astype(np.int64))

        h, w = im.shape[:2]
        s = self.size
        im = cv2.resize(im, (s, s), interpolation=cv2.INTER_AREA)

        if self.points is None:
            # Full-mask supervision. INTER_NEAREST because labels must not be interpolated.
            t = cv2.resize(read_mask(it["mask"]), (s, s), interpolation=cv2.INTER_NEAREST)
        else:
            # Point supervision: start from an all-IGNORE canvas and write back only the
            # sampled pixels, rescaled to the training resolution. Everything else stays
            # unlabelled, and the loss will skip it.
            ys, xs, labels = self.points[it["id"]]
            t = np.full((s, s), IGNORE, np.uint8)
            t[ys * s // h, xs * s // w] = labels

        # Augmentation limited to the dihedral group: rotations and flips move the points with
        # the image and introduce no interpolation, so a sparse label map survives intact.
        k = np.random.randint(4)
        im, t = np.rot90(im, k), np.rot90(t, k)
        if np.random.rand() < 0.5:
            im, t = im[:, ::-1], t[:, ::-1]

        return normalize(np.ascontiguousarray(im)), torch.from_numpy(
            np.ascontiguousarray(t).astype(np.int64))
