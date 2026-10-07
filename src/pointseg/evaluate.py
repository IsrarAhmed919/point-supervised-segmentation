"""Evaluation, separated from training.

Scoring is always against the **full dense validation masks at native 1024x1024**, for every run
including the point-supervised ones. Point labels are a property of training only. Evaluating a
weakly supervised model on anything less than complete ground truth would make the headline
number meaningless.

    pointseg-evaluate --checkpoint checkpoints/pts20_balanced_focal.pth
    pointseg-evaluate --all-checkpoints
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pointseg.config import CFG, CHECKPOINT_DIR, IGNORE, NC, RESULTS_DIR
from pointseg.data import LoveDA
from pointseg.metrics import scores
from pointseg.model import build_model


def confusion(model: torch.nn.Module, items: list[dict], device: torch.device,
              batch_size: int = 2, amp: bool = True) -> np.ndarray:
    """Accumulate the confusion matrix over a split.

    Accumulated with ``bincount`` on GPU-free integer tensors rather than by storing predictions,
    so memory stays flat regardless of split size. ``IGNORE`` pixels are dropped, which is what
    keeps LoveDA no-data regions from counting as either correct or incorrect.
    """
    loader = DataLoader(LoveDA(items, train=False), batch_size=batch_size,
                        num_workers=CFG["num_workers"], pin_memory=True)
    cm = torch.zeros(NC, NC, dtype=torch.long)
    model.eval()
    use_amp = amp and device.type == "cuda"
    with torch.no_grad():
        for x, y in tqdm(loader, desc="evaluating", leave=False):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                pred = model(x.to(device)).argmax(1).cpu()
            k = y != IGNORE
            cm += torch.bincount(y[k] * NC + pred[k], minlength=NC * NC).reshape(NC, NC)
    return cm.numpy()


def example_indices(items: list[dict], k: int = 4) -> list[int]:
    """The same handful of test images for every run, so the qualitative figure compares
    like with like."""
    return sorted(set(np.linspace(0, len(items) - 1, k).astype(int).tolist()))


def predict_examples(model: torch.nn.Module, items: list[dict], device: torch.device,
                     k: int = 4) -> list[np.ndarray]:
    model.eval()
    out = []
    with torch.no_grad():
        for i in example_indices(items, k):
            x, _ = LoveDA(items, train=False)[i]
            out.append(model(x[None].to(device)).argmax(1)[0].cpu().numpy().astype(np.uint8))
    return out


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(path, map_location=device)
    model = build_model(encoder_weights=None).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def main() -> None:
    from pointseg.data import list_split

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, help="one .pth to evaluate")
    ap.add_argument("--all-checkpoints", action="store_true",
                    help="evaluate every .pth in checkpoints/")
    ap.add_argument("--split", default="Val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "evaluation.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = (sorted(CHECKPOINT_DIR.glob("*.pth")) if args.all_checkpoints
             else [args.checkpoint] if args.checkpoint else [])
    if not paths:
        ap.error("pass --checkpoint PATH or --all-checkpoints.\n"
                 "No weights yet? The published scores are already in results/results.json; "
                 "run `pointseg-report` to reproduce every number from them without a GPU.")

    items = list_split(args.split)
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} images from {args.split}, device {device}")

    out = {}
    for path in paths:
        model, ckpt = load_checkpoint(path, device)
        cm = confusion(model, items, device)
        s = scores(cm)
        out[ckpt.get("name", path.stem)] = dict(cm=cm.tolist(), **s)
        print(f"{path.stem:24} mIoU {s['mIoU']:.4f}  OA {s['OA']:.4f}  fwIoU {s['fwIoU']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
