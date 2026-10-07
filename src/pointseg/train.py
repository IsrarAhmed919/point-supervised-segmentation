"""Training. One experiment per invocation, or the whole grid.

Unlike the original notebook this **saves a checkpoint per run**, so evaluation is a separate
step that can be repeated without retraining. That is the main structural difference between
this package and the exploratory notebook it came from.

    pointseg-train --all
    pointseg-train --name pts20_balanced_focal
    pointseg-train --name smoke --points 20 --strategy class_balanced --gamma 2.0 --epochs 1
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pointseg.config import (CFG, CHECKPOINT_DIR, EXPERIMENTS, NC, PREDS_NPZ, RESULTS_DIR,
                             RESULTS_JSON)
from pointseg.data import LoveDA, list_split
from pointseg.evaluate import confusion, predict_examples
from pointseg.losses import PartialCrossEntropy
from pointseg.metrics import scores
from pointseg.model import build_model
from pointseg.points import build_point_labels

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one(name: str, train_items: list[dict], test_items: list[dict],
              supervision: str = "points", n: int = 20, strategy: str = "uniform",
              gamma: float = 0.0, epochs: int | None = None,
              save_checkpoint: bool = True) -> tuple[dict, list[np.ndarray]]:
    """Train a single configuration and evaluate it on the full test split."""
    seed_all(CFG["seed"])
    epochs = epochs or CFG["epochs"]

    points = (build_point_labels(train_items, n, strategy, seed=0)
              if supervision == "points" else None)

    loader = DataLoader(
        LoveDA(train_items, points, train=True, size=CFG["size"]),
        batch_size=CFG["batch_size"], shuffle=True, drop_last=True,
        num_workers=CFG["num_workers"], pin_memory=True, persistent_workers=True,
    )

    model = build_model().to(DEVICE)
    criterion = PartialCrossEntropy(gamma=gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                                  weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CFG["lr"], total_steps=epochs * len(loader), pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    # Cheap mid-training signal on a subset; the real number comes from the full split at the end.
    val_items = test_items[:: max(1, len(test_items) // CFG["val_subset"])]
    history, t0 = [], time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in tqdm(loader, desc=f"{name} epoch {epoch}/{epochs}", leave=False):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()

        row = dict(epoch=epoch, loss=running / len(loader))
        if epoch % CFG["val_every"] == 0 or epoch == epochs:
            row["val_mIoU"] = scores(confusion(model, val_items, DEVICE))["mIoU"]
        history.append(row)
        print(name, row, f"{(time.time() - t0) / 60:.1f} min", flush=True)

    cm = confusion(model, test_items, DEVICE)
    preds = predict_examples(model, test_items, DEVICE)

    if save_checkpoint:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"name": name, "state_dict": model.state_dict(), "config": CFG,
                    "supervision": supervision, "n": n, "strategy": strategy, "gamma": gamma},
                   CHECKPOINT_DIR / f"{name}.pth")

    result = dict(name=name, supervision=supervision, n=n, strategy=strategy, gamma=gamma,
                  history=history, cm=cm.tolist(), minutes=(time.time() - t0) / 60,
                  **{k: v for k, v in scores(cm).items()
                     if k in ("mIoU", "OA", "IoU")})
    return result, preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="run every experiment in config.EXPERIMENTS")
    ap.add_argument("--name", help="run one named experiment, or label a custom run")
    ap.add_argument("--supervision", choices=["points", "full"], default="points")
    ap.add_argument("--points", type=int, default=20, dest="n")
    ap.add_argument("--strategy", choices=["uniform", "class_balanced"], default="uniform")
    ap.add_argument("--gamma", type=float, default=0.0, help="focal exponent; 0 = plain pCE")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap training images, for a smoke test")
    ap.add_argument("--force", action="store_true", help="rerun even if the name is in results.json")
    args = ap.parse_args()

    train_items, test_items = list_split("Train"), list_split("Val")
    if args.limit:
        train_items, test_items = train_items[: args.limit], test_items[: args.limit]
    print(f"train {len(train_items)} images, test {len(test_items)} images, device {DEVICE}")

    if args.all:
        queue = list(EXPERIMENTS)
    elif args.name and any(e["name"] == args.name for e in EXPERIMENTS):
        queue = [e for e in EXPERIMENTS if e["name"] == args.name]
    else:
        queue = [dict(name=args.name or "custom", supervision=args.supervision,
                      n=args.n, strategy=args.strategy, gamma=args.gamma)]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else {}
    preds = dict(np.load(PREDS_NPZ)) if PREDS_NPZ.exists() else {}

    for exp in queue:
        exp = dict(exp)
        name = exp.pop("name")
        if name in results and not args.force:
            print(f"already done, skipping: {name}")
            continue
        exp.setdefault("supervision", "points")
        result, pr = train_one(name, train_items, test_items, epochs=args.epochs, **exp)
        results[name] = result
        for i, p in enumerate(pr):
            preds[f"{name}__{i}"] = p
        RESULTS_JSON.write_text(json.dumps(results))
        np.savez_compressed(PREDS_NPZ, **preds)
        print(f"==> {name}: mIoU {result['mIoU']:.4f}  OA {result['OA']:.4f} "
              f"({result['minutes']:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
