"""Scoring and reporting, reproducible without a GPU.

Every number in the README comes from this module. It reads the stored confusion matrices in
``results/results.json`` and recomputes each metric from scratch, so nothing is taken on trust:
the first thing it does is assert that the recomputed mIoU and overall accuracy match the values
recorded at training time. A confusion matrix fully determines every metric here, which is why
storing it rather than just the summary numbers was worth doing.

    pointseg-report                 # tables to stdout, markdown and figures to results/
    pointseg-report --no-figures    # tables only, no matplotlib
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pointseg.config import CLASSES, FIGURES_DIR, NC, PALETTE, PREDS_NPZ, RESULTS_DIR, RESULTS_JSON
from pointseg.metrics import normalize_confusion, scores

ORDER = ["full_mask", "pts5", "pts20", "pts100", "pts500",
         "pts20_focal", "pts20_balanced", "pts20_balanced_focal"]


def load_results(path: Path = RESULTS_JSON, verify: bool = True) -> dict:
    """Load results and re-derive every metric from the confusion matrices.

    ``verify`` re-checks the stored mIoU and OA against the recomputation and raises if they
    disagree, which would mean the file had been edited or produced by different code.
    """
    raw = json.loads(Path(path).read_text())
    out = {}
    for name, run in raw.items():
        cm = np.array(run["cm"], dtype=np.int64)
        s = scores(cm)
        if verify:
            for key in ("mIoU", "OA"):
                if key in run and not np.isclose(s[key], run[key], atol=1e-9):
                    raise ValueError(
                        f"{name}: stored {key}={run[key]!r} does not match {s[key]!r} "
                        f"recomputed from the confusion matrix")
        out[name] = {**run, **s, "cm": cm}
    return out


def ordered(results: dict) -> list[str]:
    known = [n for n in ORDER if n in results]
    return known + [n for n in results if n not in ORDER]


def _supervision_label(run: dict) -> tuple[str, str, str]:
    if run["supervision"] == "full":
        return "all 1,048,576", "dense mask", "CE"
    loss = f"pCE, focal gamma={run['gamma']:g}" if run["gamma"] else "pCE"
    return f"{run['n']}", run["strategy"].replace("_", "-"), loss


def summary_table(results: dict) -> str:
    """The headline table: one row per run."""
    head = ("| run | points / image | sampling | loss | mIoU | OA | fwIoU | mF1 | min |\n"
            "|---|---:|---|---|---:|---:|---:|---:|---:|\n")
    rows = []
    for name in ordered(results):
        r = results[name]
        pts, strat, loss = _supervision_label(r)
        rows.append(
            f"| `{name}` | {pts} | {strat} | {loss} | **{r['mIoU']:.4f}** | {r['OA']:.4f} | "
            f"{r['fwIoU']:.4f} | {r['mF1']:.4f} | {r.get('minutes', float('nan')):.0f} |")
    return head + "\n".join(rows)


def per_class_table(results: dict) -> str:
    """Per-class IoU. This is where point supervision visibly succeeds or fails, because the
    mean hides which class moved."""
    head = "| run | " + " | ".join(CLASSES) + " |\n" + "|---" * (NC + 1) + "|\n"
    rows = []
    for name in ordered(results):
        iou = results[name]["IoU"]
        rows.append(f"| `{name}` | " + " | ".join(f"{v:.3f}" for v in iou) + " |")
    return head + "\n".join(rows)


def budget_table(results: dict) -> str:
    """Experiment 1: how performance moves with the point budget, uniform sampling, plain pCE."""
    runs = [n for n in ordered(results)
            if results[n]["supervision"] == "points"
            and results[n]["strategy"] == "uniform" and not results[n]["gamma"]]
    runs.sort(key=lambda n: results[n]["n"])
    full = results.get("full_mask", {}).get("mIoU")
    head = ("| points / image | labelled fraction | mIoU | OA | % of dense-mask mIoU |\n"
            "|---:|---:|---:|---:|---:|\n")
    rows = []
    for n in runs:
        r = results[n]
        frac = r["n"] / (1024 * 1024)
        pct = f"{100 * r['mIoU'] / full:.1f}%" if full else "n/a"
        rows.append(f"| {r['n']} | {frac:.6%} | {r['mIoU']:.4f} | {r['OA']:.4f} | {pct} |")
    if full:
        rows.append(f"| all (1,048,576) | 100% | {full:.4f} | "
                    f"{results['full_mask']['OA']:.4f} | 100.0% |")
    return head + "\n".join(rows)


def sampling_loss_table(results: dict) -> str:
    """Experiment 2: sampling strategy crossed with loss, at a fixed budget of 20 points."""
    grid = {("uniform", 0.0): "pts20", ("uniform", 2.0): "pts20_focal",
            ("class_balanced", 0.0): "pts20_balanced",
            ("class_balanced", 2.0): "pts20_balanced_focal"}
    head = "| sampling | pCE (gamma=0) | focal pCE (gamma=2) |\n|---|---:|---:|\n"
    rows = []
    for strat in ("uniform", "class_balanced"):
        cells = []
        for gamma in (0.0, 2.0):
            name = grid.get((strat, gamma))
            cells.append(f"{results[name]['mIoU']:.4f}" if name in results else "n/a")
        rows.append(f"| {strat.replace('_', '-')} | " + " | ".join(cells) + " |")
    return head + "\n".join(rows)


# --------------------------------------------------------------------------- figures
def make_figures(results: dict, out_dir: Path = FIGURES_DIR) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "axes.grid": False, "font.size": 10})
    written: list[Path] = []
    names = ordered(results)

    # 1. mIoU and OA per run -------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    colors = ["#2e8b57" if results[n]["supervision"] == "full" else "#2f6fb2" for n in names]
    for a, key, title in ((ax[0], "mIoU", "mean IoU (the metric that matters)"),
                          (ax[1], "OA", "overall accuracy (flattered by class imbalance)")):
        vals = [results[n][key] for n in names]
        bars = a.bar(range(len(names)), vals, color=colors)
        a.set_xticks(range(len(names)))
        a.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
        a.set_title(title)
        a.set_ylim(0, max(vals) * 1.22)
        for b, v in zip(bars, vals):
            a.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:.3f}",
                   ha="center", fontsize=8, fontweight="bold")
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", alpha=.25)
    fig.tight_layout(); p = out_dir / "01_scores_per_run.png"; fig.savefig(p); plt.close(fig)
    written.append(p)

    # 2. budget curve --------------------------------------------------------
    pts = sorted([results[n] for n in names
                  if results[n]["supervision"] == "points"
                  and results[n]["strategy"] == "uniform" and not results[n]["gamma"]],
                 key=lambda r: r["n"])
    if pts:
        fig, a = plt.subplots(figsize=(7, 4.4))
        a.plot([r["n"] for r in pts], [r["mIoU"] for r in pts], "o-", lw=2,
               color="#2f6fb2", label="point supervision, uniform, pCE")
        for r in pts:
            a.annotate(f"{r['mIoU']:.3f}", (r["n"], r["mIoU"]),
                       textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        if "full_mask" in results:
            a.axhline(results["full_mask"]["mIoU"], ls="--", c="#2e8b57",
                      label=f"dense masks ({results['full_mask']['mIoU']:.3f})")
        if "pts20_balanced_focal" in results:
            a.axhline(results["pts20_balanced_focal"]["mIoU"], ls=":", c="#c0392b",
                      label=f"20 pts, balanced + focal "
                            f"({results['pts20_balanced_focal']['mIoU']:.3f})")
        a.set_xscale("log")
        a.set_xlabel("labelled pixels per image (log scale, out of 1,048,576)")
        a.set_ylabel("test mIoU")
        a.set_title("More points help, but how you choose them helps more")
        a.legend(fontsize=8); a.grid(alpha=.25)
        a.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); p = out_dir / "02_budget_curve.png"; fig.savefig(p); plt.close(fig)
        written.append(p)

    # 3. per-class IoU heatmap ----------------------------------------------
    mat = np.array([results[n]["IoU"] for n in names], dtype=float)
    fig, a = plt.subplots(figsize=(10, 0.52 * len(names) + 2.2))
    im = a.imshow(mat, cmap="viridis", vmin=0, vmax=np.nanmax(mat), aspect="auto")
    a.set_xticks(range(NC)); a.set_xticklabels(CLASSES, rotation=30, ha="right")
    a.set_yticks(range(len(names))); a.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(NC):
            a.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                   color="white" if mat[i, j] < np.nanmax(mat) * 0.6 else "black")
    a.set_title("Per-class IoU: the mean hides which class actually moved")
    fig.colorbar(im, ax=a, fraction=.025)
    fig.tight_layout(); p = out_dir / "03_per_class_iou.png"; fig.savefig(p); plt.close(fig)
    written.append(p)

    # 4. confusion matrices --------------------------------------------------
    show = [n for n in ("pts20", "pts20_balanced_focal", "full_mask") if n in results]
    if show:
        fig, axes = plt.subplots(1, len(show), figsize=(5.4 * len(show), 4.8))
        axes = np.atleast_1d(axes)
        for a, n in zip(axes, show):
            cmn = normalize_confusion(results[n]["cm"])
            a.imshow(cmn, cmap="magma", vmin=0, vmax=1)
            a.set_xticks(range(NC)); a.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
            a.set_yticks(range(NC)); a.set_yticklabels(CLASSES, fontsize=7)
            a.set_title(f"{n}\nmIoU {results[n]['mIoU']:.3f}", fontsize=9)
            a.set_xlabel("predicted"); a.set_ylabel("true")
            for i in range(NC):
                for j in range(NC):
                    if cmn[i, j] > .02:
                        a.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6,
                               color="white" if cmn[i, j] < .55 else "black")
        fig.suptitle("Row-normalised confusion: where each true class ends up", y=1.01)
        fig.tight_layout(); p = out_dir / "04_confusion.png"; fig.savefig(p, bbox_inches="tight")
        plt.close(fig); written.append(p)

    # 5. training curves -----------------------------------------------------
    fig, a = plt.subplots(figsize=(7.5, 4.4))
    for n in names:
        h = results[n].get("history") or []
        if h:
            a.plot([r["epoch"] for r in h], [r["loss"] for r in h], label=n, lw=1.5)
    a.set_xlabel("epoch"); a.set_ylabel("training loss")
    a.set_title("Training loss. Point runs start lower: fewer, easier pixels, not better models")
    a.legend(fontsize=7); a.grid(alpha=.25)
    a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); p = out_dir / "05_training_loss.png"; fig.savefig(p); plt.close(fig)
    written.append(p)

    # 6. qualitative predictions --------------------------------------------
    if PREDS_NPZ.exists():
        z = np.load(PREDS_NPZ)
        runs = [n for n in ("pts5", "pts20", "pts20_balanced_focal", "full_mask")
                if f"{n}__0" in z.files]
        n_ex = len([k for k in z.files if k.startswith(f"{runs[0]}__")]) if runs else 0
        if runs and n_ex:
            fig, axes = plt.subplots(len(runs), n_ex, figsize=(3.1 * n_ex, 3.1 * len(runs)))
            axes = np.atleast_2d(axes)
            for i, n in enumerate(runs):
                for j in range(n_ex):
                    m = z[f"{n}__{j}"]
                    rgb = np.zeros((*m.shape, 3), np.uint8)
                    valid = m < NC
                    rgb[valid] = PALETTE[m[valid]]
                    axes[i, j].imshow(rgb); axes[i, j].axis("off")
                    if j == 0:
                        axes[i, j].set_title(f"{n}  (mIoU {results[n]['mIoU']:.3f})",
                                             loc="left", fontsize=9)
            fig.legend(handles=[mpatches.Patch(facecolor=PALETTE[i] / 255, edgecolor="k",
                                               label=c) for i, c in enumerate(CLASSES)],
                       loc="lower center", ncol=NC, fontsize=8, frameon=False)
            fig.suptitle("Same four test images, predicted by each run", y=1.0)
            fig.tight_layout(rect=[0, 0.05, 1, 1])
            p = out_dir / "06_predictions.png"; fig.savefig(p, bbox_inches="tight")
            plt.close(fig); written.append(p)

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=RESULTS_JSON)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "SCORES.md")
    args = ap.parse_args()

    results = load_results(args.results)
    print(f"loaded {len(results)} runs from {args.results}")
    print("integrity: every stored mIoU and OA matches the confusion matrix\n")

    blocks = [
        ("## Summary, all runs", summary_table(results)),
        ("## Experiment 1: point budget (uniform sampling, plain pCE)", budget_table(results)),
        ("## Experiment 2: sampling strategy x loss, at 20 points per image (mIoU)",
         sampling_loss_table(results)),
        ("## Per-class IoU", per_class_table(results)),
    ]
    doc = ["# Scores\n",
           "Recomputed from the confusion matrices in `results/results.json` by "
           "`pointseg-report`. Test split: the full LoveDA validation set, 1,669 images, "
           "dense masks, native 1024x1024.\n"]
    for title, table in blocks:
        print(title); print(table); print()
        doc += [title, "", table, ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(doc))
    print(f"wrote {args.out}")

    if not args.no_figures:
        for p in make_figures(results):
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
