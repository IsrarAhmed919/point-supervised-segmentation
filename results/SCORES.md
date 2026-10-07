# Scores

Recomputed from the confusion matrices in `results/results.json` by `pointseg-report`. Test split: the full LoveDA validation set, 1,669 images, dense masks, native 1024x1024.

## Summary, all runs

| run | points / image | sampling | loss | mIoU | OA | fwIoU | mF1 | min |
|---|---:|---|---|---:|---:|---:|---:|---:|
| `full_mask` | all 1,048,576 | dense mask | CE | **0.3481** | 0.5614 | 0.3668 | 0.5001 | 29 |
| `pts5` | 5 | uniform | pCE | **0.3227** | 0.5671 | 0.3775 | 0.4752 | 28 |
| `pts20` | 20 | uniform | pCE | **0.3296** | 0.5457 | 0.3495 | 0.4835 | 28 |
| `pts100` | 100 | uniform | pCE | **0.3423** | 0.5460 | 0.3534 | 0.4938 | 28 |
| `pts500` | 500 | uniform | pCE | **0.3422** | 0.5541 | 0.3589 | 0.4947 | 28 |
| `pts20_focal` | 20 | uniform | pCE, focal gamma=2 | **0.3194** | 0.5448 | 0.3484 | 0.4722 | 29 |
| `pts20_balanced` | 20 | class-balanced | pCE | **0.3520** | 0.5293 | 0.3461 | 0.5130 | 28 |
| `pts20_balanced_focal` | 20 | class-balanced | pCE, focal gamma=2 | **0.3658** | 0.5433 | 0.3598 | 0.5276 | 27 |

## Experiment 1: point budget (uniform sampling, plain pCE)

| points / image | labelled fraction | mIoU | OA | % of dense-mask mIoU |
|---:|---:|---:|---:|---:|
| 5 | 0.000477% | 0.3227 | 0.5671 | 92.7% |
| 20 | 0.001907% | 0.3296 | 0.5457 | 94.7% |
| 100 | 0.009537% | 0.3423 | 0.5460 | 98.3% |
| 500 | 0.047684% | 0.3422 | 0.5541 | 98.3% |
| all (1,048,576) | 100% | 0.3481 | 0.5614 | 100.0% |

## Experiment 2: sampling strategy x loss, at 20 points per image (mIoU)

| sampling | pCE (gamma=0) | focal pCE (gamma=2) |
|---|---:|---:|
| uniform | 0.3296 | 0.3194 |
| class-balanced | 0.3520 | 0.3658 |

## Per-class IoU

| run | background | building | road | water | barren | forest | agriculture |
|---|---|---|---|---|---|---|---|
| `full_mask` | 0.450 | 0.479 | 0.335 | 0.434 | 0.079 | 0.417 | 0.242 |
| `pts5` | 0.458 | 0.351 | 0.273 | 0.435 | 0.081 | 0.326 | 0.334 |
| `pts20` | 0.445 | 0.439 | 0.320 | 0.436 | 0.119 | 0.332 | 0.215 |
| `pts100` | 0.434 | 0.467 | 0.300 | 0.478 | 0.101 | 0.411 | 0.204 |
| `pts500` | 0.446 | 0.466 | 0.334 | 0.469 | 0.098 | 0.362 | 0.219 |
| `pts20_focal` | 0.444 | 0.384 | 0.292 | 0.446 | 0.108 | 0.334 | 0.228 |
| `pts20_balanced` | 0.405 | 0.457 | 0.404 | 0.446 | 0.210 | 0.323 | 0.220 |
| `pts20_balanced_focal` | 0.422 | 0.471 | 0.443 | 0.463 | 0.234 | 0.294 | 0.233 |
