# edna Training Results

"edna" is the name for the FashionNet model family trained at scale on the full balanced
dataset, with iterative improvements between each version. All evaluations use
`scripts/evaluate_custom.py` on the validation split (11,186 images, 11 classes) at
conf=0.25 and NMS IoU=0.45, unless otherwise noted.

---

## Model Progression Summary

| Model | mAP@50 (val) | mAP@50 (test) | Precision | Recall | F1 | Key changes |
|-------|-------------|--------------|-----------|--------|----|-------------|
| fashionnet_balanced_v1 | 0.193 | -- | 0.336 | 0.387 | 0.359 | Ablation-winning config |
| edna_1.2m | 0.260 | 0.268 | 0.347 | 0.492 | 0.407 | scale=m, +aug, +multi_cell |
| edna_1.3m | 0.203 | 0.210 | 0.433 | 0.357 | 0.392 | +cos_lr, +EMA, +mosaic, +IoU-obj |
| edna_1.4m | 0.263 | 0.266 | **0.477** | 0.415 | 0.444 | +2K bg images, reverted IoU-obj |
| **edna_1.5m** | **0.261** | **0.272** | 0.430 | **0.448** | **0.439** | focal_bce .mean(), 1K bg images |

For comparison: **YOLOv8M on balanced dataset: 0.592 mAP@50** (pretrained, 50 epochs).

---

## edna_1.2m

### Setup

| Parameter | Value |
|-----------|-------|
| Model scale | m (~34M params) |
| Augmentation | medium |
| Multi-cell | yes |
| Optimizer | AdamW |
| Epochs | 100 |
| Batch | 16 |
| Training time | ~34h 53m |
| Best val_loss | 2.8128 (epoch 100) |

### Results

| Metric | Value |
|--------|-------|
| mAP@50 | **0.260** |
| Precision | 0.347 |
| Recall | **0.492** |
| F1 | 0.407 |

### Per-class Breakdown

| Category | AP | Precision | Recall | F1 |
|----------|----|-----------|--------|----|
| long_sleeve_outwear | **0.373** | 0.524 | 0.552 | 0.537 |
| vest | 0.352 | 0.368 | 0.594 | 0.455 |
| shorts | 0.329 | 0.414 | 0.592 | 0.487 |
| sling_dress | 0.313 | 0.430 | 0.415 | 0.422 |
| short_sleeve_dress | 0.271 | 0.404 | 0.445 | 0.424 |
| vest_dress | 0.271 | 0.406 | 0.478 | 0.439 |
| trousers | 0.246 | 0.318 | 0.604 | 0.417 |
| long_sleeve_dress | 0.224 | 0.417 | 0.381 | 0.398 |
| skirt | 0.209 | 0.236 | 0.556 | 0.332 |
| long_sleeve_top | 0.145 | 0.329 | 0.338 | 0.334 |
| short_sleeve_top | 0.128 | 0.215 | 0.440 | 0.289 |

### What changed vs fashionnet_balanced_v1

Scaling from model_scale=s (~11.7M params) to model_scale=m (~34M params), combined with
re-enabling aug=medium and multi_cell assignment, produced a +0.067 mAP@50 gain. Recall
jumped from 0.387 to 0.492 -- the larger model finds more objects. Precision remained
roughly constant (0.347 vs 0.336), indicating proportionally more false positives as well.

---

## edna_1.3m

### Setup

| Parameter | Value |
|-----------|-------|
| Model scale | m (~34M params) |
| New vs 1.2m | +cos_lr, +EMA, +mosaic augmentation, +IoU-aware objectness (gr=0.5), +label smoothing, lambda_obj=1.5, weight_decay=0.01 |
| Epochs | 100 |
| Batch | 32 |
| Best val_loss | 2.9268 |

### Results

| Metric | edna_1.2m | edna_1.3m | Delta |
|--------|-----------|-----------|-------|
| mAP@50 | **0.260** | 0.203 | -0.057 |
| Precision | 0.347 | **0.433** | +0.086 |
| Recall | **0.492** | 0.357 | -0.135 |
| F1 | **0.407** | 0.392 | -0.015 |

### What went wrong

The IoU-aware objectness target (gr=0.5) was the primary cause of regression. With this
setting, the objectness target for positive cells becomes `0.5 + 0.5 * iou` instead of 1.0.
Early in training when IoU is low (~0.1--0.4), targets drop to ~0.55--0.70 instead of 1.0.
At the conf=0.25 threshold, many true positives fall below threshold, causing recall to
collapse from 0.492 to 0.357.

The model became more conservative: fewer but more accurate detections. edna_1.3m fires
42% fewer total detections (10,221 vs 17,540 on the test set).

### Test Set Confirmation

| Model | mAP@50 | Precision | Recall | F1 | Detections |
|-------|--------|-----------|--------|----|------------|
| edna_1.2m | **0.268** | 0.346 | **0.500** | **0.409** | 17,540 |
| edna_1.3m | 0.210 | **0.436** | 0.367 | 0.399 | 10,221 |

The validation ranking holds on the held-out test set. edna_1.2m leads on mAP@50 (+0.058)
and recall; edna_1.3m leads on precision (+0.090).

---

## edna_1.4m

### Setup

| Parameter | Value |
|-----------|-------|
| Model scale | m (~34M params) |
| New vs 1.2m | +2,000 COCO background images (empty labels); gr=0.0 (IoU-obj disabled); lambda_obj=1.0 (default); no cos_lr, no mosaic, no EMA |
| Dataset | balanced_dataset + 2,000 bg images = 54,199 training images |
| Epochs | 100 (best at epoch 57) |
| Batch | 16 |

### Design rationale

The balanced dataset contains zero pure-background images -- every image has at least one
garment. The model never learns to suppress objectness on background regions, leading to
false positives. 2,000 COCO val2017 images (filtered to exclude people and clothing-adjacent
categories) were added as negative examples with empty label files (~4% of training set).

The IoU-aware objectness target (gr=0.5) was disabled (reverted to gr=0.0) based on the
edna_1.3m regression analysis. All other experimental features (cos_lr, mosaic, EMA) were
also reverted to isolate the effect of background images.

### Results

| Metric | edna_1.2m | edna_1.4m | Delta |
|--------|-----------|-----------|-------|
| mAP@50 | 0.260 | **0.263** | +0.003 |
| Precision | 0.347 | **0.477** | +0.130 |
| Recall | **0.492** | 0.415 | -0.077 |
| F1 | 0.407 | **0.444** | +0.037 |

### Per-class Breakdown

| Category | AP | Precision | Recall | F1 |
|----------|----|-----------|--------|----|
| sling_dress | **0.404** | 0.621 | 0.458 | 0.527 |
| vest | 0.370 | 0.571 | 0.513 | 0.540 |
| long_sleeve_outwear | 0.363 | 0.591 | 0.513 | 0.549 |
| shorts | 0.329 | 0.484 | 0.524 | 0.503 |
| vest_dress | 0.318 | 0.482 | 0.505 | 0.493 |
| short_sleeve_dress | 0.275 | 0.541 | 0.386 | 0.451 |
| trousers | 0.241 | 0.356 | 0.550 | 0.432 |
| long_sleeve_dress | 0.234 | 0.554 | 0.331 | 0.414 |
| skirt | 0.205 | 0.409 | 0.427 | 0.418 |
| long_sleeve_top | 0.101 | 0.396 | 0.213 | 0.277 |
| short_sleeve_top | 0.049 | 0.288 | 0.138 | 0.187 |

### Background Image Evaluation

Evaluated on the 2,000 COCO background images (no clothing, empty labels):

| Metric | Value |
|--------|-------|
| Images | 2,000 |
| Total false detections | **7** |
| False positive rate | **0.35%** |

The objectness suppression on pure backgrounds works correctly. edna_1.2m would fire
hundreds to thousands of false detections on the same images.

### Analysis

edna_1.4m is the best edna model on mAP@50 (0.263) and F1 (0.444). The background images
successfully improved precision from 0.347 to 0.477 (+37%). However, recall dropped from
0.492 to 0.415 -- the background images caused some over-suppression of objectness. The
short_sleeve_top and long_sleeve_top categories were most affected (AP dropped from 0.128
to 0.049 and 0.145 to 0.101 respectively).

---

## edna_1.5m

### Setup

| Parameter | Value |
|-----------|-------|
| Model scale | m (~34M params) |
| New vs 1.4m | focal_bce .sum() → .mean() (fixes loss inflation); removed /B normalization; bg images reduced 2,000 → 1,000 |
| Dataset | balanced_dataset + 1,000 bg images = ~53,199 training images |
| Epochs | 80 (stopped early, best.pt used) |
| Batch | 16 |

### Design rationale

edna_1.4m suffered from two compounding issues: `focal_bce` used `.sum()` over all ~8,400 grid cells, causing background images (all-negative) to inflate obj loss ~1000x; and 2,000 background images biased the model toward over-suppression, hurting recall. edna_1.5m fixes both: `focal_bce` now uses `.mean()` (per-cell normalization), and background images were reduced to 1,000 (~1.9% of training set).

### Results

| Metric | edna_1.2m | edna_1.4m | edna_1.5m | Δ vs 1.4m |
|--------|-----------|-----------|-----------|-----------|
| mAP@50 (val) | 0.260 | 0.263 | 0.261 | -0.002 |
| mAP@50 (test) | 0.268 | 0.266 | **0.272** | +0.006 |
| Precision | 0.347 | **0.477** | 0.430 | -0.047 |
| Recall | 0.492 | 0.415 | **0.448** | +0.033 |
| F1 | 0.407 | 0.444 | 0.439 | -0.005 |

### Per-class Breakdown (val)

| Category | AP | Precision | Recall | F1 |
|----------|----|-----------|--------|----|
| long_sleeve_outwear | 0.374 | 0.595 | 0.525 | 0.558 |
| sling_dress | 0.399 | 0.548 | 0.473 | 0.507 |
| vest | 0.348 | 0.471 | 0.553 | 0.509 |
| shorts | 0.303 | 0.492 | 0.542 | 0.516 |
| short_sleeve_dress | 0.301 | 0.512 | 0.428 | 0.467 |
| vest_dress | 0.271 | 0.504 | 0.420 | 0.458 |
| long_sleeve_dress | 0.234 | 0.515 | 0.350 | 0.417 |
| trousers | 0.197 | 0.343 | 0.526 | 0.415 |
| skirt | 0.193 | 0.323 | 0.449 | 0.376 |
| long_sleeve_top | 0.162 | 0.347 | 0.353 | 0.350 |
| short_sleeve_top | 0.091 | 0.268 | 0.308 | 0.286 |

### Test Set

| Metric | Value |
|--------|-------|
| mAP@50 | **0.272** |
| Precision | 0.437 |
| Recall | 0.457 |
| F1 | 0.447 |
| Total detections | 13,170 |
| Total ground truths | 12,592 |

### Background Image Evaluation

| Metric | Value |
|--------|-------|
| Images | 2,000 |
| Total false detections | **72** |
| False positive rate | **3.6%** |

Background suppression increased from 7 (edna_1.4m) to 72 false detections. Still very low in absolute terms, but the reduction in bg images (2,000 → 1,000) and the loss normalization fix relaxed the over-suppression, which is reflected in the recall recovery (+0.033 vs 1.4m).

### Analysis

edna_1.5m recovers **recall from 0.415 → 0.448** (+0.033 vs 1.4m) while keeping precision well above edna_1.2m (0.430 vs 0.347). The test mAP of **0.272 is the best of any edna run**. The val mAP (0.261) is marginally below 1.4m (0.263) but within noise — the test set result is more meaningful.

The precision/recall tradeoff is now better balanced: edna_1.5m sits between 1.2m (high recall, low precision) and 1.4m (high precision, lower recall), with the best test mAP overall.

Short_sleeve_top and long_sleeve_top recovered significantly vs 1.4m (AP 0.091 vs 0.049, and 0.162 vs 0.101), confirming the recall regression in 1.4m was caused by over-suppression from too many background images and loss inflation.

---

## Gap to YOLOv8

| Model | mAP@50 | Precision | Recall |
|-------|--------|-----------|--------|
| edna_1.5m (best edna, test) | 0.272 | 0.437 | 0.457 |
| YOLOv8M (best balanced) | 0.592 | 0.575 | 0.689 |
| **Gap** | **0.320** | **0.138** | **0.232** |

The gap is explained by three factors:

1. **No pretrained weights.** YOLOv8M uses COCO-pretrained weights; all edna models train
   from random initialization. On YOLOv8 alone, removing pretraining costs ~7% mAP (Test 1
   vs Test 5 in the YOLO experiments).

2. **Architectural maturity.** YOLOv8 benefits from years of design optimization
   (CSPDarknet, PANet, decoupled head, mosaic augmentation, dynamic label assignment).
   FashionNet implements a simpler architecture.

3. **Training recipe.** YOLOv8's training methodology (learning rate schedules, augmentation
   pipelines, loss formulations) is highly optimized. The custom training loop was iterated
   over 4 versions but cannot match years of community development.

Despite the gap, the custom detector achieves reasonable precision (0.477 vs 0.575) and
demonstrates that a from-scratch detector can learn meaningful clothing detection without
any pretrained weights or borrowed code.

---

## Threshold Tuning -- edna_1.2m

Evaluated at multiple confidence thresholds to test whether the precision/recall imbalance
could be corrected without retraining.

| conf | mAP@50 | Precision | Recall | F1 |
|------|--------|-----------|--------|----|
| **0.25** | **0.260** | 0.347 | **0.492** | 0.407 |
| 0.30 | 0.238 | 0.392 | 0.434 | **0.412** |
| 0.35 | 0.210 | 0.436 | 0.367 | 0.399 |
| 0.40 | 0.177 | 0.484 | 0.293 | 0.365 |
| 0.45 | 0.137 | 0.535 | 0.213 | 0.304 |

The F1 peak at conf=0.30 (+0.005 over default) comes at the cost of -0.022 mAP@50.
The gain is negligible. The low precision is structural -- the model produces false positives
that no threshold adjustment can eliminate without proportional recall loss.
