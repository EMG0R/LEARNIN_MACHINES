# OBJECTIFICATION — Multi-Class Semantic Segmentation (Layer 1)

**Status:** design
**Date:** 2026-04-24
**Replaces:** original Layer 1 (Mask R-CNN, 80 COCO classes, by Jake) in `archetecture.md`
**Folder:** `OBJECTIFICATION/`
**Target platforms:** Mac (Apple MPS) for v1; Pi 5 + Hailo AI HAT+ for v2 (deferred)

---

## 1. Goal

Replace the planned Mask R-CNN Layer 1 with a **23-class semantic segmentation U-Net** trained from scratch on a curated subset of OpenImages V7. Output is a per-pixel class map; the renderer extracts contours per class for the mesh-overlay aesthetic shared with `hand_seg` and `face_parts`. No bounding boxes are drawn — the contour mesh *is* the visual.

The cascade gating contract is preserved: when `person` pixels are present, the downstream face/hand/pose branches still fire; per-class contour data passes to the renderer.

---

## 2. Class List — 23 classes + background (24 channels)

Aggressive merges keep the head small and the masks robust on webcam.

| # | Class | Composed of (OpenImages V7 labels) |
|---|---|---|
| 1 | Person | Person |
| 2 | Vehicle | Car, Bicycle, Motorcycle, Bus, Truck |
| 3 | Skateboard | Skateboard *(broken out as unique)* |
| 4 | Phone | Mobile phone |
| 5 | Device | Television, Laptop, Computer monitor, Tablet computer, Computer keyboard, Remote control |
| 6 | Animal | Bird, Dog, Cat |
| 7 | Plant | Tree, Flower, Plant, Houseplant |
| 8 | Cup | Cup, Bottle, Wine glass, Mug |
| 9 | Spork | Fork, Knife, Spoon |
| 10 | Bowl | Bowl, Plate |
| 11 | Footwear | Footwear, Boot, Sandal, High heels, Sneakers |
| 12 | Glasses | Glasses, Sunglasses |
| 13 | Headphones | Headphones |
| 14 | Chair | Chair |
| 15 | Couch | Couch |
| 16 | Table | Coffee table, Kitchen & dining room table, Desk |
| 17 | Lamp | Lamp |
| 18 | Book | Book |
| 19 | Clock | Clock |
| 20 | Bag | Handbag, Backpack |
| 21 | Guitar | Guitar |
| 22 | Trumpet | Trumpet |
| 23 | Piano | Piano |

Class 0 = background. Final output tensor: `[B, 24, H, W]`.

The OI-label → merged-ID mapping lives in `OBJECTIFICATION/seg/class_map.json` and is the single source of truth for both the dataset pipeline and the live renderer's color palette.

**Cuts considered and dropped from earlier drafts:** Window, Door, Microphone, Ball, Pillow, Bed (low art value, noisy masks, or large flat surfaces that mesh poorly).

---

## 3. Architecture — YOLO-style backbone + U-Net-style decoder

A hybrid: YOLOv5/v8-style CSPDarknet backbone (C3 blocks + SPPF) feeding a U-Net-style top-down FPN decoder. Output is per-pixel softmax over 24 channels.

### 3.1 Input

- **Resolution:** 320×320 RGB, letterboxed.
- Matches `face_det` resolution; comfortable on MPS.

### 3.2 Backbone — CSPDarknet-lite

```
Stem:    Conv 3×3 s2 + BN + SiLU         →  32 ch   (320 → 160)
Stage 1: Conv 3×3 s2 + C3(n=1)           →  64 ch   (160 →  80)
Stage 2: Conv 3×3 s2 + C3(n=2)           → 128 ch   ( 80 →  40)
Stage 3: Conv 3×3 s2 + C3(n=3)           → 256 ch   ( 40 →  20)
Stage 4: Conv 3×3 s2 + C3(n=1) + SPPF    → 512 ch   ( 20 →  10)
```

**C3 block** (CSP bottleneck):
```
x → split into two halves
   branch A: 1×1 conv → n × Bottleneck(3×3, residual) →
   branch B: 1×1 conv (shortcut) →
concat(A, B) → 1×1 conv fuse
```

**SPPF** (Spatial Pyramid Pooling — Fast):
```
x → 1×1 conv
  → maxpool k=5 → maxpool k=5 → maxpool k=5  (in series)
  → concat all four → 1×1 conv fuse
```

All convs: BatchNorm + SiLU activation.

### 3.3 Neck — top-down FPN with lateral skips

```
P5 (10×10, 512ch)
  ─upsample ×2─→ concat P4 lateral → C3(n=1) → 256 ch  (20×20)
  ─upsample ×2─→ concat P3 lateral → C3(n=1) → 128 ch  (40×40)
  ─upsample ×2─→ concat P2 lateral → C3(n=1) →  64 ch  (80×80)
```

This is functionally a U-Net decoder built from C3 blocks instead of plain conv pairs — it preserves the encoder-decoder skip pattern that already trained well in `hand_seg` and `face_parts`, with stronger feature reuse from the CSP design.

### 3.4 Head

```
Upsample 80→320 (bilinear ×4)
  → Conv 3×3 → 32 ch + BN + SiLU
  → Conv 1×1 → 24 ch
```

Train: per-pixel softmax + cross-entropy + Dice. Inference: argmax → 320×320 class index map.

### 3.5 Param budget

~4–5M parameters. Larger than `hand_seg` (~2M) because the class set is wider, but well inside MPS comfort. The 1×1 head adds only ~kilobytes regardless of class count.

---

## 4. Dataset Pipeline

### 4.1 Source

**OpenImages V7 segmentation subset.** The full release is ~500 GB; we pull only images that contain ≥1 of our 23 merged classes. Estimated download after class filtering: **~10 GB** (~40–80K images, ~150–300K masks).

### 4.2 Download flow — `OBJECTIFICATION/data_pipeline/download.py`

1. Pull OI V7 metadata CSVs (`oidv7-class-descriptions.csv`, `oidv7-train-annotations-segmentation.csv`).
2. Filter rows whose `LabelName` maps to any of our 23 merged classes via `class_map.json`.
3. Dedupe `ImageID`s (one image may contribute multiple class instances).
4. `wget` images from `s3://open-images-dataset/...` and per-instance mask PNGs from the OI mask bucket.
5. Store under `OBJECTIFICATION/shared/datasets/openimages_v7/{train,val}/{images,masks}/`.

Resumable. Skips already-downloaded files.

### 4.3 Mask preparation — `OBJECTIFICATION/data_pipeline/prepare_masks.py`

OI ships one PNG per instance per class. We collapse these into a single `[H, W]` integer mask per image where pixel value = merged class ID:

1. For each image, gather all instance masks across its target classes.
2. For each instance mask, look up merged class ID from `class_map.json`.
3. Paint instance pixels into the per-image mask. Last-write-wins for overlapping instances (rare; warn if frequent).
4. Save `{ImageID}.png` (uint8, single channel) alongside the RGB image.

### 4.4 Augmentation — `OBJECTIFICATION/seg/augment.py`

Light, validated stack only — per `feedback_hagrid_v3_overengineering.md`:

- HFlip (p=0.5)
- Color jitter (brightness/contrast/saturation ±0.2, hue ±0.05)
- Rotation ±10°
- Random scale 0.8–1.2× then random crop to 320×320
- Letterbox if smaller than 320

**No** MixUp, CutMix, RandAugment, EMA, or AMP. Adding these on MPS triggered the gesture v3 collapse.

### 4.5 Class balance

Two complementary mechanisms:

- **Weighted batch sampler:** sampling weight per image = `1 / sqrt(min_class_freq_in_image)`. Rare classes (e.g., Trumpet, Piano) appear more often.
- **Class-weighted CE loss:** per-class weight = `median_freq / class_freq`, clipped to `[0.5, 5.0]`.

---

## 5. Training Recipe

### 5.1 Loss

```
L = 0.5 · CE(pred, target, weights) + 0.5 · Dice(pred, target)
```

Dice handles imbalance and small-object recall (Trumpet, Glasses). CE keeps confidence calibrated. Both ignore index 255 (use for unlabeled / padding regions).

### 5.2 Optimizer & schedule

- **AdamW**, weight decay 5e-4
- **Cosine LR**, peak 3e-4, warmup 1k iterations *(matches HaGRID gesture profile that gave F1 0.984)*
- **Batch size 16** at 320×320 on MPS; drop to 8 if memory-constrained
- **60 epochs** target; early stop on 5-epoch mIoU plateau

### 5.3 Validation

- **Macro mIoU** across 23 foreground classes (background excluded from average)
- Per-class IoU logged for diagnosis
- Pixel accuracy logged but not optimized (dominated by background)
- Checkpoint best macro-mIoU → `OBJECTIFICATION/seg/checkpoints/best.pt`

### 5.4 Anti-patterns to avoid

Do not stack: AMP + EMA + MixUp + RandAugment. The gesture v3 incident showed this collapses to F1 0.017 on MPS. Keep the recipe minimal.

### 5.5 Time estimate

~6–10h on a 64GB M-series Mac via MPS. Slots into the existing `train_all.py` orchestration pattern with thermal watchdog.

---

## 6. Project Layout

Mirrors `HAND_JOB/` and `FACE_JOB/` exactly.

```
OBJECTIFICATION/
├── seg/
│   ├── model.py              # YOLO+U-Net hybrid (CSP backbone, FPN decoder, seg head)
│   ├── train.py              # training loop
│   ├── dataset.py            # OI V7 dataset class
│   ├── augment.py            # transforms
│   ├── eval.py               # per-class IoU + macro mIoU
│   ├── checkpoints/          # best.pt + last.pt
│   └── class_map.json        # OI label → merged class ID (source of truth)
├── data_pipeline/
│   ├── download.py           # filter OI CSVs + S3 fetch
│   └── prepare_masks.py      # per-instance PNGs → single integer mask
├── shared/
│   └── datasets/openimages_v7/
│       ├── train/{images,masks}/
│       └── val/{images,masks}/
└── train_all.py              # orchestrator (port of HAND_JOB pattern w/ thermal watchdog)
```

---

## 7. Live Integration

OBJECTIFICATION becomes the new Layer 1 in the cascade. Output flow per frame:

1. Model produces a 24-channel softmax tensor at 320×320.
2. `argmax` → integer class map.
3. For each non-background class with non-empty pixels:
   - Threshold + binary mask
   - `cv2.connectedComponents` → per-blob masks
   - `cv2.findContours` per blob → polygon vertices
   - Compute centroid, area, bbox per blob
4. Pass per-blob data to the renderer via the result dict.
5. If `Person` class has any blob with area ≥ `PERSON_MIN_AREA`, signal downstream face/hand/pose branches with the person crop region (preserves cascade gating contract from `archetecture.md`).

### 7.1 Renderer

Per-class color palette baked from `class_map.json`. Each blob's contour is drawn at 1–2px stroke with class color; optional 15% fill. Identical aesthetic philosophy to the hand mesh in `archetecture.md` §Overlays — the contour *is* the visual, no bounding boxes drawn.

---

## 8. Mac v1 Scope (this spec)

- Train and run inference at 320×320 on MPS
- Float32 weights, no quantization, no ARM-specific kernels
- Single-threaded inference path (matches existing live_app pattern)
- Goal: stable mIoU + visually clean meshes; perf is whatever MPS gives us

---

## 9. Pi v2 Optimization Notes (deferred — separate spec later)

Captured here so future-us doesn't get blindsided. **None of this work is in v1 scope.**

### 9.1 Target

**Pi 5 8GB + Hailo AI HAT+ (Hailo-8, 26 TOPS), 30 fps end-to-end** for the full cascade (OBJECTIFICATION + face_det + face_parts + emotion + hand_seg + gesture + body pose).

### 9.2 Achievability — confidence and caveats

Likely achievable on the **typical-frame case**, with real risks. Not an unconditional lock.

**Why it's plausible:**
- L1 OBJECTIFICATION at 320×320 on Hailo-8 (compiled INT8): estimated 50–100 fps standalone
- Hand U-Net (192×192): ~150 fps
- Face U-Net + face_parts U-Net + emotion CNN: ~100–200 fps each
- Gesture CNN: ~500 fps
- Cascade gating means most frames don't fire all branches — when only L1 + 1 branch run, 30 fps is comfortable

**Why it might miss:**
- **Body pose (HRNet 512px) is the heaviest piece** — likely 30–60 fps standalone, drops the cascade ceiling. Plan to shrink or swap for a lighter pose model in v2.
- **Worst-case frame** (person + face + hand + pose all active, sequential): could dip to 15–20 fps without pipelining. Per-stage pipelining (L1 on frame N+1 while L2 runs frame N) is required to hold 30 fps in worst case.
- **Hailo context switching.** 6+ models swapping on one accelerator has overhead. Orchestrator becomes more complex than the Mac version.

### 9.3 Quantization risk

From-scratch random-init models may lose mIoU when compiled to Hailo INT8. Mitigations, in order of cost:

1. **Post-training quantization (PTQ)** with calibration set first — check accuracy drop
2. **Quantization-aware training (QAT)** if PTQ drop is unacceptable — retrain final 5–10 epochs with fake-quant ops inserted
3. Worst case: keep float fallback for accuracy-critical classes, INT8 for the rest

### 9.4 Op compatibility on Hailo

- **Should compile cleanly:** Conv2d, BN, SiLU/ReLU, MaxPool, bilinear upsample, concat, residual add (everything in §3)
- **Verify before committing:** SiLU support varies by Hailo SDK version (may need ReLU swap); SPPF series-pool pattern works but compiler may rewrite it
- **Avoid in v2 model surgery:** custom CUDA-style ops, dynamic shape ops, anything Python-side in the forward pass

### 9.5 CPU-side post-processing

`cv2.findContours` × 24 classes per frame on Pi 5 ARM: estimated 10–20 ms by itself. Mitigations:

- Skip classes with no pixels (already in §7)
- Decimate contour vertices (Douglas-Peucker, ε=2px) before sending — also reduces bandwidth
- Move connected-component pass to a thread pool

### 9.6 Memory bandwidth

Pi 5 LPDDR4X-4267 ~17 GB/s vs Mac M-series ~100+ GB/s. The 320×320×24 float mask tensor (~1 MB) and downstream contour extraction will be more bus-bound on Pi. Should still fit comfortably; flag for profiling.

### 9.7 Combined-app orchestration

The eventual `combined_app/` (per `archetecture.md` §Combined Live App) will need a Pi-aware scheduler:

- Async Hailo job queue with priority (L1 always; L2/L3 conditional on gates)
- Frame-level pipelining across stages
- Single shared OpenCV capture, single output frame writer
- Backpressure: drop frames before they queue if Hailo is saturated

---

## 10. Out of Scope (this spec)

- Pi v2 optimization implementation (own spec)
- The combined live app integrating OBJECTIFICATION with HAND_JOB and FACE_JOB cascades (own spec, after all three are trained)
- Instance segmentation (semantic only — disconnected blobs become separate contours naturally)
- Anything outside the 23 classes in §2

---

## 11. Success Criteria

- Macro mIoU ≥ 0.55 across 23 classes on OI V7 val set (rough target; tune after first run)
- Per-class IoU ≥ 0.30 for every class (no class collapse)
- Live demo at ≥ 15 fps on M-series MPS with full contour pipeline
- Visually clean contours per class on webcam input (subjective sanity check)
- All artifacts checked in: `model.py`, `class_map.json`, `best.pt`, eval report
