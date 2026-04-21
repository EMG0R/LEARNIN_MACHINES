# HaGRID Gesture Classifier — Design

**Date:** 2026-04-21
**Scope:** Layer 3b (Gesture Classifier) of the vision pipeline in `archetecture.md`.
**Deliverable:** A single Jupyter notebook at `HAND_JOB/gesture_classifier.ipynb` that trains an 18-class gesture classifier on the local HaGRID 30k 384p sample.

---

## Non-Goals

- **No hand detection.** Bounding boxes come from HaGRID ground truth. The learned hand *detector* (Layer 2b) is a separate notebook trained on FreiHAND.
- **No hand landmarks.** The 30k sample does not include 21-point landmarks.
- **No full HaGRID download.** Uses only the data already in `HAND_JOB/training data/hagrid-sample-30k-384p/`.
- **No ONNX/CoreML export.** Pipeline integration is a later task.
- **No AMP.** Keep MPS path simple.

---

## Data

- **Source:** `HAND_JOB/training data/hagrid-sample-30k-384p/`
  - `ann_train_val/*.json` — one JSON per class (18 files). Each JSON maps image-id → `{bboxes, labels, leading_hand, leading_conf, user_id}`. `bboxes` is a list of `[x, y, w, h]` in normalized coords. `labels` is a per-bbox class-name string.
  - `hagrid_30k/train_val_<class>/` — 384p JPEGs named by image-id.
- **Classes (18):** `call, dislike, fist, four, like, mute, ok, one, palm, peace, peace_inverted, rock, stop, stop_inverted, three, three2, two_up, two_up_inverted`.
- **Instance count:** ~30k images, sometimes multiple bboxes per image → one training row per bbox.
- **`no_gesture` handling:** multi-hand images may tag the non-leading hand as `no_gesture`. **Drop** `no_gesture` rows — the classifier has no 19th class. Log how many rows were dropped.

---

## Split Strategy

**Group by `user_id`.** Hash `user_id → {train, val, test}` with ratio 80/10/10.

- **Zero user overlap** across splits (assert).
- Print per-class counts for each split. If any class has < 50 rows in val or test, warn.
- Rationale: interactive installation — at runtime the user's hand was never seen during training. Random split would leak same-person hands across splits and over-estimate accuracy.

---

## Dataset Class

`HagridGestureDataset(rows, img_root, mode, img_size=64)`

`rows`: list of `(img_path, bbox_xywh_norm, label_idx, user_id)`.

`__getitem__`:
1. Load PIL RGB image.
2. De-normalize bbox to pixel coords.
3. **Pad the bbox.** `padding = uniform(0.05, 0.25) * max(w, h)` in train mode; fixed `0.15 * max(w, h)` in val/test. Clamp to image bounds.
4. Crop, resize to 64×64 (bilinear).
5. Apply transforms (train aug vs. val/test fixed).
6. Return `(tensor, label_idx)`.

Padding jitter is load-bearing: at inference time the upstream hand detector's bboxes are sloppy; training on tight ground-truth bboxes alone breaks generalization.

---

## Augmentations (train only)

Applied **after** the padded crop and resize to 64×64:

- Horizontal flip (p=0.5)
- Random rotation ±20°
- Color jitter: brightness 0.3, contrast 0.3, saturation 0.3, hue 0.05
- Random perspective, distortion_scale=0.15 (p=0.5)
- Gaussian blur, kernel 3, sigma (0.1, 1.5) (p=0.2)
- ToTensor + ImageNet normalize
- Random erasing (p=0.25), scale (0.02, 0.15)

Val/test: resize 64×64 → ToTensor → ImageNet normalize only.

Rationale for heavy aug: 30k sample ≈ 1.6k/class after the drop of `no_gesture` and the 80/10/10 split means ~1.3k/class train. Heavy aug buys invariance to lighting, pose, and detector sloppiness. Aug is not optional at this scale.

---

## Model

`GestureNet(nn.Module)` — follows `archetecture.md` §Layer 3b. Input 3×64×64, output 18 logits.

```
Conv2d(3→32, k=3, p=1) → BN → ReLU → Dropout2d(0.1) → MaxPool(2)   # 32×32
Conv2d(32→64, k=3, p=1) → BN → ReLU → Dropout2d(0.1) → MaxPool(2)  # 16×16
Conv2d(64→128, k=3, p=1) → BN → ReLU → Dropout2d(0.1) → MaxPool(2) # 8×8
AdaptiveAvgPool2d(1) → Flatten
Linear(128→256) → ReLU → Dropout(0.3) → Linear(256→18)
```

Kaiming init on conv layers. Random init (no pretrained) — per `archetecture.md` intent.

---

## Training

- Loss: `CrossEntropyLoss(weight=1 / sqrt(class_count))`. HaGRID is close to balanced; mild reweight, not heavy.
- Optimizer: Adam, lr=3e-4, weight_decay=1e-4.
- Scheduler: `CosineAnnealingLR(T_max=epochs)`.
- Batch size: 128.
- Epochs: 30, with early stopping patience 6 on val macro-F1.
- Device: MPS.
- Seed: 42, set on `torch`, `numpy`, `random`.
- DataLoader: `num_workers=0` (notebook constraint noted in existing `CONV.ipynb`). `shuffle=True` for train.

---

## Metrics & Evaluation

Per epoch:
- Train loss (mean over epoch)
- Val loss (mean over val set)
- Val top-1 accuracy
- Val macro-F1

Save checkpoint on **best val macro-F1** (not accuracy — better signal under any class imbalance).

End of training:
- Loss curves (train vs. val)
- Val accuracy + macro-F1 curves
- Confusion matrix on val (matplotlib heatmap, 18×18)
- Per-class precision / recall / F1 table (sklearn `classification_report`)
- Held-out **test set** evaluation — run once, report top-1, macro-F1, confusion matrix. Separate from the val set used during training.

---

## Checkpointing

Save to `HAND_JOB/checkpoints/gesture_classifier_best.pt`:

```
{
  "model_state_dict": ...,
  "class_names": [...18 strings...],
  "img_size": 64,
  "normalize_mean": [0.485, 0.456, 0.406],
  "normalize_std":  [0.229, 0.224, 0.225],
  "val_macro_f1": ...,
  "epoch": ...,
}
```

Bundling class names + preprocessing params with the weights means the downstream pipeline import is a single load call — no separate config file to lose.

---

## Inference Helper

```python
def predict_crop(np_crop_bgr: np.ndarray) -> tuple[str, float]:
    """Given a BGR hand crop (any size), return (label, confidence)."""
```

- Converts BGR→RGB, resizes to 64×64, normalizes, runs model, returns argmax label + softmax confidence.
- Importable by `pipeline/` later without touching the notebook.

Demo cell: loads checkpoint, picks N random val samples, shows image + top-3 predictions grid.

---

## Notebook Cell Layout

1. Imports + device + seed
2. Paths + class list constant + `NUM_CLASSES = 18`
3. Index builder (walks the 18 JSONs, returns flat list of rows; drops `no_gesture`)
4. User-grouped split + assertions + per-class count plot
5. Augmentation transforms (train + eval)
6. `HagridGestureDataset` class
7. DataLoaders
8. Visualize an augmented batch (3×6 grid)
9. `GestureNet` + parameter count
10. Train / eval functions
11. Training run
12. Loss + metric plots
13. Confusion matrix + classification report (val)
14. Test set evaluation
15. Save checkpoint
16. `predict_crop` helper + demo

---

## Risks

- **Small dataset per class.** ~1.3k train rows/class after splits. Heavy aug mitigates; if val F1 < 0.75 for any class, consider downloading the full HaGRID later.
- **Padding-jitter calibration.** Upstream detector sloppiness at inference is unknown until Layer 2b is trained. 5–25% padding range is a reasonable first guess; may need to widen after pipeline integration.
- **MPS determinism.** Seeds do not guarantee bitwise reproducibility on MPS. Accept small run-to-run variance.
