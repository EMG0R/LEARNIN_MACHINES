# Hand Segmentation + Gesture Classifier — Retrain for Multi-User Deployment

**Audience:** Fresh LLM implementing the next training iteration. You haven't seen the prior conversation. Read everything here before touching code.

---

## 1. Project Context

**HAND_JOB** is a real-time computer vision pipeline for a creative-coding installation:

```
MacBook webcam (720p) → hand segmentation → gesture classification → OSC+NDI → TouchDesigner
```

- Deployment: **multi-user interactive installation**. Random visitors. Varied distances (1–5 ft from camera). Varied lighting (gallery/room/office). Different skin tones, hand sizes.
- All inference on Apple MPS, from-scratch PyTorch (no pretrained backbones).
- Live app at `live_app/app.py`, config at `live_app/config.py`, model wrappers at `live_app/models.py`.

---

## 2. Current State

### What exists
- **Seg model v3** (`hand_seg/checkpoints/hand_seg_v3.pt`) — U-Net with deep-supervision aux heads. IoU 0.933 on HaGRID val. Trained via `hand_seg/train_v3.py`.
- **Gesture model v4** (`gesture/checkpoints/gesture_v4.pt`) — 4-block Wide CNN, IMG_SIZE=112, F1 0.9606 on HaGRID test. Trained via `gesture/train_v4.py`.
- **Gesture model v5** (`gesture/train_v5.py`) — not yet trained. Has aggressive augmentation (wide bbox padding 0.0–1.0, RandomResizedCrop scale 0.3–1.0, RandomAffine with scale, strong ColorJitter, GaussianBlur, RandomErasing). Ready to run.
- Live app loads v3 seg + v1 gesture (`gesture_v1_wide96.pt`) in current config for stability.

### The problem we're solving
Both models perform well on HaGRID's own test split but **fail in live deployment** — hand-present detection is "almost totally random" on the MacBook webcam across multiple users. Classic train/deploy distribution gap:
- HaGRID = ~400 people, phone cameras, selfie-style framing, specific post-processing.
- Deployment = laptop webcam, varied users/distances/lighting, no selfie framing.

### Why augmentation alone won't fix it
`train_v3.py` seg already has heavy aug (skin-tone shift, random background paste/composite, webcam degradation, color jitter). `train_v5.py` gesture pushes augmentation further. There's no more signal to extract from HaGRID alone — the model is hitting the ceiling of "one-distribution training."

---

## 3. The Plan: Add EgoHands Dataset

**Goal:** train seg and gesture models on HaGRID **+ EgoHands** combined. EgoHands is first-person / laptop-view hand segmentation with multiple people and varied indoor settings — the exact distribution HaGRID is missing.

### EgoHands dataset
- URL: http://vision.soic.indiana.edu/projects/egohands/
- 4,800 frames, 48 videos, 4 participants per video interacting across 4 settings (puzzle/chess/cards/jenga) in 4 locations.
- Each frame has per-pixel polygon annotations for 4 hand instances (my_left, my_right, your_left, your_right). For our binary-mask use case: **union all four into a single foreground mask**.
- ~1.4 GB zipped.

### Why EgoHands specifically
| Property | HaGRID | EgoHands | Our deployment |
|---|---|---|---|
| Camera | Phone | Head-mounted GoPro | MacBook laptop |
| Viewpoint | Arm's length selfie | First-person | First-person-ish |
| People per frame | 1 | Up to 2 | 1+ |
| Background diversity | Indoor mostly | Multi-location indoor | Multi-location indoor |
| Hand distance variance | Low (gesture-posed) | High (natural activity) | High |

EgoHands covers exactly the "casual, varied-distance, laptop-perspective" distribution we deploy into.

### Optional secondary additions (do these only if EgoHands alone isn't enough)
- **11k Hands** — 11,000 dorsal/palmar hand images, 190 subjects. Broadens skin-tone and individual diversity. More hand-centric than situation-centric.
- Skip FreiHAND (studio backgrounds, 3D-focused) and Jester (dynamic gestures, wrong problem).

---

## 4. Implementation Steps

### Step 1: Download + preprocess EgoHands
Create `data/egohands/` alongside existing `data/hagrid-sample-30k-384p/`.

```python
# preprocess once, save binary masks as PNGs matching frame names
# output:
#   data/egohands/frames/<video_id>/<frame_idx>.jpg
#   data/egohands/masks/<video_id>/<frame_idx>.png   # binary 0/255
```

EgoHands annotations are MATLAB polygons (`metadata.mat`). Use `scipy.io.loadmat`, iterate videos, for each frame rasterize the union of all 4 hand polygons into a single binary mask via `PIL.ImageDraw.polygon`. Original frame size is 1280×720, keep full res during preprocessing.

### Step 2: Unified dataset loader
Create `hand_seg/data_mixed.py`:

```python
class MixedHandSegDataset:
    """Concatenates HaGRID (image+hand-box-derived mask) and EgoHands (image+polygon-mask).

    Returns (img_tensor, mask_tensor) at img_size x img_size. Both datasets get
    identical augmentation pipeline. Sampling ratio configurable (default equal,
    but EgoHands is smaller so we oversample EgoHands ~6x to match).
    """
```

Key design points:
- **Both datasets produce binary masks.** HaGRID has bounding boxes, not masks — existing v3 code already uses the bbox-shape as pseudo-mask. Keep that. Or: if you want a proper hand mask for HaGRID, skip HaGRID for seg and use EgoHands alone. **Recommended: use HaGRID's `leading_hand` polygon annotations if present, otherwise bbox. For gesture, keep using HaGRID bboxes.**
- **Oversample EgoHands ~6x** so per-epoch HaGRID:EgoHands ratio is roughly 1:1 in terms of samples seen. (HaGRID has ~30k, EgoHands ~4.8k.)
- **User-level splits on both.** For HaGRID use the existing `user_id` split. For EgoHands, split by `video_id` so no video leaks across train/val/test.

### Step 3: Retrain seg (v6)
Copy `hand_seg/train_v3.py` → `hand_seg/train_v6.py`. Changes:
1. Replace dataset construction with `MixedHandSegDataset`.
2. Keep the existing `HandUNetDS` architecture + deep supervision losses.
3. Keep v3's augmentation (skin-tone, scale_place, paste_bg, rotation, color jitter, webcam_deg) — all still useful.
4. Add **EgoHands-specific augmentation**: random crop within the larger frame (since EgoHands is 720p and hands occupy varied portions), RandomResizedCrop scale (0.5, 1.0).
5. Log separate IoU metrics for the HaGRID val subset and EgoHands val subset so you can watch both.

Expected outcome: IoU on HaGRID val drops slightly (maybe 0.90) but IoU on EgoHands val reaches 0.80+, and **deployment generalization improves dramatically**.

### Step 4: Retrain gesture (v6)
Gesture classification still uses HaGRID only — EgoHands has no gesture labels. BUT:
1. Copy `gesture/train_v5.py` → `gesture/train_v6.py`.
2. Keep v5's aggressive augmentation (bbox padding 0.0–1.0, RandomResizedCrop 0.3–1.0, strong color jitter, etc.).
3. Train with IMG_SIZE=112 (v4's value).
4. No EMA, no AMP, no MixUp, no label smoothing. (See "Do NOT do" section below.)

### Step 5: Wire into live app
Update `live_app/config.py`:
```python
SEG_CKPT     = BASE / "hand_seg/checkpoints/hand_seg_v6.pt"
GESTURE_CKPT = BASE / "gesture/checkpoints/gesture_v6.pt"
```
Keep `seg_model.load_state_dict(..., strict=False)` in `live_app/models.py` for the aux-head architecture.

Test with real multi-user scenarios. Tune `SEG_THRESHOLD` (currently 0.35), `HAND_MIN_AREA` (0.005), `CONF_THRESHOLD` (0.4) empirically based on live results.

---

## 5. Critical Do Not Do List (hard-learned)

An earlier iteration (v3 gesture) stacked these "improvements" and got F1 **0.017** — worse than random guessing:

| Don't | Why |
|---|---|
| `AMP=1` / fp16 autocast on MPS | MPS fp16 is unreliable. Weights memorize but fail to generalize. |
| `EMA` with decay 0.999 + `PATIENCE=8` | EMA needs ~14 epochs to converge; early stop kills training before you can see the true model. |
| MixUp α=0.2 on balanced 18-class | Unnecessary; interacts badly with class weighting. |
| Class weights baked into soft targets (`targets * class_w / targets.sum()`) | This is mathematically wrong, trains the wrong distribution. Always use `nn.CrossEntropyLoss(weight=class_w)` instead. |
| Label smoothing on top of MixUp on top of class weights | Three competing target manipulations = incoherent objective. |
| RandAugment magnitude=8 | Too destructive for fine-grained 18-class gesture discrimination. |
| RandomErasing p>=0.2 | Erases hand features that are the actual signal. |
| RandomGrayscale | Deployment is always color; teaching invariance wastes capacity. |

**Rule of thumb:** start from a minimal proven baseline, add one thing at a time, keep only what demonstrably helps on val F1/IoU.

---

## 6. What Has Actually Worked

### Seg v3 (IoU 0.933 on HaGRID val)
- Custom U-Net (`HandUNetDS`, based on `HandUNet` in `hand_seg/train.py`) with aux heads at u2/u3 decoder levels.
- Deep-supervision loss: main + 0.4*aux3 + 0.2*aux2 (see `deep_sup_loss` in train_v3.py).
- Focal Tversky loss (α=0.3, β=0.7, γ=0.75).
- IMG_SIZE=256, batch=16, 25 epochs, LR=2e-4, cosine + 2-ep warmup, EMA 0.999, AMP=1.
- Heavy aug: skin-tone shift, scale-place compositing, random-background paste, horizontal flip, ±25° rotation, color jitter, webcam degradation (noise/blur).
- **Note:** seg v3 uses AMP and EMA successfully (unlike v3 gesture). Segmentation is apparently more robust to these than classification on MPS. Keep them for seg.

### Gesture v1 (F1 0.984 on HaGRID test)
- 4-block Wide CNN (32→64→128→256 channels, 2 convs/block, Dropout2d 0.1, AdaptiveAvgPool, FC 256→512→18). ~1.3M params.
- IMG_SIZE=96, batch=128, 40 epochs, Adam LR=3e-4, cosine LR, weight_decay=1e-4.
- `nn.CrossEntropyLoss(weight=1/sqrt(class_count))`.
- Light augmentation: horizontal flip, ±15° rotation, mild color jitter. Nothing else.

### Gesture v4 (F1 0.9606 on HaGRID test)
- Same Wide CNN.
- IMG_SIZE=112 (up from 96).
- `RandomResizedCrop(112, scale=0.7-1.0)` + flip + ±15° rotation + ColorJitter(0.3, 0.3, 0.3, 0.05) + occasional GaussianBlur.
- AdamW, 2-ep warmup + cosine, early stop patience 10.
- **Proven stack — start any new gesture training from here.**

---

## 7. Validation / Success Criteria

After retraining, report:
1. **HaGRID-only val metrics** (seg IoU, gesture F1) — sanity check we didn't regress hard.
2. **EgoHands val metrics** (seg IoU) — confirm we learned the new distribution.
3. **Combined val metrics** — realistic overall score.
4. **Per-class gesture F1** on HaGRID test — spot any classes that collapsed.

**Success means:**
- Seg IoU on EgoHands val ≥ 0.75 (even while HaGRID val may drop to ~0.90).
- Gesture F1 on HaGRID test ≥ 0.93 (slight drop from v4's 0.96 is acceptable — deployment robustness is the point).
- **Qualitative:** run the live app and have 3 different people try it at 1ft, 3ft, 5ft in 2 different rooms. Hand-present detection should fire reliably (not "almost random" as today). Gesture classification confidence should exceed 0.5 for the correct class on clear gestures.

---

## 8. File Map (existing, for reference)

```
HAND_JOB/
├── hand_seg/
│   ├── train.py              # v1 seg baseline (defines base HandUNet)
│   ├── train_v3.py           # v3 seg (trained, IoU 0.933) — reference architecture
│   └── checkpoints/
│       ├── hand_seg_v1.pt    # v1 weights
│       └── hand_seg_v3.pt    # v3 weights (has aux heads)
├── gesture/
│   ├── train.py              # v1 gesture baseline
│   ├── train_v3.py           # v3 gesture — CAUTIONARY, don't emulate
│   ├── train_v4.py           # v4 gesture — clean working config
│   ├── train_v5.py           # v5 gesture — aggressive aug, not yet trained
│   └── checkpoints/
│       ├── gesture_v1_wide96.pt   # v1 (F1 0.984)
│       └── gesture_v4.pt          # v4 (F1 0.9606)
├── live_app/
│   ├── app.py                # main loop
│   ├── config.py             # paths, thresholds, smoothing knobs
│   ├── models.py             # loader + inference wrappers + GestureSmoother
│   ├── renderer.py
│   ├── osc_sender.py
│   └── ndi_sender.py
├── data/hagrid-sample-30k-384p/   # existing HaGRID subset
├── logs/
└── train_v3.sh               # orchestration: seg then gesture
```

Create new files:
- `data/egohands/` (after preprocessing)
- `hand_seg/data_mixed.py` (dataset loader)
- `hand_seg/preprocess_egohands.py` (one-time polygon → mask conversion)
- `hand_seg/train_v6.py`
- `gesture/train_v6.py`
- `train_v6.sh` (orchestration)

---

## 9. Open Questions for the Implementer

Before coding, confirm with the user:
1. Where does the user want EgoHands downloaded? (Probably `data/egohands/`.)
2. Is ~1.4 GB extra data OK on their disk?
3. Do they want to keep HaGRID in the seg training mix, or train seg purely on EgoHands? (Recommended: mix, because HaGRID's hand-box pseudo-masks still provide useful coverage of gesture-pose hands.)
4. If EgoHands alone isn't enough, are they willing to add 11k Hands as a second dataset in a future iteration?
