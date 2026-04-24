# Real-Time ML Vision Pipeline

Real-time computer vision pipeline. Python, PyTorch, OpenCV. Webcam in, cascading ML models, aesthetic skeletal/contour overlays out. All models built and trained from scratch. Layer 1 sends detection data over OSC to TouchDesigner.

The rendered OpenCV frame is feed into TouchDesigner so overlays are a baseline for the visual art. 

**Platform:** Mac, Apple Silicon (MPS)

---

## Detection Cascade

Each layer gates the next. Downstream models only run when upstream fires.

```
Webcam Frame
  │
  ▼
[Layer 1: Mask R-CNN — General Object Detection + Segmentation]
  │  Detects ALL objects. Draws contour outlines. Sends everything over OSC → TD.
  │
  ├─ ALL objects → OSC out to TouchDesigner
  │
  ├─ "person" detected → crop region internally
  │     │
  │     ├──▶ [Layer 2a: Face Detector — FCOS-style, multi-face]
  │     │      → bbox(es) per face, NMS. Gate: face_present(count ≥ 1).
  │     │      └─ face found → crop each face, run in parallel:
  │     │          ├─ [Layer 3a1: Face-Part U-Net]
  │     │          │    → multi-class masks {eye_L, eye_R, mouth, face_skin}
  │     │          │    → tight mesh on eyes + mouth, light mesh on face
  │     │          └─ [Layer 3a2: Emotion Classifier]
  │     │               → 7 classes (happy/sad/neutral/surprise/anger/fear/disgust) + confidence
  │     │
  │     ├──▶ [Layer 2b: Hand Segmentation (U-Net)]
  │     │      → binary hand mask → contour/mesh overlay, OSC mask data
  │     │      └─ gate: mask area ≥ HAND_MIN_AREA  (`hand_present(mask)`)
  │     │          │   if no hand present → SKIP Layer 3b entirely (save compute)
  │     │          ▼
  │     │          [Layer 3b: Gesture Classifier]  ← only runs when gate passes
  │     │            → 18 HaGRID classes + confidence (top-1 + runner-up)
  │     │
  │     └──▶ [Layer 2c: Body Pose Estimator]
  │            → 17 keypoints (COCO skeleton: nose, eyes, ears, shoulders, elbows,
  │              wrists, hips, knees, ankles) → draw full body skeletal trace
  │
  ▼
[Overlay Renderer] → OpenCV display
```

---

## Models

All `nn.Module` from scratch. Random init (Kaiming/Xavier).

---

### Layer 1: General Object Detection (Jake)

ResNet-50 + FPN backbone, detection head, 80 COCO classes. Being trained by Jake on COCO 2017.

**Dataset:** COCO 2017 — 118K images, 80 classes
- `pycocotools` annotations
- Status: **in training** (Jake)

---

### Layer 2a: Face Detector (FACE_JOB)

Anchor-free, FCOS-style CNN (~1.5M params). Single-scale head: per-pixel {objectness, bbox l/t/r/b, centerness}. NMS at inference. Multi-face native. **Bbox only — no landmarks** (eyes/mouth come from Layer 3a1 seg masks, mirroring `hand_seg`).

**Input:** 320×320 RGB letterboxed.

**Dataset:** WIDER FACE, **filtered to webcam-realistic faces**:
- Drop faces with shorter bbox side < 40px
- Drop images with >8 faces (crowd shots)
- Keep "easy" + "medium" difficulty, drop "hard"
- Result: ~15-20K images, ~30-60K face instances (from 32K / 393K raw)
- Source: [shuoyang1213.me/WIDERFACE](http://shuoyang1213.me/WIDERFACE)

Checkpoint: `FACE_JOB/face_det/checkpoints/best.pt`

---

### Layer 3a1: Face-Part Segmentation (FACE_JOB)

U-Net, 4-level encoder-decoder (~2M params) — **same architecture as `hand_seg`**, multi-class output head. Input: 192×192 RGB face crop (from L2a bbox + 10% padding). Output: 5-channel softmax — {background, eye_L, eye_R, mouth, face_skin}. Each part mask feeds the same mesh pipeline as hand_seg (contour glow + Delaunay triangulation).

**Dataset:** CelebAMask-HQ — 30K images, 19 per-pixel face-part classes. Merged to 5-class: {eye_L, eye_R, mouth (u_lip + l_lip + mouth), face_skin (skin), background}.

Checkpoint: `FACE_JOB/face_parts/checkpoints/best.pt`
---

### Layer 2b: Hand Segmentation ✓ TRAINED

U-Net (4-level encoder-decoder, ~2M params). Input: 192×192 RGB. Output: binary hand mask. No landmarks, no skeleton — mask shape is the hand representation throughout the entire pipeline.

**Dataset:** FreiHAND — 130,240 RGB images + 32,560 masks, augmented with Places365 backgrounds (36,500 scenes). Skin-tone augmentation for demographic diversity.
- Trained: `HAND_JOB/hand_seg/` — checkpoint at `hand_seg/checkpoints/best.pt`
- Val IoU: ~0.987

All hand overlays and OSC data are derived from the seg mask. No landmark model will be built.

**Cascade gate:** `hand_present(mask) := (mask_area / frame_area) ≥ HAND_MIN_AREA` (default 0.5%).
When the gate fails the pipeline short-circuits: Layer 3b is not run, gesture fields are null, OSC sends only `/hand/present 0` + `/hand/fps`, and the renderer falls back to the "no hand" UI state. This keeps the app cheap when the frame is empty and prevents noise-blob false gestures.

---

### Layer 2c: Body Pose Estimator

Lightweight HRNet-style backbone (or stacked hourglass), 17-point keypoint regression (COCO skeleton). Input: 512px person crop. Outputs 17 (x, y, confidence) tuples.

**COCO skeleton connections (bones to draw):**
nose↔eye, eye↔ear, shoulder↔elbow, elbow↔wrist, shoulder↔hip, hip↔knee, knee↔ankle, shoulder↔shoulder, hip↔hip

**Dataset:** Same dataset as Layer 1.
- `person_keypoints_train2017.json` contains 17-point skeleton annotations for all person instances.
- Filter: only instances with `num_keypoints >= 5` and `area > 32²`. Crop person bbox with padding, possible resize, normalize keypoints to crop-relative coords.

---

### Layer 3a2: Emotion Classifier (FACE_JOB)

4-block wide CNN + GAP + FC → 7 — **same architecture as `gesture_classifier`** with FC resized. Input: **64×64 RGB** face crop. Runs in parallel with Layer 3a1 (neither gates the other).

**Classes (7):** happy, sad, neutral, surprise, anger, fear, disgust.

**Dataset:** Combined **FER+ ∪ RAF-DB ∪ ExpW** (~140K images, ~20K/class):
- FER+ (35K, 48×48 grayscale, relabeled FER2013) → upscale to 64×64, replicate channels. Source: `microsoft/FERPlus`.
- RAF-DB (15K, 100×100 RGB) → downscale to 64×64. Source: official request form.
- ExpW (90K, web-scraped, 7 emotions) → resize to 64×64. Source: Google Drive.
- Align all three to the 7-class intersection (drop FER+'s "contempt").

**Weighting:** per-dataset loss weights (RAF-DB 1.5×, FER+ 1.0×, ExpW 0.7×) + balanced batch sampler so ExpW volume doesn't drown cleaner sets. Class-weighted CE for residual imbalance.

**Dataset swap:** `emotion/train.py --datasets` flag — drop ExpW or add AffectNet manual subset (~30GB) without model changes if quality plateaus.

Checkpoint: `FACE_JOB/emotion/checkpoints/best.pt`

---

### Layer 3b: Gesture Classifier ✓ TRAINED

4-block wide CNN, global avg pool, FC → 18. Input: **96×96** hand crop (from seg mask bbox).

**Dataset:** HaGRID 30k 384p sample — 18 classes, ~30K images
- Trained: `HAND_JOB/gesture/` — checkpoint at `gesture/checkpoints/best.pt`
- Val F1: 0.984

---

## OSC Output

Single port `127.0.0.1:9000`. All values normalized to frame dimensions (0–1) unless noted. Fires every frame. When no hand present, only `/hand/present 0` sends.

```
/hand/present              int      1 or 0
/hand/fps                  float    current inference fps

/hand/gesture              string   e.g. "like"
/hand/gesture/confidence   float    0–1
/hand/gesture/second       string   runner-up class
/hand/gesture/second_conf  float    0–1

/hand/area                 float    0–1  fraction of frame covered by mask
/hand/centroid             float float  x y normalized
/hand/bbox                 float float float float  x y w h normalized
/hand/aspect_ratio         float    bbox w/h
/hand/orientation          float    degrees, principal axis from mask moments
/hand/solidity             float    0–1  area / convex hull area (1 = convex fist)
/hand/contour              float[]  x1 y1 x2 y2 ... normalized contour vertices

/hand/velocity             float float  dx dy normalized centroid delta per frame
/hand/speed                float    magnitude of velocity
```

When Layer 1 (Jake) is integrated, schema extends with `/detection/{i}/` and `/person/{i}/` namespaces per cascade architecture.

**FACE_JOB `/face/*` namespace** (shared port 9000, per-face indexed `/face/{i}/`):

```
/face/present                  int      face count (0 if none)
/face/fps                      float    face cascade fps

/face/{i}/bbox                 float×4  x y w h normalized
/face/{i}/centroid             float×2  x y normalized
/face/{i}/confidence           float    detector confidence

/face/{i}/emotion              string   e.g. "happy"
/face/{i}/emotion/confidence   float    0-1
/face/{i}/emotion/second       string
/face/{i}/emotion/second_conf  float

/face/{i}/eye_L/centroid       float×2  normalized (frame-relative)
/face/{i}/eye_L/area           float    fraction of frame
/face/{i}/eye_L/contour        float[]  x1 y1 x2 y2 ... normalized
/face/{i}/eye_R/*              (same as eye_L)
/face/{i}/mouth/*              (same)
/face/{i}/face_skin/area       float
```

When no face present, only `/face/present 0` + `/face/fps` send.

---

## Overlays

The rendered frame is the TD source feed.

**Object contours** — Mask R-CNN binary masks → `cv2.findContours` → draw outline only (no fill). 1-2px stroke, per-class color. Clean silhouette around whatever the model recognizes.

**Face mesh overlays** — derived from Face-Part U-Net (Layer 3a1) masks, **same philosophy as hand mesh**:
- *Face-light mesh* — `face_skin` mask → thin (~1px) contour at ~15% opacity. Establishes silhouette without clutter.
- *Eye mesh* (per eye) — tight contour at ~70% opacity, bright accent color, small Delaunay triangulation (~20 points).
- *Mouth mesh* — tight contour at ~70% opacity, different accent color, Delaunay triangulation (~30 points).
- *Emotion label* — text near face bbox top, monospace, color-ramped to confidence.

No landmarks, no skeletal trace. Face-part masks are the face representation.

**Hand mesh overlay** — derived entirely from U-Net seg mask, no landmarks:
- *Contour glow* — 1-2px bright edge trace of mask outline. Opacity scales with gesture confidence.
- *Semi-transparent fill* — ~15% opacity color fill, hue per gesture class.
- *Delaunay triangulation* — ~40 sampled contour points triangulated, faint white mesh lines (~10% opacity) inside the mask.

No hand skeleton. No landmarks. Mask shape is the hand representation.

**Body pose skeleton** — 17 COCO keypoints connected as bones (see Layer 2c). Drawn as thin lines + small joint dots. Looks like a stick figure trace, not a filled silhouette.

**Confidence display** — per-detection label: class name + confidence %. Monospace font, color-ramped to confidence (low = dim/desaturated, high = bright/saturated). Small enough to not clutter, legible enough to read. Can be a fill bar instead of or alongside text.

**Aesthetic intent** — dark/transparent background survives TD compositing. Glowing or slightly bloomed strokes respond well to TD effects. All overlays hidden below confidence threshold.

---

## Project Structure

Actual layout (each cascade self-contained in its own directory, mirroring `HAND_JOB/`):

```
LEARNIN_MACHINES/
├── HAND_JOB/                  # ✓ trained, live_app working
│   ├── hand_seg/              # U-Net
│   ├── gesture/               # classifier
│   ├── live_app/              # webcam → cascade → render → NDI + OSC
│   ├── shared/
│   └── train_all.py           # sequential + thermal watchdog
├── FACE_JOB/                  # this spec
│   ├── face_det/              # FCOS-style anchor-free detector
│   ├── face_parts/            # U-Net, 5-class face-part seg
│   ├── emotion/               # 4-block CNN, 7 classes
│   ├── shared/
│   │   └── datasets/          # wider_face, celeba_mask, ferplus, rafdb, expw
│   ├── download_data.py
│   └── train_all.py           # sequential + thermal watchdog (port of hand_job)
├── combined_app/              # deferred — runs HAND_JOB + FACE_JOB cascades together
├── docs/
└── archetecture.md
```

**Deps:** `torch`, `torchvision`, `opencv-python`, `numpy`, `python-osc`, `pycocotools`, `scipy` (Delaunay)

---

## Combined Live App (Deferred)

Once FACE_JOB models are trained, a `combined_app/` will supersede `HAND_JOB/live_app/`:
- Shared webcam capture (single OpenCV source)
- Parallel cascades per frame: hand (seg → gesture) + face (detect → parts + emotion)
- Merged renderer: hand meshes + face meshes + both labels on one frame
- Merged OSC: `/hand/*` and `/face/*` on port 9000
- NDI output from combined frame

Spec'd separately after training validation.

---

## Training

From scratch throughout — no pretrained weights. Random init (Kaiming/Xavier). Data augmentation: random crops, flips, color jitter, rotation. Orchestrated by per-cascade `train_all.py` with `powermetrics`-based thermal watchdog (SIGSTOP on Trapping/Sleeping, SIGCONT on Nominal). Runs sequentially on MPS — not parallel (GPU contention).

### Time Estimates

| Model | Dataset | Images | M-series 64GB Mac (MPS) |
|-------|---------|--------|-------------------------|
| Mask R-CNN | COCO 2017 | 118K | 12–24h |
| **Face Detector (FACE_JOB)** | WIDER FACE filtered | ~20K | **4–6h** |
| **Face-Part U-Net (FACE_JOB)** | CelebAMask-HQ | 30K | **2–3h** |
| Body Pose Estimator | COCO 2017 | 118K | 6–14h |
| **Emotion Classifier (FACE_JOB)** | FER+ ∪ RAF-DB ∪ ExpW | ~140K | **2–4h** |
| Hand Seg ✓ | FreiHAND | 130K | trained |
| Gesture ✓ | HaGRID 30K | 30K | trained |

**FACE_JOB overnight total: ~8–13h** for all three models sequentially via `FACE_JOB/train_all.py`.

---