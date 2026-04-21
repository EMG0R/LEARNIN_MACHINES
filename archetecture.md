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
  │     ├──▶ [Layer 2a: Face Detector]
  │     │      → face bbox + 5 landmarks → draw face skeletal trace (eyes, nose, mouth)
  │     │      └─ face found →
  │     │          [Layer 3a: Emotion Classifier]
  │     │            → happy / sad / neutral / surprised + confidence
  │     │
  │     ├──▶ [Layer 2b: Hand Detector]
  │     │      → hand bbox + 21 landmarks → draw hand skeleton (finger joints + wrist)
  │     │      └─ hand found →
  │     │          [Layer 3b: Gesture Classifier]
  │     │            → thumbs up / thumbs down / palm / fist / gun fingers / peace + confidence
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

### Layer 2a: Face Detector

MobileNet-v2 style backbone, multi-scale SSD head, 5-point landmark regression (eyes, nose, mouth corners). 

**Dataset:** WIDER FACE — 32K images, 393K faces
- Downloaded from [shuoyang1213.me/WIDERFACE](http://shuoyang1213.me/WIDERFACE)
- Annotations: `.txt` files with `[x, y, w, h]` per face. Parse with custom reader (no standard lib).
- Possible Resize images
---

### Layer 2b: Hand Segmentation ✓ TRAINED

U-Net (4-level encoder-decoder, ~2M params). Input: 192×192 RGB. Output: binary hand mask. No landmarks, no skeleton — mask shape is the hand representation throughout the entire pipeline.

**Dataset:** FreiHAND — 130,240 RGB images + 32,560 masks, augmented with Places365 backgrounds (36,500 scenes). Skin-tone augmentation for demographic diversity.
- Trained: `HAND_JOB/hand_seg/` — checkpoint at `hand_seg/checkpoints/best.pt`
- Val IoU: ~0.987

All hand overlays and OSC data are derived from the seg mask. No landmark model will be built.

---

### Layer 2c: Body Pose Estimator

Lightweight HRNet-style backbone (or stacked hourglass), 17-point keypoint regression (COCO skeleton). Input: 512px person crop. Outputs 17 (x, y, confidence) tuples.

**COCO skeleton connections (bones to draw):**
nose↔eye, eye↔ear, shoulder↔elbow, elbow↔wrist, shoulder↔hip, hip↔knee, knee↔ankle, shoulder↔shoulder, hip↔hip

**Dataset:** Same dataset as Layer 1.
- `person_keypoints_train2017.json` contains 17-point skeleton annotations for all person instances.
- Filter: only instances with `num_keypoints >= 5` and `area > 32²`. Crop person bbox with padding, possible resize, normalize keypoints to crop-relative coords.

---

### Layer 3a: Emotion Classifier

ResNet-18 style (4 residual groups, global avg pool, FC → 4). Input: **48×48** face crop.

**Dataset:** FER2013 — 35K images
- Downloaded from Kaggle (`kaggle datasets download -d msambare/fer2013`)
- CSV format: `emotion, pixels, Usage`. Each row: space-separated pixel values → reshape to 48×48.
- Filter to 4 classes: happy (3), sad (4), neutral (6), surprised (5),  disgust (6) fear(7), anger(8)

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

---

## Overlays

The rendered frame is the TD source feed.

**Object contours** — Mask R-CNN binary masks → `cv2.findContours` → draw outline only (no fill). 1-2px stroke, per-class color. Clean silhouette around whatever the model recognizes.

**Face skeletal trace** — connect the 5 landmarks into a minimal face structure: left-eye↔right-eye, eye↔nose, nose↔mouth-left, nose↔mouth-right, mouth-left↔mouth-right. Dots at each landmark. Sparse but readable as a face.

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

```
LEARNIN_MACHINES/
├── models/
│   ├── backbone.py            # ResNet-50 + FPN
│   ├── mask_rcnn.py
│   ├── face_detector.py
│   ├── hand_detector.py
│   ├── body_pose.py           # Lightweight HRNet-style, 17 keypoints
│   ├── emotion_classifier.py
│   └── gesture_classifier.py
├── training/
│   ├── train_mask_rcnn.py
│   ├── train_face.py
│   ├── train_hand.py
│   ├── train_body_pose.py
│   ├── train_emotion.py
│   ├── train_gesture.py
│   └── datasets/
│       ├── coco.py            # Used by both Mask R-CNN and body pose
│       ├── coco_pose.py       # Pose-specific loader (keypoint annotations)
│       ├── wider_face.py
│       ├── freihand.py
│       ├── fer2013.py
│       └── hagrid.py
├── pipeline/
│   ├── cascade.py             # Detection cascade + gating
│   ├── capture.py             # Webcam via OpenCV
│   ├── renderer.py            # Overlay drawing
│   └── osc_sender.py          # OSC → TouchDesigner
├── utils/
│   ├── transforms.py
│   └── nms.py
├── data/                      # Datasets (gitignored)
├── checkpoints/               # Weights (gitignored)
├── main.py
└── requirements.txt
```

**Deps:** `torch`, `torchvision`, `opencv-python`, `numpy`, `python-osc`, `pycocotools`

---

## Training

Transfer learning throughout. Pretrained backbone weights from `torchvision` (ResNet-50, MobileNetV2). Freeze early layers, train heads, fine-tune upper backbone. Data augmentation: random crops, flips, color jitter, rotation.

### Time Estimates

| Model | Dataset | Images | M-series 64GB Mac | A100 Cloud (~$1.50/hr) |
|-------|---------|--------|-------------------|------------------------|
| Mask R-CNN | COCO 2017 | 118K | 12–24h | 2–4h |
| Face Detector | WIDER FACE | 393K faces | 4–8h | 45–90 min |
| Hand Detector | FreiHAND | 130K | 2–4h | 30–60 min |
| Body Pose Estimator | COCO 2017 | 118K | 6–14h | 1–3h |
| Emotion Classifier | FER2013 | 35K | 30–90 min | 10–20 min |
| Gesture Classifier | HaGRID (18 classes) | 552K | 3–6h | 1–2h |

**2-person timeline:** ~ Training locally would take a lot of hours while cloud would cost $20-$40. Mix of both maybe, will depend on timing. 

---