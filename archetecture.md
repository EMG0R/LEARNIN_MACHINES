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

### Layer 1: Mask R-CNN

ResNet-50 + FPN backbone, RPN, ROI Align, detection head (cls + bbox), mask head (binary pixel mask → contour). 80 COCO classes.

**Dataset:** COCO 2017 — 118K images, 80 classes
- Downloaded from `wget http://images.cocodataset.org/zips/train2017.zip` + val + annotations
- Parse with `pycocotools`. During training, possible resize down

---

### Layer 2a: Face Detector

MobileNet-v2 style backbone, multi-scale SSD head, 5-point landmark regression (eyes, nose, mouth corners). 

**Dataset:** WIDER FACE — 32K images, 393K faces
- Downloaded from [shuoyang1213.me/WIDERFACE](http://shuoyang1213.me/WIDERFACE)
- Annotations: `.txt` files with `[x, y, w, h]` per face. Parse with custom reader (no standard lib).
- Possible Resize images
---

### Layer 2b: Hand Detector

SSD-style backbone, hand bbox + 21-point landmark regression (finger joints + wrist). 

**Dataset:** FreiHAND — 130K images, 3D landmarks
- Downloaded from [lmb.informatik.uni-freiburg.de/projects/freihand](https://lmb.informatik.uni-freiburg.de/projects/freihand)
- Annotations: JSON with 3D keypoints + camera intrinsics. **Project 3D → 2D:** `uv = K @ xyz / xyz[2]` per keypoint using provided `K` matrix.
- Possible resize to 512px. 
- Derive bboxes from projected 2D landmarks with padding.

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

### Layer 3b: Gesture Classifier

Small CNN (3-4 conv blocks, global avg pool, FC → 18). Input: **64×64** hand crop.

**Dataset:** HaGRID — full dataset, all 18 classes, ~552K images, ~716GB
- Download everything. This is an interactive installation — recognizing any hand gesture seen is the point.
- Annotations: JSON per-image with bbox + label. Crop hand region, resize to 64×64, normalize.
- Script: `hagrid download` (no subset filter)

---

## OSC Output

Layer 1 sends all detections every frame. Default target: `127.0.0.1:9000`.

```
/detection/count              → int
/detection/{i}/class          → string
/detection/{i}/confidence     → float
/detection/{i}/bbox           → [x, y, w, h] normalized
/detection/{i}/mask           → [x1,y1, x2,y2, ...] contour points

/person/{i}/face/emotion      → string
/person/{i}/face/confidence   → float
/person/{i}/face/landmarks    → [x1,y1, ...] 5 points
/person/{i}/hand/gesture      → string
/person/{i}/hand/confidence   → float
/person/{i}/hand/landmarks    → [x1,y1, ...] 21 points
/person/{i}/hand/finger/{f}/extension  → float 0–1 (f: thumb/index/middle/ring/pinky)
/person/{i}/hand/finger/{f}/pinch      → float 0–1 (distance thumb tip to finger tip, normalized)
/person/{i}/hand/palm_facing           → float -1–1 (toward cam vs away)
/person/{i}/hand/rotation              → float degrees (wrist rotation around forearm axis)
/person/{i}/hand/spread                → float 0–1 (avg angle between extended fingers)
/person/{i}/pose/keypoints    → [x1,y1,c1, x2,y2,c2, ...] 17 points (x,y normalized + confidence per joint)
/person/{i}/pose/confidence   → float (mean joint confidence)
```

---

## Overlays

The rendered frame is the TD source feed.

**Object contours** — Mask R-CNN binary masks → `cv2.findContours` → draw outline only (no fill). 1-2px stroke, per-class color. Clean silhouette around whatever the model recognizes.

**Face skeletal trace** — connect the 5 landmarks into a minimal face structure: left-eye↔right-eye, eye↔nose, nose↔mouth-left, nose↔mouth-right, mouth-left↔mouth-right. Dots at each landmark. Sparse but readable as a face.

**Hand skeleton** — 21-joint tree drawn as connected bones: finger segments + wrist. Per-finger color or uniform, your call.

**Hand overlays (all derived from 21 landmarks, zero extra compute):**
- *Wrist-to-tip fan* — 5 lines from wrist to each fingertip. Morphs as fingers extend/curl. Claw-like when all curl, starburst when open.
- *Fingertip polygon* — connect tips of all extended fingers in order. Collapses as fingers fold. Pentagon when fully open, shrinks dynamically.
- *Knuckle web* — lines between MCP base knuckles across the palm (pinky↔ring↔middle↔index↔thumb base). Circuit-board look.
- *Proximity threads* — draw a line between any two fingertips within N pixels of each other. Appears/vanishes reactively as fingers approach and separate.
- *Joint curl arcs* — small arc drawn at each knuckle joint, radius proportional to bend angle. Reads as a tick mark showing how curled each segment is.
- *Palm centroid spokes* — compute centroid of all 21 points, draw lines from centroid to every landmark. Dense and radial, collapses to a tight cluster on fist.
- *Velocity trails* — if tracking hand across frames, draw fading tail lines from each fingertip. Fast movement = long bright trail.

**Per-finger computed state (all derived, no model):**
- Extension ratio per finger: `tip_distance / base_distance` → 0.0 (fully curled) to 1.0 (fully extended)
- Pinch distance: thumb tip to each other fingertip (normalized to palm width)
- Palm facing: dot product of palm normal vs camera Z axis → facing / away
- Hand rotation: wrist-to-middle-MCP angle relative to vertical
- Finger spread: angle between adjacent extended finger vectors

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