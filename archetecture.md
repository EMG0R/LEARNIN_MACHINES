# Real-Time ML Vision Pipeline

Real-time computer vision pipeline. Python, PyTorch, OpenCV. Webcam in, cascading ML models, minimalist contour overlays out. All models built and trained from scratch — no pre-trained weights. Layer 1 sends all detection data over OSC to TouchDesigner.

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
  │     │      → face bbox + 5 landmarks → draw contour + dots
  │     │      └─ face found →
  │     │          [Layer 3a: Emotion Classifier]
  │     │            → happy / sad / neutral / surprised + confidence
  │     │
  │     └──▶ [Layer 2b: Hand Detector]
  │            → hand bbox + 21 landmarks → draw skeleton
  │            └─ hand found →
  │                [Layer 3b: Gesture Classifier]
  │                  → open palm / fist / peace / thumbs up / pointing + confidence
  │
  ▼
[Overlay Renderer] → OpenCV display
```

---

## Models

All `nn.Module` from scratch. No ultralytics, detectron2, mediapipe. All weights randomly initialized (Kaiming/Xavier), trained on open-source data.

### Layer 1: Mask R-CNN

General object detection + instance segmentation. 80 COCO classes.

- ResNet-50 + FPN backbone
- Region Proposal Network (RPN)
- ROI Align
- Detection head (classification + box regression)
- Mask head (per-object binary pixel mask → contour outlines)
- Supporting utils: anchor generation, NMS

**Dataset:** COCO 2017 — 118K images, 80 classes

### Layer 2a: Face Detector (SSD-style)

Detects faces + 5 landmarks (eyes, nose, mouth corners) within person regions.

- Lightweight backbone (MobileNet-v2 style)
- Multi-scale single-shot detection head
- 5-point landmark regression

**Dataset:** WIDER FACE — 32K images, 393K faces

### Layer 2b: Hand Detector (SSD-style)

Detects hands + 21 landmarks (finger joints + wrist) within person regions.

- SSD-style backbone
- Hand bbox regression
- 21-point landmark regression

**Dataset:** FreiHAND — 130K images, 3D hand landmarks

### Layer 3a: Emotion Classifier

Classifies cropped face → 4 emotions.

- ResNet-18 style (4 residual block groups, global avg pool, FC → 4 classes)
- Input: 48x48 face crop
- Output: happy / sad / neutral / surprised + confidence

**Dataset:** FER2013 — 35K face images, filter to 4 classes

### Layer 3b: Gesture Classifier

Classifies cropped hand → 5 gestures.

- Small CNN (3-4 conv blocks, global avg pool, FC → 5 classes)
- Input: 64x64 hand crop
- Output: open palm / fist / peace / thumbs up / pointing + confidence

**Dataset:** HaGRID — 552K images, filter to 5 gestures

---

## OSC Output

Layer 1 sends all detections to TouchDesigner every frame. Person-specific data sent downstream.

```
/detection/count              → int
/detection/{i}/class          → string
/detection/{i}/confidence     → float (0-1)
/detection/{i}/bbox           → [x, y, w, h] normalized
/detection/{i}/mask           → [x1,y1, x2,y2, ...] contour points

/person/{i}/face/emotion      → string
/person/{i}/face/confidence   → float
/person/{i}/face/landmarks    → [x1,y1, ...] 5 points
/person/{i}/hand/gesture      → string
/person/{i}/hand/confidence   → float
/person/{i}/hand/landmarks    → [x1,y1, ...] 21 points
```

Default target: `127.0.0.1:9000`, configurable.

---

## Overlays

Minimalist. Drawn on full webcam frame after inference.

- **Contour outlines** — thin (1-2px), traced from segmentation masks, per-class color
- **Face landmarks** — small dots + connecting lines
- **Hand skeleton** — dots at 21 joints, lines along fingers
- **Labels** — small monospace text, `class confidence%`, semi-transparent background
- Suppressed below confidence threshold

---

## Project Structure

```
LEARNIN_MACHINES/
├── models/
│   ├── backbone.py            # ResNet-50 + FPN
│   ├── mask_rcnn.py           # Mask R-CNN
│   ├── face_detector.py       # Face detection
│   ├── hand_detector.py       # Hand detection
│   ├── emotion_classifier.py  # Emotion classification
│   └── gesture_classifier.py  # Gesture classification
├── training/
│   ├── train_mask_rcnn.py
│   ├── train_face.py
│   ├── train_hand.py
│   ├── train_emotion.py
│   ├── train_gesture.py
│   └── datasets/
│       ├── coco.py
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
│   ├── transforms.py          # Preprocessing
│   └── nms.py                 # Non-maximum suppression
├── data/                      # Datasets (gitignored)
├── checkpoints/               # Trained weights (gitignored)
├── main.py                    # Entry point
└── requirements.txt
```

**Deps:** `torch`, `torchvision`, `opencv-python`, `numpy`, `python-osc`

---

## Training

All from scratch. No pre-trained weights anywhere. Random init → train on open-source data.

| Model | Dataset | Size |
|-------|---------|------|
| Mask R-CNN | COCO 2017 | 118K images |
| Face Detector | WIDER FACE | 32K images / 393K faces |
| Hand Detector | FreiHAND | 130K images |
| Emotion Classifier | FER2013 | 35K images |
| Gesture Classifier | HaGRID | 552K images |

Data augmentation is critical (random crops, flips, color jitter, rotation).

### MPS training time estimates

| Model | Time |
|-------|------|
| Mask R-CNN (full COCO) | 3-7+ days |
| Mask R-CNN (class subset) | 12-24 hours |
| Face Detector | 6-12 hours |
| Hand Detector | 4-8 hours |
| Emotion Classifier | 30-90 min |
| Gesture Classifier | 30-90 min |

---

## Expansion

**Phase 2:** Full body pose estimation, object tracking across frames, more emotion/gesture classes
**Phase 3:** Style transfer on detected regions, richer OSC data (timelines, sequences)
**Phase 4:** Model quantization, audio reactivity, custom datasets
