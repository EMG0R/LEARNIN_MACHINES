# Real-Time ML Vision Pipeline — AI Mirror

Real-time computer vision pipeline. Python, PyTorch, OpenCV, pygame. Webcam in, cascading ML models, mesh/contour overlays rendered to a fullscreen mirror display. All models built and trained from scratch. Self-contained — no OSC, no external integrations.

**Dev platform:** Mac, Apple Silicon (MPS), float32
**Deploy target:** Pi 5 8GB + Hailo-8 AI HAT+ (26 TOPS) — Pi migration deferred after all models trained

---

## Detection Cascade

Each layer gates the next. Downstream models only run when upstream fires.

```
Webcam Frame
  │
  ▼
[Layer 1: OBJECTIFICATION — 23-class semantic seg, 320×320]
  │  Per-pixel class map → contour extraction → mesh overlays
  │
  ├─ person pixels ≥ PERSON_MIN_AREA → person crop
  │     │
  │     ├──▶ [Layer 2a: Face Detector — FCOS-style, multi-face, 320×320]
  │     │      → bbox(es) per face, NMS. Gate: face_present (count ≥ 1)
  │     │      └─ face found → crop each face, run in parallel:
  │     │          ├─ [Layer 3a1: Face-Part U-Net — 192×192, 5-class]
  │     │          │    → masks {eye_L, eye_R, mouth, face_skin, background}
  │     │          └─ [Layer 3a2: Emotion Classifier — 64×64, 7-class]
  │     │               → happy/sad/neutral/surprise/anger/fear/disgust + confidence
  │     │
  │     ├──▶ [Layer 2b: Hand Segmentation U-Net — 192×192, binary]
  │     │      → binary hand mask → contour/mesh overlay
  │     │      └─ gate: mask area ≥ HAND_MIN_AREA
  │     │          └─ [Layer 3b: Gesture Classifier — 96×96, 18-class]
  │     │               → 18 HaGRID classes + confidence (top-1 + runner-up)
  │     │
  │     └──▶ [Layer 2c: Body Pose Estimator — planned, not yet built]
  │            → 17 COCO keypoints → skeletal trace
  │
  ▼
[Render Thread] → pygame fullscreen display
```

---

## Models

All `nn.Module` from scratch. Random init (Kaiming/Xavier).

---

### Layer 1: OBJECTIFICATION ✓ IMPLEMENTED

CSPDarknet-lite backbone + U-Net FPN decoder, 24-channel semantic seg (23 classes + background). Input: 320×320 RGB letterboxed.

**Classes (23 foreground):** person, vehicle, skateboard, phone, device, animal, plant, cup, spork, bowl, footwear, glasses, headphones, chair, couch, table, lamp, book, clock, bag, guitar, trumpet, piano.

**Dataset:** OpenImages V7 filtered to 23 classes — ~10GB, ~40–80K images.

Checkpoint: `OBJECTIFICATION/seg/checkpoints/best.pt`
Spec: `docs/superpowers/specs/2026-04-24-objectification-design.md`

---

### Layer 2a: Face Detector ✓ TRAINED

Anchor-free FCOS-style CNN (~1.5M params). Input: 320×320 RGB letterboxed. Output: bboxes + objectness + centerness. Multi-face native. NMS at inference.

**Dataset:** WIDER FACE filtered (~15–20K images, webcam-realistic faces only).

Checkpoint: `FACE_JOB/face_det/checkpoints/face_det_v1.pt`

---

### Layer 3a1: Face-Part Segmentation ✓ TRAINED

U-Net 4-level encoder-decoder (~2M params). Input: 192×192 RGB face crop. Output: 5-class softmax — {background, eye_L, eye_R, mouth, face_skin}.

**Dataset:** CelebAMask-HQ — 30K images.

Checkpoint: `FACE_JOB/face_parts/checkpoints/face_parts_v1.pt`

---

### Layer 3a2: Emotion Classifier ✓ TRAINED

4-block wide CNN + GAP + FC → 7. Input: 64×64 RGB face crop.

**Classes:** happy, sad, neutral, surprise, anger, fear, disgust.

**Dataset:** FER+ ∪ RAF-DB ∪ ExpW (~140K images).

Checkpoint: `FACE_JOB/emotion/checkpoints/emotion_v1.pt`

---

### Layer 2b: Hand Segmentation ✓ TRAINED

U-Net 4-level encoder-decoder (~2M params). Input: 192×192 RGB. Output: binary hand mask.

**Dataset:** FreiHAND — 130K images + Places365 backgrounds. Val IoU ~0.987.

Checkpoint: `HAND_JOB/hand_seg/checkpoints/hand_seg_v7.pt`

---

### Layer 3b: Gesture Classifier ✓ TRAINED

4-block wide CNN, global avg pool, FC → 18. Input: 96×96 hand crop from seg mask bbox.

**Dataset:** HaGRID 30K sample. Val F1: 0.984.

Checkpoint: `HAND_JOB/gesture/checkpoints/gesture_v7.pt`

---

### Layer 2c: Body Pose Estimator (planned)

Lightweight HRNet-style, 17 COCO keypoints. Input: 512px person crop. Not yet built — slots in when ready.

---

## Overlays

**OBJECTIFICATION contours** — per-class colored 1–2px contour strokes + optional 15% fill. Per-class color palette from `class_map.json`.

**Face mesh** — face_skin thin outline at ~38% opacity, eye/mouth tight mesh at ~70% opacity, emotion label near face bbox top. Derived entirely from face-part seg masks.

**Hand mesh** — contour glow + 15% fill hue-per-gesture + Delaunay triangulation. Derived entirely from seg mask. No skeleton, no landmarks.

**Body pose skeleton** — 17 COCO keypoints connected as bones. Thin lines + small joint dots.

---

## Mirror Display

**Hardware (target):** Samsung 24" VA 1080p + glossy screen protector overlay. Frame: SK6812 RGBW LED strip in aluminum diffuser channel, separate 5V supply, Pi GPIO via `rpi-ws281x`.

**Camera (target):** Pi HQ Camera (IMX477) + 6mm f/1.2 CS-mount lens, CSI ribbon.

**Dev:** standard webcam, Mac display.

**Rendering:**
- Camera feed flipped horizontally (mirror effect)
- Slight desaturation + contrast curve (mirror tone grade) via precomputed LUT
- Ambient flow-field effects layer composited additively at ~25% opacity
- ML overlays drawn on top
- Minimal HUD: clock (top-right), detected-class strip (bottom)

---

## Project Structure

```
LEARNIN_MACHINES/
├── HAND_JOB/                  # ✓ trained
│   ├── hand_seg/
│   ├── gesture/
│   ├── live_app/              # standalone hand-only app (reference)
│   └── train_all.py
├── FACE_JOB/                  # ✓ trained
│   ├── face_det/
│   ├── face_parts/
│   ├── emotion/
│   └── train_all.py
├── OBJECTIFICATION/           # ✓ implemented, training pending
│   ├── seg/
│   └── data_pipeline/
├── combined_app/              # ← AI Mirror app
│   ├── app.py                 # main: 3 threads, pygame fullscreen
│   ├── config.py              # all constants and checkpoint paths
│   ├── models.py              # hand + face model loading + inference
│   ├── objectification_model.py  # OBJECTIFICATION wrapper
│   ├── effects.py             # ambient numpy flow-field
│   ├── tone.py                # mirror LUT + color grade
│   ├── renderer.py            # all cv2 drawing
│   └── tests/                 # unit tests for new modules
├── archetecture.md
└── docs/superpowers/
    ├── specs/
    └── plans/
```

---

## Pi Migration Path (deferred)

After all models trained and validated on Mac:

1. Per model: `PyTorch → ONNX → Hailo DFC → .hef`
2. Swap inference backend per model (same input/output interface)
3. Camera: swap `cv2.VideoCapture` for `picamera2` CSI capture
4. LEDs: `rpi-ws281x` GPIO init on startup (no-op shim on Mac)
5. Display: same pygame fullscreen, Pi drives HDMI

No app restructure needed.

---

## Training Time Estimates

| Model | Dataset | Status |
|---|---|---|
| OBJECTIFICATION | OI V7 ~40–80K | implemented, training pending |
| Face Detector ✓ | WIDER FACE ~20K | trained |
| Face-Part U-Net ✓ | CelebAMask-HQ 30K | trained |
| Emotion CNN ✓ | FER+∪RAF-DB∪ExpW ~140K | trained |
| Hand Seg ✓ | FreiHAND 130K | trained |
| Gesture ✓ | HaGRID 30K | trained |
