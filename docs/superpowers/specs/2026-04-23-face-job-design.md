# FACE_JOB — Real-Time Face Cascade Design

**Date:** 2026-04-23
**Status:** Design approved, pending implementation plan
**Mirrors:** `HAND_JOB/` — same philosophy, file layout, training orchestration, thermal protection, OSC/mesh rendering aesthetic.

---

## Goal

A face-analysis cascade that runs alongside the existing hand cascade. Detects faces (multi-face), overlays a tight mesh on eyes and mouth plus a light mesh on the face, and classifies emotion. Outputs everything over OSC to TouchDesigner. All models trained from scratch on Apple MPS, sequentially overnight, with thermal protection.

Ultimate target: a combined `live_app` that runs `HAND_JOB` + `FACE_JOB` cascades together, sharing webcam, renderer, and OSC sender.

---

## Cascade

```
Webcam Frame
  │
  ▼
[Layer 1: Face Detector — FCOS-style anchor-free]
  │  Multi-face bbox output. NMS. Gate: faces present?
  │
  ├─ No face → SKIP L2a/L2b, send only /face/present 0
  │
  └─ Face(s) found → crop each face region
        │
        ├──▶ [Layer 2a: Face-Part U-Net]          ┐  (run in PARALLEL per face crop —
        │      Multi-class masks:                 │   logically parallel, executed
        │      {eye_L, eye_R, mouth, face_skin}   │   sequentially on single MPS)
        │      → meshes per part                  │
        │                                         │
        └──▶ [Layer 2b: Emotion Classifier]       ┘
               7 classes, 64×64 RGB
               → label + confidence
```

L2a and L2b are independent: neither gates the other. Both only run when L1 fires.

---

## Models

All `nn.Module` from scratch. Random init (Kaiming/Xavier). No pretrained weights unless explicitly noted.

### Layer 1: Face Detector

**Architecture:** Anchor-free, FCOS-style. Small CNN backbone (~4 stages, ~1.5M params total), single-scale head outputting per-pixel {objectness, bbox l/t/r/b, centerness}. NMS at inference.

**Input:** 320×320 RGB (letterboxed from webcam frame).

**Output:** List of (bbox, confidence). Multi-face native via NMS.

**Dataset:** WIDER FACE, **filtered** to webcam-realistic subset:
- Drop faces with shorter bbox side < 40px
- Drop images with >8 faces (crowd shots)
- Keep "easy" + "medium" difficulty; drop "hard"
- Result: ~15-20K images, ~30-60K face instances

**Loss:** Focal loss (objectness) + GIoU (bbox) + BCE (centerness).

**Checkpoint:** `FACE_JOB/face_det/checkpoints/best.pt`

**Gate:** `face_present := (num_detections ≥ 1 after NMS @ conf ≥ 0.5)`.

---

### Layer 2a: Face-Part U-Net

**Architecture:** U-Net, 4-level encoder-decoder, ~2M params — **same as `hand_seg`** but multi-class output head.

**Input:** 192×192 RGB face crop (letterboxed bbox from L1 + 10% padding).

**Output:** 5-channel softmax mask: {background, eye_L, eye_R, mouth, face_skin}. Each non-background channel is the binary mask for that face part.

**Dataset:** CelebAMask-HQ — 30K images, 19 per-pixel face-part classes.
- Merge {l_eye} → eye_L, {r_eye} → eye_R
- Merge {u_lip, l_lip, mouth} → mouth
- {skin} → face_skin
- Everything else → background
- Result: 30K images, 5-class masks

**Loss:** Dice + CE, class-weighted (eye masks are tiny — upweight).

**Checkpoint:** `FACE_JOB/face_parts/checkpoints/best.pt`

**Renderer consumes:** each mask → contour → same mesh pipeline as hand_seg (contour glow + Delaunay triangulation). Eye/mouth meshes dense and bright; face_skin mesh faint and thin.

---

### Layer 2b: Emotion Classifier

**Architecture:** 4-block wide CNN + GAP + FC → 7. **Same as `gesture_classifier`** with last FC resized.

**Input:** 64×64 RGB face crop (from L1 bbox, ~5% padding).

**Output:** 7 classes — {happy, sad, neutral, surprise, anger, fear, disgust} + confidence.

**Dataset:** **Combined FER+ ∪ RAF-DB ∪ ExpW**, 7-class intersection.
- FER+ (35K, 48×48 grayscale, relabeled FER2013) → upscale to 64×64, replicate to 3 channels
- RAF-DB (15K, 100×100 RGB) → downscale to 64×64
- ExpW (90K, web-scraped, 7 emotions) → resize to 64×64
- Drop FER+'s "contempt" class. Align all three to the 7-class intersection.
- Combined: ~140K images, ~20K/class average.

**Per-dataset weighting:**
- Loss weights: RAF-DB 1.5×, FER+ 1.0×, ExpW 0.7× (trust cleaner data more)
- Balanced batch sampler: each batch draws from all three proportionally, so ExpW volume doesn't drown cleaner sets

**Loss:** Cross-entropy, class-weighted for residual imbalance (fear/disgust still smallest after combination).

**Checkpoint:** `FACE_JOB/emotion/checkpoints/best.pt`

**Dataset swap:** `emotion/train.py` takes a `--datasets` flag. If quality plateaus, drop ExpW or add AffectNet manual subset (~30GB) as a 4th source without code changes to the model.

---

## OSC Schema

Port `127.0.0.1:9000` shared with hand cascade. New namespace `/face/*`. Per-face indexed via `/face/{i}/` when multiple faces present.

```
/face/present                int      count of faces (0 if none)
/face/fps                    float    face-cascade inference fps

# Per-face (i = 0, 1, ...)
/face/{i}/bbox               float×4  x y w h normalized
/face/{i}/centroid           float×2  x y normalized
/face/{i}/confidence         float    detector confidence

/face/{i}/emotion            string   e.g. "happy"
/face/{i}/emotion/confidence float    0-1
/face/{i}/emotion/second     string   runner-up class
/face/{i}/emotion/second_conf float   0-1

/face/{i}/eye_L/centroid     float×2  x y normalized (frame-relative)
/face/{i}/eye_L/area         float    fraction of frame
/face/{i}/eye_L/contour      float[]  x1 y1 x2 y2 ... normalized
/face/{i}/eye_R/*            (same schema as eye_L)
/face/{i}/mouth/*            (same schema)
/face/{i}/face_skin/area     float    fraction of frame (contour optional — may be too heavy)
```

When no face present, only `/face/present 0` + `/face/fps` send.

---

## Overlays

**Face-light mesh** — `face_skin` mask → thin (~1px) contour at ~15% opacity. Establishes the face silhouette without clutter.

**Eye mesh** (per eye) — tight contour at 70% opacity, bright accent color, small Delaunay triangulation (~20 points). High-detail, draws the viewer's attention.

**Mouth mesh** — tight contour at 70% opacity, different accent color, Delaunay triangulation (~30 points).

**Emotion label** — text near face bbox top, color-ramped to confidence. Same monospace style as gesture label.

**Face bbox** — optional faint rectangle, mostly for debug.

Aesthetic consistency with hand overlays: dark/transparent background, bloomable strokes, all hidden below confidence threshold.

---

## File Layout

Mirror of `HAND_JOB/`:

```
FACE_JOB/
├── face_det/
│   ├── train.py
│   ├── model.py               # FCOS-style detector
│   └── checkpoints/
├── face_parts/
│   ├── train.py               # nearly verbatim from hand_seg/train.py, multi-class output
│   ├── model.py               # U-Net, 5-class output head
│   └── checkpoints/
├── emotion/
│   ├── train.py
│   ├── model.py               # 4-block wide CNN, FC→7
│   └── checkpoints/
├── shared/
│   ├── transforms.py
│   └── datasets/
│       ├── wider_face.py
│       ├── celeba_mask.py
│       ├── ferplus.py
│       ├── rafdb.py
│       └── expw.py
├── download_data.py           # fetch WIDER FACE, CelebAMask-HQ, FER+, RAF-DB, ExpW
├── train_all.py               # sequential orchestrator + thermal watchdog
└── logs/
```

---

## Training Orchestration

**`FACE_JOB/train_all.py`** — direct port of `HAND_JOB/train_all.py` thermal watchdog:
- `powermetrics` subprocess polls thermal level every 1s
- On `Trapping` or `Sleeping` → SIGSTOP training subprocess; on `Nominal` → SIGCONT
- Runs three trainings **sequentially**: face_det → face_parts → emotion
- Logs per-run to `FACE_JOB/logs/{run}_{timestamp}.log`
- Requires root for powermetrics (same as hand_job)

**Time estimate (M-series MPS, overnight):**
| Model | Data | Est. Time |
|---|---|---|
| Face detector | ~20K images | 4-6h |
| Face-part U-Net | 30K images | 2-3h |
| Emotion classifier | 140K images | 2-4h |
| **Total** | | **8-13h overnight** |

---

## Data Download

`FACE_JOB/download_data.py` fetches:
- WIDER FACE: `http://shuoyang1213.me/WIDERFACE` (direct http download, ~4GB)
- CelebAMask-HQ: Google Drive link (~2GB)
- FER+: GitHub (`microsoft/FERPlus`) + FER2013 Kaggle (~60MB)
- RAF-DB: requires request form on official site — **manual step, user fetches and drops into `data/rafdb/`**
- ExpW: Google Drive (~2GB)

Total disk: ~9GB.

---

## Integration With Existing App

Tonight's scope: train the three FACE_JOB models.

Follow-up scope (separate spec when ready): a `combined_app/` that subsumes `HAND_JOB/live_app/`:
- Shared webcam capture
- Parallel cascades: hand (seg → gesture) + face (detect → parts + emotion)
- Merged renderer: both overlays on one frame
- Merged OSC: `/hand/*` and `/face/*` on same port
- NDI output from combined frame

Deferred. Spec'd separately once all face models are trained and validated.

---

## Success Criteria

- **Face detector:** val AP ≥ 0.75 on filtered WIDER FACE, works on 1-5 people in front of webcam with visible stable bboxes.
- **Face-part U-Net:** val mean IoU ≥ 0.85, eye/mouth masks visually clean and mesh-able at 30fps.
- **Emotion classifier:** val F1 ≥ 0.70 macro-averaged across 7 classes. Fear/disgust may be lower — acceptable if happy/sad/neutral/surprise are ≥ 0.80.
- **Pipeline:** end-to-end inference < 33ms per frame on M-series (30fps target).

If emotion stalls below 0.70, first remedy is to drop ExpW and retrain on cleaner FER+ ∪ RAF-DB only. Second remedy is add AffectNet manual subset.

---

## Out of Scope

- Facial landmarks (68-point, etc.) — eyes/mouth come from seg masks, not points.
- Gaze estimation — eyes are masks, not iris centers. Possible future extension.
- Face recognition / identification — detection only.
- Video-temporal emotion smoothing — per-frame classification only; TD can smooth downstream.
- Combined live app — deferred to separate spec.
