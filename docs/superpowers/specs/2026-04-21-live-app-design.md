# Live App — Design

**Date:** 2026-04-21  
**Scope:** Integrated real-time inference app (`HAND_JOB/live_app/`) — the master display and output app for the full vision pipeline.  
**Deliverable:** A single-process Python app that runs trained models, renders a fullscreen aesthetic display, streams video via NDI, and sends structured data via OSC to TouchDesigner.

---

## Non-Goals

- No mouse or keyboard interactivity — pure AI-driven display.
- No hand skeleton or landmarks — ever. Hand visual is seg mask only.
- No Layer 1 (general object detection) for now — Jake's COCO model slots in later via a stub.
- No face, body pose, or emotion layers yet — stubs exist for future integration.
- No GUI controls or config UI — all settings in `config.py`.

---

## Architecture

Single process, one main loop. Efficiency is the top priority.

```
webcam frame
  → HandSegModel (U-Net)     → binary mask
  → GestureModel (CNN)       → label + confidence (top-2)
  → Renderer                 → composited frame (raw + mesh overlay + UI panels)
  → NDI sender               → rendered frame as video stream
  → OSC sender               → all structured data on one port
  → cv2.imshow               → fullscreen reference monitor
```

Models load once at startup from checkpoints. Every frame runs seg → classify → render → send. No gating on person detection — hands are detected directly from the full frame.

**Stub slots** (no-ops that return None, wired into the loop):
- `layer1_object_detection(frame)` — Jake's COCO model
- `layer2a_face(crop)` — face detector
- `layer2c_body_pose(crop)` — body pose estimator
- `layer3a_emotion(face_crop)` — emotion classifier

---

## File Structure

```
HAND_JOB/
└── live_app/
    ├── app.py           # main loop — capture, infer, render, send
    ├── models.py        # load checkpoints, run inference (seg + gesture)
    ├── renderer.py      # mesh overlay + UI panel drawing
    ├── ndi_sender.py    # NDI video output
    ├── osc_sender.py    # OSC data output
    └── config.py        # all tunable constants
```

---

## Models

Both loaded from `HAND_JOB/*/checkpoints/best.pt` at startup.

**HandSegModel** — U-Net (4-level, ~2M params). Input: 192×192 RGB. Output: binary mask resized to frame resolution. Threshold at 0.5.

**GestureModel** — 4-block CNN. Input: 96×96 RGB hand crop (from seg mask bbox). Output: 18-class softmax. Top-2 label + confidence returned.

Hand crop derived from seg mask bounding box + 10% padding. If mask is empty (no hand present), gesture model is skipped entirely.

---

## Rendering

Fullscreen OpenCV window. Raw webcam frame as base layer. Two dedicated functions in `renderer.py`:

**`draw_mesh(frame, mask, confidence, gesture_idx) → frame`**
Draws the hand mesh overlay. Called every frame when mask is non-empty.
1. **Seg mask fill** — semi-transparent color fill over mask region (~15% opacity). Color driven by gesture class index (18 distinct hues via HSV wheel).
2. **Delaunay triangulation** — sample ~40 evenly-spaced contour points, run `cv2.Subdiv2D`, draw triangle edges as faint white lines (~10% opacity) clipped inside mask bounds.
3. **Contour glow** — `cv2.drawContours` 1-2px width, bright white, opacity scales linearly with confidence (conf 0.4 → opacity 0.0, conf 1.0 → opacity 1.0).

**`draw_ui(frame, gesture, confidence, second, second_conf, fps, present) → frame`**
Draws the two frosted card panels. Called every frame regardless of hand presence.
- Bottom-left: gesture name + confidence bar + runner-up in smaller text.
- Bottom-right: fps + presence dot.
- Confidence below 0.4 → panels show "—", mesh hidden.

Both functions take a frame copy and return it — no side effects.

---

## OSC Output

Single port (default `127.0.0.1:9000`). All values normalized to frame dimensions (0–1) unless noted. Fires every frame. When no hand present, only `/hand/present 0` sends.

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
/hand/contour              float[]  x1 y1 x2 y2 ... normalized contour vertices (~40 evenly-sampled points, same set used for Delaunay triangulation — guaranteed consistent between OSC and visual)
/hand/triangle_count       int      number of Delaunay triangles in mesh
/hand/triangles            float[]  x1a y1a x2a y2a x3a y3a ... normalized triangle vertices (triangle_count * 6 floats)

/hand/velocity             float float  dx dy normalized centroid delta per frame
/hand/speed                float    magnitude of velocity
```

**Mesh data guarantee:** `/hand/contour` and `/hand/triangles` are computed once per frame from the same 40 sampled points used to render the Delaunay mesh overlay. OSC data and visual are always in sync.

---

## NDI Output

Rendered frame (composited overlay on webcam feed) streamed as NDI source named `"LEARNIN_MACHINES"`. TouchDesigner receives it as a standard video input. Uses `PyNDI` or `ndi-python` library.

---

## UI Panels

Two frosted dark cards, no interactivity:

**Bottom-left — Gesture panel**
```
[ like          ]
[████████░░ 84% ]
[ two_up  84%   ]  ← runner-up, smaller text
```

**Bottom-right — Status panel**
```
[ ● 28 fps      ]
[ hand present  ]
```

All text: monospace, white, small (16px equivalent). Card: semi-transparent dark background, 8px corner radius. No borders.

---

## Config

```python
# config.py
WEBCAM_INDEX    = 0
FRAME_W, FRAME_H = 1280, 720
OSC_IP          = "127.0.0.1"
OSC_PORT        = 9000
NDI_SOURCE_NAME = "LEARNIN_MACHINES"
SEG_CKPT        = "../hand_seg/checkpoints/best.pt"
GESTURE_CKPT    = "../gesture/checkpoints/best.pt"
CONF_THRESHOLD  = 0.4
FILL_OPACITY    = 0.15
MESH_OPACITY    = 0.10
GLOW_WIDTH      = 2
CONTOUR_POINTS  = 40   # sampled points for Delaunay
```

---

## Performance Notes

- Inference runs on MPS (Apple Silicon GPU).
- Seg model: ~192×192 input keeps latency low.
- Gesture crop: only runs when mask is non-empty.
- Contour sampling at 40 points keeps Delaunay fast.
- Target: 25–30 fps on M-series Mac.
- Threading can be added later (capture thread + inference thread) if fps drops below 20.

---

## Future Layers (stubs, no-op today)

When Jake's COCO model is ready, `layer1_object_detection(frame)` returns bounding boxes. Person crops gate face/pose/emotion layers. Hand seg runs on person crops instead of full frame. OSC schema extends with `/person/{i}/` namespace per architecture doc.
