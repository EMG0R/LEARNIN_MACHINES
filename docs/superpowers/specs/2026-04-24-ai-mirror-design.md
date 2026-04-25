# AI Mirror — Design Spec

**Status:** approved  
**Date:** 2026-04-24  
**Folder:** `combined_app/` (Mac dev), Pi deployment deferred  

---

## 1. What It Is

A standalone self-contained Python app. Webcam in, full ML cascade, rendered to a fullscreen display that looks as close to a physical mirror as possible. Dark flipped camera feed of yourself, ML contour overlays floating on top, ambient generative background running on a loop. No OSC, no external integrations, no TouchDesigner. Everything lives in one Python process with three threads.

---

## 2. Hardware Stack

| Part | Spec | Notes |
|---|---|---|
| Computer (dev) | Mac, Apple Silicon MPS | Float32, all current training |
| Computer (deploy) | Pi 5 8GB | ARM, Hailo inference backend — deferred |
| AI HAT+ | Hailo-8, 26 TOPS | Pi only; models compiled to `.hef` — deferred |
| Camera | Pi HQ Camera (IMX477) + 6mm f/1.2 CS-mount lens | CSI ribbon, ~$85 total |
| Display | Samsung 24" VA 1080p + glossy screen protector overlay | VA for deep blacks, glossy overlay for glass-like surface |
| Frame lighting | SK6812 RGBW LED strip in aluminum diffuser channel | Separate 5V supply, Pi GPIO via `rpi-ws281x` |

**Why HQ Camera + f/1.2 lens:** largest sensor of any native Pi camera (IMX477 BSI), interchangeable lens mount, f/1.2 handles both low-light and shooting through semi-transparent enclosure materials. Fixed white balance matched to SK6812 color temp — never auto WB.

**Why VA + glossy overlay:** VA panels hit 3000:1 contrast vs IPS's 1000:1 — dark areas are genuinely dark, which is what makes a digital feed read as a mirror. Clear glossy screen protector overlay converts the matte anti-glare coating to a glass-like reflective surface. Total ~$145.

---

## 3. App Architecture — Three Threads

```
┌─ Capture Thread ─────────────────────────────────────────────┐
│  picamera2 (Pi) / cv2.VideoCapture (Mac) @ 1080p 60fps       │
│  → frame_queue (maxsize=2, drop-oldest)                       │
└───────────────────────────────────────────────────────────────┘
                          │
┌─ Inference Thread ───────────────────────────────────────────┐
│  pop frame → two-path split:                                  │
│    display path: keep full 1080p, flip horizontal             │
│    inference path: resize to model inputs, run cascade        │
│  → result_queue (maxsize=2, drop-oldest)                      │
│    payload: {display_frame, overlay_data}                     │
└───────────────────────────────────────────────────────────────┘
                          │
┌─ Render Thread ──────────────────────────────────────────────┐
│  pop {display_frame, overlay_data}                            │
│  composite: effects_layer → display_frame → overlays → HUD   │
│  → cv2.imshow fullscreen                                      │
└───────────────────────────────────────────────────────────────┘
```

`maxsize=2` drop-oldest on both queues: if inference falls behind, frames are dropped rather than queued. Display always shows the freshest available result, never accumulates lag.

---

## 4. Camera Signal Split

The camera feed serves two completely separate purposes processed independently:

```
Raw 1080p frame
  ├─ Display path
  │    horizontal flip (mirror effect)
  │    → tone grade (see §6)
  │    → used as background layer in compositor
  │
  └─ Inference path
       resize 1080p → 320×320  → OBJECTIFICATION
       resize 1080p → 192×192  → hand_seg, face_parts
       resize 1080p →  64×64   → emotion classifier
       resize 1080p →  96×96   → gesture classifier
       (all resizes are letterbox with padding, matching each model's training pipeline)
```

Inference never sees the full-resolution frame. The display path never has model artifacts applied to it before grading.

---

## 5. ML Cascade

Identical cascade structure to `archetecture.md` (updated version — see §9), with two changes:

1. **No OSC anywhere.** All inference results are Python dicts passed via `result_queue`. No sockets, no external output.
2. **OBJECTIFICATION replaces Mask R-CNN as Layer 1.** See `docs/superpowers/specs/2026-04-24-objectification-design.md` for full spec.

```
Inference frame (320×320)
  │
  ▼
[OBJECTIFICATION — 23-class semantic seg, 320×320]
  │  per-pixel class map → contour extraction per class
  │
  ├─ person pixels ≥ PERSON_MIN_AREA → person crop
  │     │
  │     ├──▶ [face_det — FCOS anchor-free, 320×320]
  │     │      └─ face found → per-face crop:
  │     │          ├─ [face_parts U-Net — 192×192, 5-class]
  │     │          └─ [emotion CNN — 64×64, 7-class]
  │     │
  │     ├──▶ [hand_seg U-Net — 192×192, binary]
  │     │      └─ mask area ≥ HAND_MIN_AREA:
  │     │          └─ [gesture CNN — 96×96, 18-class]
  │     │
  │     └──▶ [body pose — planned, not yet built]
  │
  ▼
overlay_data dict → result_queue
```

---

## 6. Rendering Pipeline — Mirror Aesthetic

### 6.1 Tone grade (display path)

Applied per frame via a precomputed OpenCV LUT — single array lookup, negligible cost:

- Slight desaturation (~10%) — mirrors are slightly less saturated than raw camera
- Contrast curve: lift blacks gently, boost midtones — mirrors have a slight luminance rolloff
- No color cast — white balance locked to SK6812 LED color temp at calibration time

### 6.2 Layer composite (render thread)

Layers composited in order, bottom to top:

```
[1] Effects layer        — ambient generative background, ~25% opacity
[2] Graded display frame — full 1080p flipped, ~100% (replaces layer 1 where pixels exist)
[3] ML overlays          — contour meshes, fills, labels (see §6.3)
[4] HUD                  — clock, label strip (see §7)
```

The effects layer is only visible in areas where the camera frame is very dark (near-black), which is most of the frame area in normal room conditions. The VA panel's deep blacks make the transition natural.

### 6.3 ML overlays

Same contour mesh aesthetic defined in `archetecture.md` — unchanged:

- **OBJECTIFICATION:** per-class colored 1–2px contour strokes, optional 15% fill
- **Face parts:** tight eye/mouth mesh + glow, light face_skin outline, emotion label
- **Hand:** contour glow + 15% fill hue-per-gesture + Delaunay triangulation
- **Body pose:** stick figure skeleton (when built)
- All overlays hidden below confidence threshold
- All drawn on a transparent layer, composited onto the display frame

---

## 7. Background Effects — Ambient Generative Loop

Simplex/Perlin noise flow field rendered into a pre-allocated `[H, W, 3]` uint8 numpy buffer:

- Two noise fields offset in phase drive per-pixel hue and luminance
- Time-stepped by ~0.005 per frame — slow, smooth color drift (aurora / lava lamp feel)
- Fully vectorized numpy ops — no Python per-pixel loops
- Target cost: ≤5ms per frame on Pi ARM
- Alpha-composited at 25% opacity as the bottom layer
- **No reactivity to model outputs** — pure ambient loop, fire and forget

The buffer is updated in the render thread between frame composites. If the render thread is busy it skips the effect tick — display never blocks on effects.

---

## 8. Frame Lighting — SK6812 RGBW LEDs

SK6812 RGBW (not WS2812B RGB) for accurate white — dedicated white LED chip vs combined RGB which reads yellow-green on skin.

- Mounted in aluminum channel extrusion with frosted diffuser — soft even fill, no hotspots
- Positioned on the inner face of the 3D-printed frame, angled toward the viewer
- Controlled via `rpi-ws281x` from Pi GPIO (one pin, minimal CPU)
- On Mac dev: LEDs absent, lighting handled by room environment
- Startup: set warm white matched to calibrated white balance and leave it
- Separate 5V supply — ~75 LEDs at full brightness = ~4.5A, not safe to pull from Pi rail

---

## 9. Extra Features

Minimal — the ML overlays are already visually dense:

- **Clock** — current time, small monospace text, top corner, updated every second
- **Label strip** — bottom of frame: active OBJECTIFICATION classes + current gesture + current emotion, updated each inference cycle
- **FPS counter** — dev only, toggled with a keypress (`f`), hidden in normal use

Nothing else. No weather, no widgets.

---

## 10. Architecture Doc Changes (`archetecture.md`)

The existing `archetecture.md` needs a rewrite to reflect current state:

- **Remove:** all Jake/class/group references — solo project
- **Remove:** all OSC sections and OSC schema — no external output
- **Remove:** Mask R-CNN Layer 1 — replaced by OBJECTIFICATION
- **Add:** OBJECTIFICATION as Layer 1 (reference the spec)
- **Add:** AI Mirror as the target deployment context
- **Update:** "Combined Live App" section → becomes the AI Mirror app description
- **Keep:** Pi v2 notes, updated to reflect Hailo `.hef` migration workflow
- **Keep:** body pose as planned/not yet built

The OBJECTIFICATION spec (`docs/superpowers/specs/2026-04-24-objectification-design.md`) also has OSC references in §7.1 that need to be removed or reframed as internal data only.

---

## 11. Pi Migration Path (Deferred)

All models trained and validated on Mac first. Pi migration is a separate phase:

1. Per model: `PyTorch → ONNX → Hailo DFC → .hef`
2. Swap inference backend: one file per model, same input/output interface as Mac version
3. Camera: swap `cv2.VideoCapture` for `picamera2` CSI capture
4. LEDs: `rpi-ws281x` GPIO control (no-op on Mac)
5. Display: same OpenCV fullscreen, Pi drives HDMI

No app restructure needed. The three-thread architecture runs identically on both platforms. The only platform-specific code is in the camera capture module and the inference backend module.

---

## 12. Project Layout

```
combined_app/
├── app.py              # main: starts three threads, handles shutdown
├── capture.py          # capture thread: camera → frame_queue
├── inference.py        # inference thread: cascade orchestration, two-path split
├── renderer.py         # render thread: compositor, effects tick, cv2.imshow
├── effects.py          # ambient noise flow field, pre-allocated buffer
├── overlays.py         # all cv2 drawing: contours, meshes, labels, HUD
├── tone.py             # precomputed LUT, white balance calibration
├── models/             # inference backend wrappers (one file per model)
│   ├── objectification.py
│   ├── face_det.py
│   ├── face_parts.py
│   ├── emotion.py
│   ├── hand_seg.py
│   └── gesture.py
└── config.py           # thresholds, paths, display size, LED config
```

---

## 13. Success Criteria

- Fullscreen display at ≥30fps on Mac MPS
- Mirror feel: dark background, horizontally flipped, overlays only appear when models fire
- Background effects visually present and smooth, not distracting, ≤5ms render cost
- All six model wrappers integrated with clean interfaces
- Architecture doc reflects current reality, no stale references
- Pi path documented but no Pi-specific code in v1 scope
