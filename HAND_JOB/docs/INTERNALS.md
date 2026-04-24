# HAND JOB — Internal Architecture

Last updated: 2026-04-22 by Claude Sonnet 4.6

---

## Model Architecture

### Segmentation — `_HandUNet`
- 4-block encoder/decoder UNet, ~1.9M params
- Input: 192×192 RGB, ImageNet norm
- Output: single-channel sigmoid prob map, resized to frame resolution
- Trained on FreiHAND + heavy augmentation (JPEG, motion blur, gamma, noise, background replacement)
- Best checkpoint: `hand_seg/checkpoints/hand_seg_v7.pt` — val IoU 0.9467
- `AUG=heavy` lowers output probabilities vs clean training — threshold must be 0.10–0.15, not 0.3+

### Gesture — `_Wide` CNN
- 4-block wide CNN, ~1.3M params
- Input: 96×96 RGB crop of hand region, ImageNet norm
- Output: softmax over 18 classes
- Trained on HaGRID 30k + HaGRID 500k (~430k studio images) + 8,869 webcam captures
- User-based train/val split to prevent data leakage across HaGRID users
- Best checkpoint: `gesture/checkpoints/gesture_v7.pt` — val F1 0.9896, test F1 0.9897
- `two_up_inverted` prob is merged into `middle_finger` at inference (no retrain)

---

## Live Pipeline (`live_app/app.py`)

```
webcam frame (1280×720)
    │
    ├─► BackgroundSubtractorMOG2 → fg_mask (moving pixels)
    │
    ├─► 3 overlapping square crops (720×720): left / center / right
    │       └─► run_seg_prob_batch() → 3 prob maps (single batched forward pass)
    │               └─► postprocess_mask() × 3 (EMA smooth, threshold, morph close, CC filter)
    │                       └─► bitwise_and with fg_mask (remove static background)
    │                               └─► composite → mask_disp (full frame)
    │
    ├─► hand_present(mask_disp) → bool
    │
    ├─► [if present] run_gesture(best_crop, best_mask) → probs
    │       └─► GestureSmoother (confidence²-weighted, 6-frame window, hard flush on switch)
    │
    ├─► mesh_fade (0→1 in 7 frames, 1→0 in 40 frames)
    │
    ├─► draw_blobs() — faint white mesh on non-hand moving objects
    ├─► draw_mesh() — hand mesh with bloom, fill, delaunay, constellation, feedback trail
    ├─► draw_ui()  — title bar + bottom status bar
    │
    ├─► OSCSender → 127.0.0.1:9000
    └─► NDISender → "LEARNIN_MACHINES"
```

---

## Key Config (`live_app/config.py`)

| Param | Value | Notes |
|---|---|---|
| `SEG_THRESHOLD` | 0.10 | Low because AUG=heavy shifts prob distribution down |
| `CONF_THRESHOLD` | 0.25 | Show gesture at lower confidence |
| `HAND_MIN_AREA` | 0.0005 | Fraction of frame — catches distant/small hands |
| `MASK_EMA_ALPHA` | 0.15 | Slow mask decay — mesh lingers after hand leaves |
| `MIN_CC_AREA_PX` | 80 | Min connected component to keep in mask |
| `VOTE_WINDOW` | 6 | Frames for gesture smoother — fast switching |

---

## Training (`train_all.py`)

Single orchestrator, runs both stages as subprocesses with thermal protection.

- **Stage 1** — seg: `hand_seg/train.py` with `AUG=heavy IMG_SIZE=192 EPOCHS=35`
- **Stage 2** — gesture: `gesture/train_v7.py` with `IMG_SIZE=96 EPOCHS=40`
- Thermal watchdog: `powermetrics` polls every 3s in background thread. SIGSTOP on Trapping/Sleeping, SIGCONT on Nominal. Requires `sudo`.
- Auto-patches `live_app/config.py` checkpoint paths when done.

Run with: `sudo .venv/bin/python3 train_all.py`

---

## Data

| Source | Type | Count |
|---|---|---|
| FreiHAND | Seg ground truth | ~130k |
| HaGRID 30k | Gesture annotations | ~25k |
| HaGRID 500k | Gesture annotations | ~405k |
| Webcam (`data/webcam/`) | Self-collected, auto-labeled | ~8,869 |

Webcam data routes entirely to training split (never val). User-based split on HaGRID prevents leakage.

`middle_finger` and `background` classes have no HaGRID data — webcam only.

---

## Data Collection

```bash
python3 collect.py              # all classes
python3 collect.py --only rock  # specific class
python3 review.py               # browse/delete captures
```

SPACE = capture, BACKSPACE = delete last, N/P = next/prev class, Q = quit.

---

## Known Issues / Future Work

- `two_up`, `two_up_inverted`, `three2`, `stop_inverted` have few/zero webcam captures — weakest in real use
- Background masking (zero non-hand pixels before gesture inference) not yet implemented — would improve accuracy
- Gesture model saw studio data for most classes; more self-collected data in real usage conditions would close the domain gap
- EgoHands dataset was unavailable at training time (IU server migration) — worth adding if accessible
