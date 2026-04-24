# HAND JOB

Real-time hand gesture recognition system running entirely on your laptop. No cloud, no API — just your camera and your GPU.

## What it does

A webcam feeds live video into two AI models running back-to-back. The first finds your hand in the frame and draws a mask over it. The second classifies which of 18 gestures you're making. The result updates dozens of times per second.

Gesture labels, confidence scores, and the hand mask stream out over OSC so any software — TouchDesigner, Ableton, Max/MSP — can receive and react to them live. Video output goes over NDI for compositing into any live video setup.

A wireframe mesh tracks the shape of your hand, glows in a color unique to each gesture, and leaves a fading trail as you move. Any other moving objects in frame get their own faint wireframe too.

## Gestures

call, dislike, fist, four, like, ok, one, palm, peace, peace_inverted, rock, stop, stop_inverted, three, three2, two_up, middle_finger, background

## Requirements

- macOS (Apple Silicon)
- Python 3.11+
- Webcam

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Collect your own training data
python3 collect.py

# Review and delete bad captures
python3 review.py

# Train both models overnight
sudo .venv/bin/python3 train_all.py

# Run the live app
python3 -m live_app.app
```

## OSC Output

All data sent to `127.0.0.1:9000`

| Address | Type | Description |
|---|---|---|
| `/gesture` | string | Current gesture label |
| `/confidence` | float | 0.0 – 1.0 |
| `/present` | bool | Hand detected |
| `/gesture_idx` | int | Class index |
| `/mesh` | float[] | Contour point coords |

## NDI

Source name: `LEARNIN_MACHINES` — receive in TouchDesigner, OBS, or any NDI-compatible software.
