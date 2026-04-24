import colorsys
from pathlib import Path

REPO = Path(__file__).parent.parent

HAND_SEG_CKPT    = REPO / "HAND_JOB/hand_seg/checkpoints/hand_seg_v7.pt"
GESTURE_CKPT     = REPO / "HAND_JOB/gesture/checkpoints/gesture_v7.pt"
FACE_DET_CKPT    = REPO / "FACE_JOB/face_det/checkpoints/face_det_v1.pt"
FACE_PARTS_CKPT  = REPO / "FACE_JOB/face_parts/checkpoints/face_parts_v1.pt"
EMOTION_CKPT     = REPO / "FACE_JOB/emotion/checkpoints/emotion_v1.pt"

WEBCAM_INDEX = 0
FRAME_W      = 1280
FRAME_H      = 720

# ── Hand inference ────────────────────────────────────────────────────────────
SEG_THRESHOLD  = 0.10
CONF_THRESHOLD = 0.60   # label only shown above this
HAND_MIN_AREA  = 0.0005
MASK_EMA_ALPHA = 0.15
MIN_CC_AREA_PX = 80
VOTE_WINDOW    = 6

# ── Face inference ────────────────────────────────────────────────────────────
FACE_DET_SCORE_THR = 0.30
FACE_DET_IOU_THR   = 0.45
FACE_DET_IMG       = 320
FACE_PARTS_IMG     = 192
EMOTION_IMG        = 64
FACE_EMA_ALPHA     = 0.20
FACE_CONF_THR      = 0.25
MAX_FACES          = 4

# ── Hand colors: same hue-cycle as original HAND_JOB ─────────────────────────
GESTURE_COLORS_BGR = [
    tuple(int(c * 255) for c in reversed(colorsys.hsv_to_rgb(i / 18, 0.8, 0.9)))
    for i in range(18)
]

# ── Face colors: cool independent palette (one per emotion) ───────────────────
# order: happy sad neutral surprise anger fear disgust
EMOTION_COLORS_BGR = [
    (220, 190,  20),   # happy     — bright cyan
    (200,  40, 180),   # sad       — deep violet-blue
    (160, 120, 200),   # neutral   — soft purple
    (255, 220,   0),   # surprise  — electric cyan
    ( 60,   0, 200),   # anger     — indigo
    (180,   0, 220),   # fear      — magenta-purple
    ( 80, 200, 180),   # disgust   — teal
]

# ── Valence weights (kept for potential future use) ───────────────────────────
GESTURE_VALENCE = {
    "like": 0.9, "peace": 0.8, "ok": 0.8, "two_up": 0.6,
    "palm": 0.3, "one": 0.2, "call": 0.1, "stop": -0.2,
    "mute": 0.0, "three": 0.0, "four": 0.0, "three2": 0.0,
    "fist": -0.3, "stop_inverted": -0.3, "rock": -0.1,
    "peace_inverted": -0.4, "dislike": -0.9,
    "middle_finger": -0.9, "two_up_inverted": -0.9,
}
EMOTION_VALENCE = {
    "happy": 0.9, "surprise": 0.4, "neutral": 0.0,
    "sad": -0.6, "fear": -0.7, "anger": -0.9, "disgust": -0.8,
}

# face-parts class indices
FACE_CLASS_BACKGROUND = 0
FACE_CLASS_EYE_L      = 1
FACE_CLASS_EYE_R      = 2
FACE_CLASS_MOUTH      = 3
FACE_CLASS_SKIN       = 4
