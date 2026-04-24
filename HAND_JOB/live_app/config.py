import colorsys
from pathlib import Path

BASE = Path(__file__).parent.parent

SEG_CKPT     = BASE / "hand_seg/checkpoints/hand_seg_v7.pt"
GESTURE_CKPT = BASE / "gesture/checkpoints/gesture_v7.pt"

WEBCAM_INDEX = 0
FRAME_W      = 1280
FRAME_H      = 720

OSC_IP   = "127.0.0.1"
OSC_PORT = 9000

NDI_SOURCE_NAME = "LEARNIN_MACHINES"

SEG_THRESHOLD  = 0.10
CONF_THRESHOLD = 0.25
HAND_MIN_AREA  = 0.0005

# --- Temporal smoothing ---
MASK_EMA_ALPHA  = 0.15
MIN_CC_AREA_PX  = 80      # catch smaller/distant hands
VOTE_WINDOW     = 6

GESTURE_COLORS_BGR = [
    tuple(int(c * 255) for c in reversed(colorsys.hsv_to_rgb(i / 18, 0.8, 0.9)))
    for i in range(18)
]
