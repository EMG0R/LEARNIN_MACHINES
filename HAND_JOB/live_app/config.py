import colorsys
from pathlib import Path

BASE = Path(__file__).parent.parent

SEG_CKPT     = BASE / "hand_seg/checkpoints/hand_seg_v1.pt"
GESTURE_CKPT = BASE / "gesture/checkpoints/gesture_v1_wide96.pt"

WEBCAM_INDEX = 0
FRAME_W      = 1280
FRAME_H      = 720

OSC_IP   = "127.0.0.1"
OSC_PORT = 9000

NDI_SOURCE_NAME = "LEARNIN_MACHINES"

SEG_THRESHOLD  = 0.5
CONF_THRESHOLD = 0.4

FILL_OPACITY   = 0.15
MESH_OPACITY   = 0.10
GLOW_WIDTH     = 2
CONTOUR_POINTS = 40

GESTURE_COLORS_BGR = [
    tuple(int(c * 255) for c in reversed(colorsys.hsv_to_rgb(i / 18, 0.8, 0.9)))
    for i in range(18)
]
