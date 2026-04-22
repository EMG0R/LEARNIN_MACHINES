import cv2
import numpy as np
from scipy.spatial import Delaunay
from live_app.config import (
    CONF_THRESHOLD, FILL_OPACITY, MESH_OPACITY, GLOW_WIDTH,
    CONTOUR_POINTS, GESTURE_COLORS_BGR,
)


def _sample_contour(mask: np.ndarray, n: int):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea).squeeze()
    if cnt.ndim < 2 or len(cnt) < 3:
        return None
    idx = np.linspace(0, len(cnt) - 1, n, dtype=int)
    return cnt[idx]


def draw_mesh(frame: np.ndarray, mask: np.ndarray,
              confidence: float, gesture_idx: int) -> np.ndarray:
    out = frame.copy()
    if confidence < CONF_THRESHOLD or mask is None:
        return out

    pts = _sample_contour(mask, CONTOUR_POINTS)
    if pts is None:
        return out

    color = GESTURE_COLORS_BGR[gesture_idx % len(GESTURE_COLORS_BGR)]
    h, w  = mask.shape

    fill_layer = out.copy()
    cv2.drawContours(fill_layer, [pts.reshape(-1, 1, 2)], -1, color, -1)
    cv2.addWeighted(fill_layer, FILL_OPACITY, out, 1 - FILL_OPACITY, 0, out)

    if len(pts) >= 3:
        tri        = Delaunay(pts)
        mesh_layer = out.copy()
        for simplex in tri.simplices:
            p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            cx = int((p1[0] + p2[0] + p3[0]) // 3)
            cy = int((p1[1] + p2[1] + p3[1]) // 3)
            if 0 <= cy < h and 0 <= cx < w and mask[cy, cx] > 0:
                cv2.line(mesh_layer, tuple(p1), tuple(p2), (255, 255, 255), 1)
                cv2.line(mesh_layer, tuple(p2), tuple(p3), (255, 255, 255), 1)
                cv2.line(mesh_layer, tuple(p3), tuple(p1), (255, 255, 255), 1)
        cv2.addWeighted(mesh_layer, MESH_OPACITY, out, 1 - MESH_OPACITY, 0, out)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        glow_layer = out.copy()
        cv2.drawContours(glow_layer, contours, -1, (255, 255, 255), GLOW_WIDTH)
        cv2.addWeighted(glow_layer, float(confidence), out, 1 - float(confidence), 0, out)

    return out


def _frosted_card(frame: np.ndarray, x: int, y: int, w: int, h: int,
                  alpha: float = 0.55) -> np.ndarray:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def draw_ui(frame: np.ndarray, gesture, confidence: float,
            second, second_conf: float, fps: float, present: bool) -> np.ndarray:
    out    = frame.copy()
    fh, fw = out.shape[:2]
    font   = cv2.FONT_HERSHEY_DUPLEX
    small  = 0.45
    med    = 0.60

    px, py, pw, ph = 16, fh - 90, 260, 80
    out = _frosted_card(out, px, py, pw, ph)

    if present and gesture and confidence >= CONF_THRESHOLD:
        cv2.putText(out, gesture.upper(),
                    (px + 10, py + 26), font, med, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, f"{confidence * 100:.0f}%",
                    (px + pw - 55, py + 26), font, small, (200, 200, 200), 1, cv2.LINE_AA)
        bar_x, bar_y, bar_w, bar_h = px + 10, py + 36, pw - 20, 8
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(out, (bar_x, bar_y),
                      (bar_x + int(bar_w * confidence), bar_y + bar_h), (220, 220, 220), -1)
        if second:
            cv2.putText(out, f"{second}  {second_conf*100:.0f}%",
                        (px + 10, py + 66), font, small, (140, 140, 140), 1, cv2.LINE_AA)
    else:
        cv2.putText(out, "—", (px + 10, py + 40), font, med, (100, 100, 100), 1, cv2.LINE_AA)

    sw, sh = 180, 56
    sx, sy = fw - sw - 16, fh - sh - 16
    out = _frosted_card(out, sx, sy, sw, sh)
    dot_color = (80, 220, 80) if present else (80, 80, 80)
    cv2.circle(out, (sx + 16, sy + 20), 6, dot_color, -1)
    cv2.putText(out, f"{fps:.1f} fps",
                (sx + 30, sy + 25), font, small, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, "hand present" if present else "no hand",
                (sx + 10, sy + 46), font, small, (160, 160, 160), 1, cv2.LINE_AA)

    return out
