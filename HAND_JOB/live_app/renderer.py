import cv2
import numpy as np
from scipy.spatial import Delaunay
from live_app.config import CONF_THRESHOLD, GESTURE_COLORS_BGR

FILL_OPACITY    = 0.55
MESH_OPACITY    = 0.92
GLOW_WIDTH      = 2
CONTOUR_POINTS  = 10
INTERIOR_POINTS = 3
NETWORK_DIST    = 160
NETWORK_MAX     = 14
BLOB_MIN_AREA   = 800

MESH_BASE_W     = 1
NET_BASE_W      = 1
MAX_THICK_MULT  = 2.0
MESH_PTS_UPDATE = 0

_feedback_buf: np.ndarray | None = None
_pts_cache: np.ndarray | None = None
_pts_age: int = 0


# ── helpers ───────────────────────────────────────────────────────────────────

def _sample_contour(mask: np.ndarray, n: int):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea).squeeze()
    if cnt.ndim < 2 or len(cnt) < 3:
        return None
    idx = np.linspace(0, len(cnt) - 1, n, dtype=int)
    return cnt[idx]


def _interior_points(mask: np.ndarray, n: int) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) < n:
        return np.empty((0, 2), dtype=int)
    cx, cy = int(xs.mean()), int(ys.mean())
    idx    = np.random.choice(len(xs), size=min(n, len(xs)), replace=False)
    pts    = np.stack([xs[idx], ys[idx]], axis=1)
    return np.vstack([pts, [[cx, cy]]])


def _dim(color, f):
    return tuple(min(255, int(c * f)) for c in color)


def _edge_thickness(p1, p2, dist_map: np.ndarray, base: int) -> int:
    """Scale line thickness by how deep inside the mask the edge midpoint sits."""
    if dist_map is None:
        return base
    mx = int((p1[0] + p2[0]) / 2)
    my = int((p1[1] + p2[1]) / 2)
    h, w = dist_map.shape
    if not (0 <= my < h and 0 <= mx < w):
        return base
    max_d = dist_map.max()
    if max_d == 0:
        return base
    t = float(dist_map[my, mx]) / max_d
    mult = 1.0 + (MAX_THICK_MULT - 1.0) * t
    return max(1, int(base * mult))


def _bloom(canvas, color, mask, fade):
    glow = np.zeros_like(canvas)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return canvas
    cv2.drawContours(glow, contours, -1, color, -1)
    glow = cv2.GaussianBlur(glow, (61, 61), 0)
    return cv2.addWeighted(canvas, 1.0, glow, 0.35 * fade, 0)


# ── main draw ─────────────────────────────────────────────────────────────────

def draw_mesh(frame: np.ndarray, mask: np.ndarray,
              confidence: float, gesture_idx: int, fade: float = 1.0) -> np.ndarray:
    global _feedback_buf, _pts_cache, _pts_age
    out      = frame.copy()
    eff_fade = fade * max(confidence, 0.15)
    if eff_fade < 0.01 or mask is None:
        _feedback_buf = None
        _pts_cache    = None
        _pts_age      = 0
        return out

    h, w  = frame.shape[:2]
    color = GESTURE_COLORS_BGR[gesture_idx % len(GESTURE_COLORS_BGR)]

    if _feedback_buf is not None and _feedback_buf.shape == (h, w, 3):
        M      = cv2.getRotationMatrix2D((w / 2, h / 2), 0.0, 0.97)
        warped = cv2.warpAffine(_feedback_buf, M, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
        out = cv2.addWeighted(out, 1.0, warped, 0.35 * eff_fade, 0)

    out = _bloom(out, _dim(color, 0.7), mask, eff_fade)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        fill_layer = out.copy()
        cv2.drawContours(fill_layer, contours, -1, color, -1)
        cv2.addWeighted(fill_layer, FILL_OPACITY * eff_fade, out, 1 - FILL_OPACITY * eff_fade, 0, out)

    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    contour_pts  = _sample_contour(mask, CONTOUR_POINTS)
    if contour_pts is not None and len(contour_pts) >= 3:
        if _pts_cache is None or _pts_age >= MESH_PTS_UPDATE:
            interior_pts = _interior_points(mask, INTERIOR_POINTS)
            all_pts      = np.vstack([contour_pts, interior_pts]) if len(interior_pts) else contour_pts
            _pts_cache   = all_pts
            _pts_age     = 0
        else:
            all_pts  = _pts_cache
            _pts_age = _pts_age + 1

        try:
            tri        = Delaunay(all_pts)
            fill_layer = out.copy()
            mesh_layer = out.copy()
            for simplex in tri.simplices:
                p1 = tuple(all_pts[simplex[0]])
                p2 = tuple(all_pts[simplex[1]])
                p3 = tuple(all_pts[simplex[2]])
                cx_ = int((p1[0] + p2[0] + p3[0]) // 3)
                cy_ = int((p1[1] + p2[1] + p3[1]) // 3)
                if not (0 <= cy_ < h and 0 <= cx_ < w):
                    continue
                # check all three edge midpoints inside mask to avoid filling
                # triangles that merely have a centroid near the boundary
                mids = [
                    (int((p1[0]+p2[0])//2), int((p1[1]+p2[1])//2)),
                    (int((p2[0]+p3[0])//2), int((p2[1]+p3[1])//2)),
                    (int((p3[0]+p1[0])//2), int((p3[1]+p1[1])//2)),
                ]
                inside = (mask[cy_, cx_] > 0 and
                          all(0 <= my < h and 0 <= mx < w and mask[my, mx] > 0
                              for mx, my in mids))
                if inside:
                    pts_tri = np.array([p1, p2, p3], dtype=np.int32)
                    cv2.fillPoly(fill_layer, [pts_tri], color)
                    cv2.line(mesh_layer, p1, p2, (255, 255, 255), _edge_thickness(p1, p2, dist_map, MESH_BASE_W))
                    cv2.line(mesh_layer, p2, p3, (255, 255, 255), _edge_thickness(p2, p3, dist_map, MESH_BASE_W))
                    cv2.line(mesh_layer, p3, p1, (255, 255, 255), _edge_thickness(p3, p1, dist_map, MESH_BASE_W))
            cv2.addWeighted(fill_layer, 0.22 * eff_fade, out, 1 - 0.22 * eff_fade, 0, out)
            cv2.addWeighted(mesh_layer, MESH_OPACITY * eff_fade, out, 1 - MESH_OPACITY * eff_fade, 0, out)
        except Exception:
            pass

        pts_f = all_pts.astype(np.float32)
        pairs = []
        for i in range(len(pts_f)):
            for j in range(i + 1, len(pts_f)):
                d = float(np.linalg.norm(pts_f[i] - pts_f[j]))
                if d < NETWORK_DIST:
                    pairs.append((d, i, j))
        pairs.sort(key=lambda t: -t[0])
        net_layer = out.copy()
        for d, i, j in pairs[:NETWORK_MAX]:
            brightness = 0.3 + 0.5 * (d / NETWORK_DIST)
            c = tuple(min(255, int(255 * brightness)) for _ in range(3))
            p1, p2 = tuple(all_pts[i]), tuple(all_pts[j])
            cv2.line(net_layer, p1, p2, c, _edge_thickness(p1, p2, dist_map, NET_BASE_W))
        cv2.addWeighted(net_layer, 0.70 * eff_fade, out, 1 - 0.70 * eff_fade, 0, out)

        node_layer = out.copy()
        for pt in all_pts:
            x_, y_ = int(pt[0]), int(pt[1])
            if 0 <= x_ < w and 0 <= y_ < h:
                cv2.circle(node_layer, (x_, y_), 3, color, -1)
                cv2.circle(node_layer, (x_, y_), 1, (255, 255, 255), -1)
        cv2.addWeighted(node_layer, 0.55 * eff_fade, out, 1 - 0.55 * eff_fade, 0, out)

    if contours:
        max_d = float(dist_map.max()) if dist_map is not None else 0.0
        h_approx = max(frame.shape[:2])
        depth_t = min(1.0, max_d / (h_approx * 0.15))   # 0..1
        glow_w = max(GLOW_WIDTH, int(GLOW_WIDTH * (1.0 + (MAX_THICK_MULT - 1.0) * depth_t)))
        glow_layer = out.copy()
        cv2.drawContours(glow_layer, contours, -1, color, glow_w * 2)
        cv2.addWeighted(glow_layer, 0.6 * eff_fade, out, 1 - 0.6 * eff_fade, 0, out)

    _feedback_buf = cv2.subtract(out, frame)
    return out


def draw_blobs(frame: np.ndarray, fg_mask: np.ndarray,
               hand_mask: np.ndarray) -> np.ndarray:
    """Light white mesh over any moving non-hand blobs."""
    if fg_mask is None:
        return frame
    out = frame.copy()
    h, w = frame.shape[:2]

    other = fg_mask.copy()
    if hand_mask is not None:
        other[hand_mask > 0] = 0

    contours, _ = cv2.findContours(other, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < BLOB_MIN_AREA:
            continue
        cnt_sq = cnt.squeeze()
        if cnt_sq.ndim < 2 or len(cnt_sq) < 3:
            continue
        n   = min(12, len(cnt_sq))
        idx = np.linspace(0, len(cnt_sq) - 1, n, dtype=int)
        pts = cnt_sq[idx]
        try:
            tri   = Delaunay(pts)
            layer = out.copy()
            for simplex in tri.simplices:
                p1, p2, p3 = tuple(pts[simplex[0]]), tuple(pts[simplex[1]]), tuple(pts[simplex[2]])
                cv2.line(layer, p1, p2, (200, 200, 200), 1)
                cv2.line(layer, p2, p3, (200, 200, 200), 1)
                cv2.line(layer, p3, p1, (200, 200, 200), 1)
            cv2.addWeighted(layer, 0.18, out, 0.82, 0, out)
        except Exception:
            pass
    return out


# ── UI ────────────────────────────────────────────────────────────────────────

def _card(frame, x, y, w, h, alpha=0.6):
    blurred = cv2.GaussianBlur(frame[y:y+h, x:x+w], (21, 21), 0)
    overlay = frame.copy()
    overlay[y:y+h, x:x+w] = blurred
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (10, 10, 10), -1)
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.rectangle(result, (x, y), (x+w, y+h), (50, 50, 50), 1)
    return result


def draw_ui(frame, gesture, confidence, second, second_conf, fps, present):
    out    = frame.copy()
    fh, fw = out.shape[:2]
    font   = cv2.FONT_HERSHEY_DUPLEX

    out = _card(out, 0, 0, fw, 44, alpha=0.65)
    title = "HAND JOB"
    (tw, _), _ = cv2.getTextSize(title, font, 0.85, 1)
    cv2.putText(out, title, ((fw - tw) // 2, 30), font, 0.85, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"{fps:.0f} fps", (fw - 80, 30), font, 0.42, (100, 100, 100), 1, cv2.LINE_AA)

    bh, bx = 56, 16
    by     = fh - bh
    bw     = fw - 32
    out    = _card(out, bx, by, bw, bh, alpha=0.65)

    if present and gesture and confidence >= CONF_THRESHOLD:
        label = gesture.upper().replace("_", " ")
        pct   = f"{confidence * 100:.0f}%"
        (lw, _), _ = cv2.getTextSize(label + "   ", font, 0.80, 1)
        (fw2, _), _ = cv2.getTextSize(label + "   " + pct, font, 0.80, 1)
        tx = bx + (bw - fw2) // 2
        cv2.putText(out, label, (tx, by + 30), font, 0.80, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, pct, (tx + lw, by + 30), font, 0.80, (160, 160, 160), 1, cv2.LINE_AA)
        bar_w = bw // 3
        bar_x = bx + (bw - bar_w) // 2
        bar_y = by + 40
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + 5), (40, 40, 40), -1)
        fill_w = int(bar_w * confidence)
        if fill_w > 0:
            bar_color = (int(60 + confidence * 195), int(180 + confidence * 75), int(60 + confidence * 60))
            cv2.rectangle(out, (bar_x, bar_y), (bar_x + fill_w, bar_y + 5), bar_color, -1)
    else:
        msg = "no hand"
        (tw, _), _ = cv2.getTextSize(msg, font, 0.70, 1)
        cv2.putText(out, msg, (bx + (bw - tw) // 2, by + 34), font, 0.70, (80, 80, 80), 1, cv2.LINE_AA)

    return out
